"""
SAM2ACT Agent Flask API服务

本脚本提供Flask服务，将训练好的SAM2ACT模型的agent.act方法暴露为HTTP API。
主要功能：
1. 启动Flask服务器
2. 初始化agent模型
3. 提供/act端点用于动作预测
4. 提供/health端点用于健康检查

使用场景：
- 将agent模型部署为Web服务
- 通过HTTP API调用agent进行动作预测
- 支持远程客户端调用
"""

import os
import sys
import torch
import numpy as np
import socket
import argparse
from typing import Any, Dict as DictType

# ============================================================================
# 路径配置：添加 sam2act 目录到 Python 路径
# ============================================================================
# 获取当前文件的绝对路径，然后获取 sam2act 目录
# 当前文件位于 sam2act/historybench_eval/ 子目录下
# 需要将 sam2act 目录添加到 sys.path 才能让 mvt 模块被正确导入
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sam2act_dir = os.path.dirname(current_file_dir)  # 获取 sam2act 目录
if sam2act_dir not in sys.path:
    sys.path.insert(0, sam2act_dir)

from sam2act.eval import load_agent
from rlbench.backend.observation import Observation
import clip
import torch.nn.functional as F
from pyrep.objects import VisionSensor

# Plotly for pointcloud visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: Plotly未安装，点云可视化功能将不可用。安装: pip install plotly")

# Flask imports for API service
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("错误: Flask未安装，请先安装: pip install flask")
    sys.exit(1)


def get_local_ip():
    """
    获取本机IP地址
    
    尝试连接到外部地址来获取本机IP，如果失败则返回localhost
    """
    try:
        # 创建一个UDP socket（不需要实际连接）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到外部地址（不会实际发送数据）
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 如果失败，返回localhost
        return "127.0.0.1"


def is_port_in_use(host, port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_port(host, start_port, max_attempts=10):
    """查找可用端口，从start_port开始尝试"""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(host, port):
            return port
    return None


# ============================================================================
# Flask API Service for Agent.act
# ============================================================================

# 全局agent实例
agent_instance = None
device_id = 0

# 创建Flask应用
app = Flask(__name__)


# ============================================================================
# 相机参数转换函数
# ============================================================================
def convert_camera_matrix_maniskill_to_coppeliasim(
    extrinsics_opencv: np.ndarray,
    intrinsics_opencv: np.ndarray
) -> tuple:
    """
    将 Maniskill/OpenCV 格式的相机参数转换为 CoppeliaSim 格式
    
    参数:
        extrinsics_opencv: (4, 4) float32, OpenCV 格式外参（世界→相机）
        intrinsics_opencv: (3, 3) float32, OpenCV 格式内参
    
    返回:
        extrinsics_coppeliasim: (4, 4) float32, CoppeliaSim 格式外参（相机→世界）
        intrinsics_coppeliasim: (3, 3) float32, CoppeliaSim 格式内参
    """
    # 1. 转换外参：OpenCV 是"世界→相机"，CoppeliaSim 需要"相机→世界"
    # 直接求逆矩阵即可
    extrinsics_coppeliasim = np.linalg.inv(extrinsics_opencv)
    
    # 2. 内参格式相同，直接返回（OpenCV 和 CoppeliaSim 都使用标准针孔相机模型）
    intrinsics_coppeliasim = intrinsics_opencv.copy()
    
    return extrinsics_coppeliasim, intrinsics_coppeliasim


def extract_obs(obs_dict: DictType[str, Any], curr_idx: int, lang_goal: str | None, 
                episode_length: int = 25, time_in_state: bool = True):
    """
    从序列化的观察字典中提取特征字典
    
    Args:
        obs_dict: 序列化的观察字典（从Observation对象转换而来）
        curr_idx: 当前时间步索引
        lang_goal: 语言目标描述字符串
        episode_length: episode长度，默认25
        time_in_state: 是否在状态中包含时间信息，默认True
        
    Returns:
        obs_dict: 包含提取特征的字典，格式与agent.act所需格式一致
    """
    # 从字典重建Observation对象（简化版，只包含需要的字段）
    obs = Observation(
        left_shoulder_rgb=None, left_shoulder_depth=None, left_shoulder_mask=None, left_shoulder_point_cloud=None,
        right_shoulder_rgb=None, right_shoulder_depth=None, right_shoulder_mask=None, right_shoulder_point_cloud=None,
        overhead_rgb=None, overhead_depth=None, overhead_mask=None, overhead_point_cloud=None,
        wrist_rgb=None, wrist_depth=None, wrist_mask=None, wrist_point_cloud=None,
        front_rgb=None, front_depth=None, front_mask=None, front_point_cloud=None,
        joint_velocities=None, joint_positions=None, joint_forces=None,
        gripper_open=None, gripper_pose=None, gripper_matrix=None, gripper_joint_positions=None, gripper_touch_forces=None,
        task_low_dim_state=None, ignore_collisions=None, misc=None
    )
    
    # 从字典恢复Observation对象的属性
    for key, val in obs_dict.items():
        if hasattr(obs, key):
            # 将列表转换回numpy数组
            if isinstance(val, list):
                val = np.array(val)
            setattr(obs, key, val)
    
    # 确保misc存在
    if 'misc' in obs_dict:
        obs.misc = obs_dict['misc']
    else:
        obs.misc = {}
    
    # ============================================================================
    # 处理原始输入数据 (Raw Inputs Processing)
    # ============================================================================
    
    # --- 1. RGB 图像 ---
    if 'image' in obs_dict:
        obs.front_rgb = np.array(obs_dict['image'])
        # 假设输入已经是128x128，或者不需要处理
        
    if 'wrist_image' in obs_dict:
        obs.wrist_rgb = np.array(obs_dict['wrist_image'])

    # --- 2. 深度图像 ---
    if 'base_camera_depth' in obs_dict:
        depth = np.array(obs_dict['base_camera_depth']).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        # 降采样到 128x128
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        obs.front_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
    
    if 'wrist_camera_depth' in obs_dict:
        depth = np.array(obs_dict['wrist_camera_depth']).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        # 降采样到 128x128
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        obs.wrist_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

    # --- 3. 机器人状态 ---
    if 'robot_endeffector_p' in obs_dict and 'robot_endeffector_q' in obs_dict:
        gripper_pos = np.array(obs_dict['robot_endeffector_p']).flatten()
        gripper_quat = np.array(obs_dict['robot_endeffector_q']).flatten()
        obs.gripper_pose = np.concatenate([gripper_pos, gripper_quat])

    if 'action' in obs_dict:
        action = np.array(obs_dict['action'])
        obs.gripper_open = float(action[-1])
        
    # 设置 gripper_joint_positions (如果没有直接提供)
    if obs.gripper_joint_positions is None and obs.gripper_open is not None:
        if obs.gripper_open > 0:
            obs.gripper_joint_positions = np.array([0.04, 0.04], dtype=np.float32)
        else:
            obs.gripper_joint_positions = np.array([0.0, 0.0], dtype=np.float32)
            
    if obs.ignore_collisions is None:
        obs.ignore_collisions = 0
            
    # --- 4. 相机参数 ---
    if 'base_camera_extrinsic_opencv' in obs_dict and 'base_camera_intrinsic_opencv' in obs_dict:
        ext_opencv = np.array(obs_dict['base_camera_extrinsic_opencv'])
        intr_opencv = np.array(obs_dict['base_camera_intrinsic_opencv'])
        if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
        if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]
        
        if ext_opencv.shape == (3, 4):
            ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
            
        ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
            ext_opencv, intr_opencv
        )
        obs.misc['front_camera_extrinsics'] = ext_coppeliasim
        obs.misc['front_camera_intrinsics'] = intr_coppeliasim
        
    if 'wrist_camera_extrinsic_opencv' in obs_dict and 'wrist_camera_intrinsic_opencv' in obs_dict:
        ext_opencv = np.array(obs_dict['wrist_camera_extrinsic_opencv'])
        intr_opencv = np.array(obs_dict['wrist_camera_intrinsic_opencv'])
        if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
        if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]

        if ext_opencv.shape == (3, 4):
            ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])

        ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
            ext_opencv, intr_opencv
        )
        obs.misc['wrist_camera_extrinsics'] = ext_coppeliasim
        obs.misc['wrist_camera_intrinsics'] = intr_coppeliasim

    # --- 5. 生成点云 ---
    if obs.front_point_cloud is None and 'front_camera_extrinsics' in obs.misc and obs.front_depth is not None:
        obs.front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs.front_depth,
            obs.misc['front_camera_extrinsics'],
            obs.misc['front_camera_intrinsics']
        )
        
    if obs.wrist_point_cloud is None and 'wrist_camera_extrinsics' in obs.misc and obs.wrist_depth is not None:
        obs.wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs.wrist_depth,
            obs.misc['wrist_camera_extrinsics'],
            obs.misc['wrist_camera_intrinsics']
        )
    
    # 填充缺失的点云
    if obs.front_point_cloud is None:
        obs.front_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)
    if obs.wrist_point_cloud is None:
        obs.wrist_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)
    
    # 备份需要恢复的值
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    
    # 修改观察对象，移除不需要的属性
    obs.joint_velocities = None
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    
    # 限制夹爪关节位置在合理范围内 [0, 0.04]
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)
    
    # 提取观察特征
    channels_last = False
    obs_dict_extracted = vars(obs)
    obs_dict_extracted = {k: v for k, v in obs_dict_extracted.items() if v is not None}
    
    # 移除机器人状态相关的键
    ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']
    obs_dict_extracted = {k: v for k, v in obs_dict_extracted.items() if k not in ROBOT_STATE_KEYS}
    
    # 处理图像和深度数据的维度
    if not channels_last:
        new_obs_dict = {}
        for k, v in obs_dict_extracted.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 3:  # RGB图像或点云 (H, W, 3)
                    new_obs_dict[k] = np.transpose(v, [2, 0, 1])
                elif v.ndim == 2:  # 深度图 (H, W)
                    new_obs_dict[k] = np.expand_dims(v, 0)
                else:
                    new_obs_dict[k] = np.expand_dims(v, 0) if v.ndim == 0 else v
            else:
                new_obs_dict[k] = v
        obs_dict_extracted = new_obs_dict
    else:
        new_obs_dict = {}
        for k, v in obs_dict_extracted.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 2:
                    new_obs_dict[k] = np.expand_dims(v, -1)
                else:
                    new_obs_dict[k] = v
            else:
                new_obs_dict[k] = v
        obs_dict_extracted = new_obs_dict
    
    # 添加碰撞忽略信息
    obs_dict_extracted['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    
    # 确保点云数据为float32类型
    for k, v in obs_dict_extracted.items():
        if 'point_cloud' in k and isinstance(v, np.ndarray):
            obs_dict_extracted[k] = v.astype(np.float32)
    
    # 添加相机内参和外参（从misc中读取）
    camera_names = ['left_shoulder', 'right_shoulder', 'front', 'wrist', 'overhead']
    for name in camera_names:
        if '%s_camera_extrinsics' % name in obs.misc:
            obs_dict_extracted['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
        if '%s_camera_intrinsics' % name in obs.misc:
            obs_dict_extracted['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    
    # ============================================================================
    # 构造 low_dim_state
    # ============================================================================
    # TODO: 在这里手写代码构造 low_dim_state
    # 格式要求：
    # - 如果 time_in_state=True: [gripper_open, left_finger_joint, right_finger_joint, time]
    # - 如果 time_in_state=False: [gripper_open, left_finger_joint, right_finger_joint]
    # 类型: np.ndarray, dtype=np.float32

    gripper_open_val = obs.gripper_open if obs.gripper_open is not None else 1.0
    left_finger = obs.gripper_joint_positions[0] if obs.gripper_joint_positions is not None else 0.0
    right_finger = obs.gripper_joint_positions[1] if obs.gripper_joint_positions is not None else 0.0
    
    if time_in_state:
        time = (1. - (curr_idx / float(episode_length - 1))) * 2. - 1.
        obs_dict_extracted['low_dim_state'] = np.array(
            [gripper_open_val, left_finger, right_finger, time], dtype=np.float32
        )
    else:
        obs_dict_extracted['low_dim_state'] = np.array(
            [gripper_open_val, left_finger, right_finger], dtype=np.float32
        )
    
    # 如果提供了语言目标，进行tokenize并添加到观察中
    if lang_goal is not None:
        tokens = clip.tokenize([lang_goal]).numpy()
        obs_dict_extracted['lang_goal_tokens'] = tokens
    
    # 恢复观察对象的原始属性
    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose
    
    # ============================================================================
    # 可视化 front 和 wrist pointcloud
    # ============================================================================

    
    return obs_dict_extracted


def initialize_agent(model_folder, model_name, device, exp_cfg_path=None, 
                     mvt_cfg_path=None, use_input_place_with_mean=False):
    """
    初始化agent模型
    
    Args:
        model_folder: 模型文件夹路径
        model_name: 模型文件名
        device: GPU设备编号
        exp_cfg_path: 实验配置文件路径（可选）
        mvt_cfg_path: MVT配置文件路径（可选）
        use_input_place_with_mean: 是否使用输入place with mean（可选）
    """
    global agent_instance, device_id
    
    device_id = device
    model_path = os.path.join(model_folder, model_name) if model_name else None
    
    print(f"正在初始化agent模型...")
    print(f"模型路径: {model_path}")
    print(f"设备: cuda:{device_id}")
    
    try:
        agent_instance = load_agent(
            model_path=model_path,
            peract_official=False,
            peract_model_dir=None,
            exp_cfg_path=exp_cfg_path,
            mvt_cfg_path=mvt_cfg_path,
            eval_log_dir="",
            device=device_id,
            use_input_place_with_mean=use_input_place_with_mean
        )
        agent_instance.eval()
        
        # 如果agent有load_clip方法，加载CLIP模型
        if hasattr(agent_instance, 'load_clip'):
            agent_instance.load_clip()
        
        print("Agent模型初始化成功！")
        return True
    except Exception as e:
        print(f"Agent模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    if agent_instance is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Agent not initialized"
        }), 503
    return jsonify({
        "status": "healthy",
        "message": "Agent is ready"
    })


@app.route('/act', methods=['POST'])
def act_endpoint():
    """
    调用agent.act方法
    
    接收JSON格式的请求：
    {
        "obs_obj": dict,  # 序列化的Observation对象字典
        "curr_idx": int,  # 当前时间步索引
        "lang_goal": str,  # 语言目标描述
        "deterministic": bool (可选，默认True)
    }
    
    返回JSON格式的响应：
    {
        "action": [float, ...],
        "step": int,
        "success": bool,
        "message": str
    }
    
    obs_obj格式应该包含Observation对象的所有属性（序列化为字典）：
    - front_rgb, front_depth, front_point_cloud等
    - gripper_open, gripper_joint_positions等
    - misc字典（包含相机参数等）
    """
    global agent_instance, device_id
    
    if agent_instance is None:
        return jsonify({
            "error": "Agent not initialized"
        }), 503
    
    try:
        # 获取JSON请求数据
        data = request.get_json()
        if data is None:
            return jsonify({
                "error": "Invalid JSON request"
            }), 400
        
        obs_obj = data.get('obs_obj', {})
        curr_idx = data.get('curr_idx', 0)
        lang_goal = data.get('lang_goal', None)
        deterministic = data.get('deterministic', True)
        episode_length = data.get('episode_length', 25)
        
        if not obs_obj:
            return jsonify({
                "error": "obs_obj is required"
            }), 400
        
        # 调用extract_obs提取观察特征
        observation = extract_obs(obs_obj, curr_idx, lang_goal, episode_length)
        
        # 将observation字典转换为torch tensor格式
        prepped_data = {}
        
        for key, val in observation.items():
            # 将值转换为numpy数组（如果还不是）
            if isinstance(val, list):
                val = np.array(val)
            elif not isinstance(val, np.ndarray):
                val = np.array(val)
            
            # 确保数据类型正确（避免float64导致的类型错误）
            if key == 'lang_goal_tokens':
                # lang_goal_tokens应该是整数类型（int64）
                if val.dtype in [np.float64, np.float32]:
                    val = val.astype(np.int64)
                elif val.dtype not in [np.int64, np.int32]:
                    val = val.astype(np.int64)
            else:
                # 其他数值数据都转换为float32
                if val.dtype == np.float64:
                    val = val.astype(np.float32)
                elif val.dtype in [np.int64, np.int32, np.int16, np.int8]:
                    val = val.astype(np.float32)
            
            # 转换为tensor并移动到GPU
            if key == 'lang_goal_tokens':
                val_tensor = torch.tensor(np.array([val]), device=f"cuda:{device_id}", dtype=torch.long)
            else:
                val_tensor = torch.tensor(np.array([val]), device=f"cuda:{device_id}", dtype=torch.float32)
            
            # 对于非语言token的观察，需要添加时间维度（unsqueeze）
            if key != 'lang_goal_tokens':
                val_tensor = val_tensor.unsqueeze(1)  # 添加时间维度
            
            prepped_data[key] = val_tensor
        
        # 调用agent.act方法
        with torch.no_grad():
            act_result = agent_instance.act(
                curr_idx, 
                prepped_data, 
                deterministic=deterministic
            )
        
        # 获取预测的动作
        # 格式：[x, y, z, qx, qy, qz, qw, grip, coll]
        pred_action = act_result.action
        
        # 转换为列表格式（如果是tensor或numpy数组）
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.cpu().numpy()
        if isinstance(pred_action, np.ndarray):
            pred_action = pred_action.tolist()
        if isinstance(pred_action, list) and len(pred_action) > 0 and isinstance(pred_action[0], (torch.Tensor, np.ndarray)):
            pred_action = [float(x.item() if hasattr(x, 'item') else x) for x in pred_action]
        else:
            pred_action = [float(x) for x in pred_action]
        
        return jsonify({
            "action": pred_action,
            "step": curr_idx,
            "success": True,
            "message": "Action predicted successfully"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error during action prediction: {str(e)}"
        }), 500


def run_api_server(host="0.0.0.0", port=8000, debug=False):
    """运行API服务器"""
    app.run(host=host, port=port, debug=debug, threaded=True)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    主程序入口 - 启动Flask服务，暴露agent.act为HTTP API
    
    使用示例：
python sam2act/historybench_eval/agent_api_server.py \
  --model_folder /home/hongzefu/sam2act_historybench/sam2act/runs/sam2act_binfill2 \
  --model_name model_last.pth \
  --device 0 \
  --port 8002

    """
    parser = argparse.ArgumentParser(description='SAM2ACT Agent Flask API服务')
    parser.add_argument('--model_folder', type=str, 
                       default=os.getenv("MODEL_FOLDER", "/home/hongzefu/sam2act_historybench/sam2act/runs/sam2act_binfill2"),
                       help='模型文件夹路径')
    parser.add_argument('--model_name', type=str,
                       default=os.getenv("MODEL_NAME", "model_last.pth"),
                       help='模型文件名')
    parser.add_argument('--device', type=int,
                       default=int(os.getenv("DEVICE", "0")),
                       help='GPU设备编号')
    parser.add_argument('--exp_cfg_path', type=str, default=None,
                       help='实验配置文件路径')
    parser.add_argument('--mvt_cfg_path', type=str, default=None,
                       help='MVT配置文件路径')
    parser.add_argument('--use_input_place_with_mean', action='store_true',
                       help='是否使用输入place with mean')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='监听地址（默认: 0.0.0.0）')
    parser.add_argument('--port', type=int, default=None,
                       help='监听端口（默认: 8000，如果被占用则自动查找可用端口）')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    # 初始化agent模型
    success = initialize_agent(
        model_folder=args.model_folder,
        model_name=args.model_name,
        device=args.device,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        use_input_place_with_mean=args.use_input_place_with_mean
    )
    
    if not success:
        print("❌ Agent模型初始化失败，退出")
        sys.exit(1)
    
    # 处理端口配置
    api_host = args.host
    api_port = args.port if args.port is not None else 8000
    
    # 如果用户明确指定了端口，检查是否被占用
    if args.port is not None:
        if is_port_in_use(api_host, api_port):
            print(f"⚠️  错误: 端口 {api_port} 已被占用！")
            print(f"   请使用其他端口，例如: --port 8001")
            print(f"   或者先停止占用该端口的进程")
            sys.exit(1)
    else:
        # 如果没有指定端口，自动查找可用端口
        original_port = api_port
        if is_port_in_use(api_host, api_port):
            print(f"⚠️  端口 {api_port} 已被占用，正在查找可用端口...")
            available_port = find_available_port(api_host, api_port)
            if available_port:
                api_port = available_port
                print(f"✅ 找到可用端口: {api_port}")
            else:
                print(f"❌ 错误: 无法找到可用端口（已尝试 {original_port}-{original_port + 9}）")
                sys.exit(1)
    
    # 显示启动信息
    print("=" * 60)
    print("SAM2ACT Agent Flask API 服务")
    print("=" * 60)
    print(f"本机IP地址: {get_local_ip()}")
    print(f"监听地址: {api_host}")
    print(f"端口: {api_port}")
    if api_host == "0.0.0.0":
        print(f"服务地址: http://{get_local_ip()}:{api_port}")
    else:
        print(f"服务地址: http://{api_host}:{api_port}")
    print(f"健康检查: http://{api_host}:{api_port}/health")
    print(f"动作预测: http://{api_host}:{api_port}/act")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    # 启动API服务器
    try:
        run_api_server(host=api_host, port=api_port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n服务已停止")
