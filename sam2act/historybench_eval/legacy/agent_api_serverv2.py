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
        "step": int,
        "observation": dict,
        "deterministic": bool (可选，默认True)
    }
    
    返回JSON格式的响应：
    {
        "action": [float, ...],
        "step": int,
        "success": bool,
        "message": str
    }
    
    observation格式应该包含：
    - low_dim_state: 低维状态数组
    - {camera}_rgb: RGB图像 (C, H, W) 格式
    - {camera}_depth: 深度图像 (1, H, W) 格式
    - {camera}_point_cloud: 点云 (3, H, W) 格式
    - {camera}_camera_extrinsics: 相机外参 (4, 4) 格式
    - {camera}_camera_intrinsics: 相机内参 (3, 3) 格式
    - lang_goal_tokens: 语言目标tokens (可选)
    - ignore_collisions: 碰撞忽略标志
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
        
        step = data.get('step', 0)
        observation = data.get('observation', {})
        deterministic = data.get('deterministic', True)
        
        if not observation:
            return jsonify({
                "error": "Observation is required"
            }), 400
        
        # 将observation字典转换为torch tensor格式
        # 参考offline_eval_Hbench_api.py第720-728行的预处理逻辑
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
                # 如果从JSON反序列化后变成float，需要转换回int
                if val.dtype in [np.float64, np.float32]:
                    val = val.astype(np.int64)
                elif val.dtype not in [np.int64, np.int32]:
                    val = val.astype(np.int64)
            else:
                # 其他数值数据（包括点云、相机参数等）都转换为float32
                if val.dtype == np.float64:
                    val = val.astype(np.float32)
                elif val.dtype in [np.int64, np.int32, np.int16, np.int8]:
                    # 整数类型转换为float32（除了lang_goal_tokens）
                    val = val.astype(np.float32)
            
            # 转换为tensor并移动到GPU
            # 使用dtype参数确保tensor类型正确
            if key == 'lang_goal_tokens':
                val_tensor = torch.tensor(np.array([val]), device=f"cuda:{device_id}", dtype=torch.long)
            else:
                val_tensor = torch.tensor(np.array([val]), device=f"cuda:{device_id}", dtype=torch.float32)
            
            # 对于非语言token的观察，需要添加时间维度（unsqueeze）
            # lang_goal_tokens已经是正确的形状，不需要unsqueeze
            if key != 'lang_goal_tokens':
                val_tensor = val_tensor.unsqueeze(1)  # 添加时间维度
            
            prepped_data[key] = val_tensor
        
        # 调用agent.act方法
        with torch.no_grad():
            act_result = agent_instance.act(
                step, 
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
            "step": step,
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
