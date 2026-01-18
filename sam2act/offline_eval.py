"""
离线评估脚本

本脚本用于对训练好的SAM2ACT模型进行离线评估，无需实际运行RLBench环境。
主要功能包括：
1. 从存储的演示数据中加载任务数据
2. 使用训练好的agent模型对演示数据进行预测
3. 计算预测动作与真实动作之间的误差（平移误差、旋转误差、夹爪误差）
4. 生成评估报告并保存为CSV文件

使用场景：
- 快速评估模型性能，无需启动完整的RLBench环境
- 批量评估多个任务
- 分析模型在不同任务上的表现
"""

import os
import torch
import numpy as np
import pickle
import clip
import csv
from typing import List, Dict
import argparse
from scipy.spatial.transform import Rotation

from rlbench.backend.observation import Observation
from rlbench.backend.utils import task_file_to_task_class
from rlbench.utils import get_stored_demos
import rlbench.backend.task as rlbench_task

from yarr.utils.observation_type import ObservationElement
from yarr.envs.rlbench_env import _extract_obs, _observation_elements

from sam2act.eval import load_agent
from sam2act.libs.peract.helpers import utils
from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE, SCENE_BOUNDS
from sam2act.utils.rvt_utils import get_eval_parser, RLBENCH_TASKS


class OfflineEnvMock:
    """
    离线环境模拟器类
    
    用于模拟RLBench环境，从存储的观察数据中提取特征，无需实际运行环境。
    重新实现了CustomMultiTaskRLBenchEnv的部分功能，以避免需要完整的环境实例。
    
    主要功能：
    - 从RLBench观察对象中提取特征字典
    - 添加时间状态信息
    - 处理语言目标token化
    """
    def __init__(self, observation_config, episode_length=25, time_in_state=True):
        """
        初始化离线环境模拟器
        
        Args:
            observation_config: 观察配置对象，定义需要提取的观察类型和格式
            episode_length: 每个episode的长度（时间步数），默认25
            time_in_state: 是否在状态中包含时间信息，默认True
        """
        self._observation_config = observation_config  # 观察配置
        self._episode_length = episode_length  # Episode长度
        self._time_in_state = time_in_state  # 是否在状态中包含时间
        self._channels_last = False  # 图像通道顺序：False表示通道在前（CHW），这是YARR/PerAct的默认格式

    def extract_obs(self, obs: Observation, t: int, lang_goal: str | None = None):
        """
        从RLBench观察对象中提取特征字典
        
        该方法实现了与CustomMultiTaskRLBenchEnv2/CustomRLBenchEnv相同的观察提取逻辑，
        将RLBench的Observation对象转换为模型所需的特征字典格式。
        
        Args:
            obs: RLBench观察对象，包含图像、状态等信息
            t: 当前时间步（在episode中的索引）
            lang_goal: 语言目标描述字符串，可选。如果提供，会被tokenize后添加到观察中
        
        Returns:
            obs_dict: 包含提取特征的字典，包括：
                - 图像观察（根据observation_config配置的相机）
                - 低维状态（low_dim_state）
                - 时间状态（如果time_in_state=True）
                - 语言目标tokens（如果提供了lang_goal）
        """
        # 备份需要恢复的值，因为_extract_obs可能会修改obs对象
        grip_mat = obs.gripper_matrix  # 夹爪变换矩阵
        grip_pose = obs.gripper_pose  # 夹爪位姿
        joint_pos = obs.joint_positions  # 关节位置
        
        # 修改观察对象，移除不需要的属性（这些属性可能会干扰YARR的提取函数）
        obs.joint_velocities = None  # 关节速度
        obs.gripper_pose = None  # 夹爪位姿
        obs.gripper_matrix = None  # 夹爪变换矩阵
        obs.wrist_camera_matrix = None  # 腕部相机变换矩阵
        obs.joint_positions = None  # 关节位置
        
        # 限制夹爪关节位置在合理范围内 [0, 0.04]
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        # 调用YARR的_extract_obs函数提取观察特征
        # 该函数会根据observation_config提取图像、状态等特征
        obs_dict = _extract_obs(obs, self._channels_last, self._observation_config)
        
        # 添加时间状态信息
        # 时间编码公式：time = (1 - t/(L-1)) * 2 - 1
        # 其中t是当前时间步，L是episode长度
        # 这样时间从1（开始）线性映射到-1（结束）
        if self._time_in_state:
            time = (1. - (t / float(self._episode_length - 1))) * 2. - 1.
            # 将时间信息拼接到低维状态中
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        # 如果提供了语言目标，进行tokenize并添加到观察中
        if lang_goal is not None:
            # 使用CLIP的tokenizer将语言目标转换为tokens
            tokens = clip.tokenize([lang_goal]).numpy()
            obs_dict['lang_goal_tokens'] = tokens

        # 恢复观察对象的原始属性（避免影响后续使用）
        obs.gripper_matrix = grip_mat
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        
        return obs_dict

def calculate_metrics(pred_action, gt_pose, gt_open):
    """
    计算预测动作与真实动作之间的误差指标
    
    计算三个误差指标：
    1. 平移误差（Translation Error）：预测位置与真实位置的欧氏距离
    2. 旋转误差（Rotation Error）：预测旋转与真实旋转之间的角度差（使用四元数距离）
    3. 夹爪误差（Gripper Error）：预测夹爪状态与真实状态的二值化误差
    
    Args:
        pred_action: 预测动作数组，格式为 [x, y, z, qx, qy, qz, qw, grip, coll]
            - x, y, z: 预测的末端执行器位置
            - qx, qy, qz, qw: 预测的末端执行器旋转（四元数）
            - grip: 预测的夹爪状态（0=关闭，1=打开）
            - coll: 碰撞标志（本函数中未使用）
        gt_pose: 真实位姿数组，格式为 [x, y, z, qx, qy, qz, qw]
            - x, y, z: 真实的末端执行器位置
            - qx, qy, qz, qw: 真实的末端执行器旋转（四元数）
        gt_open: 真实的夹爪状态（整数，0=关闭，1=打开）
    
    Returns:
        tuple: (trans_err, rot_err_deg, grip_err)
            - trans_err: 平移误差（米）
            - rot_err_deg: 旋转误差（度）
            - grip_err: 夹爪误差（0.0表示正确，1.0表示错误）
    """
    # 从预测动作中提取位置、旋转和夹爪状态
    pred_trans = pred_action[:3]  # 预测位置 [x, y, z]
    pred_quat = pred_action[3:7]  # 预测旋转四元数 [qx, qy, qz, qw]
    pred_grip = pred_action[7]  # 预测夹爪状态（0或1）
    
    # 从真实位姿中提取位置和旋转
    gt_trans = gt_pose[:3]  # 真实位置 [x, y, z]
    gt_quat = gt_pose[3:7]  # 真实旋转四元数 [qx, qy, qz, qw]
    
    # 计算平移误差：使用L2范数（欧氏距离）
    # ||pred_trans - gt_trans||_2
    trans_err = np.linalg.norm(pred_trans - gt_trans)
    
    # 计算旋转误差：使用四元数距离公式
    # 四元数距离公式：d(q1, q2) = 2 * arccos(|<q1, q2>|)
    # 其中 <q1, q2> 是两个四元数的点积
    # 首先确保四元数归一化（单位四元数）
    pred_quat = pred_quat / np.linalg.norm(pred_quat)
    gt_quat = gt_quat / np.linalg.norm(gt_quat)
    
    # 计算两个四元数的点积（内积）
    # 使用绝对值是因为q和-q表示相同的旋转
    dot_prod = np.abs(np.dot(pred_quat, gt_quat))
    # 限制点积在[0, 1]范围内，避免数值误差导致arccos出错
    dot_prod = np.clip(dot_prod, 0.0, 1.0)
    # 计算旋转误差（弧度）
    rot_err = 2 * np.arccos(dot_prod)
    # 转换为角度
    rot_err_deg = np.degrees(rot_err)
    
    # 计算夹爪误差（二值化比较）
    # pred_grip是通过argmax得到的类别索引（0或1）
    # 在sam2act_agent.get_pred中：pred_grip = grip_q.argmax(1, keepdim=True)
    # 所以pred_grip是0（关闭）或1（打开）
    # gt_open也是0（关闭）或1（打开）
    # 如果两者不同，误差为1.0；相同则为0.0
    grip_err = 1.0 if pred_grip != gt_open else 0.0
    
    return trans_err, rot_err_deg, grip_err

def eval_offline(args):
    """
    离线评估主函数
    
    对训练好的agent模型进行离线评估，使用存储的演示数据计算预测误差。
    评估流程：
    1. 加载训练好的agent模型
    2. 为每个任务加载演示数据
    3. 对每个演示的关键点进行预测
    4. 计算预测动作与真实动作的误差
    5. 聚合结果并保存为CSV文件
    
    Args:
        args: 命令行参数对象，包含以下关键属性：
            - device: 使用的GPU设备编号
            - model_folder: 模型文件夹路径
            - model_name: 模型文件名
            - tasks: 要评估的任务列表
            - eval_datafolder: 评估数据文件夹路径
            - eval_episodes: 每个任务评估的episode数量
            - episode_length: 每个episode的长度
            - eval_log_dir: 评估日志保存目录
            - 其他模型加载相关参数
    """
    device = args.device
    
    # 加载训练好的agent模型
    print(f"Loading agent from {args.model_folder}...")
    agent = load_agent(
        model_path=os.path.join(args.model_folder, args.model_name) if args.model_name else None,
        peract_official=args.peract_official,
        peract_model_dir=args.peract_model_dir,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=args.eval_log_dir,
        device=device,
        use_input_place_with_mean=args.use_input_place_with_mean
    )
    # 将模型设置为评估模式（禁用dropout等训练时的操作）
    agent.eval()
    # 如果agent有load_clip方法，加载CLIP模型（用于语言理解）
    if hasattr(agent, 'load_clip'):
        agent.load_clip()

    # 创建观察配置
    # 定义相机分辨率（图像大小）
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    # 创建观察配置，指定使用的相机、分辨率等信息
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="", use_mask_from_replay=False)
    
    # 创建离线环境模拟器
    env_mock = OfflineEnvMock(obs_config, episode_length=args.episode_length)
    
    # 处理任务列表：如果第一个任务是"all"，则评估所有RLBench任务
    tasks = args.tasks
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        
    # 存储所有任务的评估结果
    results = {}
    # 存储所有详细记录（用于保存每次计算的详细信息）
    all_detailed_records = []
    
    # 遍历每个任务进行评估
    for task_name in tasks:
        print(f"Evaluating task: {task_name}")
        # 初始化当前任务的误差指标列表
        task_metrics = {
            "trans_err": [],  # 平移误差列表
            "rot_err": [],    # 旋转误差列表
            "grip_err": []    # 夹爪误差列表
        }
        # 初始化当前任务的详细记录列表
        detailed_records = []
        
        try:
            # 从存储的数据中加载演示数据
            # get_stored_demos会从指定路径加载RLBench任务的演示数据
            demos = get_stored_demos(
                amount=args.eval_episodes,  # 加载的episode数量
                image_paths=False,  # 不返回图像路径，直接返回图像数据
                dataset_root=args.eval_datafolder,  # 数据集根目录
                variation_number=-1,  # -1表示加载所有variation
                task_name=task_name,  # 任务名称
                obs_config=obs_config,  # 观察配置
                random_selection=False,  # 不随机选择，按顺序加载
                from_episode_number=args.start_episode  # 起始episode编号
            )
        except Exception as e:
            # 如果加载失败，打印错误信息并跳过该任务
            print(f"Failed to load demos for {task_name}: {e}")
            continue
            
        # 遍历每个演示episode
        for i, demo in enumerate(demos):
            # 关键点发现：从演示数据中提取关键时间步
            # 关键点表示任务执行过程中的重要状态转换点
            # 使用启发式方法（heuristic）来发现关键点
            keypoints = keypoint_discovery(demo, method='heuristic')
            # 如果关键点少于2个，无法形成有效的状态转换，跳过该episode
            if len(keypoints) < 2:
                print(f"Skipping episode {i} of {task_name}: Not enough keypoints")
                continue
                
            # 获取语言目标描述
            # 语言目标存储在第一个观察的misc['descriptions']中
            descriptions = demo[0].misc.get('descriptions', ['unknown goal'])
            lang_goal = descriptions[0]  # 使用第一个描述作为语言目标
            
            print(f"  Episode {i}: {len(keypoints)} keypoints, Goal: {lang_goal}")
            
            # 如果agent使用了MVT（Multi-View Transformer）网络，重置记忆库
            # 这确保每个episode开始时，agent的记忆是干净的
            if hasattr(agent, '_network') and hasattr(agent._network, 'mvt1'):
                 if hasattr(agent._network.mvt1, 'reset_memory_bank'):
                    agent._network.mvt1.reset_memory_bank()
                 if hasattr(agent._network.mvt2, 'reset_memory_bank'):
                    agent._network.mvt2.reset_memory_bank()

            # 遍历关键点之间的转换
            # 对于每对相邻的关键点，使用当前关键点的观察预测下一个关键点的动作
            for k in range(len(keypoints) - 1):
                curr_idx = keypoints[k]      # 当前关键点的索引
                next_idx = keypoints[k+1]     # 下一个关键点的索引
                
                # 获取当前和目标的观察对象
                obs_obj = demo[curr_idx]           # 当前状态的观察
                target_obs_obj = demo[next_idx]   # 目标状态的观察（用于计算真实动作）
                
                # 从当前观察中提取特征字典
                # 传入当前时间步索引和语言目标
                obs_dict = env_mock.extract_obs(obs_obj, curr_idx, lang_goal)
                
                # 准备批次数据（批次大小为1）
                # 将numpy数组转换为torch tensor并移动到指定设备
                prepped_data = {}
                for key, val in obs_dict.items():
                    # 将值转换为tensor并移动到GPU
                    val = torch.tensor(np.array([val]), device=f"cuda:{device}")
                    # 对于非语言token的观察，需要添加时间维度（unsqueeze）
                    # lang_goal_tokens已经是正确的形状，不需要unsqueeze
                    if key != 'lang_goal_tokens':
                        val = val.unsqueeze(1)  # 添加时间维度
                    prepped_data[key] = val
                
                # 使用agent进行动作预测
                # step_signal通常用于epsilon-greedy或调度，这里传入关键点索引k
                # 在eval.py中传入的是Value对象，这里传入整数
                # deterministic=True表示使用确定性策略（不采样）
                with torch.no_grad():  # 禁用梯度计算以节省内存和加速
                    act_result = agent.act(k, prepped_data, deterministic=True)
                
                # 获取预测的动作
                # 格式：[x, y, z, qx, qy, qz, qw, grip, coll]
                # x,y,z: 位置；qx,qy,qz,qw: 旋转四元数；grip: 夹爪状态；coll: 碰撞标志
                pred_action = act_result.action
                
                # 获取真实动作（ground truth）
                gt_pose = target_obs_obj.gripper_pose  # 真实夹爪位姿 [x,y,z,qx,qy,qz,qw]
                gt_open = target_obs_obj.gripper_open  # 真实夹爪状态（浮点数，0.0-1.0）
                
                # 将真实夹爪状态二值化，以便与预测的夹爪状态比较
                # RLBench中：gripper_open 1.0表示打开，0.0表示关闭
                # PerAct中：action_grip_one_hot [closed, open]，argmax=0表示关闭，argmax=1表示打开
                # 预测的pred_grip是0（关闭）或1（打开）
                # 所以如果gt_open > 0.5，则认为是打开（1），否则是关闭（0）
                gt_open_int = 1 if gt_open > 0.5 else 0
                
                # 计算预测动作与真实动作之间的误差
                t_err, r_err, g_err = calculate_metrics(pred_action, gt_pose, gt_open_int)
                
                # 将误差添加到任务指标列表中
                task_metrics["trans_err"].append(t_err)  # 平移误差
                task_metrics["rot_err"].append(r_err)    # 旋转误差
                task_metrics["grip_err"].append(g_err)   # 夹爪误差
                
                # 记录详细的预测动作、真实动作和误差信息
                detailed_record = {
                    "task_name": task_name,
                    "episode_idx": i,
                    "keypoint_pair_idx": k,
                    "curr_keypoint_idx": int(curr_idx),
                    "next_keypoint_idx": int(next_idx),
                    # 预测动作完整值
                    "pred_x": float(pred_action[0]),
                    "pred_y": float(pred_action[1]),
                    "pred_z": float(pred_action[2]),
                    "pred_qx": float(pred_action[3]),
                    "pred_qy": float(pred_action[4]),
                    "pred_qz": float(pred_action[5]),
                    "pred_qw": float(pred_action[6]),
                    "pred_grip": float(pred_action[7]),
                    # 真实动作完整值
                    "gt_x": float(gt_pose[0]),
                    "gt_y": float(gt_pose[1]),
                    "gt_z": float(gt_pose[2]),
                    "gt_qx": float(gt_pose[3]),
                    "gt_qy": float(gt_pose[4]),
                    "gt_qz": float(gt_pose[5]),
                    "gt_qw": float(gt_pose[6]),
                    "gt_grip": float(gt_open_int),
                    # 误差值
                    "trans_err": float(t_err),
                    "rot_err": float(r_err),
                    "grip_err": float(g_err)
                }
                detailed_records.append(detailed_record)
        
        # 聚合任务指标：计算平均误差
        if len(task_metrics["trans_err"]) > 0:
            # 计算所有状态转换的平均误差
            avg_trans = np.mean(task_metrics["trans_err"])  # 平均平移误差（米）
            avg_rot = np.mean(task_metrics["rot_err"])      # 平均旋转误差（度）
            avg_grip = np.mean(task_metrics["grip_err"])    # 平均夹爪误差（0-1之间）
            
            # 保存任务结果
            results[task_name] = {
                "trans_err": avg_trans,  # 平均平移误差
                "rot_err": avg_rot,      # 平均旋转误差
                "grip_err": avg_grip,    # 平均夹爪误差
                "steps": len(task_metrics["trans_err"])  # 有效状态转换的数量
            }
            # 打印任务评估结果
            print(f"  Result {task_name}: Trans={avg_trans:.4f}, Rot={avg_rot:.4f}, Grip={avg_grip:.4f}")
        else:
            # 如果没有有效的状态转换，打印警告信息
            print(f"  No valid transitions for {task_name}")
        
        # 将当前任务的详细记录添加到全局列表
        all_detailed_records.extend(detailed_records)

    # 保存评估结果到CSV文件
    if args.eval_log_dir:
        # 创建评估日志目录（如果不存在）
        os.makedirs(args.eval_log_dir, exist_ok=True)
        # 构建CSV文件路径
        csv_path = os.path.join(args.eval_log_dir, "offline_eval_results.csv")
        # 写入CSV文件
        with open(csv_path, "w") as f:
            # 写入表头
            f.write("task,trans_err,rot_err,grip_err,steps\n")
            # 写入每个任务的结果
            for task, mets in results.items():
                f.write(f"{task},{mets['trans_err']},{mets['rot_err']},{mets['grip_err']},{mets['steps']}\n")
        print(f"Results saved to {csv_path}")
        
        # 保存详细记录到CSV文件
        if len(all_detailed_records) > 0:
            detailed_csv_path = os.path.join(args.eval_log_dir, "detailed_eval_results.csv")
            # 定义CSV列名
            fieldnames = [
                "task_name", "episode_idx", "keypoint_pair_idx", 
                "curr_keypoint_idx", "next_keypoint_idx",
                "pred_x", "pred_y", "pred_z", "pred_qx", "pred_qy", "pred_qz", "pred_qw", "pred_grip",
                "gt_x", "gt_y", "gt_z", "gt_qx", "gt_qy", "gt_qz", "gt_qw", "gt_grip",
                "trans_err", "rot_err", "grip_err"
            ]
            
            # 写入详细记录CSV文件
            with open(detailed_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_detailed_records)
            print(f"Detailed results saved to {detailed_csv_path} ({len(all_detailed_records)} records)")

if __name__ == "__main__":
    """
    主程序入口
    
    解析命令行参数，设置默认值，并调用离线评估函数。
    
    使用示例：
        python sam2act/offline_eval.py \
          --tasks open_drawer \
          --model_folder /path/to/model \
          --eval_datafolder /path/to/data \
          --eval_episodes 10 \
          --device 0
    """
    # 获取评估参数解析器（从rvt_utils导入）
    parser = get_eval_parser()
    
    # 设置默认参数值
    # 这些默认值可以在命令行中被覆盖
    parser.set_defaults(
        tasks=["close_jar"],  # 默认评估任务
        model_folder="/home/hongzefu/sam2act_historybench/sam2act/runs/sam2act_test",  # 默认模型文件夹
        model_name="model_last.pth",  # 默认模型文件名
        eval_datafolder="/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/dataset/rlbench-18-tasks/data/train"  # 默认数据文件夹
    )
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果未指定日志名称，使用默认值
    if args.log_name is None:
        args.log_name = "offline_eval"
        
    # 根据是否使用官方PerAct模型，设置评估日志目录
    if not (args.peract_official):
        # 使用SAM2ACT模型的目录结构
        args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    else:
        # 使用官方PerAct模型的目录结构
        args.eval_log_dir = os.path.join(args.peract_model_dir, "eval", args.log_name)

    # 执行离线评估
    eval_offline(args)


# 使用示例（命令行）：
# python sam2act/offline_eval.py \
#   --tasks open_drawer \
#   --model_folder /path/to/model \
#   --eval_datafolder /path/to/data \
#   --eval_episodes 10 \
#   --device 0