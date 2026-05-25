#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_dataset_temporal 函数的数据读取功能

本文件用于测试时序强化学习数据集的创建和读取功能。主要功能包括：
1. 创建时序 replay buffer（支持多时
2. 从 RLBench 数据集中加载演示数据并填充到 replay buffer
3. 使用 CLIP 模型提取语言特征
4. 将 replay buffer 包装为 PyTorch 数据集，便于训练使用

使用场景：
- 测试数据集创建流程是否正常
- 验证数据格式和内容是否正确
- 调试数据加载过程中的问题

主要测试内容：
- 数据集创建：调用 get_dataset_temporal 函数创建训练和测试数据集
- 数据读取：从数据集中读取样本并打印其结构和内容
- 数据验证：检查数据的形状、类型等属性是否符合预期
"""

# ============================================================================
# 标准库导入
# ============================================================================
import os      # 用于文件路径操作，如 os.path.join, os.path.exists 等
import sys     # 用于系统相关操作，如 sys.path 路径管理
import shutil  # 用于高级文件操作，如删除目录树 shutil.rmtree
import pickle  # 用于序列化和反序列化 Python 对象，用于加载任务描述文件
import logging # 用于日志记录，用于记录 replay buffer 保存到磁盘的信息
from typing import List  # 用于类型注解，指定列表类型
import h5py    # 用于读取 HDF5 文件

# ============================================================================
# 第三方库导入
# ============================================================================
import torch   # PyTorch 深度学习框架，用于张量操作和 GPU 管理
import torch.nn.functional as F  # PyTorch 函数式接口，用于图像插值等操作
import clip    # OpenAI CLIP 模型，用于提取语言特征和图像-文本匹配
import numpy as np  # NumPy 数值计算库，用于数组操作和数值计算
from PIL import Image  # 用于图像处理
from rlbench.backend.utils import image_to_float_array  # 用于深度图像转换
from pyrep.objects import VisionSensor  # 用于点云生成
import plotly.graph_objects as go  # Plotly 用于 3D 可视化

# ============================================================================
# 路径配置：添加项目根目录到 Python 路径
# ============================================================================
# 获取当前文件的绝对路径，然后获取其所在目录（项目根目录）
# 测试文件位于项目根目录，sam2act 包在 sam2act/ 子目录下
# 需要将项目根目录添加到 sys.path 才能正确导入 sam2act 模块
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# 项目内部模块导入
# ============================================================================
# 从 sam2act.utils.dataset 导入时序 replay buffer 的创建函数
# create_replay_temporal: 创建支持时序数据的 replay buffer
# 注意：fill_replay_temporal 函数将在本文件中定义，不再从 dataset 模块导入
from sam2act.utils.dataset import create_replay_temporal

# 导入工具函数和类
# peract_colab.arm.utils: 包含动作离散化相关的工具函数（如四元数归一化、体素索引计算等）
import peract_colab.arm.utils as utils

# 从 peract_colab.rlbench.utils 导入演示数据加载函数
# get_stored_demo: 从磁盘加载存储的演示数据
# 注意：get_stored_demo 函数定义已拷贝到本文件中，不再从外部模块导入
# from peract_colab.rlbench.utils import get_stored_demo

# 从 sam2act.libs.peract.helpers 导入观察提取函数
# extract_obs: 从观察对象中提取特征（图像、深度、点云等）
# 注意：keypoint_discovery 函数已在本文件中定义，不再从外部模块导入
from sam2act.libs.peract.helpers.utils import extract_obs

# 从 rlbench 导入观察和演示数据类型
# Observation: RLBench 的观察对象，包含相机图像、机器人状态等信息
# Demo: RLBench 的演示对象，包含一个完整演示的所有帧
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

# 从 yarr.replay_buffer 导入 ReplayBuffer 基类
# ReplayBuffer: replay buffer 的抽象基类，定义了添加和采样数据的方法
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_temporal import UniformReplayBuffer_temporal

# Patch UniformReplayBuffer_temporal to relax signature check
def patched_check_add_types(self, kwargs, signature):
    # Relaxed check: do not enforce len(kwargs) == len(signature)
    # Just check that signature elements exist in kwargs and have correct shape
    for store_element in signature:
        if store_element.name not in kwargs:
             raise ValueError('Element {} missing from kwargs'.format(store_element.name))
        arg_element = kwargs[store_element.name]
        if isinstance(arg_element, np.ndarray):
            arg_shape = arg_element.shape
        elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
            arg_shape = np.array(arg_element).shape
        else:
            arg_shape = tuple()
        store_element_shape = tuple(store_element.shape)
        if arg_shape != store_element_shape:
            raise ValueError('arg {} has shape {}, expected {}'.format(
                store_element.name, arg_shape, store_element_shape))

UniformReplayBuffer_temporal._check_add_types = patched_check_add_types


# ============================================================================
# 常量定义：用于 get_stored_demo 函数
# ============================================================================
EPISODE_FOLDER = 'episode%d'
CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_FORMAT = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'
DEPTH_SCALE = 2**24 - 1

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

# ============================================================================
# 辅助函数：关键点发现
# ============================================================================
def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    """
    判断机器人是否在某个时刻停止
    
    该函数用于检测演示中的停止状态，通过检查关节速度和抓取器状态来判断
    机器人是否在某个时刻停止运动。这是关键点发现算法的一部分。
    
    参数:
        demo (Demo): 完整的演示对象，包含演示的所有帧
        i (int): 当前帧的索引
        obs (Observation): 当前帧的观察对象
        stopped_buffer (int): 停止缓冲区计数器，用于平滑停止检测
        delta (float): 速度阈值，用于判断关节速度是否接近零（默认 0.1）
    
    返回:
        bool: 如果机器人停止则返回 True，否则返回 False
    """
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    """
    从演示数据中发现关键帧（关键点）
    
    该函数从演示数据中识别出重要的时刻（关键点），这些关键点通常对应
    任务执行中的重要动作，例如抓取、放置、打开/关闭等。关键点用于
    将长演示序列分解为更短的子任务，便于强化学习训练。
    
    参数:
        demo (Demo): 完整的演示对象，包含演示的所有帧（Observation 对象列表）
        stopping_delta (float): 停止检测的速度阈值，用于判断机器人是否停止（默认 0.1）
        method (str): 关键点发现方法，可选值：
                     - 'heuristic': 启发式方法，基于抓取器状态变化和停止检测（默认）
                     - 'random': 随机选择关键点
                     - 'fixed_interval': 固定间隔选择关键点
    
    返回:
        List[int]: 关键点帧索引列表，例如 [10, 25, 40, 55]
                  表示第 10、25、40、55 帧是关键点
    
    工作流程（heuristic 方法）:
        1. 遍历演示的所有帧
        2. 检测抓取器状态变化（打开/关闭）
        3. 检测机器人停止状态
        4. 在状态变化或停止时标记为关键点
        5. 清理相邻的关键点（如果两个关键点太接近，移除其中一个）
    
    注意:
        - heuristic 方法是最常用的方法，能够自动识别任务中的重要时刻
        - random 方法用于数据增强或实验对比
        - fixed_interval 方法用于均匀采样，适用于长演示序列
    """
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    elif method == 'dataset':
        episode_keypoints = []
        for i, obs in enumerate(demo):
             # Check if keypoint_type exists and is valid (not None and not string "None")
             if 'keypoint_type' in obs.misc and obs.misc['keypoint_type'] is not None and obs.misc['keypoint_type'] != 'None':
                 episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


# ============================================================================
# 辅助函数：从磁盘加载存储的演示数据
# ============================================================================
def get_stored_demo(data_path, index):
    """
    从磁盘加载存储的演示数据
    
    该函数从指定的数据路径加载指定索引的演示数据，包括：
    1. 低维观察数据（从 pickle 文件加载）
    2. 任务变体编号
    3. 所有相机视角的 RGB 图像
    4. 所有相机视角的深度图像（转换为浮点数并应用近远平面缩放）
    5. 所有相机视角的点云数据
    
    参数:
        data_path (str): 演示数据的根目录路径
        index (int): 演示的索引编号
    
    返回:
        obs: 观察对象列表，每个元素是一个 Observation 对象，包含该帧的所有观察信息
    """
    if data_path.endswith('.h5'):
        class DemoList(list):
            pass
        obs = DemoList()
        
        with h5py.File(data_path, 'r') as f:
            # 假设只有一个环境 env_BinFill 或类似的，取第一个键作为环境名
            # 或者搜索包含 episode_X 的组
            env_name = list(f.keys())[0]
            episode_name = f'episode_{index}'
            
            # 如果直接在根目录下找不到，尝试进入环境目录
            if episode_name not in f[env_name]:
                 # 也许结构是 env_name/episode_0
                 if episode_name in f:
                     # 这种情况不太可能，基于之前的 inspect，是 env_BinFill/episode_0
                     pass
                 else:
                     # 尝试在 env_name 下查找
                     if episode_name not in f[env_name]:
                         raise ValueError(f"Episode {index} not found in {data_path} under {env_name}")
                     ep_grp = f[env_name][episode_name]
            else:
                ep_grp = f[env_name][episode_name]

            # 获取所有时间步，按索引排序
            # 过滤出 record_timestep_X 的键
            timesteps = sorted([k for k in ep_grp.keys() if k.startswith('record_timestep_')], 
                               key=lambda x: int(x.split('_')[-1]))
            
            for ts_name in timesteps:
                ts_grp = ep_grp[ts_name]
                
                # 创建 Observation 对象
                # 初始化所有参数为 None，后续填充
                current_obs = Observation(
                    left_shoulder_rgb=None, left_shoulder_depth=None, left_shoulder_mask=None, left_shoulder_point_cloud=None,
                    right_shoulder_rgb=None, right_shoulder_depth=None, right_shoulder_mask=None, right_shoulder_point_cloud=None,
                    overhead_rgb=None, overhead_depth=None, overhead_mask=None, overhead_point_cloud=None,
                    wrist_rgb=None, wrist_depth=None, wrist_mask=None, wrist_point_cloud=None,
                    front_rgb=None, front_depth=None, front_mask=None, front_point_cloud=None,
                    joint_velocities=None, joint_positions=None, joint_forces=None,
                    gripper_open=None, gripper_pose=None, gripper_matrix=None, gripper_joint_positions=None, gripper_touch_forces=None,
                    task_low_dim_state=None, ignore_collisions=None, misc=None
                )
                
                # --- 1. 加载 RGB 图像 ---
                if 'image' in ts_grp:
                    img = np.array(ts_grp['image'])
                    # 数据集中的图像已经是 128x128，不需要下变换
                    current_obs.front_rgb = img
                else:
                    current_obs.front_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                    
                if 'wrist_image' in ts_grp:
                    img = np.array(ts_grp['wrist_image'])
                    # 数据集中的图像已经是 128x128，不需要下变换
                    current_obs.wrist_rgb = img
                else:
                    current_obs.wrist_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                
                # 填充缺失的相机
                # current_obs.left_shoulder_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                # current_obs.right_shoulder_rgb = np.zeros((128, 128, 3), dtype=np.uint8)

                # --- 2. 加载深度图像 ---
                # 假设 int16 是毫米，转换为米
                if 'base_camera_depth' in ts_grp:
                     depth = np.array(ts_grp['base_camera_depth']).astype(np.float32) / 1000.0
                     if depth.ndim == 3:
                         depth = depth.squeeze(-1)
                     # 降采样到 128x128
                     depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                     depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                     current_obs.front_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
                else:
                    current_obs.front_depth = np.zeros((128, 128), dtype=np.float32)

                if 'wrist_camera_depth' in ts_grp:
                     depth = np.array(ts_grp['wrist_camera_depth']).astype(np.float32) / 1000.0
                     if depth.ndim == 3:
                         depth = depth.squeeze(-1)
                     # 降采样到 128x128
                     depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                     depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                     current_obs.wrist_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
                else:
                     current_obs.wrist_depth = np.zeros((128, 128), dtype=np.float32)
                
                # 填充缺失深度
                # current_obs.left_shoulder_depth = np.zeros((128, 128), dtype=np.float32)
                # current_obs.right_shoulder_depth = np.zeros((128, 128), dtype=np.float32)

                # --- 3. 加载机器人状态 ---
                # robot_endeffector_p: (1, 3)
                # robot_endeffector_q: (1, 4)
                gripper_pos = np.array(ts_grp['robot_endeffector_p']).flatten()
                gripper_quat = np.array(ts_grp['robot_endeffector_q']).flatten()
                # 将四元数从 wxyz 格式转换为 xyzw 格式
                # 输入格式: [w, x, y, z] -> 输出格式: [x, y, z, w]
                if len(gripper_quat) == 4:
                    gripper_quat = np.array([gripper_quat[1], gripper_quat[2], gripper_quat[3], gripper_quat[0]])
                
                current_obs.gripper_pose = np.concatenate([gripper_pos, gripper_quat])
                
                if 'action' in ts_grp:
                    action = np.array(ts_grp['action'])
                    # 映射action[-1]到gripper_open:
                    # action[-1] = -1 表示关闭夹爪 -> gripper_open = 0 (关闭)
                    # action[-1] = 1 表示打开夹爪 -> gripper_open = 1 (打开)
                    # 使用公式 (action[-1] + 1) / 2 进行映射: -1 -> 0, 1 -> 1
                    current_obs.gripper_open = float((action[-1] + 1) / 2)

                
                # 设置 gripper_joint_positions（左右手指关节位置）
                # 对于 HDF5 文件，如果没有这个字段，使用默认值
                # 根据 gripper_open 状态设置：打开时为 [0.04, 0.04]，关闭时为 [0.0, 0.0]
                # 注意：extract_obs 会将值裁剪到 [0.0, 0.04] 范围
                if current_obs.gripper_open > 0:
                    # 夹爪打开时，手指关节位置接近最大值 0.04
                    current_obs.gripper_joint_positions = np.array([0.04, 0.04], dtype=np.float32)
                else:
                    # 夹爪关闭时，手指关节位置为 0.0
                    current_obs.gripper_joint_positions = np.array([0.0, 0.0], dtype=np.float32)
                
                current_obs.ignore_collisions = 0
                
                # --- 4. 构建 misc ---
                current_obs.misc = {}

                # --- Read Keypoint Info ---
                if 'keypoint_type' in ts_grp:
                    val = ts_grp['keypoint_type'][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    current_obs.misc['keypoint_type'] = val
                else:
                    current_obs.misc['keypoint_type'] = None

                if 'keypoint_solve_function' in ts_grp:
                    val = ts_grp['keypoint_solve_function'][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    current_obs.misc['keypoint_solve_function'] = val
                else:
                    current_obs.misc['keypoint_solve_function'] = None
                
                if 'keypoint_gripper_open' in ts_grp:
                     current_obs.misc['keypoint_gripper_open'] = bool(ts_grp['keypoint_gripper_open'][()])
                else:
                     current_obs.misc['keypoint_gripper_open'] = None
                
                # Extrinsics and Intrinsics
                if 'base_camera_extrinsic_opencv' in ts_grp and 'base_camera_intrinsic_opencv' in ts_grp:
                    ext_opencv = np.array(ts_grp['base_camera_extrinsic_opencv'])
                    intr_opencv = np.array(ts_grp['base_camera_intrinsic_opencv'])
                    if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
                    if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]
                    if ext_opencv.shape == (3, 4):
                        ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
                    
                    # 转换为 CoppeliaSim 格式
                    ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
                        ext_opencv, intr_opencv
                    )
                    current_obs.misc['front_camera_extrinsics'] = ext_coppeliasim
                    current_obs.misc['front_camera_intrinsics'] = intr_coppeliasim

                if 'wrist_camera_extrinsic_opencv' in ts_grp and 'wrist_camera_intrinsic_opencv' in ts_grp:
                    ext_opencv = np.array(ts_grp['wrist_camera_extrinsic_opencv'])
                    intr_opencv = np.array(ts_grp['wrist_camera_intrinsic_opencv'])
                    if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
                    if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]
                    if ext_opencv.shape == (3, 4):
                        ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
                    
                    # 转换为 CoppeliaSim 格式
                    ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
                        ext_opencv, intr_opencv
                    )
                    current_obs.misc['wrist_camera_extrinsics'] = ext_coppeliasim
                    current_obs.misc['wrist_camera_intrinsics'] = intr_coppeliasim

                # Mock missing cameras misc
                # current_obs.misc['left_shoulder_camera_extrinsics'] = np.eye(4)
                # current_obs.misc['left_shoulder_camera_intrinsics'] = np.eye(3)
                # current_obs.misc['right_shoulder_camera_extrinsics'] = np.eye(4)
                # current_obs.misc['right_shoulder_camera_intrinsics'] = np.eye(3)

                # --- 5. 生成点云 ---
                if 'front_camera_extrinsics' in current_obs.misc:
                    current_obs.front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        current_obs.front_depth,
                        current_obs.misc['front_camera_extrinsics'],
                        current_obs.misc['front_camera_intrinsics']
                    )
                else:
                    # 如果没有相机参数，生成零填充的点云（图像格式）
                    current_obs.front_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)
                     
                # 为 left_shoulder 和 right_shoulder 生成点云（即使深度为零，也要生成正确形状的点云）
                # 使用相机参数生成点云，即使深度为零也会生成正确形状的点云 (128, 128, 3)
                # current_obs.left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                #     current_obs.left_shoulder_depth,
                #     current_obs.misc['left_shoulder_camera_extrinsics'],
                #     current_obs.misc['left_shoulder_camera_intrinsics']
                # )
                # current_obs.right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                #     current_obs.right_shoulder_depth,
                #     current_obs.misc['right_shoulder_camera_extrinsics'],
                #     current_obs.misc['right_shoulder_camera_intrinsics']
                # )
                
                # 如果有 wrist camera 参数，生成点云
                if 'wrist_camera_extrinsics' in current_obs.misc:
                     current_obs.wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        current_obs.wrist_depth,
                        current_obs.misc['wrist_camera_extrinsics'],
                        current_obs.misc['wrist_camera_intrinsics']
                    )
                else:
                    # 如果没有 wrist camera 参数，生成零填充的点云（图像格式）
                    current_obs.wrist_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)

                obs.append(current_obs)
            
            # --- 6. 读取 episode 级别的 setup_language_goal ---
            # 尝试从 episode 组中读取 setup_language_goal
            if 'setup_language_goal' in ep_grp:
                lang_goal_val = ep_grp['setup_language_goal'][()]
                if isinstance(lang_goal_val, bytes):
                    lang_goal_val = lang_goal_val.decode('utf-8')
                elif isinstance(lang_goal_val, np.ndarray):
                    # 如果是 numpy 数组，尝试转换为字符串
                    if lang_goal_val.dtype == object or lang_goal_val.dtype.type == np.str_:
                        lang_goal_val = str(lang_goal_val.item())
                    else:
                        lang_goal_val = lang_goal_val.tobytes().decode('utf-8').strip('\x00')
                obs.setup_language_goal = lang_goal_val
            elif 'language_goal' in ep_grp:
                # 兼容其他可能的字段名
                lang_goal_val = ep_grp['language_goal'][()]
                if isinstance(lang_goal_val, bytes):
                    lang_goal_val = lang_goal_val.decode('utf-8')
                elif isinstance(lang_goal_val, np.ndarray):
                    if lang_goal_val.dtype == object or lang_goal_val.dtype.type == np.str_:
                        lang_goal_val = str(lang_goal_val.item())
                    else:
                        lang_goal_val = lang_goal_val.tobytes().decode('utf-8').strip('\x00')
                obs.setup_language_goal = lang_goal_val
            else:
                # 如果没有找到，设置为 None，后续使用默认值
                obs.setup_language_goal = None
        
        obs.variation_number = 0
        return obs

    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
    
    # 加载低维观察数据（pickle 文件）
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)
    
    # 加载任务变体编号
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)
    
    # 遍历所有帧，加载图像和深度数据
    num_steps = len(obs)
    for i in range(num_steps):
        # 加载 RGB 图像
        obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        # obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        # obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))
        
        # 加载并处理深度图像（front 相机）
        obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[i].front_depth = near + obs[i].front_depth * (far - near)
        
        # 加载并处理深度图像（left_shoulder 相机）
        # obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        # obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        # near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
        # far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
        # obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)
        
        # 加载并处理深度图像（right_shoulder 相机）
        # obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        # near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
        # far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
        # obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)
        
        # 加载并处理深度图像（wrist 相机）
        obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)
        
        # 生成点云数据（从深度图像和相机参数）
        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].front_depth, 
            obs[i].misc['front_camera_extrinsics'],
            obs[i].misc['front_camera_intrinsics']
        )
        # obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #     obs[i].left_shoulder_depth, 
        #     obs[i].misc['left_shoulder_camera_extrinsics'],
        #     obs[i].misc['left_shoulder_camera_intrinsics']
        # )
        # obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #     obs[i].right_shoulder_depth, 
        #     obs[i].misc['right_shoulder_camera_extrinsics'],
        #     obs[i].misc['right_shoulder_camera_intrinsics']
        # )
        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].wrist_depth, 
            obs[i].misc['wrist_camera_extrinsics'],
            obs[i].misc['wrist_camera_intrinsics']
        )
    
    return obs

# ============================================================================
# 辅助函数：动作离散化
# ============================================================================
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    """
    将连续动作离散化为体素索引和旋转索引
    
    该函数将机器人的连续动作（位置、旋转、抓取器状态）离散化为离散的动作索引，
    以便在离散动作空间中进行强化学习。主要处理：
    1. 平移动作：将 3D 位置离散化为体素网格索引（支持多尺度）
    2. 旋转动作：将四元数旋转离散化为欧拉角索引
    3. 抓取器动作：将抓取器开合状态转换为二进制索引
    4. 碰撞忽略标志：从观察中提取是否忽略碰撞的标志
    
    参数:
        obs_tp1 (Observation): 目标时刻的观察对象，包含目标位置和姿态信息
                              obs_tp1.gripper_pose 包含 [x, y, z, qx, qy, qz, qw]（位置+四元数）
        obs_tm1 (Observation): 前一时刻的观察对象，用于提取碰撞忽略标志
        rlbench_scene_bounds (List[float]): 场景的 3D 边界，格式为 [x_min, x_max, y_min, y_max, z_min, z_max]
                                           用于定义体素网格的空间范围
        voxel_sizes (List[int]): 体素大小列表，例如 [100, 200] 表示两个尺度的体素网格
                                每个尺度对应不同的空间分辨率
        rotation_resolution (int): 旋转动作的离散化分辨率，将连续旋转空间划分为固定数量的离散动作
                                  例如 5 表示每个欧拉角轴有 5 个离散值
        crop_augmentation (bool): 是否启用裁剪增强（当前版本未使用，保留以兼容接口）
    
    返回:
        tuple: 包含以下元素的元组
            - trans_indicies (List[int]): 平移动作的体素索引列表
                                        长度为 3 * len(voxel_sizes)，每个尺度对应 3 个坐标轴（x, y, z）
            - rot_and_grip_indicies (List[int]): 旋转和抓取器动作索引列表
                                               长度为 4：[rot_x_idx, rot_y_idx, rot_z_idx, grip_idx]
            - ignore_collisions (int): 碰撞忽略标志，0 或 1
            - action (np.ndarray): 连续动作值，形状为 (8,)，包含 [x, y, z, qx, qy, qz, qw, grip]
            - attention_coordinates (List[np.ndarray]): 注意力坐标列表，每个尺度对应一个坐标
                                                      用于多尺度注意力机制
    
    工作流程:
        1. 归一化四元数并转换为离散欧拉角索引
        2. 对每个体素尺度，将 3D 位置转换为体素索引
        3. 计算每个尺度的注意力坐标（体素中心位置）
        4. 提取抓取器状态并转换为索引
        5. 返回所有离散索引和连续动作值
    """
    # ========================================================================
    # 步骤 1: 处理旋转动作（四元数 -> 离散欧拉角索引）
    # ========================================================================
    # 从目标观察中提取四元数（gripper_pose 的后 4 个元素）
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    
    # 确保四元数的 w 分量为正（四元数的标准形式）
    # 四元数 q 和 -q 表示相同的旋转，但为了保持一致性，我们选择 w > 0 的表示
    if quat[-1] < 0:
        quat = -quat
    
    # 将归一化的四元数转换为离散的欧拉角索引
    # disc_rot 是一个包含 3 个整数的列表，分别对应绕 x、y、z 轴的旋转索引
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    
    # ========================================================================
    # 步骤 2: 处理平移动作（3D 位置 -> 体素索引）
    # ========================================================================
    # 初始化注意力坐标为抓取器的当前位置（前 3 个元素是 x, y, z）
    attention_coordinate = obs_tp1.gripper_pose[:3]
    
    # 初始化平移索引和注意力坐标列表
    trans_indicies = []  # 存储所有尺度的体素索引
    attention_coordinates = []  # 存储所有尺度的注意力坐标
    
    # 将场景边界转换为 NumPy 数组，便于计算
    bounds = np.array(rlbench_scene_bounds)
    
    # 从前一时刻的观察中提取碰撞忽略标志（0 或 1）
    ignore_collisions = int(obs_tm1.ignore_collisions)
    
    # 对每个体素尺度进行处理（支持多尺度表示）
    for depth, vox_size in enumerate(voxel_sizes):
        # 将 3D 位置转换为体素网格索引
        # index 是一个包含 3 个整数的数组，表示在体素网格中的位置 [i, j, k]
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3],  # 目标位置的 3D 坐标
            vox_size,                  # 当前尺度的体素大小
            bounds                      # 场景边界
        )
        
        # 将索引添加到平移索引列表中（每个尺度贡献 3 个索引：x, y, z）
        trans_indicies.extend(index.tolist())
        
        # 计算体素的分辨率（每个体素对应的实际空间大小）
        res = (bounds[3:] - bounds[:3]) / vox_size
        
        # 计算当前尺度的注意力坐标（体素中心的位置）
        # 这是用于多尺度注意力机制的关键坐标
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)
    
    # ========================================================================
    # 步骤 3: 处理抓取器动作
    # ========================================================================
    # 将旋转索引转换为列表
    rot_and_grip_indicies = disc_rot.tolist()
    
    # 提取抓取器的开合状态（True/False -> 1.0/0.0）
    grip = float(obs_tp1.gripper_open)


    
    # 将抓取器状态添加到旋转索引列表的末尾
    # 最终列表长度为 4：[rot_x_idx, rot_y_idx, rot_z_idx, grip_idx]
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    
    # ========================================================================
    # 步骤 4: 构建连续动作值
    # ========================================================================
    # 将抓取器姿态和抓取器状态拼接成连续动作向量
    # 形状为 (8,)：[x, y, z, qx, qy, qz, qw, grip]
    action = np.concatenate([obs_tp1.gripper_pose, np.array([grip])])
    
    # 返回所有离散索引和连续动作值
    return (
        trans_indicies,           # 平移体素索引列表
        rot_and_grip_indicies,     # 旋转和抓取器索引列表
        ignore_collisions,         # 碰撞忽略标志
        action,                    # 连续动作值
        attention_coordinates,     # 注意力坐标列表
    )


# ============================================================================
# 辅助函数：CLIP 文本编码
# ============================================================================
def _clip_encode_text(clip_model, text):
    """
    使用 CLIP 模型提取文本的语言特征
    
    该函数使用预训练的 CLIP 模型将文本描述编码为高维特征向量。
    CLIP (Contrastive Language-Image Pre-training) 是一个多模态模型，
    能够将文本和图像映射到同一个特征空间，便于进行跨模态匹配。
    
    参数:
        clip_model: CLIP 模型对象，包含 token embedding、transformer 等组件
        text (torch.Tensor): 已 tokenize 的文本张量，形状为 (batch_size, seq_len)
                           其中 seq_len 通常是 77（CLIP 的最大序列长度）
    
    返回:
        tuple: 包含以下元素的元组
            - lang_feats (torch.Tensor): 语言特征向量，形状为 (batch_size, feature_dim)
                                       用于条件化策略学习的主要特征
            - lang_embs (torch.Tensor): 语言嵌入序列，形状为 (batch_size, seq_len, emb_dim)
                                      包含序列中每个 token 的嵌入表示
    
    工作流程:
        1. 将文本 token 转换为嵌入向量
        2. 添加位置编码
        3. 通过 Transformer 编码器处理
        4. 提取全局特征和序列嵌入
    """
    # ========================================================================
    # 步骤 1: Token 嵌入
    # ========================================================================
    # 将文本 token 转换为嵌入向量
    # x 的形状为 (batch_size, seq_len, d_model)，其中 d_model 是嵌入维度
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    
    # ========================================================================
    # 步骤 2: 添加位置编码
    # ========================================================================
    # CLIP 使用可学习的位置编码，将其添加到 token 嵌入中
    # 位置编码的形状为 (seq_len, d_model)，会自动广播到 batch 维度
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    
    # ========================================================================
    # 步骤 3: Transformer 编码
    # ========================================================================
    # 将张量从 (batch, seq, dim) 转换为 (seq, batch, dim) 以适配 Transformer
    # Transformer 期望输入格式为 (sequence_length, batch_size, feature_dim)
    x = x.permute(1, 0, 2)  # NLD -> LND (N=batch, L=seq, D=dim)
    
    # 通过 Transformer 编码器处理
    # Transformer 包含多层自注意力机制和前馈网络
    x = clip_model.transformer(x)
    
    # 将张量转换回 (batch, seq, dim) 格式
    x = x.permute(1, 0, 2)  # LND -> NLD
    
    # 应用最终的层归一化
    x = clip_model.ln_final(x).type(clip_model.dtype)
    
    # ========================================================================
    # 步骤 4: 提取特征
    # ========================================================================
    # 克隆完整的嵌入序列（用于后续使用）
    emb = x.clone()  # 形状: (batch_size, seq_len, emb_dim)
    
    # 提取全局特征：选择每个序列中最后一个有效 token 的特征
    # text.argmax(dim=-1) 找到每个序列中最后一个非零 token 的位置（通常是 EOS token）
    # 然后通过文本投影层得到最终的语言特征
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    # x 的形状: (batch_size, feature_dim)
    
    # 返回全局特征和序列嵌入
    return x, emb


# ============================================================================
# 辅助函数：添加关键点到 Replay Buffer
# ============================================================================
def _add_keypoints_to_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    """
    将关键点数据添加到时序 replay buffer 中
    
    该函数处理一个演示中的关键点序列，将每个关键点的观察、动作、奖励等信息
    添加到 replay buffer 中。关键点是演示中的重要时刻（例如抓取、放置等动作）。
    函数会从起始关键点开始，逐个处理后续的所有关键点，为每个关键点创建
    一个训练样本。
    
    参数:
        replay (ReplayBuffer): 目标 replay buffer 对象，用于存储训练数据
        task (str): 任务名称，例如 'close_jar'、'open_drawer' 等
        task_replay_storage_folder (str): replay buffer 的磁盘存储目录路径
        episode_idx (int): 当前演示的索引（演示编号）
        sample_frame (int): 起始采样帧的索引（演示中的帧编号）
        inital_obs (Observation): 初始观察对象，作为第一个关键点的观察
        demo (Demo): 完整的演示对象，包含演示的所有帧
        episode_keypoints (List[int]): 关键点帧索引列表，例如 [10, 25, 40]
                                     表示第 10、25、40 帧是关键点
        cameras (List[str]): 相机名称列表，例如 ['front', 'left_shoulder', 'wrist']
        rlbench_scene_bounds (List[float]): 场景的 3D 边界
        voxel_sizes (List[int]): 体素大小列表
        rotation_resolution (int): 旋转动作的离散化分辨率
        crop_augmentation (bool): 是否启用裁剪增强
        next_keypoint_idx (int): 下一个要处理的关键点索引（在 episode_keypoints 中的位置）
        description (str): 任务描述文本，例如 "close the jar"
        clip_model: CLIP 模型对象，用于提取语言特征
        device (str): 计算设备，'cuda:X' 或 'cpu'
    
    工作流程:
        1. 从 next_keypoint_idx 开始遍历所有后续关键点
        2. 对每个关键点：
           a. 提取关键点帧和前一帧的观察
           b. 计算离散动作和连续动作
           c. 提取观察特征（图像、深度、点云等）
           d. 使用 CLIP 提取语言特征
           e. 构建完整的样本数据
           f. 添加到 replay buffer
        3. 添加最终状态观察（用于时序学习）
    
    注意:
        - 函数会为每个关键点创建一个训练样本
        - 最后一个关键点会被标记为 terminal（终止状态）
        - 奖励只在最后一个关键点时为 1.0，其他为 0.0
        - 时间步 t 是相对于起始关键点的偏移量
    """
    # ========================================================================
    # 初始化
    # ========================================================================
    # 前一个动作（初始为 None，第一个关键点没有前一个动作）
    prev_action = None
    
    # 当前观察（初始为传入的初始观察）
    obs = inital_obs
    
    # 记录初始帧索引（用于元数据）
    initial_frame = sample_frame
    
    # ========================================================================
    # 遍历所有后续关键点
    # ========================================================================
    # 从 next_keypoint_idx 开始，处理所有后续的关键点
    # k 是关键点在 episode_keypoints 列表中的索引
    for k in range(next_keypoint_idx, len(episode_keypoints)):
        # 获取当前关键点的帧索引（演示中的帧编号）
        keypoint = episode_keypoints[k]
        
        # 提取关键点帧的观察（目标状态）
        obs_tp1 = demo[keypoint]
        
        # 提取关键点前一帧的观察（用于计算动作变化）
        # max(0, keypoint - 1) 确保索引不会为负
        obs_tm1 = demo[max(0, keypoint - 1)]
        
        # ====================================================================
        # 计算动作（离散化和连续值）
        # ====================================================================
        (
            trans_indicies,        # 平移体素索引列表
            rot_grip_indicies,     # 旋转和抓取器索引列表
            ignore_collisions,     # 碰撞忽略标志
            action,                # 连续动作值
            attention_coordinates, # 注意力坐标列表
        ) = _get_action(
            obs_tp1,               # 目标观察
            obs_tm1,               # 前一帧观察
            rlbench_scene_bounds,  # 场景边界
            voxel_sizes,           # 体素大小列表
            rotation_resolution,   # 旋转分辨率
            crop_augmentation,     # 裁剪增强标志
        )
        
        # ====================================================================
        # 确定终止状态和奖励
        # ====================================================================
        # 如果当前关键点是最后一个关键点，则标记为终止状态
        terminal = k == len(episode_keypoints) - 1
        
        # 奖励设计：只在最后一个关键点（任务完成）时给予奖励 1.0
        # 其他关键点的奖励为 0.0（稀疏奖励设置）
        reward = float(terminal) * 1.0 if terminal else 0
        
        # ====================================================================
        # 提取观察特征
        # ====================================================================
        # 从当前观察中提取特征（图像、深度、点云、相机参数等）
        # t 是时间步，相对于起始关键点的偏移量
        # 例如：如果从关键点 0 开始，当前是关键点 2，则 t = 2 - 0 = 2
        obs_dict = extract_obs(
            obs,                   # 当前观察对象
            CAMERAS,               # 相机列表（使用全局常量）
            t=k - next_keypoint_idx,  # 时间步偏移量
            prev_action=prev_action,  # 前一个动作（用于时序建模）
            episode_length=25,     # 演示的最大长度（用于时间归一化）
        )
        
        # ====================================================================
        # 提取语言特征
        # ====================================================================
        # 使用 CLIP tokenizer 将文本描述转换为 token
        tokens = clip.tokenize([description]).numpy()
        
        # 将 token 转换为 PyTorch 张量并移动到指定设备
        token_tensor = torch.from_numpy(tokens).to(device)
        
        # 使用 CLIP 模型提取语言特征（禁用梯度计算以节省内存）
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        
        # 将语言嵌入添加到观察字典中
        # lang_embs[0] 是第一个（也是唯一一个）样本的嵌入
        # 转换为 NumPy 数组并移动到 CPU
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
        
        # 保存当前动作作为下一个关键点的前一个动作
        prev_action = np.copy(action)
        
        # ====================================================================
        # 构建元数据
        # ====================================================================
        # 确定前一个关键点的帧索引
        if k == 0:
            # 如果是第一个关键点，前一个关键点帧设为 -1（表示没有前一个关键点）
            keypoint_frame = -1
        else:
            # 否则，前一个关键点帧是 episode_keypoints[k-1]
            keypoint_frame = episode_keypoints[k - 1]
        
        # 构建其他元数据字典
        others = {
            "demo": True,                      # 标记为演示数据（而非真实交互数据）
            "keypoint_idx": k,                 # 当前关键点在关键点列表中的索引
            "episode_idx": episode_idx,        # 演示索引
            "keypoint_frame": keypoint_frame,  # 前一个关键点的帧索引
            "next_keypoint_frame": keypoint,    # 下一个关键点的帧索引（即当前关键点）
            "sample_frame": sample_frame,      # 采样起始帧
            "initial_frame": initial_frame,    # 初始帧（用于追踪）
        }
        
        # 构建最终观察数据（包含动作索引和目标信息）
        final_obs = {
            "trans_action_indicies": trans_indicies,      # 平移动作索引
            "rot_grip_action_indicies": rot_grip_indicies,  # 旋转和抓取器动作索引
            "gripper_pose": obs_tp1.gripper_pose,        # 抓取器姿态（位置+四元数）
            "lang_goal": np.array([description], dtype=object),  # 语言目标（文本描述）
        }
        
        # 合并所有数据字典
        others.update(final_obs)  # 先添加最终观察数据
        others.update(obs_dict)    # 再添加观察特征（会覆盖同名的键）
        
        # ====================================================================
        # 添加到 Replay Buffer
        # ====================================================================
        # timeout 标志（当前未使用，设为 False）
        timeout = False
        
        # 将样本添加到 replay buffer
        replay.add(
            task,                          # 任务名称
            task_replay_storage_folder,    # 存储目录
            action,                        # 连续动作值
            reward,                        # 奖励值
            terminal,                      # 终止标志
            timeout,                       # 超时标志
            **others                       # 其他所有数据（观察、动作索引、元数据等）
        )
        
        # ====================================================================
        # 更新状态
        # ====================================================================
        # 更新当前观察为目标观察（为下一个关键点准备）
        obs = obs_tp1
        
        # 更新采样帧为当前关键点帧
        sample_frame = keypoint
    
    # ========================================================================
    # 添加最终状态观察
    # ========================================================================
    # 只有在处理了至少一个关键点的情况下，才添加最终状态观察
    # 如果循环没有执行（next_keypoint_idx >= len(episode_keypoints)），则跳过
    if next_keypoint_idx < len(episode_keypoints):
        # 提取最后一个关键点的观察特征（用于时序学习）
        # 这是最终状态的观察，不包含动作（因为任务已完成）
        obs_dict_tp1 = extract_obs(
            obs_tp1,                           # 最后一个关键点的观察
            CAMERAS,                           # 相机列表
            t=k + 1 - next_keypoint_idx,       # 时间步（比最后一个关键点多 1）
            prev_action=prev_action,           # 最后一个动作
            episode_length=25,                 # 演示长度
        )
        
        # 添加语言嵌入（使用之前计算的嵌入）
        obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
        
        # 移除不需要的键（如果存在）
    obs_dict_tp1.pop("wrist_world_to_cam", None)
    
    # 添加最终观察数据（动作索引和目标信息）
    obs_dict_tp1.update(final_obs)
    others.update(obs_dict_tp1)

    # 将最终状态添加到 replay buffer（使用特殊的 add_final 方法）
    replay.add_final(task, task_replay_storage_folder, **others)


# ============================================================================
# 主函数：填充时序 Replay Buffer
# ============================================================================
def fill_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    rank,
    clip_model=None,
    device="cpu",
):
    """
    将 RLBench 演示数据填充到时序 replay buffer 中
    
    该函数是填充 replay buffer 的主函数，负责：
    1. 检查磁盘上是否已存在处理好的 replay buffer 数据
    2. 如果存在，直接从磁盘加载（快速恢复）
    3. 如果不存在，从原始演示数据中处理并填充：
       a. 加载演示数据
       b. 提取关键点
       c. 处理每个演示的每一帧（根据数据增强设置）
       d. 将关键点数据添加到 replay buffer
    4. 保存终止状态信息到磁盘
    
    参数:
        replay (ReplayBuffer): 目标 replay buffer 对象
        task (str): 任务名称，例如 'close_jar'
        task_replay_storage_folder (str): replay buffer 的磁盘存储目录路径
        start_idx (int): 起始演示索引，从第几个演示开始处理
        num_demos (int): 要处理的演示数量
        demo_augmentation (bool): 是否启用演示数据增强
                                如果为 True，会从每个演示中采样多个起始帧
                                如果为 False，只使用第一个帧作为起始点
        demo_augmentation_every_n (int): 数据增强采样间隔，每 N 帧采样一次
                                        例如 10 表示每 10 帧采样一个起始帧
        cameras (List[str]): 相机名称列表
        rlbench_scene_bounds (List[float]): 场景的 3D 边界
        voxel_sizes (List[int]): 体素大小列表
        rotation_resolution (int): 旋转动作的离散化分辨率
        crop_augmentation (bool): 是否启用裁剪增强（当前未使用）
        data_path (str): 原始演示数据的路径
        episode_folder (str): 演示数据文件夹名称，通常为 'episodes' 或 'episode_%d'
        variation_desriptions_pkl (str): 任务变体描述文件的名称，例如 'variation_descriptions.pkl'
        rank (int): 进程排名，用于多进程训练时的日志输出控制
        clip_model: CLIP 模型对象，用于提取语言特征
        device (str): 计算设备，'cuda:X' 或 'cpu'
    
    工作流程:
        1. 检查磁盘上是否已存在 replay buffer 数据
        2. 如果存在，直接恢复（跳过处理）
        3. 如果不存在：
           a. 创建存储目录
           b. 遍历所有演示（从 start_idx 到 start_idx + num_demos）
           c. 对每个演示：
              - 加载演示数据
              - 加载任务描述
              - 提取关键点
              - 遍历演示的每一帧（根据数据增强设置）
              - 对每个采样帧，添加从该帧开始的关键点序列
           d. 保存终止状态信息到磁盘
    
    注意:
        - 如果 replay buffer 已存在于磁盘，函数会直接加载，不会重新处理
        - 数据增强通过采样不同的起始帧来增加数据多样性
        - 每个演示的关键点序列会被多次添加到 replay buffer（如果启用数据增强）
        - 终止状态信息会保存到 replay_info.npy 文件中
    """
    # ========================================================================
    # 步骤 1: 检查磁盘上是否已存在 Replay Buffer 数据
    # ========================================================================
    disk_exist = False  # 标记磁盘数据是否存在
    
    # 如果 replay buffer 启用了磁盘存储功能
    if replay._disk_saving:
        # 检查存储目录是否存在
        if os.path.exists(task_replay_storage_folder):
            # 如果存在，打印信息（只在主进程打印，避免多进程重复输出）
            if rank == 0:
                print(
                    "[Info] Replay dataset already exists in the disk: {}".format(
                        task_replay_storage_folder
                    ),
                    flush=True,
                )
            # 标记为已存在
            disk_exist = True
        else:
            # 如果不存在，创建存储目录并记录日志
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)
    
    # ========================================================================
    # 步骤 2: 恢复或填充 Replay Buffer
    # ========================================================================
    if disk_exist:
        # 如果磁盘数据已存在，直接从磁盘恢复
        # 这会加载之前保存的所有数据，跳过重新处理
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        # 如果磁盘数据不存在，需要从原始演示数据中处理并填充
        print("Filling replay ...:", task)
        
        # 遍历所有要处理的演示
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            
            # ====================================================================
            # 步骤 2.1: 加载演示数据
            # ====================================================================
            # 从磁盘加载存储的演示数据
            # demo 是一个 Demo 对象，包含演示的所有帧（Observation 对象列表）
            demo = get_stored_demo(data_path=data_path, index=d_idx)
            
            # ====================================================================
            # 步骤 2.2: 加载任务描述
            # ====================================================================
            # 检查是否是 HDF5 文件
            if data_path.endswith('.h5'):
                # 对于 HDF5 文件，从 demo 对象中读取 setup_language_goal
                # 如果 demo 对象有 setup_language_goal 属性且不为 None，使用它
                if hasattr(demo, 'setup_language_goal') and demo.setup_language_goal is not None:
                    descs = [demo.setup_language_goal]
                else:
                    # 如果未找到，尝试直接从 HDF5 文件读取（作为备用方案）
                    try:
                        with h5py.File(data_path, 'r') as f:
                            env_name = list(f.keys())[0]
                            episode_name = f'episode_{d_idx}'
                            if episode_name in f[env_name]:
                                ep_grp = f[env_name][episode_name]
                                if 'setup_language_goal' in ep_grp:
                                    lang_goal_val = ep_grp['setup_language_goal'][()]
                                    if isinstance(lang_goal_val, bytes):
                                        lang_goal_val = lang_goal_val.decode('utf-8')
                                    elif isinstance(lang_goal_val, np.ndarray):
                                        if lang_goal_val.dtype == object or lang_goal_val.dtype.type == np.str_:
                                            lang_goal_val = str(lang_goal_val.item())
                                        else:
                                            lang_goal_val = lang_goal_val.tobytes().decode('utf-8').strip('\x00')
                                    descs = [lang_goal_val]
                                elif 'language_goal' in ep_grp:
                                    lang_goal_val = ep_grp['language_goal'][()]
                                    if isinstance(lang_goal_val, bytes):
                                        lang_goal_val = lang_goal_val.decode('utf-8')
                                    elif isinstance(lang_goal_val, np.ndarray):
                                        if lang_goal_val.dtype == object or lang_goal_val.dtype.type == np.str_:
                                            lang_goal_val = str(lang_goal_val.item())
                                        else:
                                            lang_goal_val = lang_goal_val.tobytes().decode('utf-8').strip('\x00')
                                    descs = [lang_goal_val]
                                else:
                                    # 如果都找不到，使用默认任务描述
                                    descs = [f"{task} task"]
                            else:
                                descs = [f"{task} task"]
                    except Exception as e:
                        # 如果读取失败，使用默认任务描述
                        print(f"警告: 无法从 HDF5 文件读取 setup_language_goal，使用默认描述: {e}")
                        descs = [f"{task} task"]
            else:
                # 对于标准的 RLBench 数据结构，从 pickle 文件中加载任务描述
                # 构建任务变体描述文件的路径
                # episode_folder 通常是 'episode_%d' 或 'episodes'，需要格式化
                varation_descs_pkl_file = os.path.join(
                    data_path,                          # 数据根目录
                    episode_folder % d_idx,             # 演示文件夹（格式化后的路径）
                    variation_desriptions_pkl           # 描述文件名
                )
                
                # 从 pickle 文件中加载任务描述列表
                # descs 是一个字符串列表，包含不同变体的任务描述
                with open(varation_descs_pkl_file, "rb") as f:
                    descs = pickle.load(f)
            
            # ====================================================================
            # 步骤 2.3: 提取关键点
            # ====================================================================
            # 从演示数据中发现关键点（重要时刻的帧索引）
            # episode_keypoints 是一个整数列表，例如 [10, 25, 40, 55]
            # 表示第 10、25、40、55 帧是关键点
            
            # Determine keypoint discovery method


            episode_keypoints = keypoint_discovery(demo, method='dataset')
            
            # 初始化下一个关键点索引（用于追踪当前处理到哪个关键点）
            next_keypoint_idx = 0

            # ====================================================================
            # 步骤 2.4: 遍历演示的每一帧（数据增强）
            # ====================================================================
            # 遍历演示的所有帧（除了最后一帧，因为最后一帧没有下一帧）
            for i in range(len(demo) - 1):
                # 如果未启用数据增强，且不是第一帧，则跳出循环
                # 这意味着只使用第一帧作为起始点
                if not demo_augmentation and i > 0:
                    break
                
                # 根据数据增强采样间隔，只处理每 N 帧
                # 例如 demo_augmentation_every_n=10 表示每 10 帧采样一次
                if i % demo_augmentation_every_n != 0:
                    continue
                
                # 获取当前帧的观察
                obs = demo[i]
                
                # 获取任务描述（使用第一个变体的描述）
                # 确保 desc 是字符串类型
                desc = str(descs[0]) if descs and descs[0] is not None else f"{task} task"
                
                # ============================================================
                # 步骤 2.5: 更新关键点索引
                # ============================================================
                # 如果当前采样帧已经超过了某个关键点，需要更新 next_keypoint_idx
                # 这样可以确保只处理从当前帧开始的后续关键点
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                
                # 如果当前帧已经超过了所有关键点，跳出循环
                # 这意味着从这个帧开始没有可用的关键点序列
                if next_keypoint_idx == len(episode_keypoints):
                    break
                
                # ============================================================
                # 步骤 2.6: 添加关键点序列到 Replay Buffer
                # ============================================================
                # 调用辅助函数，将从当前帧开始的所有后续关键点添加到 replay buffer
                _add_keypoints_to_replay_temporal(
                    replay,                      # replay buffer 对象
                    task,                       # 任务名称
                    task_replay_storage_folder, # 存储目录
                    d_idx,                     # 演示索引
                    i,                          # 采样帧索引
                    obs,                        # 当前帧的观察
                    demo,                       # 完整演示对象
                    episode_keypoints,          # 关键点列表
                    cameras,                    # 相机列表
                    rlbench_scene_bounds,       # 场景边界
                    voxel_sizes,                # 体素大小列表
                    rotation_resolution,        # 旋转分辨率
                    crop_augmentation,         # 裁剪增强标志
                    next_keypoint_idx=next_keypoint_idx,  # 下一个关键点索引
                    description=desc,          # 任务描述
                    clip_model=clip_model,      # CLIP 模型
                    device=device,              # 计算设备
                )
        
        # ========================================================================
        # 步骤 3: 保存终止状态信息到磁盘
        # ========================================================================
        # 获取任务在 replay buffer 中的索引
        task_idx = replay._task_index[task]
        
        # 构建保存路径
        replay_info_path = os.path.join(task_replay_storage_folder, "replay_info.npy")
        
        # 保存终止状态信息到 NumPy 文件
        # 这包含了所有样本的终止标志，用于后续的数据分析和采样
        with open(replay_info_path, "wb") as fp:
            np.save(
                fp,
                # 提取该任务的所有终止标志
                # 使用切片操作获取从任务开始索引到结束索引的所有终止标志
                replay._store["terminal"][
                    replay._task_replay_start_index[task_idx]  # 任务开始索引
                    : replay._task_replay_start_index[task_idx]  # 任务结束索引
                    + replay._task_add_count[task_idx].value    # 任务样本数量
                ],
            )
        
        # 打印完成信息
        print("Replay filled with demos.")


# 从 sam2act.utils.peract_utils 导入配置常量
# CAMERAS: 相机名称列表，定义使用哪些相机视角（如 'front', 'left_shoulder' 等）
# SCENE_BOUNDS: 场景边界，定义机器人工作空间的 3D 边界范围 [x_min, x_max, y_min, y_max, z_min, z_max]
# EPISODE_FOLDER: 演示数据文件夹名称，通常为 'episodes'
# VARIATION_DESCRIPTIONS_PKL: 任务变体描述文件的名称，包含不同变体的文本描述
# DEMO_AUGMENTATION_EVERY_N: 数据增强采样间隔，每 N 帧采样一次用于数据增强
# ROTATION_RESOLUTION: 旋转动作的离散化分辨率，将连续旋转空间离散化为固定数量的旋转动作
# VOXEL_SIZES: 体素大小列表，用于将 3D 空间离散化为体素网格，支持多尺度表示
from sam2act.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
CAMERAS = ['front', 'wrist']

# 从 yarr 库导入 PyTorch replay buffer 包装器
# PyTorchReplayBuffer: 将 replay buffer 包装为 PyTorch Dataset，支持 DataLoader 使用
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset_temporal(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    num_maskmem,
    rank,
    sample_distribution_mode="transition_uniform",
):
    """
    创建时序强化学习数据集
    
    该函数负责创建用于训练和测试的时序数据集。主要流程包括：
    1. 创建时序 replay buffer（支持存储多时间步的数据）
    2. 加载 CLIP 模型用于提取语言特征
    3. 遍历所有任务，从 RLBench 数据集中加载演示数据
    4. 将演示数据填充到 replay buffer（支持磁盘存储和恢复）
    5. 将 replay buffer 包装为 PyTorch Dataset
    
    参数:
        tasks (list[str]): 任务名称列表，例如 ['close_jar', 'open_drawer']
        BATCH_SIZE_TRAIN (int): 训练集的批次大小，用于创建 replay buffer
        BATCH_SIZE_TEST (int): 测试集的批次大小，如果为 None 且 only_train=False 会报错
        TRAIN_REPLAY_STORAGE_DIR (str): 训练集 replay buffer 的磁盘存储目录路径
        TEST_REPLAY_STORAGE_DIR (str): 测试集 replay buffer 的磁盘存储目录路径，如果为 None 则只创建训练集
        DATA_FOLDER (str): RLBench 原始数据集的根目录路径
        NUM_TRAIN (int): 每个任务使用的训练演示数量
        NUM_VAL (int): 每个任务使用的验证/测试演示数量，如果为 None 且 only_train=False 会报错
        refresh_replay (bool): 是否刷新已有的 replay buffer 数据
                              如果为 True，会删除已存在的 replay buffer 目录并重新生成
        device (str): 计算设备，'cuda:X' 或 'cpu'，用于 CLIP 模型推理
        num_workers (int): 数据加载的并行工作进程数
        only_train (bool): 是否只创建训练集，如果为 True 则不创建测试集
        num_maskmem (int): 时序记忆模块的数量，用于存储历史掩码信息
        rank (int): 进程排名，用于多进程训练时的标识（单进程训练通常为 0）
        sample_distribution_mode (str): 采样分布模式，默认为 "transition_uniform"
                                        "transition_uniform" 表示均匀采样所有转换
        
    返回:
        tuple: (train_dataset, test_dataset)
            - train_dataset: PyTorch Dataset 对象，用于训练数据加载
            - test_dataset: PyTorch Dataset 对象，用于测试数据加载，如果 only_train=True 则为 None
    
    注意:
        - 函数会加载 CLIP 模型提取语言特征，提取完成后会释放模型以节省显存
        - 如果 replay buffer 已存在于磁盘且 refresh_replay=False，会直接加载已有数据
        - 支持多任务数据集创建，会遍历 tasks 列表中的所有任务
    """

    # ========================================================================
    # 步骤 1: 创建时序 Replay Buffer
    # ========================================================================
    # 创建训练集的时序 replay buffer
    # replay buffer 用于存储和采样强化学习经验数据（状态、动作、奖励等）
    # 时序版本支持存储多时间步的数据，便于处理序列决策问题
    train_replay_buffer = create_replay_temporal(
        batch_size=BATCH_SIZE_TRAIN,  # 批次大小，影响采样时的批量大小
        timesteps=1,                   # 时间步数，当前设置为 1（单步数据）
        disk_saving=True,              # 启用磁盘存储，可以将数据保存到磁盘以便后续快速加载
        cameras=CAMERAS,               # 相机配置，定义需要存储哪些相机视角的数据
        voxel_sizes=VOXEL_SIZES,       # 体素大小列表，用于多尺度 3D 表示
        num_maskmem=num_maskmem,       # 时序记忆模块数量，用于存储历史掩码信息
    )
    
    # 如果不仅创建训练集，还需要创建测试集的 replay buffer
    if not only_train:
        # 创建测试集的时序 replay buffer，配置与训练集类似
        test_replay_buffer = create_replay_temporal(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            num_maskmem=num_maskmem,
        )

    # ========================================================================
    # 步骤 2: 加载预训练语言模型（CLIP）
    # ========================================================================
    # CLIP (Contrastive Language-Image Pre-training) 用于提取任务描述的语言特征
    # 这些语言特征会被存储到 replay buffer 中，用于条件化策略学习
    try:
        # 加载 CLIP-ResNet50 模型，先在 CPU 上加载以避免显存问题
        clip_model, _ = clip.load("RN50", device="cpu")  # RN50 表示 ResNet-50 作为视觉编码器
        # 将模型移动到指定设备（GPU 或 CPU）
        clip_model = clip_model.to(device)
        # 设置为评估模式，禁用 dropout 和 batch normalization 的训练行为
        clip_model.eval()
    except RuntimeError:
        # 如果加载失败（例如模型文件不存在），打印警告并设置为 None
        # 注意：如果 replay buffer 不在磁盘上，这会导致后续处理失败
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    # ========================================================================
    # 步骤 3: 遍历所有任务，加载并处理演示数据
    # ========================================================================
    # 对每个任务分别处理，将演示数据加载到对应的 replay buffer 中
    for task in tasks:  # 遍历任务列表中的每个任务
        # 检查 DATA_FOLDER 是否是 HDF5 文件路径
        if DATA_FOLDER.endswith('.h5'):
            # 如果是 HDF5 文件，直接使用该路径
            data_path_train = DATA_FOLDER
            data_path_val = DATA_FOLDER  # 对于 HDF5 文件，训练集和验证集使用同一个文件
        else:
            # 检查是否存在以 record_dataset_ 开头的 HDF5 文件模式
            # 如果 DATA_FOLDER 是目录，尝试根据任务名称构建数据文件路径
            task_data_file = os.path.join(DATA_FOLDER, f"record_dataset_{task}.h5")
            if os.path.exists(task_data_file):
                # 如果找到对应任务的数据文件，使用该文件
                data_path_train = task_data_file
                data_path_val = task_data_file
            else:
                # 对于标准的 RLBench 数据结构，构建训练集和验证集的演示数据路径
                # RLBench 数据集的目录结构：train/{task}/all_variations/episodes/
                EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
                EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
                # 拼接完整的数据路径
                data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
                data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        
        # 构建 replay buffer 的存储路径（每个任务有独立的存储目录）
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # ====================================================================
        # 步骤 3.1: 如果需要刷新 replay buffer，删除已存在的数据
        # ====================================================================
        # 当 refresh_replay=True 时，删除已有的 replay buffer 数据，强制重新生成
        # 这通常用于数据格式更新或需要重新处理数据的情况
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            # 检查并删除训练集的 replay buffer 目录
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)  # 递归删除整个目录树
                print(f"remove {train_replay_storage_folder}")
            # 检查并删除测试集的 replay buffer 目录
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # ====================================================================
        # 步骤 3.2: 填充训练集 Replay Buffer
        # ====================================================================
        # 从 RLBench 数据集中加载演示数据，处理后填充到训练集 replay buffer
        # 如果 replay buffer 已存在于磁盘，会直接加载；否则会处理原始数据并保存
        fill_replay_temporal(
            replay=train_replay_buffer,                    # 目标 replay buffer 对象
            task=task,                                      # 当前任务名称
            task_replay_storage_folder=train_replay_storage_folder,  # replay buffer 存储路径
            start_idx=0,                                    # 起始演示索引，从第 0 个演示开始
            num_demos=NUM_TRAIN,                            # 使用的演示数量
            demo_augmentation=False,                         # 启用演示数据增强，通过采样不同帧增加数据多样性
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,  # 每 N 帧采样一次用于增强
            cameras=CAMERAS,                                # 相机配置列表
            rlbench_scene_bounds=SCENE_BOUNDS,              # 场景 3D 边界，用于动作离散化
            voxel_sizes=VOXEL_SIZES,                        # 体素大小列表，用于多尺度 3D 表示
            rotation_resolution=ROTATION_RESOLUTION,        # 旋转动作的离散化分辨率
            crop_augmentation=False,                         # 禁用裁剪增强（当前不使用）
            data_path=data_path_train,                      # 原始演示数据的路径
            episode_folder=EPISODE_FOLDER,                  # 演示数据文件夹名称
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,  # 任务变体描述文件
            clip_model=clip_model,                          # CLIP 模型，用于提取语言特征
            device=device,                                  # 计算设备
            rank=rank,                                      # 进程排名，用于多进程训练
        )

        # ====================================================================
        # 步骤 3.3: 填充测试集 Replay Buffer（如果启用）
        # ====================================================================
        # 如果 only_train=False，还需要处理测试集数据
        # 测试集的处理流程与训练集相同，但使用验证集的演示数据
        if not only_train:
            fill_replay_temporal(
                replay=test_replay_buffer,                  # 测试集 replay buffer 对象
                task=task,                                  # 当前任务名称
                task_replay_storage_folder=test_replay_storage_folder,  # 测试集存储路径
                start_idx=0,                                # 起始演示索引
                num_demos=NUM_VAL,                          # 使用的验证演示数量
                demo_augmentation=False,                     # 同样启用数据增强
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,                    # 验证集数据路径（与训练集不同）
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                rank=rank,                                  # 进程排名（与训练集相同）
                clip_model=clip_model,
                device=device,
            )

    # ========================================================================
    # 步骤 4: 释放 CLIP 模型资源
    # ========================================================================
    # 语言特征已经提取并存储到 replay buffer 中，不再需要 CLIP 模型
    # 删除模型并清空 GPU 缓存以释放显存
    del clip_model
    # 在指定设备上清空 CUDA 缓存
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # ========================================================================
    # 步骤 5: 将 Replay Buffer 包装为 PyTorch Dataset
    # ========================================================================
    # 将 replay buffer 包装为 PyTorch Dataset，使其可以与 DataLoader 配合使用
    # 训练集使用随机采样模式，测试集使用枚举模式（按顺序遍历）
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,                    # 训练集 replay buffer
        sample_mode="random",                   # 随机采样模式，每次随机选择样本
        num_workers=num_workers,                # 数据加载的并行工作进程数
        sample_distribution_mode=sample_distribution_mode,  # 采样分布模式
    )
    # 获取 PyTorch Dataset 对象，可以直接用于 DataLoader
    train_dataset = train_wrapped_replay.dataset()

    # 根据 only_train 标志决定是否创建测试集
    if only_train:
        # 如果只创建训练集，测试集设为 None
        test_dataset = None
    else:
        # 创建测试集的 PyTorch Dataset
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,                 # 测试集 replay buffer
            sample_mode="enumerate",             # 枚举模式，按顺序遍历所有样本（用于评估）
            num_workers=num_workers,
            # 注意：测试集通常不需要 sample_distribution_mode，因为使用枚举模式
        )
        test_dataset = test_wrapped_replay.dataset()
    
    # 返回训练集和测试集的 Dataset 对象
    return train_dataset, test_dataset

# ============================================================================
# 数据集路径配置
# ============================================================================
# 训练集 replay buffer 的存储目录
# replay buffer 处理后的数据会保存在此目录下，每个任务有独立的子目录
TRAIN_REPLAY_STORAGE_DIR = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/sam2act/test_buffer"

# RLBench 原始数据集的根目录
# 包含所有任务的演示数据，目录结构为：{DATA_FOLDER}/train/{task}/all_variations/episodes/
# 如果使用 HDF5 文件，设置为数据文件的目录路径（不包含文件名）
# 代码会根据任务名称自动构建对应的数据文件路径：{DATA_FOLDER}/record_dataset_{task}.h5
DATA_FOLDER = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate"

# ============================================================================
# 数据集创建参数
# ============================================================================
# 任务列表：要处理的任务名称列表
# 可以包含多个任务，例如：["close_jar", "open_drawer", "pick_and_lift_simple"]
# 对于 HDF5 文件，任务名称可以自定义（用于 replay buffer 存储目录的组织）
#tasks = ["BinFill","VideoUnmask","VideoPlaceOrder","RouteStick"]
tasks = ["RouteStick"]   # 当前使用 HDF5 数据，任务名称自定义

# 批次大小配置
BATCH_SIZE_TRAIN = 8        # 训练集的批次大小，影响 replay buffer 的采样批量
BATCH_SIZE_TEST = None      # 测试集的批次大小，如果 only_train=True 可以设为 None

# 测试集 replay buffer 存储目录（如果 only_train=False 需要设置）
TEST_REPLAY_STORAGE_DIR = None

# 演示数量配置
NUM_TRAIN = 2   # 每个任务使用的训练演示数量，从演示数据集中选择前 NUM_TRAIN 个演示
NUM_VAL = None   # 每个任务使用的验证演示数量，如果 only_train=True 可以设为 None

# 数据刷新标志
# 如果为 True，会删除已存在的 replay buffer 数据并重新生成
# 如果为 False，会直接加载已存在的 replay buffer 数据（如果存在）
refresh_replay = True  # 刷新已有数据，重新生成 replay buffer

# 计算设备配置
# 自动检测是否有可用的 CUDA 设备，如果有则使用 GPU，否则使用 CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据加载配置
num_workers = 4  # DataLoader 的并行工作进程数，用于加速数据加载

# 数据集模式配置
only_train = True  # 是否只创建训练集，如果为 True 则不创建测试集

# 时序记忆配置
num_maskmem = 8  # 时序记忆模块的数量，用于存储历史掩码信息（默认值）

# 多进程配置
rank = 0  # 进程排名，单进程训练时通常为 0，多进程训练时用于区分不同进程

# 采样分布模式
# "transition_uniform" 表示均匀采样所有状态转换，确保每个转换被采样的概率相等
sample_distribution_mode = "transition_uniform"

# ============================================================================
# 辅助函数：批量将3D点投影到2D像素坐标
# ============================================================================
def project_points_to_pixels(points, extrinsics, intrinsics):
    """
    批量将3D点云投影到2D像素坐标
    
    参数:
        points: (N, 3) numpy数组，世界坐标系下的3D点
        extrinsics: (4, 4) numpy数组，相机外参（CoppeliaSim格式：相机→世界）
        intrinsics: (3, 3) numpy数组，相机内参
    
    返回:
        pixel_coords: (N, 2) numpy数组，像素坐标 (u, v)
        valid_mask: (N,) bool数组，标记哪些点在相机前方（深度>0）
    """
    # 转换为齐次坐标 (N, 4)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    
    # 外参是CoppeliaSim格式（相机→世界），需要求逆得到世界→相机转换
    world_to_cam = np.linalg.inv(extrinsics)
    
    # 将世界坐标转换为相机坐标 (N, 4)
    points_cam_homo = (world_to_cam @ points_homo.T).T
    
    # 提取相机坐标 (N, 3)
    points_cam = points_cam_homo[:, :3]
    
    # 检查哪些点在相机前方（深度>0）
    valid_mask = points_cam[:, 2] > 0
    
    # 使用内参将相机坐标投影到像素坐标
    # 透视投影: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # 避免除以0，对于深度<=0的点使用1
    z = np.maximum(points_cam[:, 2], 1e-8)
    
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    
    # 组合为 (N, 2) 数组
    pixel_coords = np.stack([u, v], axis=1)
    
    return pixel_coords, valid_mask

def main():
    """
    主测试函数
    
    该函数使用 get_dataset_temporal 创建数据集并迭代读取数据：
    1. 打印配置信息
    2. 调用 get_dataset_temporal 创建训练数据集
    3. 创建数据迭代器并读取样本
    4. 打印样本信息以验证数据格式
    """
    # ========================================================================
    # 步骤 0: 打印配置信息
    # ========================================================================
    # 打印测试标题和分隔线
    print("=" * 80)
    print("测试 get_dataset_temporal 函数")
    print("=" * 80)
    # 打印关键配置参数，便于调试和验证
    print(f"TRAIN_REPLAY_STORAGE_DIR: {TRAIN_REPLAY_STORAGE_DIR}")
    print(f"DATA_FOLDER: {DATA_FOLDER}")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE_TRAIN}")
    print(f"Number of training samples: {NUM_TRAIN}")
    print("=" * 80)
    
    try:
        # ====================================================================
        # 步骤 1: 创建数据集
        # ====================================================================
        # 调用 get_dataset_temporal 函数创建训练集和测试集
        # 该函数会处理所有任务，加载演示数据，填充 replay buffer，并返回 PyTorch Dataset
        print("\n[1/3] 调用 get_dataset_temporal...")
        train_dataset, test_dataset = get_dataset_temporal(
            tasks=tasks,
            BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
            BATCH_SIZE_TEST=BATCH_SIZE_TEST,
            TRAIN_REPLAY_STORAGE_DIR=TRAIN_REPLAY_STORAGE_DIR,
            TEST_REPLAY_STORAGE_DIR=TEST_REPLAY_STORAGE_DIR,
            DATA_FOLDER=DATA_FOLDER,
            NUM_TRAIN=NUM_TRAIN,
            NUM_VAL=NUM_VAL,
            refresh_replay=refresh_replay,
            device=device,
            num_workers=num_workers,
            only_train=only_train,
            num_maskmem=num_maskmem,
            rank=rank,
            sample_distribution_mode=sample_distribution_mode,
        )
        print("✓ 数据集创建成功!")
        
        # ====================================================================
        # 步骤 2: 创建数据迭代器
        # ====================================================================
        # 为训练集创建迭代器，用于逐个读取样本
        # 注意：这里使用的是训练集，因为 only_train=True
        print("\n[2/3] 创建数据迭代器...")
        data_iter = iter(train_dataset)
        print("✓ 数据迭代器创建成功!")
        
        # ====================================================================
        # 步骤 3: 读取并打印样本信息
        # ====================================================================
        # 读取前 5 个样本，打印每个样本的详细信息
        # 这有助于验证数据格式是否正确，以及了解数据的结构
        print("\n[3/4] 读取并打印样本信息:")
        print("-" * 80)
        
        # 尝试读取 5 个样本
        for i in range(5):
            try:
                print(f"\n样本 {i+1}:")
                # 从迭代器中获取下一个样本（batch）
                batch = next(data_iter)
                
                # 打印样本中所有的键（数据字段名称）
                # 这些键通常包括：观察（observations）、动作（actions）、奖励（rewards）等
                print(f"  键 (keys): {list(batch.keys())}")
                
                # 遍历每个键，打印其详细信息
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # 如果是 PyTorch 张量，打印形状和数据类型
                        # 形状信息有助于理解数据的维度结构
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        # 如果是列表或元组，打印类型、长度，以及第一个元素的形状（如果是张量）
                        print(f"    {key}: type={type(value).__name__}, length={len(value)}")
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            print(f"      [0]: shape={value[0].shape}, dtype={value[0].dtype}")
                    else:
                        # 其他类型，打印类型和前 100 个字符的值
                        print(f"    {key}: type={type(value).__name__}, value={str(value)[:100]}")
                
            except StopIteration:
                # 如果数据集中的样本数量少于 5 个，会触发 StopIteration 异常
                print(f"\n警告: 数据集只有 {i} 个样本，无法读取第 {i+1} 个样本")
                break
            except Exception as e:
                # 如果读取过程中出现其他错误，打印错误信息并停止
                print(f"\n错误: 读取样本 {i+1} 时出错: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈信息
                break
        
        # ====================================================================
        # 步骤 4: 读取并可视化 front_point_cloud
        # ====================================================================
        print("\n[4/4] 读取并可视化 replay 中的 front_point_cloud:")
        print("-" * 80)
        
        # 重新创建迭代器以读取第一个样本
        data_iter = iter(train_dataset)
        
        try:
            # 读取第一个样本
            batch = next(data_iter)
            batch = next(data_iter)
            
            # 检查是否包含 front_point_cloud
            if 'front_point_cloud' not in batch:
                print("  警告: batch 中未找到 'front_point_cloud' 键")
                print(f"  可用的键: {list(batch.keys())}")
            else:
                # 提取 front_point_cloud
                front_pc = batch['front_point_cloud']
                print(f"  front_point_cloud 形状: {front_pc.shape}")
                print(f"  front_point_cloud 数据类型: {front_pc.dtype}")
                
                # 如果是 PyTorch 张量，转换为 NumPy
                if isinstance(front_pc, torch.Tensor):
                    front_pc = front_pc.cpu().numpy()
                
                # 选择第一个样本（batch 的第一个元素）
                # 形状可能是多种格式：
                # - (batch_size, timesteps, 3, H, W) - 时序数据
                # - (batch_size, 3, H, W) - 标准格式
                # - (batch_size, H, W, 3) - 另一种格式
                # - (3, H, W) 或 (H, W, 3) - 单样本格式
                if len(front_pc.shape) == 5:
                    # (batch_size, timesteps, 3, H, W) 格式 - 时序数据
                    batch_size, timesteps, channels, H, W = front_pc.shape
                    print(f"  批次大小: {batch_size}, 时序步数: {timesteps}, 通道数: {channels}, 高度: {H}, 宽度: {W}")
                    
                    # 选择第一个样本和第一个时间步，并转换为 (H, W, 3) 格式
                    pc_sample = front_pc[0, 0]  # (3, H, W) - 第一个batch，第一个时间步
                    pc_sample = np.transpose(pc_sample, (1, 2, 0))  # (H, W, 3)
                elif len(front_pc.shape) == 4:
                    # (batch_size, 3, H, W) 或 (batch_size, H, W, 3) 格式
                    batch_size = front_pc.shape[0]
                    if front_pc.shape[1] == 3:
                        # (batch_size, 3, H, W) 格式
                        channels, H, W = front_pc.shape[1], front_pc.shape[2], front_pc.shape[3]
                        print(f"  批次大小: {batch_size}, 通道数: {channels}, 高度: {H}, 宽度: {W}")
                        
                        # 选择第一个样本并转换为 (H, W, 3) 格式
                        pc_sample = front_pc[0]  # (3, H, W)
                        pc_sample = np.transpose(pc_sample, (1, 2, 0))  # (H, W, 3)
                    else:
                        # (batch_size, H, W, 3) 格式
                        H, W, channels = front_pc.shape[1], front_pc.shape[2], front_pc.shape[3]
                        print(f"  批次大小: {batch_size}, 高度: {H}, 宽度: {W}, 通道数: {channels}")
                        
                        # 选择第一个样本
                        pc_sample = front_pc[0]  # (H, W, 3)
                elif len(front_pc.shape) == 3:
                    # 可能是 (3, H, W) 或 (H, W, 3)
                    if front_pc.shape[0] == 3:
                        # (3, H, W) 格式
                        pc_sample = np.transpose(front_pc, (1, 2, 0))  # (H, W, 3)
                    else:
                        # (H, W, 3) 格式
                        pc_sample = front_pc
                else:
                    print(f"  错误: 不支持的点云形状 {front_pc.shape}")
                    pc_sample = None
                
                if pc_sample is not None:
                    # 重塑为 (N, 3) 格式，其中 N = H * W
                    H, W, _ = pc_sample.shape
                    point_cloud = pc_sample.reshape(-1, 3)
                    print(f"  重塑后点云形状: {point_cloud.shape}")
                    
                    # 移除无效点（包含 NaN 或 Inf 的点，以及全零的点）
                    valid_mask = np.isfinite(point_cloud).all(axis=1)
                    # 移除全零的点（通常表示无效的深度）
                    zero_mask = ~np.all(point_cloud == 0, axis=1)
                    valid_mask = valid_mask & zero_mask
                    valid_mask = valid_mask & zero_mask
                    point_cloud = point_cloud[valid_mask]
                    print(f"  有效点数: {len(point_cloud)}")
                    
                    if len(point_cloud) > 0:
                        # 如果点太多，进行下采样以提高可视化性能
                        max_points = 50000
                        if len(point_cloud) > max_points:
                            indices = np.random.choice(len(point_cloud), max_points, replace=False)
                            point_cloud = point_cloud[indices]
                            print(f"  下采样到: {len(point_cloud)} 个点")
                        
                        # 提取 x, y, z 坐标
                        x = point_cloud[:, 0]
                        y = point_cloud[:, 1]
                        z = point_cloud[:, 2]
                        
                        # 尝试从RGB图像采样颜色
                        rgb_colors = None
                        if 'front_rgb' in batch:
                            try:
                                front_rgb = batch['front_rgb']
                                if isinstance(front_rgb, torch.Tensor):
                                    front_rgb = front_rgb.cpu().numpy()
                                
                                # 选择第一个样本，处理多种形状格式
                                if len(front_rgb.shape) == 5:
                                    # (batch_size, timesteps, 3, H, W) 或 (batch_size, timesteps, H, W, 3)
                                    if front_rgb.shape[2] == 3:
                                        # (batch_size, timesteps, 3, H, W)
                                        rgb_image = front_rgb[0, 0]  # (3, H, W) - 第一个batch，第一个时间步
                                        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # (H, W, 3)
                                    else:
                                        # (batch_size, timesteps, H, W, 3)
                                        rgb_image = front_rgb[0, 0]  # (H, W, 3)
                                elif len(front_rgb.shape) == 4:
                                    # (batch_size, 3, H, W) 或 (batch_size, H, W, 3)
                                    if front_rgb.shape[1] == 3:
                                        rgb_image = front_rgb[0]  # (3, H, W)
                                        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # (H, W, 3)
                                    else:
                                        rgb_image = front_rgb[0]  # (H, W, 3)
                                elif len(front_rgb.shape) == 3:
                                    # (3, H, W) 或 (H, W, 3)
                                    if front_rgb.shape[0] == 3:
                                        rgb_image = np.transpose(front_rgb, (1, 2, 0))
                                    else:
                                        rgb_image = front_rgb
                                else:
                                    rgb_image = None
                                
                                if rgb_image is not None:
                                    # 确保 RGB 值在 [0, 255] 范围内
                                    if rgb_image.max() <= 1.0:
                                        rgb_image = (rgb_image * 255).astype(np.uint8)
                                    else:
                                        rgb_image = rgb_image.astype(np.uint8)
                                    
                                    # 检查 RGB 图像和点云的尺寸是否匹配
                                    img_h, img_w = rgb_image.shape[:2]
                                    pc_h, pc_w = pc_sample.shape[:2]
                                    
                                    if img_h == pc_h and img_w == pc_w:
                                        # 尺寸匹配，直接从对应位置采样颜色
                                        # 将 RGB 图像重塑为 (H*W, 3)，然后应用相同的 valid_mask
                                        rgb_flat = rgb_image.reshape(-1, 3)
                                        rgb_colors = rgb_flat[valid_mask]
                                        
                                        # 确保颜色数量与点云数量匹配
                                        if len(rgb_colors) == len(point_cloud):
                                            # 转换为 Plotly 需要的 RGB 字符串格式 'rgb(r,g,b)'
                                            rgb_strings = [f'rgb({r},{g},{b})' for r, g, b in rgb_colors]
                                            print(f"  使用 RGB 颜色着色，成功采样 {len(rgb_colors)} 个点的颜色")
                                        else:
                                            print(f"  警告: RGB 颜色数量 ({len(rgb_colors)}) 与点云数量 ({len(point_cloud)}) 不匹配，使用 z 坐标颜色")
                                            rgb_strings = None
                                    else:
                                        print(f"  警告: RGB 图像尺寸 ({img_h}, {img_w}) 与点云尺寸 ({pc_h}, {pc_w}) 不匹配，使用 z 坐标颜色")
                                        rgb_strings = None
                                else:
                                    rgb_strings = None
                            except Exception as e:
                                print(f"  警告: 无法从 RGB 图像采样颜色: {e}")
                                rgb_strings = None
                        else:
                            print("  未找到 front_rgb，使用 z 坐标颜色")
                            rgb_strings = None
                        
                        # 创建 3D 散点图
                        if rgb_strings is not None:
                            # 使用 RGB 颜色
                            fig = go.Figure(data=[go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                mode='markers',
                                marker=dict(
                                    size=1,
                                    color=rgb_strings,  # 使用 RGB 颜色
                                    opacity=0.8
                                ),
                                name='点云（RGB颜色）'
                            )])
                        else:
                            # 使用 z 坐标颜色（回退方案）
                            fig = go.Figure(data=[go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                mode='markers',
                                marker=dict(
                                    size=1,
                                    color=z,  # 使用 z 坐标作为颜色
                                    colorscale='Viridis',
                                    opacity=0.8,
                                    colorbar=dict(title="Z坐标")
                                ),
                                name='点云'
                            )])
                        
                        # 设置图表布局
                        title_suffix = "（RGB颜色）" if rgb_strings is not None else "（Z坐标颜色）"
                        fig.update_layout(
                            title=f'Replay Front Point Cloud{title_suffix}',
                            scene=dict(
                                xaxis_title='X (米)',
                                yaxis_title='Y (米)',
                                zaxis_title='Z (米)',
                                aspectmode='data'  # 保持坐标轴比例
                            ),
                            width=1200,
                            height=800
                        )
                        
                        # 保存为 HTML 文件
                        output_file = 'replay_front_point_cloud_visualization.html'
                        fig.write_html(output_file)
                        print(f"✓ 可视化已保存到: {output_file}")
                    else:
                        print("  警告: 没有有效的点云数据")
                else:
                    print("  警告: 无法处理点云数据")
        except StopIteration:
            print("  警告: 数据集为空，无法读取样本")
        except Exception as e:
            print(f"  错误: 读取和可视化 front_point_cloud 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 打印测试完成信息
        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
    except Exception as e:
        # 如果数据集创建过程中出现错误，捕获异常并打印详细信息
        # 这有助于快速定位问题所在
        print(f"\n错误: 调用 get_dataset_temporal 时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈信息
        sys.exit(1)  # 以错误状态退出程序

# ============================================================================
# 程序入口
# ============================================================================
# 当脚本被直接运行时（而不是被导入），执行 main 函数
# 这是 Python 脚本的标准入口点模式
if __name__ == "__main__":
    main()
