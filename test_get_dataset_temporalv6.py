#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_dataset_temporal 函数的数据读取功能 (版本 v5)

本文件提取了 sam2act 训练管线中用于生成时序数据集的核心函数，并将其整合为一个独立的脚本。
主要用于调试和验证数据集生成的逻辑，包括：
1. 演示数据加载 (RLBench Demo Loading)
2. 关键点发现 (Keypoint Discovery)
3. 动作离散化 (Action Discretization)
4. 观察特征提取 (Observation Extraction)
5. 语言目标编码 (CLIP Language Encoding)
6. 时序 Replay Buffer 的创建与填充

使用方法:
    python test_get_dataset_temporalv5.py
"""

import os
import sys
import shutil
import pickle
import logging
import numpy as np
import torch
import clip
from typing import List
from PIL import Image
from scipy.spatial.transform import Rotation

# ============================================================================
# 路径配置
# ============================================================================
# 获取当前文件的绝对路径，并将项目根目录添加到系统路径中，以便导入 sam2act 模块
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# 第三方库导入
# ============================================================================
# YARR: 用于强化学习的 Replay Buffer 库
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.utils.observation_type import ObservationElement
try:
    # 尝试导入时序 Replay Buffer (支持存储多时间步数据)
    from yarr.replay_buffer.uniform_replay_buffer_temporal import UniformReplayBuffer_temporal
except ImportError:
    # 如果 yarr 是作为子模块存在于 sam2act/libs 中，则手动添加路径
    sys.path.append(os.path.join(project_root, 'sam2act/libs/YARR'))
    from yarr.replay_buffer.uniform_replay_buffer_temporal import UniformReplayBuffer_temporal

# RLBench: 机器人学习基准环境
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor
from rlbench.backend.const import DEPTH_SCALE

# ============================================================================
# 全局常量配置
# ============================================================================
# 从项目配置中导入关键参数
from sam2act.utils.peract_utils import (
    CAMERAS,                      # 使用的相机列表 (e.g., front, wrist, etc.)
    SCENE_BOUNDS,                 # 场景的三维边界 [x_min, x_max, y_min, y_max, z_min, z_max]
    EPISODE_FOLDER,               # 演示数据存储的文件夹模板
    VARIATION_DESCRIPTIONS_PKL,   # 任务语言描述文件名
    DEMO_AUGMENTATION_EVERY_N,    # 数据增强采样间隔
    ROTATION_RESOLUTION,          # 旋转离散化分辨率 (每个轴的离散份数)
    VOXEL_SIZES,                  # 体素化网格大小 (用于动作空间离散化)
    LOW_DIM_SIZE,                 # 低维状态向量的长度
    IMAGE_SIZE                    # 图像尺寸 (e.g., 128x128)
)

# get_stored_demo 函数使用的常量
CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_FORMAT = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

# 需要从 Observation 字典中移除的低维键，避免冗余存储
REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']


# ============================================================================
# 辅助函数：数学与几何变换 (from sam2act.libs.peract.helpers.utils)
# ============================================================================

def normalize_quaternion(quat):
    """
    归一化四元数。
    参数:
        quat: 输入四元数 [x, y, z, w]
    返回:
        归一化后的四元数
    """
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)

def quaternion_to_discrete_euler(quaternion, resolution):
    """
    将连续的四元数转换为离散的欧拉角索引。
    用于 PerAct 等算法中的离散动作空间。
    参数:
        quaternion: 输入四元数
        resolution: 每个欧拉角轴的离散份数 (e.g., 72 表示每 5 度一份)
    返回:
        disc: [3] 形状的整数数组，表示 XYZ 欧拉角的离散索引
    """
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

def point_to_voxel_index(
    point: np.ndarray,
    voxel_size: np.ndarray,
    coord_bounds: np.ndarray):
    """
    将 3D 空间中的连续点坐标转换为体素网格索引。
    参数:
        point: [3] 3D 点坐标 (x, y, z)
        voxel_size: 体素大小 (例如 100，表示将场景某轴划分为 100 份)
        coord_bounds: 场景边界 [min_x, max_x, min_y, max_y, min_z, max_z]
    返回:
        voxel_indicy: [3] 体素网格索引 (ix, iy, iz)
    """
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)  # 每个体素的实际物理尺寸
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy

def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False,
                episode_length: int = 10):
    """
    从 RLBench Observation 对象中提取并处理观察数据。
    主要功能：
    1. 移除不必要的数据（如关节速度等）。
    2. 调整图像数据的通道顺序 (CHW vs HWC)。
    3. 添加时间步编码到低维状态中。
    4. 提取相机内外参。
    
    参数:
        obs: 当前帧的 Observation 对象
        cameras: 需要保留的相机列表
        t: 当前时间步
        prev_action: 上一步的动作 (未使用，但为了接口兼容保留)
        channels_last: 是否将通道放在最后 (False 为 PyTorch 格式 CHW)
        episode_length: 任务的最大步长，用于归一化时间步
    """
    # 临时保存需要保留的数据，置空 Observation 中不需要的大数组以节省内存（如果 obs 是引用）
    # 但这里主要是为了整理 obs_dict
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    # 过滤掉 None 值
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    
    # 构造机器人状态向量 [夹爪开度, 夹爪关节位置...]
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions])
    
    # 移除不需要的键
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    # 处理图像数据维度
    if not channels_last:
        # swap channels from last dim to 1st dim (H, W, C) -> (C, H, W)
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # 碰撞检测忽略标志
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    
    # 确保点云数据是 float32
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    # 只保留 cameras 列表中指定的相机数据，过滤掉其他相机
    # 支持的相机数据类型：rgb, depth, point_cloud
    camera_suffixes = ['_rgb', '_depth', '_point_cloud']
    keys_to_remove = []
    for key in obs_dict.keys():
        # 检查是否是相机相关的键
        for camera_name in ['left_shoulder', 'right_shoulder', 'front', 'wrist']:
            for suffix in camera_suffixes:
                if key == camera_name + suffix:
                    # 如果该相机不在 cameras 列表中，则标记为需要删除
                    if camera_name not in cameras:
                        keys_to_remove.append(key)
                        break
    
    # 删除不需要的相机数据
    for key in keys_to_remove:
        obs_dict.pop(key, None)

    # 提取相机内外参矩阵
    for camera_name in cameras:
        obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # 将归一化的时间步 [-1, 1] 添加到低维状态中
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    # 恢复 obs 对象的原始数据（如果在函数外还需要使用）
    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict

# ============================================================================
# 关键点发现 (Keypoint Discovery) 相关函数
# ============================================================================

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    """
    判断机器人在当前帧是否处于静止状态。
    依据：关节速度接近0，且夹爪状态在前后几帧内保持不变。
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
    从完整的演示中提取关键帧（Keypoints）。
    关键帧通常是机器人动作发生显著变化（如夹爪开闭、停顿）的时刻。
    
    参数:
        demo: 完整的演示数据列表
        method: 发现方法 ('heuristic', 'random', 'fixed_interval')
    返回:
        episode_keypoints: 关键帧的索引列表
    """
    episode_keypoints = []
    if method == 'heuristic':
        # 启发式方法：检测夹爪状态变化或机器人静止
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # 如果夹爪状态改变，或者是最后一帧，或者是静止状态，则标记为关键点
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        
        # 移除重复或相邻过近的关键点
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # 随机选择 20 个关键点
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # 固定间隔采样
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError

# ============================================================================
# 数据加载与动作处理 (Dataset Helpers)
# ============================================================================

def get_stored_demo(data_path, index):
    """
    从磁盘加载单个演示数据。
    需要读取各个相机的 RGB 图、深度图，并计算点云。
    
    参数:
        data_path: 数据根目录
        index: 演示的索引 ID
    """
    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
    
    # 加载低维观测数据和变体编号
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)
    
    num_steps = len(obs)
    for i in range(num_steps):
        # 1. 加载各个视角的 RGB 图像
        obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))
        
        # 2. 加载各个视角的深度图像，并转换为真实深度值（米）
        # image_to_float_array 会将 24bit 编码的深度图解码
        obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[i].front_depth = near + obs[i].front_depth * (far - near) # 反归一化
        
        obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
        obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)
        
        obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
        obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)
        
        obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)
        
        # 3. 根据深度图和相机参数生成点云
        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].front_depth, obs[i].misc['front_camera_extrinsics'], obs[i].misc['front_camera_intrinsics']
        )
        obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].left_shoulder_depth, obs[i].misc['left_shoulder_camera_extrinsics'], obs[i].misc['left_shoulder_camera_intrinsics']
        )
        obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].right_shoulder_depth, obs[i].misc['right_shoulder_camera_extrinsics'], obs[i].misc['right_shoulder_camera_intrinsics']
        )
        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].wrist_depth, obs[i].misc['wrist_camera_extrinsics'], obs[i].misc['wrist_camera_intrinsics']
        )
    return obs

def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    """
    将从当前帧(t-1)到下一帧(t+1)的变化转换为离散动作。
    动作包含：平移(体素索引)、旋转(离散欧拉角)、夹爪开闭、是否忽略碰撞。
    
    参数:
        obs_tp1: 下一关键帧 (t+1) 的观测
        obs_tm1: 当前帧/上一帧 (t-1) 的观测 (用于判断 collision ignore 标志)
    """
    # 1. 处理旋转动作
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    
    # 2. 处理平移动作 (将目标位置转换为体素索引)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    
    # 支持多尺度的体素化（虽然通常只用一个尺度）
    for depth, vox_size in enumerate(voxel_sizes):
        index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    # 3. 处理夹爪动作 (开/闭)
    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    
    # 返回：离散平移索引、离散旋转+夹爪索引、忽略碰撞标志、连续动作向量、注意力坐标
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )

def _clip_encode_text(clip_model, text):
    """使用 CLIP 模型编码文本指令，返回投影后的特征和原始嵌入。"""
    # 1. 词嵌入 + 位置编码
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    # 2. Transformer 编码
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # 3. 获取序列嵌入
    emb = x.clone()
    # 4. 获取 EOS token 的特征并投影
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return x, emb

def create_replay_temporal(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    num_maskmem,
    replay_size=3e5,
):
    """
    创建时序 Replay Buffer。定义存储在 Buffer 中的数据结构。
    """
    trans_indicies_size = 3 * len(voxel_sizes)  # XYZ * 尺度数量
    rot_and_grip_indicies_size = 3 + 1          # RPY + Gripper
    gripper_pose_size = 7                       # Pos(3) + Quat(4)
    ignore_collisions_size = 1
    max_token_seq_len = 77                      # CLIP 最大序列长度
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # 定义观测元素 (ObservationElements) - 图像、点云等
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    for cname in cameras:
        # RGB 图像 (C, H, W)
        observation_elements.append(
            ObservationElement("%s_rgb" % cname, (3, IMAGE_SIZE, IMAGE_SIZE), np.float32)
        )
        # 深度图像 (1, H, W)
        observation_elements.append(
            ObservationElement("%s_depth" % cname, (1, IMAGE_SIZE, IMAGE_SIZE), np.float32)
        )
        # 点云 (3, H, W)
        observation_elements.append(
            ObservationElement("%s_point_cloud" % cname, (3, IMAGE_SIZE, IMAGE_SIZE), np.float32)
        )
        # 相机外参
        observation_elements.append(
            ObservationElement("%s_camera_extrinsics" % cname, (4, 4), np.float32)
        )
        # 相机内参
        observation_elements.append(
            ObservationElement("%s_camera_intrinsics" % cname, (3, 3), np.float32)
        )

    # 定义其他存储元素 (ReplayElements) - 动作、语言嵌入、元数据
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement("rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement("lang_goal_embs", (max_token_seq_len, lang_emb_dim), np.float32),
            ReplayElement("lang_goal", (1,), object),
        ]
    )

    # 额外的元数据，用于辅助训练和调试
    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
        ReplayElement("initial_frame", (), int),
    ]

    # 初始化 Buffer
    replay_buffer = UniformReplayBuffer_temporal(
        disk_saving=disk_saving,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
        num_maskmem=num_maskmem,
    )
    return replay_buffer

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
    将从当前 sample_frame 开始的一系列关键点转换作为样本加入 Replay Buffer。
    
    逻辑:
    从 next_keypoint_idx 开始遍历剩余的关键点。
    对于每一个关键点 k:
    1. obs = 当前观测 (初始为 initial_obs, 之后更新为上一个关键点的观测)
    2. obs_tp1 = 关键点 k 对应的演示帧 (作为目标)
    3. 计算从 obs 到 obs_tp1 的动作 (Action)
    4. 提取 obs 的特征 (Extract Obs)
    5. 编码语言目标
    6. 将 (obs, action, reward, terminal, ...) 存入 Buffer
    """
    prev_action = None
    obs = inital_obs
    initial_frame = sample_frame
    
    # 遍历后续的所有关键点
    for k in range(next_keypoint_idx, len(episode_keypoints)):
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)] # 用于获取 ignore_collisions
        
        # 计算离散动作和连续动作
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        # 仅在最后一个关键点给予奖励 1.0，并标记为 terminal
        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        # 提取当前帧的观测特征
        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k - next_keypoint_idx, # 相对时间步
            prev_action=prev_action,
            episode_length=25,
        )
        
        # 提取语言特征
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
            
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
            "initial_frame": initial_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        # 将样本添加到 Buffer
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        
        # 更新当前观测为目标观测，准备下一次迭代
        obs = obs_tp1
        sample_frame = keypoint

    # 存储最后一个状态 (Terminal State) 的观测
    # Replay Buffer 通常需要存储 s_T+1
    if next_keypoint_idx < len(episode_keypoints):
        obs_dict_tp1 = extract_obs(
            obs_tp1,
            CAMERAS,
            t=k + 1 - next_keypoint_idx,
            prev_action=prev_action,
            episode_length=25,
        )
        obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
        obs_dict_tp1.pop("wrist_world_to_cam", None)
        obs_dict_tp1.update(final_obs)
        replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


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
    填充 Replay Buffer 的主循环。
    遍历指定数量的演示，进行数据增强（如果启用），并将生成的数据存入 Buffer。
    """
    # 1. 检查磁盘上是否已有数据
    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            if rank == 0:
                print(f"[Info] Replay dataset already exists in the disk: {task_replay_storage_folder}", flush=True)
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    # 2. 如果存在，直接恢复
    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        # 3. 否则，开始处理演示数据
        print("Filling replay ...:", task)
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            # 加载演示数据
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # 加载任务描述
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # 提取关键点
            episode_keypoints = keypoint_discovery(demo)
            next_keypoint_idx = 0
            
            # 数据增强循环：遍历演示中的每一帧作为可能的起始帧
            for i in range(len(demo) - 1):
                # 如果不增强，只使用第一帧 (i=0)
                if not demo_augmentation and i > 0:
                    break
                # 按照间隔采样起始帧
                if i % demo_augmentation_every_n != 0:
                    continue

                obs = demo[i]
                desc = descs[0]
                
                # 找到当前帧之后的第一个关键点
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                
                # 如果没有后续关键点，则停止当前演示的处理
                if next_keypoint_idx == len(episode_keypoints):
                    break
                
                # 将从当前帧开始的序列添加到 Buffer
                _add_keypoints_to_replay_temporal(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                )

        # 保存所有样本的 terminal 标记到单独的文件，方便后续索引
        task_idx = replay._task_index[task]
        with open(os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb") as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[task_idx] : 
                    replay._task_replay_start_index[task_idx] + replay._task_add_count[task_idx].value
                ],
            )
        print("Replay filled with demos.")


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
    数据集生成管线入口。
    
    功能:
    1. 初始化 Replay Buffers (训练/测试)。
    2. 加载 CLIP 模型用于提取语言特征。
    3. 遍历任务列表，调用 fill_replay_temporal 填充数据。
    4. 将 Replay Buffer 包装为 PyTorch Dataset。
    
    参数:
        tasks: 任务列表
        BATCH_SIZE_*: 批大小
        *_REPLAY_STORAGE_DIR: Buffer 存储路径
        DATA_FOLDER: 原始数据路径
        NUM_*: 演示数量
        refresh_replay: 是否强制重新生成数据
        device: 计算设备
    """
    # 1. 创建 Replay Buffer 对象
    train_replay_buffer = create_replay_temporal(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        num_maskmem=num_maskmem,
    )
    if not only_train:
        test_replay_buffer = create_replay_temporal(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            num_maskmem=num_maskmem,
        )

    # 2. 加载 CLIP 模型
    try:
        clip_model, _ = clip.load("RN50", device="cpu")
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    # 3. 遍历并处理每个任务
    for task in tasks:
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # 如果请求刷新，则删除旧数据
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(train_replay_storage_folder):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(test_replay_storage_folder):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # 填充训练集 Buffer
        fill_replay_temporal(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
            rank=rank,
        )

        # 填充测试集 Buffer
        if not only_train:
            fill_replay_temporal(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                rank=rank,
                clip_model=clip_model,
                device=device,
            )

    # 释放模型显存
    if clip_model is not None:
        del clip_model
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # 4. 包装为 PyTorch Dataset
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset


# ============================================================================
# 主执行入口 (Main Execution)
# ============================================================================

def main():
    # 配置参数
    TRAIN_REPLAY_STORAGE_DIR = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/buffer/"
    DATA_FOLDER = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/dataset/rlbench-18-tasks/data"
    TEST_REPLAY_STORAGE_DIR = None
    tasks = ["close_jar"]       # 测试的任务
    BATCH_SIZE_TRAIN = 4        # 训练 Batch 大小
    BATCH_SIZE_TEST = None
    NUM_TRAIN = 100              # 使用的演示数量
    NUM_VAL = None
    refresh_replay = True       # 是否重新生成 Replay Buffer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = 2             # DataLoader 线程数
    only_train = True           # 仅生成训练集
    num_maskmem = 8
    rank = 0
    sample_distribution_mode = "transition_uniform"

    print("=" * 80)
    print("Testing get_dataset_temporal (v5 - Single File Pipeline)")
    print("=" * 80)
    print(f"TRAIN_REPLAY_STORAGE_DIR: {TRAIN_REPLAY_STORAGE_DIR}")
    print(f"DATA_FOLDER: {DATA_FOLDER}")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}")
    
    try:
        # 1. 创建数据集
        print("\n[1/3] Creating dataset...")
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
        print("Dataset created successfully!")

        # 2. 创建迭代器
        print("\n[2/3] Creating iterator...")
        data_iter = iter(train_dataset)

        # 3. 尝试读取样本以验证
        print("\n[3/3] Reading samples...")
        for i in range(5):
            try:
                print(f"\nSample {i+1}:")
                batch = next(data_iter)
                print(f"  Keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    {key}: type={type(value).__name__}, length={len(value)}")
                    else:
                        print(f"    {key}: type={type(value).__name__}")
            except StopIteration:
                break
            except Exception as e:
                print(f"Error reading sample {i+1}: {e}")
                break

        print("\nTest Complete!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
