#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动作变换验证脚本 - HDF5版本

本脚本用于验证从HDF5文件加载的机器人演示数据中的动作变换是否正确。
主要功能包括：
1. 从HDF5文件加载演示数据（包括RGB图像、深度图、机器人状态等）
2. 将世界坐标系中的抓取器位置转换为局部归一化坐标系（模拟训练时的变换）
3. 将局部归一化坐标转换回世界坐标系（模拟推理时的逆变换）
4. 验证双向变换的一致性，确保变换过程没有误差

该验证对于确保训练和推理时坐标变换的一致性至关重要。
"""

# ==================== 标准库导入 ====================
import os  # 用于文件路径操作
import sys  # 用于系统路径管理
import h5py  # 用于读取HDF5格式的数据文件
import numpy as np  # 用于数值计算和数组操作
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式API，用于插值等操作
from PIL import Image  # 用于图像处理和调整大小
import plotly.graph_objects as go  # 用于交互式3D可视化
from plotly.subplots import make_subplots  # 用于创建子图

# ==================== 第三方库导入 ====================
from rlbench.backend.observation import Observation  # RLBench的观察数据结构
from rlbench.backend.utils import image_to_float_array  # 图像格式转换工具（未使用但保留）
from pyrep.objects import VisionSensor  # CoppeliaSim的视觉传感器，用于点云生成

# ==================== 项目路径设置 ====================
# 将项目根目录添加到Python路径中，以便能够导入sam2act模块
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==================== sam2act模块导入 ====================
from sam2act.utils.peract_utils import CAMERAS, SCENE_BOUNDS  # 相机配置和场景边界常量
import sam2act.utils.rvt_utils as rvt_utils  # RVT工具函数，用于点云和图像特征提取
import sam2act.mvt.utils as mvt_utils  # MVT工具函数，用于点云归一化变换
from sam2act.libs.peract.helpers.utils import stack_on_channel  # 通道堆叠工具（未使用但保留）

# ==================== 常量定义 ====================
# 演示数据文件夹命名格式，%d会被替换为演示编号
EPISODE_FOLDER = 'episode%d'

# 相机名称常量
CAMERA_FRONT = 'front'  # 前置相机
CAMERA_LS = 'left_shoulder'  # 左肩相机
CAMERA_RS = 'right_shoulder'  # 右肩相机
CAMERA_WRIST = 'wrist'  # 手腕相机

# 图像类型标识
IMAGE_RGB = 'rgb'  # RGB彩色图像
IMAGE_DEPTH = 'depth'  # 深度图像

# 图像文件命名格式，%d会被替换为时间步编号
IMAGE_FORMAT = '%d.png'

# 深度值缩放因子：2^24 - 1 = 16777215
# 用于将深度值从整数范围转换到浮点数范围（通常用于深度图的存储格式）
DEPTH_SCALE = 2**24 - 1

def convert_camera_matrix_maniskill_to_coppeliasim(extrinsics_opencv, intrinsics_opencv):
    """
    将Maniskill/OpenCV格式的相机参数转换为CoppeliaSim格式
    
    不同的仿真环境使用不同的坐标系约定：
    - OpenCV/Maniskill: 外参矩阵表示从世界坐标系到相机坐标系的变换（World->Camera）
    - CoppeliaSim: 外参矩阵表示从相机坐标系到世界坐标系的变换（Camera->World）
    
    因此需要对外参矩阵进行求逆操作，而内参矩阵在两个系统中是相同的。
    
    参数:
        extrinsics_opencv (np.ndarray): OpenCV格式的外参矩阵，形状为(4, 4)
                                        表示从世界坐标系到相机坐标系的齐次变换矩阵
        intrinsics_opencv (np.ndarray): OpenCV格式的内参矩阵，形状为(3, 3)
                                        包含焦距、主点等相机内部参数
    
    返回:
        extrinsics_coppeliasim (np.ndarray): CoppeliaSim格式的外参矩阵，形状为(4, 4)
                                             表示从相机坐标系到世界坐标系的齐次变换矩阵
        intrinsics_coppeliasim (np.ndarray): CoppeliaSim格式的内参矩阵，形状为(3, 3)
                                             与输入的内参矩阵相同
    """
    # 1. 外参矩阵转换：OpenCV使用"World->Camera"，CoppeliaSim需要"Camera->World"
    #    因此需要对矩阵求逆，使得变换方向相反
    #    如果 T_opencv: World -> Camera，则 T_coppeliasim = T_opencv^(-1): Camera -> World
    extrinsics_coppeliasim = np.linalg.inv(extrinsics_opencv)
    
    # 2. 内参矩阵保持不变
    #    内参矩阵描述的是相机内部的投影参数（焦距、主点等），与坐标系约定无关
    #    因此OpenCV和CoppeliaSim使用相同的内参矩阵
    intrinsics_coppeliasim = intrinsics_opencv.copy()
    
    return extrinsics_coppeliasim, intrinsics_coppeliasim

def get_stored_demo(data_path, index):
    """
    从HDF5文件中加载存储的演示数据
    
    HDF5文件结构：
    {
        "环境名称": {
            "episode_0": {
                "record_timestep_0": {
                    "image": RGB图像数据,
                    "wrist_image": 手腕相机RGB图像,
                    "base_camera_depth": 基础相机深度图,
                    "wrist_camera_depth": 手腕相机深度图,
                    "robot_endeffector_p": 机器人末端执行器位置,
                    "robot_endeffector_q": 机器人末端执行器四元数,
                    "action": 动作向量,
                    "base_camera_extrinsic_opencv": 基础相机外参,
                    "base_camera_intrinsic_opencv": 基础相机内参,
                    "wrist_camera_extrinsic_opencv": 手腕相机外参,
                    "wrist_camera_intrinsic_opencv": 手腕相机内参,
                    ...
                },
                "record_timestep_1": {...},
                ...
            },
            "episode_1": {...},
            ...
        }
    }
    
    参数:
        data_path (str): HDF5数据文件的路径
        index (int): 要加载的演示编号（episode索引）
    
    返回:
        obs (DemoList): 包含所有时间步观察数据的列表，每个元素是一个Observation对象
                        DemoList继承自list，并添加了variation_number属性
    """
    if data_path.endswith('.h5'):
        # DemoList是一个简单的列表子类，用于存储观察数据
        # 它继承list的所有功能，并可以添加额外的属性（如variation_number）
        class DemoList(list):
            pass
        obs = DemoList()
        
        # 以只读模式打开HDF5文件
        with h5py.File(data_path, 'r') as f:
            # 获取第一个环境名称（HDF5文件的顶层键）
            env_name = list(f.keys())[0]
            # 构建episode名称，格式为 "episode_0", "episode_1" 等
            episode_name = f'episode_{index}'
            
            # 查找episode数据组
            # 首先尝试在环境名称下查找，如果不存在则尝试在根目录下查找
            if episode_name not in f[env_name]:
                 if episode_name in f:
                     pass  # 如果episode在根目录下，后续会处理
                 else:
                     if episode_name not in f[env_name]:
                         raise ValueError(f"Episode {index} not found in {data_path} under {env_name}")
                     ep_grp = f[env_name][episode_name]
            else:
                ep_grp = f[env_name][episode_name]

            # 获取所有时间步的键，并按时间步编号排序
            # 时间步键的格式为 "record_timestep_0", "record_timestep_1" 等
            timesteps = sorted([k for k in ep_grp.keys() if k.startswith('record_timestep_')], 
                               key=lambda x: int(x.split('_')[-1]))
            
            # 遍历每个时间步，加载观察数据
            for ts_name in timesteps:
                ts_grp = ep_grp[ts_name]
                
                # 创建Observation对象，初始化所有字段为None
                # Observation是RLBench定义的观察数据结构，包含多个相机的RGB、深度、点云等
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
                
                # ==================== 1. 加载RGB图像 ====================
                # 前置相机RGB图像
                if 'image' in ts_grp:
                    # 从HDF5读取图像数据并转换为numpy数组
                    img = np.array(ts_grp['image'])
                    # 转换为PIL Image对象以便进行图像处理
                    img_pil = Image.fromarray(img)
                    # 调整图像大小到128x128，使用LANCZOS重采样（高质量插值）
                    img_pil = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
                    # 转换回numpy数组并存储到观察对象中
                    current_obs.front_rgb = np.array(img_pil)
                else:
                    # 如果HDF5中没有图像数据，创建全零图像作为占位符
                    current_obs.front_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                    
                # 手腕相机RGB图像
                if 'wrist_image' in ts_grp:
                    img = np.array(ts_grp['wrist_image'])
                    img_pil = Image.fromarray(img)
                    img_pil = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
                    current_obs.wrist_rgb = np.array(img_pil)
                else:
                    current_obs.wrist_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                
                # 左肩和右肩相机在HDF5数据中不存在，使用全零图像填充
                # 这样可以保持Observation对象的结构一致性
                current_obs.left_shoulder_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
                current_obs.right_shoulder_rgb = np.zeros((128, 128, 3), dtype=np.uint8)

                # ==================== 2. 加载深度图像 ====================
                # 前置相机深度图
                if 'base_camera_depth' in ts_grp:
                     # 从HDF5读取深度数据，转换为float32类型
                     # 除以1000.0将深度值从毫米转换为米（假设HDF5中存储的是毫米）
                     depth = np.array(ts_grp['base_camera_depth']).astype(np.float32) / 1000.0
                     # 如果深度图是3维的（可能是(H, W, 1)），压缩最后一个维度
                     if depth.ndim == 3:
                         depth = depth.squeeze(-1)
                     # 转换为PyTorch张量，并添加批次和通道维度：(H, W) -> (1, 1, H, W)
                     # 这是为了使用PyTorch的插值函数
                     depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                     # 使用双线性插值将深度图调整到128x128大小
                     # align_corners=False使用更现代的插值方式，与align_corners=True相比更准确
                     depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                     # 移除批次和通道维度，转换回numpy数组：(1, 1, 128, 128) -> (128, 128)
                     current_obs.front_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
                else:
                    # 如果HDF5中没有深度数据，创建全零深度图作为占位符
                    current_obs.front_depth = np.zeros((128, 128), dtype=np.float32)

                # 手腕相机深度图
                if 'wrist_camera_depth' in ts_grp:
                     depth = np.array(ts_grp['wrist_camera_depth']).astype(np.float32) / 1000.0
                     if depth.ndim == 3:
                         depth = depth.squeeze(-1)
                     depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                     depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                     current_obs.wrist_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
                else:
                     current_obs.wrist_depth = np.zeros((128, 128), dtype=np.float32)
                
                # 左肩和右肩相机的深度图在HDF5数据中不存在，使用全零深度图填充
                current_obs.left_shoulder_depth = np.zeros((128, 128), dtype=np.float32)
                current_obs.right_shoulder_depth = np.zeros((128, 128), dtype=np.float32)

                # ==================== 3. 加载机器人状态 ====================
                # 机器人末端执行器位置（3D坐标，单位：米）
                gripper_pos = np.array(ts_grp['robot_endeffector_p']).flatten()
                # 机器人末端执行器姿态（四元数，表示旋转）
                gripper_quat = np.array(ts_grp['robot_endeffector_q']).flatten()
                # 将位置和四元数拼接成7维向量：[x, y, z, qx, qy, qz, qw]
                # 这是RLBench标准的gripper_pose格式
                current_obs.gripper_pose = np.concatenate([gripper_pos, gripper_quat])
                
                # 从动作向量中提取抓取器开合状态
                # 动作向量的最后一个元素通常表示抓取器的开合（0=闭合，1=张开）
                if 'action' in ts_grp:
                    action = np.array(ts_grp['action'])
                    current_obs.gripper_open = float(action[-1])

                # 根据抓取器开合状态设置关节位置
                # 如果抓取器张开（gripper_open > 0），关节位置为0.04米（完全张开）
                # 如果抓取器闭合（gripper_open <= 0），关节位置为0.0米（完全闭合）
                if current_obs.gripper_open > 0:
                    current_obs.gripper_joint_positions = np.array([0.04, 0.04], dtype=np.float32)
                else:
                    current_obs.gripper_joint_positions = np.array([0.0, 0.0], dtype=np.float32)
                
                # 忽略碰撞标志（0表示不忽略碰撞检测）
                current_obs.ignore_collisions = 0
                
                # ==================== 4. 加载相机参数（存储在misc字典中）====================
                # misc字典用于存储额外的元数据，如相机参数等
                current_obs.misc = {}

                # 前置相机参数转换
                if 'base_camera_extrinsic_opencv' in ts_grp and 'base_camera_intrinsic_opencv' in ts_grp:
                    # 读取OpenCV格式的外参和内参矩阵
                    ext_opencv = np.array(ts_grp['base_camera_extrinsic_opencv'])
                    intr_opencv = np.array(ts_grp['base_camera_intrinsic_opencv'])
                    # 如果矩阵是3维的（可能是(1, 4, 4)或(1, 3, 3)），取第一个元素
                    if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
                    if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]
                    # 如果外参矩阵是(3, 4)格式（非齐次），补齐为(4, 4)齐次矩阵
                    # 添加最后一行 [0, 0, 0, 1] 使其成为齐次变换矩阵
                    if ext_opencv.shape == (3, 4):
                        ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
                    
                    # 将OpenCV格式转换为CoppeliaSim格式
                    ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
                        ext_opencv, intr_opencv
                    )
                    # 存储转换后的相机参数到misc字典中
                    current_obs.misc['front_camera_extrinsics'] = ext_coppeliasim
                    current_obs.misc['front_camera_intrinsics'] = intr_coppeliasim

                # 手腕相机参数转换（处理方式与前置相机相同）
                if 'wrist_camera_extrinsic_opencv' in ts_grp and 'wrist_camera_intrinsic_opencv' in ts_grp:
                    ext_opencv = np.array(ts_grp['wrist_camera_extrinsic_opencv'])
                    intr_opencv = np.array(ts_grp['wrist_camera_intrinsic_opencv'])
                    if ext_opencv.ndim == 3: ext_opencv = ext_opencv[0]
                    if intr_opencv.ndim == 3: intr_opencv = intr_opencv[0]
                    if ext_opencv.shape == (3, 4):
                        ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
                    
                    ext_coppeliasim, intr_coppeliasim = convert_camera_matrix_maniskill_to_coppeliasim(
                        ext_opencv, intr_opencv
                    )
                    current_obs.misc['wrist_camera_extrinsics'] = ext_coppeliasim
                    current_obs.misc['wrist_camera_intrinsics'] = intr_coppeliasim

                # 为缺失的相机创建虚拟参数（使用单位矩阵）
                # 左肩和右肩相机在HDF5数据中不存在，但Observation对象需要这些字段
                # 使用单位矩阵作为占位符，这样点云生成函数不会出错
                current_obs.misc['left_shoulder_camera_extrinsics'] = np.eye(4)  # 4x4单位矩阵（无变换）
                current_obs.misc['left_shoulder_camera_intrinsics'] = np.eye(3)  # 3x3单位矩阵（无投影）
                current_obs.misc['right_shoulder_camera_extrinsics'] = np.eye(4)
                current_obs.misc['right_shoulder_camera_intrinsics'] = np.eye(3)

                # ==================== 5. 生成点云 ====================
                # 点云是通过深度图和相机参数反投影得到的3D点集合
                # 每个像素的深度值结合相机内参和外参，可以计算出该像素对应的3D点坐标
                
                # 前置相机点云生成
                if 'front_camera_extrinsics' in current_obs.misc:
                    # 使用CoppeliaSim的VisionSensor工具函数从深度图生成点云
                    # 输入：深度图(H, W)、外参矩阵(4, 4)、内参矩阵(3, 3)
                    # 输出：点云(H, W, 3)，每个像素对应一个3D点(x, y, z)
                    current_obs.front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        current_obs.front_depth,
                        current_obs.misc['front_camera_extrinsics'],
                        current_obs.misc['front_camera_intrinsics']
                    )
                else:
                    # 如果没有相机参数，创建全零点云作为占位符
                    current_obs.front_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)
                     
                # 左肩相机点云生成（使用虚拟的单位矩阵参数）
                current_obs.left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                    current_obs.left_shoulder_depth,
                    current_obs.misc['left_shoulder_camera_extrinsics'],
                    current_obs.misc['left_shoulder_camera_intrinsics']
                )
                # 右肩相机点云生成（使用虚拟的单位矩阵参数）
                current_obs.right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                    current_obs.right_shoulder_depth,
                    current_obs.misc['right_shoulder_camera_extrinsics'],
                    current_obs.misc['right_shoulder_camera_intrinsics']
                )
                
                # 手腕相机点云生成
                if 'wrist_camera_extrinsics' in current_obs.misc:
                     current_obs.wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        current_obs.wrist_depth,
                        current_obs.misc['wrist_camera_extrinsics'],
                        current_obs.misc['wrist_camera_intrinsics']
                    )
                else:
                    # 如果没有相机参数，创建全零点云作为占位符
                    current_obs.wrist_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)

                # 将当前时间步的观察数据添加到列表中
                obs.append(current_obs)
        
        # 设置演示的变体编号（variation number），用于标识不同的任务变体
        obs.variation_number = 0
        return obs
    else:
        # 如果文件路径不是.h5格式，抛出错误
        raise ValueError("Data path must end with .h5")

def _norm_rgb(x):
    """
    将RGB图像从[0, 255]范围归一化到[-1, 1]范围
    
    归一化公式：normalized = (x / 255.0) * 2.0 - 1.0
    - 输入范围：[0, 255]
    - 输出范围：[-1, 1]
    - 0 -> -1.0
    - 127.5 -> 0.0
    - 255 -> 1.0
    
    这种归一化方式常用于深度学习模型，将像素值映射到对称区间，
    有助于训练稳定性和收敛速度。
    
    参数:
        x (torch.Tensor): RGB图像张量，值域为[0, 255]，形状为(B, C, H, W)
    
    返回:
        torch.Tensor: 归一化后的RGB图像张量，值域为[-1, 1]，形状为(B, C, H, W)
    """
    return (x.float() / 255.0) * 2.0 - 1.0

def _preprocess_inputs(obs_item, cameras):
    """
    预处理观察数据，将其转换为模型输入格式
    
    本函数模拟sam2act.utils.peract_utils._preprocess_inputs的功能，
    但假设输入是单个Observation对象（而不是批次）。
    为了模拟批次大小为1的情况，会为每个张量添加批次维度。
    
    处理流程：
    1. 从Observation对象中提取每个相机的RGB图像和点云
    2. 将numpy数组转换为PyTorch张量
    3. 调整张量维度顺序：从(H, W, C)转换为(C, H, W)
    4. 添加批次维度：从(C, H, W)转换为(1, C, H, W)
    5. 对RGB图像进行归一化处理
    
    参数:
        obs_item (Observation): 单个观察对象，包含多个相机的RGB和点云数据
        cameras (list): 相机名称列表，例如 ["front", "left_shoulder", "right_shoulder", "wrist"]
    
    返回:
        obs (list): 预处理后的观察数据列表
                   每个元素是一个列表 [rgb_tensor, pcd_tensor]
                   rgb_tensor: 归一化后的RGB图像，形状为(1, 3, 128, 128)
                   pcd_tensor: 点云数据，形状为(1, 3, 128, 128)
        pcds (list): 点云张量列表，每个元素形状为(1, 3, 128, 128)
                    用于后续的点云特征提取
    """
    obs = []
    pcds = []
    
    # 遍历每个相机，提取并预处理其RGB图像和点云
    for n in cameras:
        # 从Observation对象中获取指定相机的RGB图像和点云
        # 使用getattr动态获取属性，例如 obs_item.front_rgb, obs_item.front_point_cloud
        # 形状都是 (H, W, 3)，即 (128, 128, 3)
        rgb = getattr(obs_item, f"{n}_rgb")
        pcd = getattr(obs_item, f"{n}_point_cloud")
        
        # 将numpy数组转换为PyTorch张量，并调整维度顺序
        # permute(2, 0, 1): 将 (H, W, C) 转换为 (C, H, W)
        # unsqueeze(0): 添加批次维度，将 (C, H, W) 转换为 (1, C, H, W)
        # 最终形状：(128, 128, 3) -> (1, 3, 128, 128)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        pcd_tensor = torch.from_numpy(pcd).permute(2, 0, 1).unsqueeze(0).float()
        
        # 对RGB图像进行归一化，将值域从[0, 255]映射到[-1, 1]
        rgb_tensor = _norm_rgb(rgb_tensor)

        # 将RGB和点云张量组合在一起，添加到观察列表
        # obs的结构：[[rgb_front, pcd_front], [rgb_left_shoulder, pcd_left_shoulder], ...]
        obs.append([rgb_tensor, pcd_tensor])
        # 单独保存点云张量，用于后续的点云特征提取
        pcds.append(pcd_tensor)
        
    return obs, pcds

def main():
    """
    主函数：验证动作变换的正确性
    
    本函数执行完整的验证流程：
    1. 从HDF5文件加载演示数据
    2. 提取真实值（ground truth）抓取器位置（世界坐标系）
    3. 模拟训练时的变换：将世界坐标转换为局部归一化坐标
    4. 模拟推理时的变换：将局部归一化坐标转换回世界坐标
    5. 计算误差并验证变换的一致性
    
    如果双向变换是一致的，那么重建的世界坐标应该与原始坐标非常接近（误差 < 1e-5）。
    """
    print("=" * 80)
    print("Verifying Action Transformation from HDF5")
    print("=" * 80)
    
    # HDF5数据文件路径
    data_path = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/record_dataset_BinFill.h5"
    print(f"Loading data from: {data_path}")
    
    try:
        # ==================== 步骤1：加载演示数据 ====================
        # 从HDF5文件加载第0个演示（episode 0）
        # get_stored_demo会返回一个包含所有时间步观察数据的列表
        demo = get_stored_demo(data_path, 0)
        print(f"Loaded demo with {len(demo)} frames")
        
        # 选择一个时间步进行测试（例如第10帧）
        # 可以选择任意帧，但通常选择中间帧可以更好地验证变换
        frame_idx = 497
        # 如果指定的帧索引超出范围，使用第0帧
        if frame_idx >= len(demo):
            frame_idx = 0
        
        # 获取指定时间步的观察数据
        obs = demo[frame_idx]
        print(f"Processing frame {frame_idx}")
        
        # ==================== 步骤2：提取真实值动作（世界坐标系）====================
        # gripper_pose的前3个元素是抓取器的3D位置坐标（x, y, z）
        # 后4个元素是四元数（qx, qy, qz, qw），表示姿态
        # 这里我们只关心位置，用于验证坐标变换
        gt_gripper_pos = obs.gripper_pose[:3]
        print(f"Ground Truth Gripper Position (World): {gt_gripper_pos}")
        
        # ==================== 步骤3：模拟训练时的变换（世界坐标 -> 局部归一化坐标）====================
        print("\n--- Simulating Training Transform (World -> Local) ---")
        
        # 准备输入数据用于get_pc_img_feat函数
        # 需要按照agent训练时的方式预处理输入数据
        # CAMERAS常量定义：["front", "left_shoulder", "right_shoulder", "wrist"]
        # 注意：get_stored_demo函数会为缺失的相机填充零值，以保持数据结构一致性
        
        # 预处理观察数据：将numpy数组转换为PyTorch张量，调整维度，归一化RGB
        # 返回格式：obs是[[rgb, pcd], ...]的列表，pcds是点云张量列表
        preprocessed_obs, preprocessed_pcds = _preprocess_inputs(obs, CAMERAS)
        
        # 检测并选择计算设备（GPU或CPU）
        # 如果CUDA可用，使用GPU加速；否则使用CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 将预处理后的数据移动到指定设备（GPU或CPU）
        obs_tensor = []
        pcds_tensor = []
        
        # 将每个相机的RGB和点云张量移动到设备
        for o in preprocessed_obs:
            obs_tensor.append([item.to(device) for item in o])
        
        # 将点云张量移动到设备
        for p in preprocessed_pcds:
            pcds_tensor.append(p.to(device))
            
        # 提取点云和图像特征
        # rvt_utils.get_pc_img_feat函数会：
        # 1. 合并所有相机的点云
        # 2. 提取图像特征并与点云对齐
        # 返回：
        #   pc: 合并后的点云，形状为(B, N, 3)，其中B是批次大小，N是点的数量，3是xyz坐标
        #   img_feat: 图像特征，形状为(B, N, C)，其中C是特征维度
        pc, img_feat = rvt_utils.get_pc_img_feat(obs_tensor, pcds_tensor)
        
        print(f"Point Cloud Shape: {pc.shape}")
        
        # 创建真实值路径点（waypoint）张量
        # waypoint是机器人应该到达的目标位置
        # 形状：(B, 3)，其中B=1（批次大小为1），3是xyz坐标
        gt_wpt = torch.from_numpy(gt_gripper_pos).unsqueeze(0).float().to(device)
        
        # ==================== 归一化/变换点云和路径点 ====================
        # mvt_utils.place_pc_in_cube函数用于将点云和路径点归一化到局部坐标系
        # 返回：
        #   app_pc: 变换后的点云（归一化后的点云）
        #   rev_trans: 逆变换函数，用于将归一化坐标转换回世界坐标
        
        # 重要说明：
        # place_pc_in_cube函数会将点云居中并缩放到单位立方体内
        # 函数签名：place_pc_in_cube(pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False)
        # - pc: 输入点云，形状为(N, 3)
        # - app_pc: 需要同时变换的点（如路径点），形状为(M, 3)，如果为None则只变换点云
        # - with_mean_or_bounds: 如果True，使用点云均值进行居中；如果False，使用scene_bounds
        # - scene_bounds: 场景边界，当with_mean_or_bounds=False时使用
        # 返回的app_pc是变换后的点云，rev_trans是逆变换函数
        
        # 在sam2act_agent.py的update()方法中，变换过程如下：
        # a, b = mvt_utils.place_pc_in_cube(
        #     _pc,  # 单个点云 (N, 3)
        #     _wpt,  # 单个路径点 (3,) 或 (1, 3)
        #     with_mean_or_bounds=self._place_with_mean,
        #     scene_bounds=None if self._place_with_mean else self.scene_bounds,
        # )
        # 这样点云和路径点会使用相同的变换（相同的平移和缩放），保持相对位置关系
        
        # 从批次中提取单个点云和路径点
        pc_0 = pc[0]  # 形状：(N, 3)，N是点的数量
        wpt_0 = gt_wpt[0]  # 形状：(3,)，xyz坐标
        
        # 将路径点重塑为(1, 3)，因为place_pc_in_cube期望输入形状为(num_points, 3)
        wpt_0_reshaped = wpt_0.unsqueeze(0)
        
        # 使用与sam2act_agent相同的默认设置：place_with_mean=True
        # 这意味着使用点云的均值进行居中，而不是使用固定的场景边界
        print("Transforming using place_pc_in_cube (with_mean_or_bounds=True)...")
        wpt_local, rev_trans = mvt_utils.place_pc_in_cube(
            pc_0,  # 输入点云
            wpt_0_reshaped,  # 需要同时变换的路径点
            with_mean_or_bounds=True,  # 使用点云均值进行居中
            scene_bounds=None  # 不使用固定场景边界
        )
        
        # wpt_local是变换后的路径点，形状为(1, 3)
        # 打印归一化后的路径点坐标（在局部归一化坐标系中）
        print(f"Local Normalized Waypoint: {wpt_local.cpu().numpy()[0]}")
        
        # 计算归一化后的点云（用于可视化）
        pc_local, _ = mvt_utils.place_pc_in_cube(
            pc_0,
            with_mean_or_bounds=True,
            scene_bounds=None
        )
        
        # ==================== 步骤4：模拟推理时的变换（局部归一化坐标 -> 世界坐标）====================
        print("\n--- Simulating Inference Transform (Local -> World) ---")
        
        # 在评估（eval）过程中，模型会预测归一化后的路径点 `pred_wpt_local`
        # 然后调用逆变换函数 `rev_trans(pred_wpt_local)` 来获取世界坐标系下的路径点 `pred_wpt`
        
        # 这里我们使用刚才计算得到的 `wpt_local` 作为"预测值"
        # 然后验证 `rev_trans` 函数是否能将其正确转换回原始的 `gt_gripper_pos`
        # 如果变换是正确的，重建的位置应该与原始位置非常接近
        
        # 应用逆变换，将归一化坐标转换回世界坐标
        reconstructed_wpt = rev_trans(wpt_local)
        # 提取重建的位置坐标（从张量转换为numpy数组）
        reconstructed_pos = reconstructed_wpt.cpu().numpy()[0]
        
        print(f"Reconstructed Position (World): {reconstructed_pos}")
        
        # ==================== 步骤5：计算误差并验证 ====================
        # 计算原始位置和重建位置之间的欧氏距离（L2范数）
        # 如果变换是正确的，这个误差应该非常小（接近0）
        error = np.linalg.norm(gt_gripper_pos - reconstructed_pos)
        print(f"\nError: {error:.8f}")
        
        # 判断变换是否成功
        # 如果误差小于1e-5，认为变换是正确的
        # 这个阈值很小，允许浮点数运算的微小误差
        if error < 1e-5:
            print("\nSUCCESS: Transformation verified! Forward and Inverse transforms are consistent.")
            print("说明：正向变换（世界->局部）和逆向变换（局部->世界）是一致的。")
        else:
            print("\nFAILURE: Significant error detected.")
            print("警告：检测到显著误差，变换可能存在问题。")
        
        # ==================== 步骤6：可视化点云和路径点 ====================
        print("\n--- Visualizing Point Cloud and Waypoints ---")
        
        # 转换为numpy数组用于可视化
        pc_0_np = pc_0.cpu().numpy()
        pc_local_np = pc_local.cpu().numpy()
        wpt_0_np = wpt_0.cpu().numpy()
        wpt_local_np = wpt_local.cpu().numpy()[0]
        reconstructed_pos_np = reconstructed_pos
        
        # 创建两个子图：世界坐标系和局部坐标系
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('世界坐标系 - 点云和路径点', '局部归一化坐标系 - 点云和路径点'),
            horizontal_spacing=0.1
        )
        
        # 对点云进行下采样以便可视化（如果点太多）
        if len(pc_0_np) > 10000:
            indices = np.random.choice(len(pc_0_np), 10000, replace=False)
            pc_0_vis = pc_0_np[indices]
        else:
            pc_0_vis = pc_0_np
        
        # 子图1：世界坐标系
        # 绘制原始点云（浅灰色，小点）
        fig.add_trace(
            go.Scatter3d(
                x=pc_0_vis[:, 0],
                y=pc_0_vis[:, 1],
                z=pc_0_vis[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightgray',
                    opacity=0.3
                ),
                name='点云 (世界坐标系)',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 绘制原始路径点（红色大点）
        fig.add_trace(
            go.Scatter3d(
                x=[wpt_0_np[0]],
                y=[wpt_0_np[1]],
                z=[wpt_0_np[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                name='原始路径点',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 绘制重建后的路径点（绿色大点）
        fig.add_trace(
            go.Scatter3d(
                x=[reconstructed_pos_np[0]],
                y=[reconstructed_pos_np[1]],
                z=[reconstructed_pos_np[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    symbol='square',
                    line=dict(width=1, color='black')
                ),
                name='重建路径点',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 绘制从原始到重建的连线
        fig.add_trace(
            go.Scatter3d(
                x=[wpt_0_np[0], reconstructed_pos_np[0]],
                y=[wpt_0_np[1], reconstructed_pos_np[1]],
                z=[wpt_0_np[2], reconstructed_pos_np[2]],
                mode='lines',
                line=dict(color='blue', width=5, dash='dash'),
                name=f'误差: {error:.6f}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 子图2：局部归一化坐标系
        # 对归一化点云进行下采样
        if len(pc_local_np) > 10000:
            indices = np.random.choice(len(pc_local_np), 10000, replace=False)
            pc_local_vis = pc_local_np[indices]
        else:
            pc_local_vis = pc_local_np
        
        # 绘制归一化后的点云（浅蓝色，小点）
        fig.add_trace(
            go.Scatter3d(
                x=pc_local_vis[:, 0],
                y=pc_local_vis[:, 1],
                z=pc_local_vis[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightblue',
                    opacity=0.3
                ),
                name='点云 (归一化坐标系)',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 绘制归一化后的路径点（红色大点）
        fig.add_trace(
            go.Scatter3d(
                x=[wpt_local_np[0]],
                y=[wpt_local_np[1]],
                z=[wpt_local_np[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=1, color='black')
                ),
                name='归一化路径点',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 绘制单位立方体的边界（-1到1）
        cube_edges = [
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]],  # 底面
            [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]],  # 顶面
            [[-1, -1, -1], [-1, -1, 1]],  # 垂直边
            [[1, -1, -1], [1, -1, 1]],
            [[1, 1, -1], [1, 1, 1]],
            [[-1, 1, -1], [-1, 1, 1]]
        ]
        for edge in cube_edges:
            edge = np.array(edge)
            fig.add_trace(
                go.Scatter3d(
                    x=edge[:, 0],
                    y=edge[:, 1],
                    z=edge[:, 2],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="点云和路径点可视化",
            title_x=0.5,
            height=700,
            width=1600,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )
        
        # 更新场景设置（子图需要单独更新）
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=1
        )

        #         fig.update_scenes(
        #     xaxis_title='X (m)',
        #     yaxis_title='Y (m)',
        #     zaxis_title='Z (m)',
        #     xaxis=dict(range=[-3, 3]),  # 设置X轴范围
        #     yaxis=dict(range=[-3, 3]),  # 设置Y轴范围
        #     zaxis=dict(range=[0, 3]),  # 设置Z轴范围
        #     aspectmode='data',
        #     row=1, col=1
        # )
        
        fig.update_scenes(
            xaxis_title='X (归一化)',
            yaxis_title='Y (归一化)',
            zaxis_title='Z (归一化)',
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[-1.2, 1.2]),
            aspectmode='cube',
            row=1, col=2
        )
        
        # 保存为HTML
        output_path = 'pointcloud_waypoint_visualization.html'
        fig.write_html(output_path)
        print(f"\n可视化HTML已保存到: {output_path}")
        
        # 打印统计信息
        print(f"\n点云统计信息:")
        print(f"  原始点云点数: {len(pc_0_np)}")
        print(f"  原始点云范围: X[{pc_0_np[:, 0].min():.3f}, {pc_0_np[:, 0].max():.3f}], "
              f"Y[{pc_0_np[:, 1].min():.3f}, {pc_0_np[:, 1].max():.3f}], "
              f"Z[{pc_0_np[:, 2].min():.3f}, {pc_0_np[:, 2].max():.3f}]")
        print(f"  归一化点云范围: X[{pc_local_np[:, 0].min():.3f}, {pc_local_np[:, 0].max():.3f}], "
              f"Y[{pc_local_np[:, 1].min():.3f}, {pc_local_np[:, 1].max():.3f}], "
              f"Z[{pc_local_np[:, 2].min():.3f}, {pc_local_np[:, 2].max():.3f}]")
        print(f"\n路径点坐标:")
        print(f"  原始路径点 (世界): [{wpt_0_np[0]:.6f}, {wpt_0_np[1]:.6f}, {wpt_0_np[2]:.6f}]")
        print(f"  归一化路径点 (局部): [{wpt_local_np[0]:.6f}, {wpt_local_np[1]:.6f}, {wpt_local_np[2]:.6f}]")
        print(f"  重建路径点 (世界): [{reconstructed_pos_np[0]:.6f}, {reconstructed_pos_np[1]:.6f}, {reconstructed_pos_np[2]:.6f}]")
            
    except Exception as e:
        # 异常处理：如果过程中出现任何错误，打印错误信息并显示完整的堆栈跟踪
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

# 当脚本直接运行时（而不是被导入为模块），执行主函数
if __name__ == "__main__":
    main()
