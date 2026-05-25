#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作正变换和反变换验证脚本

该脚本从数据集中读取动作数据，进行正变换（连续动作 -> 离散索引），
然后进行反变换（离散索引 -> 连续动作），验证变换的可逆性。

主要验证内容：
1. 平移动作：连续3D坐标 <-> 体素索引
2. 旋转动作：四元数 <-> 离散欧拉角索引
3. 夹爪动作：连续值 <-> 离散索引（0/1）
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块
from test_load_and_pointcloud_create_Hbench6 import (
    get_dataset_temporal,
    _get_action,
    get_stored_demo,
    keypoint_discovery,
)
from sam2act.utils.peract_utils import (
    SCENE_BOUNDS,
    VOXEL_SIZES,
    ROTATION_RESOLUTION,
    CAMERAS,
)
import peract_colab.arm.utils as utils
from sam2act.mvt.aug_utils import discrete_euler_to_quaternion
from rlbench.backend.observation import Observation

# ============================================================================
# 反变换函数
# ============================================================================

def voxel_index_to_point_np(voxel_index, voxel_size, coord_bounds):
    """
    将体素索引转换回3D坐标点
    
    参数:
        voxel_index: (3,) numpy数组，体素索引 [i, j, k]
        voxel_size: int，体素大小
        coord_bounds: (6,) numpy数组，场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
    
    返回:
        point: (3,) numpy数组，3D坐标点（体素中心位置）
    
    注意:
        这个函数计算的是体素中心的位置，与正变换中的 attention_coordinate 一致。
        正变换公式: attention_coordinate = bounds[:3] + res * index
        其中 res = (bounds[3:] - bounds[:3]) / voxel_size
    """
    bounds = np.array(coord_bounds)
    res = (bounds[3:] - bounds[:3]) / voxel_size
    # 计算体素中心位置（与正变换中的 attention_coordinate 计算方式一致）
    point = bounds[:3] + res * voxel_index
    return point


def inverse_transform_action(
    trans_indicies,
    rot_and_grip_indicies,
    voxel_sizes,
    scene_bounds,
    rotation_resolution,
):
    """
    将离散动作索引反变换回连续动作
    
    参数:
        trans_indicies: List[int]，平移体素索引列表，长度为 3 * len(voxel_sizes)
        rot_and_grip_indicies: List[int]，旋转和夹爪索引，长度为 4 [rot_x, rot_y, rot_z, grip]
        voxel_sizes: List[int]，体素大小列表
        scene_bounds: List[float]，场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
        rotation_resolution: int，旋转分辨率
    
    返回:
        action: (8,) numpy数组，连续动作 [x, y, z, qx, qy, qz, qw, grip]
    """
    # 1. 反变换平移：从体素索引转换回3D坐标
    # trans_indicies 的长度是 3 * len(voxel_sizes)
    # 我们使用第一个尺度的索引（前3个元素）
    if len(trans_indicies) >= 3:
        voxel_idx = np.array(trans_indicies[:3], dtype=np.int32)
        voxel_size = voxel_sizes[0]
        trans_point = voxel_index_to_point_np(voxel_idx, voxel_size, scene_bounds)
    else:
        raise ValueError(f"trans_indicies 长度不足: {len(trans_indicies)}")
    
    # 2. 反变换旋转：从离散欧拉角索引转换回四元数
    if len(rot_and_grip_indicies) >= 3:
        disc_rot = np.array(rot_and_grip_indicies[:3], dtype=np.int32)
        rot_quat = discrete_euler_to_quaternion(disc_rot, rotation_resolution)
    else:
        raise ValueError(f"rot_and_grip_indicies 长度不足: {len(rot_and_grip_indicies)}")
    
    # 3. 反变换夹爪：直接使用索引值（0或1）
    if len(rot_and_grip_indicies) >= 4:
        grip_value = float(rot_and_grip_indicies[3])
    else:
        grip_value = 0.0
    
    # 4. 组合成连续动作向量
    action = np.concatenate([
        trans_point,      # [x, y, z]
        rot_quat,         # [qx, qy, qz, qw]
        np.array([grip_value])  # [grip]
    ])
    
    return action


# ============================================================================
# 辅助函数：安全的关键点发现
# ============================================================================

def safe_keypoint_discovery(demo, method='fixed_interval'):
    """
    安全的关键点发现函数，处理 joint_velocities 为 None 的情况
    
    参数:
        demo: Demo 对象，演示数据
        method: str，关键点发现方法
    
    返回:
        List[int]: 关键点索引列表
    """
    if method == 'fixed_interval':
        # 固定间隔方法，最安全
        keypoints = []
        segment_length = max(1, len(demo) // 20)
        for i in range(0, len(demo), segment_length):
            keypoints.append(i)
        if len(keypoints) == 0:
            keypoints = [0]
        if keypoints[-1] != len(demo) - 1:
            keypoints.append(len(demo) - 1)
        return keypoints
    
    elif method == 'simple':
        # 简单采样：每10帧一个关键点
        keypoints = list(range(10, len(demo), 10))
        if len(keypoints) == 0:
            keypoints = [len(demo) - 1]
        if keypoints[-1] != len(demo) - 1:
            keypoints.append(len(demo) - 1)
        return keypoints
    
    elif method == 'dataset':
        # 尝试使用 dataset 方法
        try:
            return keypoint_discovery(demo, method='dataset')
        except:
            return safe_keypoint_discovery(demo, method='fixed_interval')
    
    else:
        # 其他方法，如果失败则回退到 fixed_interval
        try:
            return keypoint_discovery(demo, method=method)
        except Exception as e:
            print(f"  警告: {method} 方法失败 ({e})，使用固定间隔方法")
            return safe_keypoint_discovery(demo, method='fixed_interval')


# ============================================================================
# 验证函数
# ============================================================================

def verify_single_action(obs_tp1, obs_tm1, scene_bounds, voxel_sizes, rotation_resolution):
    """
    验证单个动作的正变换和反变换
    
    参数:
        obs_tp1: Observation，目标时刻的观察
        obs_tm1: Observation，前一时刻的观察
        scene_bounds: List[float]，场景边界
        voxel_sizes: List[int]，体素大小列表
        rotation_resolution: int，旋转分辨率
    
    返回:
        dict: 包含验证结果的字典
    """
    # 原始连续动作
    original_action = np.concatenate([
        obs_tp1.gripper_pose,
        np.array([float(obs_tp1.gripper_open)])
    ])
    
    # 正变换：连续动作 -> 离散索引
    (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        action_from_get_action,
        attention_coordinates,
    ) = _get_action(
        obs_tp1,
        obs_tm1,
        scene_bounds,
        voxel_sizes,
        rotation_resolution,
        crop_augmentation=False,
    )
    
    # 反变换：离散索引 -> 连续动作
    recovered_action = inverse_transform_action(
        trans_indicies,
        rot_and_grip_indicies,
        voxel_sizes,
        scene_bounds,
        rotation_resolution,
    )
    
    # 计算误差
    # 注意：平移误差不会为0，因为离散化会将点映射到体素中心
    # 误差应该在体素分辨率范围内（体素大小对应的实际空间大小）
    trans_error = np.linalg.norm(original_action[:3] - recovered_action[:3])
    
    # 对于旋转，比较四元数（注意四元数 q 和 -q 表示相同旋转）
    orig_quat = original_action[3:7]
    recov_quat = recovered_action[3:7]
    
    # 归一化四元数
    orig_quat = orig_quat / np.linalg.norm(orig_quat)
    recov_quat = recov_quat / np.linalg.norm(recov_quat)
    
    # 如果点积为负，取反（因为 q 和 -q 表示相同旋转）
    if np.dot(orig_quat, recov_quat) < 0:
        recov_quat = -recov_quat
    
    # 计算旋转误差（四元数差异）
    rot_error = np.linalg.norm(orig_quat - recov_quat)
    
    # 或者使用角度误差
    orig_rot = Rotation.from_quat(orig_quat)
    recov_rot = Rotation.from_quat(recov_quat)
    rot_diff = orig_rot * recov_rot.inv()
    angle_error = np.abs(rot_diff.as_rotvec())
    angle_error_deg = np.linalg.norm(angle_error) * 180 / np.pi
    
    # 夹爪误差
    grip_error = abs(original_action[7] - recovered_action[7])
    
    return {
        'original_action': original_action,
        'recovered_action': recovered_action,
        'trans_indicies': trans_indicies,
        'rot_and_grip_indicies': rot_and_grip_indicies,
        'trans_error': trans_error,
        'rot_error_quat': rot_error,
        'rot_error_angle_deg': angle_error_deg,
        'grip_error': grip_error,
        'attention_coordinates': attention_coordinates,
    }


def verify_from_dataset(data_path, num_samples=10):
    """
    从数据集中读取数据并验证正反变换
    
    参数:
        data_path: str，数据集路径（HDF5文件或目录）
        num_samples: int，验证的样本数量
    """
    print("=" * 80)
    print("动作正变换和反变换验证")
    print("=" * 80)
    print(f"数据集路径: {data_path}")
    print(f"验证样本数: {num_samples}")
    print(f"场景边界: {SCENE_BOUNDS}")
    print(f"体素大小: {VOXEL_SIZES}")
    print(f"旋转分辨率: {ROTATION_RESOLUTION}")
    print("=" * 80)
    
    # 读取演示数据
    print("\n[1/3] 读取演示数据...")
    demo = get_stored_demo(data_path, index=0)
    print(f"✓ 成功读取演示，共 {len(demo)} 帧")
    
    # 发现关键点
    print("\n[2/3] 发现关键点...")
    # 使用安全的关键点发现方法，避免 joint_velocities 为 None 的问题
    keypoints = safe_keypoint_discovery(demo, method='fixed_interval')
    print(f"✓ 发现 {len(keypoints)} 个关键点: {keypoints[:10]}...")
    
    # 验证关键点的动作
    print("\n[3/3] 验证动作变换...")
    print("-" * 80)
    
    errors = {
        'trans_errors': [],
        'rot_errors_deg': [],
        'grip_errors': [],
    }
    
    # 验证前 num_samples 个关键点
    num_to_verify = min(num_samples, len(keypoints))
    for i, kp_idx in enumerate(keypoints[:num_to_verify]):
        if kp_idx == 0:
            continue  # 跳过第一帧（没有前一帧）
        
        obs_tp1 = demo[kp_idx]
        obs_tm1 = demo[kp_idx - 1]
        
        try:
            result = verify_single_action(
                obs_tp1,
                obs_tm1,
                SCENE_BOUNDS,
                VOXEL_SIZES,
                ROTATION_RESOLUTION,
            )
            
            errors['trans_errors'].append(result['trans_error'])
            errors['rot_errors_deg'].append(result['rot_error_angle_deg'])
            errors['grip_errors'].append(result['grip_error'])
            
            print(f"\n关键点 {i+1}/{num_to_verify} (帧 {kp_idx}):")
            print(f"  平移误差: {result['trans_error']:.6f} m")
            print(f"  旋转误差: {result['rot_error_angle_deg']:.6f} 度")
            print(f"  夹爪误差: {result['grip_error']:.6f}")
            print(f"  原始动作: trans={result['original_action'][:3]}, "
                  f"grip={result['original_action'][7]:.1f}")
            print(f"  恢复动作: trans={result['recovered_action'][:3]}, "
                  f"grip={result['recovered_action'][7]:.1f}")
            
        except Exception as e:
            print(f"\n关键点 {i+1} (帧 {kp_idx}) 验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("验证统计:")
    print("=" * 80)
    if errors['trans_errors']:
        print(f"平移误差:")
        print(f"  平均: {np.mean(errors['trans_errors']):.6f} m")
        print(f"  最大: {np.max(errors['trans_errors']):.6f} m")
        print(f"  最小: {np.min(errors['trans_errors']):.6f} m")
        print(f"  中位数: {np.median(errors['trans_errors']):.6f} m")
    
    if errors['rot_errors_deg']:
        print(f"\n旋转误差:")
        print(f"  平均: {np.mean(errors['rot_errors_deg']):.6f} 度")
        print(f"  最大: {np.max(errors['rot_errors_deg']):.6f} 度")
        print(f"  最小: {np.min(errors['rot_errors_deg']):.6f} 度")
        print(f"  中位数: {np.median(errors['rot_errors_deg']):.6f} 度")
    
    if errors['grip_errors']:
        print(f"\n夹爪误差:")
        print(f"  平均: {np.mean(errors['grip_errors']):.6f}")
        print(f"  最大: {np.max(errors['grip_errors']):.6f}")
        print(f"  最小: {np.min(errors['grip_errors']):.6f}")
        print(f"  中位数: {np.median(errors['grip_errors']):.6f}")
    
    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)
    
    # 判断验证是否通过
    # 平移误差应该很小（体素分辨率范围内）
    voxel_res = (np.array(SCENE_BOUNDS[3:]) - np.array(SCENE_BOUNDS[:3])) / VOXEL_SIZES[0]
    max_voxel_res = np.max(voxel_res)
    
    trans_ok = np.mean(errors['trans_errors']) < max_voxel_res * 2
    # 旋转误差应该在分辨率范围内（360度 / rotation_resolution）
    max_rot_res = 360.0 / ROTATION_RESOLUTION
    rot_ok = np.mean(errors['rot_errors_deg']) < max_rot_res * 2
    # 夹爪误差应该为0（因为只有0和1两个值）
    grip_ok = np.max(errors['grip_errors']) < 0.1
    
    if trans_ok and rot_ok and grip_ok:
        print("\n✓ 验证通过！正变换和反变换基本可逆。")
    else:
        print("\n✗ 验证未完全通过，请检查变换函数。")
        if not trans_ok:
            print(f"  平移误差过大（期望 < {max_voxel_res * 2:.6f} m）")
        if not rot_ok:
            print(f"  旋转误差过大（期望 < {max_rot_res * 2:.6f} 度）")
        if not grip_ok:
            print(f"  夹爪误差过大（期望 < 0.1）")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数
    
    使用方法:
        1. 修改下面的 data_path 变量为你的数据集路径
        2. 运行脚本: python verify_action_transform.py
        3. 查看验证结果，检查正变换和反变换的误差
    """
    # 数据集路径（请修改为你的数据集路径）
    # 可以是 HDF5 文件路径或包含演示数据的目录路径
    data_path = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/record_dataset_BinFill.h5"
    
    # 如果路径不存在，提示用户
    if not os.path.exists(data_path):
        print(f"警告: 数据路径不存在: {data_path}")
        print("\n请修改脚本中的 data_path 变量为正确的数据集路径")
        print("例如:")
        print("  - HDF5 文件: data_path = '/path/to/dataset.h5'")
        print("  - 目录路径: data_path = '/path/to/dataset_dir'")
        print("\n或者从 test_load_and_pointcloud_create_Hbench6.py 中获取正确的路径")
        return
    
    # 验证动作变换
    print("\n开始验证动作正变换和反变换...")
    verify_from_dataset(data_path, num_samples=20)


if __name__ == "__main__":
    main()
