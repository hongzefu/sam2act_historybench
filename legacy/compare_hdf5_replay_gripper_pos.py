#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比hdf5文件中keyframe的gripper位置和replay buffer中的action

本文件用于：
1. 从hdf5文件中读取每个episode的keyframe及其gripper位置
2. 从replay buffer中读取对应的action数据
3. 对比两者的gripper位置（hdf5中的robot_endeffector_p vs replay buffer action的前3个元素）
"""

# ============================================================================
# 标准库导入
# ============================================================================
import os
import sys
import h5py
import pickle
import numpy as np
from typing import Dict, List, Tuple

# ============================================================================
# 路径配置：添加项目根目录到 Python 路径
# ============================================================================
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# 项目内部模块导入
# ============================================================================
# 不需要导入replay buffer相关模块，因为我们直接从磁盘读取.replay文件

# ============================================================================
# 配置参数（从参考文件获取）
# ============================================================================
DATA_FOLDER = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/record_dataset_BinFill.h5"
TRAIN_REPLAY_STORAGE_DIR = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/sam2act/test_buffer"
TASK_NAME = "BinFill"
BATCH_SIZE_TRAIN = 4
NUM_MASKMEM = 8

# ============================================================================
# 辅助函数：从hdf5读取keyframe gripper位置
# ============================================================================
def read_keyframe_gripper_positions_from_hdf5(
    hdf5_path: str,
    num_episodes: int
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    从hdf5文件中读取每个episode的keyframe及其gripper位置
    
    参数:
        hdf5_path: hdf5文件路径
        num_episodes: episode数量
    
    返回:
        Dict[episode_idx, Dict[keypoint_frame, gripper_pos]]
        gripper_pos是(3,)数组，包含[x, y, z]
    """
    result = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取环境名（通常是第一个键）
        env_name = list(f.keys())[0]
        
        for episode_idx in range(num_episodes):
            episode_name = f'episode_{episode_idx}'
            if episode_name not in f[env_name]:
                print(f"Warning: Episode {episode_idx} not found in hdf5 file")
                continue
            
            ep_grp = f[env_name][episode_name]
            
            # 获取所有时间步
            timesteps = sorted([k for k in ep_grp.keys() if k.startswith('record_timestep_')], 
                             key=lambda x: int(x.split('_')[-1]))
            
            episode_keyframes = {}
            
            # 遍历所有时间步，找到keyframe（有keypoint_type字段的）
            for ts_idx, ts_name in enumerate(timesteps):
                ts_grp = ep_grp[ts_name]
                
                # 检查是否是keyframe（有keypoint_type字段且不为None）
                if 'keypoint_type' in ts_grp:
                    val = ts_grp['keypoint_type'][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    if val is not None and val != 'None' and val != '':
                        # 这是一个keyframe，读取gripper位置
                        gripper_pos = np.array(ts_grp['robot_endeffector_p']).flatten()
                        if len(gripper_pos) >= 3:
                            episode_keyframes[ts_idx] = gripper_pos[:3]
            
            if len(episode_keyframes) > 0:
                result[episode_idx] = episode_keyframes
    
    return result

# ============================================================================
# 辅助函数：从replay buffer读取action数据
# ============================================================================
def read_action_data_from_replay_buffer(
    replay_storage_dir: str,
    task_name: str
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    从replay buffer中读取action数据
    
    参数:
        replay_storage_dir: replay buffer存储目录
        task_name: 任务名称
    
    返回:
        Dict[(episode_idx, keypoint_frame), action_pos]
        action_pos是(3,)数组，包含action的前3个元素（位置）
    """
    task_replay_storage_folder = os.path.join(replay_storage_dir, task_name)
    
    if not os.path.exists(task_replay_storage_folder):
        raise ValueError(f"Replay buffer directory not found: {task_replay_storage_folder}")
    
    # 直接从磁盘读取所有.replay文件
    replay_files = [f for f in os.listdir(task_replay_storage_folder) if f.endswith('.replay')]
    replay_indices = [int(f.split('.')[0]) for f in replay_files]
    replay_indices.sort()
    
    print(f"Found {len(replay_indices)} replay files")
    
    result = {}
    
    # 读取每个replay文件
    for idx in replay_indices:
        replay_file = os.path.join(task_replay_storage_folder, f'{idx}.replay')
        try:
            with open(replay_file, 'rb') as f:
                data = pickle.load(f)
            
            # 提取action（格式为(8,)，前3个是位置）
            if 'action' not in data:
                continue
            
            action = np.array(data['action'])
            action_pos = action[:3]  # 提取位置部分
            
            # 提取元数据
            if 'episode_idx' not in data or 'next_keypoint_frame' not in data:
                continue
            
            episode_idx = int(data['episode_idx'])
            next_keypoint_frame = int(data['next_keypoint_frame'])
            
            # 存储结果
            key = (episode_idx, next_keypoint_frame)
            result[key] = action_pos
            
        except Exception as e:
            print(f"Warning: Failed to read {replay_file}: {e}")
            continue
    
    return result

# ============================================================================
# 辅助函数：对比gripper位置
# ============================================================================
def compare_gripper_positions(
    hdf5_data: Dict[int, Dict[int, np.ndarray]],
    replay_data: Dict[Tuple[int, int], np.ndarray]
):
    """
    对比hdf5和replay buffer中的gripper位置
    
    参数:
        hdf5_data: 从hdf5读取的数据
        replay_data: 从replay buffer读取的数据
    """
    print("=" * 80)
    print("对比结果")
    print("=" * 80)
    
    all_differences = []
    
    for episode_idx in sorted(hdf5_data.keys()):
        episode_keyframes = hdf5_data[episode_idx]
        
        print(f"\nEpisode {episode_idx}:")
        print("-" * 80)
        
        for keypoint_frame in sorted(episode_keyframes.keys()):
            hdf5_pos = episode_keyframes[keypoint_frame]
            
            # 在replay buffer中查找对应的数据
            key = (episode_idx, keypoint_frame)
            
            if key not in replay_data:
                print(f"  Keyframe {keypoint_frame}: 在replay buffer中未找到对应数据")
                continue
            
            replay_pos = replay_data[key]
            
            # 计算L2距离
            diff = np.linalg.norm(hdf5_pos - replay_pos)
            all_differences.append(diff)
            
            # 打印对比结果
            print(f"  Keyframe {keypoint_frame}:")
            print(f"    HDF5 gripper pos:    [{hdf5_pos[0]:.6f}, {hdf5_pos[1]:.6f}, {hdf5_pos[2]:.6f}]")
            print(f"    Replay action pos:   [{replay_pos[0]:.6f}, {replay_pos[1]:.6f}, {replay_pos[2]:.6f}]")
            print(f"    L2 distance:         {diff:.6f}")
    
    # 打印统计信息
    if len(all_differences) > 0:
        print("\n" + "=" * 80)
        print("统计信息")
        print("=" * 80)
        print(f"总对比数量: {len(all_differences)}")
        print(f"平均L2距离: {np.mean(all_differences):.6f}")
        print(f"最大L2距离: {np.max(all_differences):.6f}")
        print(f"最小L2距离: {np.min(all_differences):.6f}")
        print(f"标准差:     {np.std(all_differences):.6f}")

# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    print("=" * 80)
    print("对比hdf5和replay buffer中的gripper位置")
    print("=" * 80)
    print(f"HDF5文件路径: {DATA_FOLDER}")
    print(f"Replay buffer路径: {TRAIN_REPLAY_STORAGE_DIR}")
    print(f"任务名称: {TASK_NAME}")
    print("=" * 80)
    
    # 首先确定episode数量
    # 从hdf5文件中检查有多少个episode
    num_episodes = 0
    with h5py.File(DATA_FOLDER, 'r') as f:
        env_name = list(f.keys())[0]
        episode_names = [k for k in f[env_name].keys() if k.startswith('episode_')]
        if len(episode_names) > 0:
            # 提取最大的episode索引
            episode_indices = [int(name.split('_')[1]) for name in episode_names]
            num_episodes = max(episode_indices) + 1
        else:
            print("Error: No episodes found in hdf5 file")
            return
    
    print(f"\n找到 {num_episodes} 个episodes")
    
    # 步骤1: 从hdf5读取keyframe gripper位置
    print("\n[1/2] 从hdf5文件读取keyframe gripper位置...")
    hdf5_data = read_keyframe_gripper_positions_from_hdf5(DATA_FOLDER, num_episodes)
    print(f"成功读取 {len(hdf5_data)} 个episodes的数据")
    total_keyframes = sum(len(keyframes) for keyframes in hdf5_data.values())
    print(f"总共找到 {total_keyframes} 个keyframes")
    
    # 步骤2: 从replay buffer读取action数据
    print("\n[2/2] 从replay buffer读取action数据...")
    try:
        replay_data = read_action_data_from_replay_buffer(TRAIN_REPLAY_STORAGE_DIR, TASK_NAME)
        print(f"成功读取 {len(replay_data)} 个action数据")
    except Exception as e:
        print(f"Error reading replay buffer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 对比
    print("\n[3/3] 对比数据...")
    compare_gripper_positions(hdf5_data, replay_data)
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
