#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_dataset_temporal 函数的数据读取功能
"""

import os
import sys
import torch

# 添加项目根目录到路径，确保可以导入 sam2act 模块
# 测试文件位于项目根目录，sam2act 包在 sam2act/ 子目录下
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sam2act.utils.get_dataset import get_dataset_temporal

# 设置参数
TRAIN_REPLAY_STORAGE_DIR = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/test_buffer"
DATA_FOLDER = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/dataset/rlbench-18-tasks/data"

# 其他参数
tasks = ["close_jar"]  # 单个任务
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = None
TEST_REPLAY_STORAGE_DIR = None
NUM_TRAIN = 10  # 训练样本数量
NUM_VAL = None
refresh_replay = True  # 刷新已有数据，重新生成 replay buffer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_workers = 2
only_train = True
num_maskmem = 8  # 默认值
rank = 0
sample_distribution_mode = "transition_uniform"

def main():
    print("=" * 80)
    print("测试 get_dataset_temporal 函数")
    print("=" * 80)
    print(f"TRAIN_REPLAY_STORAGE_DIR: {TRAIN_REPLAY_STORAGE_DIR}")
    print(f"DATA_FOLDER: {DATA_FOLDER}")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE_TRAIN}")
    print(f"Number of training samples: {NUM_TRAIN}")
    print("=" * 80)
    
    try:
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
        
        print("\n[2/3] 开始读取测试样本...")
        data_iter = iter(train_dataset)
        
        print("\n[3/3] 读取并打印样本信息:")
        print("-" * 80)
        
        for i in range(5):  # 读取5个样本
            try:
                print(f"\n样本 {i+1}:")
                batch = next(data_iter)
                
                # 打印所有键
                print(f"  键 (keys): {list(batch.keys())}")
                
                # 打印每个键的形状信息
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    {key}: type={type(value).__name__}, length={len(value)}")
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            print(f"      [0]: shape={value[0].shape}, dtype={value[0].dtype}")
                    else:
                        print(f"    {key}: type={type(value).__name__}, value={str(value)[:100]}")
                
            except StopIteration:
                print(f"\n警告: 数据集只有 {i} 个样本，无法读取第 {i+1} 个样本")
                break
            except Exception as e:
                print(f"\n错误: 读取样本 {i+1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: 调用 get_dataset_temporal 时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
