#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_dataset_temporal 函数的数据读取功能

本文件用于测试时序强化学习数据集的创建和读取功能。主要功能包括：
1. 创建时序 replay buffer（支持多时间步的数据存储）
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

# ============================================================================
# 第三方库导入
# ============================================================================
import torch   # PyTorch 深度学习框架，用于张量操作和 GPU 管理
import clip    # OpenAI CLIP 模型，用于提取语言特征和图像-文本匹配

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
# 从 sam2act.utils.dataset 导入时序 replay buffer 的创建和填充函数
# create_replay_temporal: 创建支持时序数据的 replay buffer
# fill_replay_temporal: 将演示数据填充到 replay buffer 中
from sam2act.utils.dataset import create_replay_temporal, fill_replay_temporal

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
        # 构建训练集和验证集的演示数据路径
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
            demo_augmentation=True,                         # 启用演示数据增强，通过采样不同帧增加数据多样性
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
                demo_augmentation=True,                     # 同样启用数据增强
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,                    # 验证集数据路径（与训练集不同）
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
                # 注意：测试集不需要 rank 参数，因为通常只在单进程下使用
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
TRAIN_REPLAY_STORAGE_DIR = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/test_buffer"

# RLBench 原始数据集的根目录
# 包含所有任务的演示数据，目录结构为：{DATA_FOLDER}/train/{task}/all_variations/episodes/
DATA_FOLDER = "/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/dataset/rlbench-18-tasks/data"

# ============================================================================
# 数据集创建参数
# ============================================================================
# 任务列表：要处理的任务名称列表
# 可以包含多个任务，例如：["close_jar", "open_drawer", "pick_and_lift_simple"]
tasks = ["close_jar"]  # 当前只测试单个任务

# 批次大小配置
BATCH_SIZE_TRAIN = 4        # 训练集的批次大小，影响 replay buffer 的采样批量
BATCH_SIZE_TEST = None      # 测试集的批次大小，如果 only_train=True 可以设为 None

# 测试集 replay buffer 存储目录（如果 only_train=False 需要设置）
TEST_REPLAY_STORAGE_DIR = None

# 演示数量配置
NUM_TRAIN = 10   # 每个任务使用的训练演示数量，从演示数据集中选择前 NUM_TRAIN 个演示
NUM_VAL = None   # 每个任务使用的验证演示数量，如果 only_train=True 可以设为 None

# 数据刷新标志
# 如果为 True，会删除已存在的 replay buffer 数据并重新生成
# 如果为 False，会直接加载已存在的 replay buffer 数据（如果存在）
refresh_replay = True  # 刷新已有数据，重新生成 replay buffer

# 计算设备配置
# 自动检测是否有可用的 CUDA 设备，如果有则使用 GPU，否则使用 CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据加载配置
num_workers = 2  # DataLoader 的并行工作进程数，用于加速数据加载

# 数据集模式配置
only_train = True  # 是否只创建训练集，如果为 True 则不创建测试集

# 时序记忆配置
num_maskmem = 8  # 时序记忆模块的数量，用于存储历史掩码信息（默认值）

# 多进程配置
rank = 0  # 进程排名，单进程训练时通常为 0，多进程训练时用于区分不同进程

# 采样分布模式
# "transition_uniform" 表示均匀采样所有状态转换，确保每个转换被采样的概率相等
sample_distribution_mode = "transition_uniform"

def main():
    """
    主测试函数
    
    该函数执行完整的数据集创建和读取测试流程：
    1. 打印配置信息
    2. 调用 get_dataset_temporal 创建数据集
    3. 从数据集中读取样本并打印详细信息
    4. 验证数据格式和内容是否正确
    
    测试流程分为三个步骤：
    - [1/3] 创建数据集：调用 get_dataset_temporal 函数
    - [2/3] 创建迭代器：为数据集创建迭代器用于读取样本
    - [3/3] 读取样本：读取并打印前 5 个样本的详细信息
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
        print("\n[2/3] 开始读取测试样本...")
        data_iter = iter(train_dataset)
        
        # ====================================================================
        # 步骤 3: 读取并打印样本信息
        # ====================================================================
        # 读取前 5 个样本，打印每个样本的详细信息
        # 这有助于验证数据格式是否正确，以及了解数据的结构
        print("\n[3/3] 读取并打印样本信息:")
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
