#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Buffer数据可视化脚本

该脚本用于可视化replay buffer数据，包括：
1. 每个.replay文件的基本信息（索引、文件大小、元数据等）
2. init_frame_to_idx.pkl的映射关系（表格形式）
3. replay_info.npy的终止状态统计信息

使用方法：
    python visualize_replay_data.py <replay_directory>
    
示例：
    python visualize_replay_data.py /path/to/replay_buffer/BinFill
    python visualize_replay_data.py /path/to/replay_buffer/BinFill --view-replay 0  # 查看第0个replay文件
    python visualize_replay_data.py /path/to/replay_buffer/BinFill --save-output report.txt  # 保存统计信息到文件
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_replay_info(replay_dir: str) -> Optional[np.ndarray]:
    """
    加载replay_info.npy文件
    
    参数:
        replay_dir: replay buffer目录路径
    
    返回:
        replay_info数组，如果文件不存在则返回None
    """
    replay_info_path = os.path.join(replay_dir, 'replay_info.npy')
    if not os.path.exists(replay_info_path):
        print(f"警告: 未找到 replay_info.npy 文件: {replay_info_path}")
        return None
    
    try:
        with open(replay_info_path, 'rb') as f:
            replay_info = np.load(f)
        return replay_info
    except Exception as e:
        print(f"错误: 无法加载 replay_info.npy: {e}")
        return None


def load_init_frame_to_idx(replay_dir: str) -> Optional[Dict]:
    """
    加载init_frame_to_idx.pkl文件
    
    参数:
        replay_dir: replay buffer目录路径
    
    返回:
        init_frame_to_idx字典，结构为 {demo_idx: {init_frame: [replay_idx1, ...]}}
        如果文件不存在则返回None
    """
    init_frame_to_idx_path = os.path.join(replay_dir, 'init_frame_to_idx.pkl')
    if not os.path.exists(init_frame_to_idx_path):
        print(f"警告: 未找到 init_frame_to_idx.pkl 文件: {init_frame_to_idx_path}")
        return None
    
    try:
        with open(init_frame_to_idx_path, 'rb') as f:
            init_frame_to_idx = pickle.load(f)
        return init_frame_to_idx
    except Exception as e:
        print(f"错误: 无法加载 init_frame_to_idx.pkl: {e}")
        return None


def get_replay_files_info(replay_dir: str) -> List[Tuple[int, str, int]]:
    """
    获取所有replay文件的信息
    
    参数:
        replay_dir: replay buffer目录路径
    
    返回:
        replay文件信息列表，每个元素为 (索引, 文件路径, 文件大小(字节))
    """
    replay_files = []
    
    if not os.path.isdir(replay_dir):
        print(f"错误: 目录不存在: {replay_dir}")
        return replay_files
    
    # 扫描所有.replay文件
    for filename in os.listdir(replay_dir):
        if filename.endswith('.replay'):
            try:
                # 提取文件索引
                index = int(filename.split('.')[0])
                filepath = os.path.join(replay_dir, filename)
                filesize = os.path.getsize(filepath)
                replay_files.append((index, filepath, filesize))
            except ValueError:
                # 如果不是数字文件名（如initial.replay），跳过
                continue
    
    # 按索引排序
    replay_files.sort(key=lambda x: x[0])
    return replay_files


def format_size(size_bytes: float) -> str:
    """将字节数格式化为人类可读的格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_replay_files_statistics(replay_files: List[Tuple[int, str, int]], output_file=None):
    """
    打印replay文件的统计信息
    
    参数:
        replay_files: replay文件信息列表
        output_file: 输出文件对象，如果为None则打印到stdout
    """
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, file=output_file, **kwargs)
        else:
            print(*args, **kwargs)
    
    print_func("=" * 80)
    print_func("1. Replay文件统计")
    print_func("=" * 80)
    
    if not replay_files:
        print_func("未找到任何.replay文件")
        return
    
    total_files = len(replay_files)
    total_size = sum(size for _, _, size in replay_files)
    indices = [idx for idx, _, _ in replay_files]
    sizes = [size for _, _, size in replay_files]
    
    print_func(f"\n总文件数量: {total_files}")
    print_func(f"文件索引范围: {min(indices)} - {max(indices)}")
    print_func(f"总文件大小: {format_size(total_size)}")
    print_func(f"平均文件大小: {format_size(float(total_size) / total_files)}")
    
    if sizes:
        print_func(f"最小文件大小: {format_size(min(sizes))}")
        print_func(f"最大文件大小: {format_size(max(sizes))}")
    
    # 显示前10个文件的信息
    print_func(f"\n前10个文件详情:")
    print_func("-" * 80)
    print_func(f"{'索引':<10} {'文件大小':<15} {'文件路径'}")
    print_func("-" * 80)
    
    for idx, filepath, size in replay_files[:10]:
        print_func(f"{idx:<10} {format_size(size):<15} {os.path.basename(filepath)}")
    
    if total_files > 10:
        print_func(f"... 还有 {total_files - 10} 个文件未显示")
    
    print_func()


def print_init_frame_to_idx_statistics(init_frame_to_idx: Dict, output_file=None):
    """
    打印init_frame_to_idx映射关系的统计信息
    
    参数:
        init_frame_to_idx: 映射字典，结构为 {demo_idx: {init_frame: [replay_idx1, ...]}}
        output_file: 输出文件对象，如果为None则打印到stdout
    """
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, file=output_file, **kwargs)
        else:
            print(*args, **kwargs)
    
    print_func("=" * 80)
    print_func("2. Init Frame to Index 映射关系")
    print_func("=" * 80)
    
    if not init_frame_to_idx:
        print_func("未找到 init_frame_to_idx 数据")
        return
    
    # 统计信息
    total_demos = len(init_frame_to_idx)
    total_init_frames = sum(len(frames) for frames in init_frame_to_idx.values())
    total_replay_indices = sum(
        len(replay_indices)
        for frames in init_frame_to_idx.values()
        for replay_indices in frames.values()
    )
    
    print_func(f"\n总Demo数量: {total_demos}")
    print_func(f"总Init Frame数量: {total_init_frames}")
    print_func(f"总Replay索引数量: {total_replay_indices}")
    
    # 按demo分组显示
    print_func(f"\n按Demo分组的映射关系:")
    print_func("-" * 80)
    
    # 对demo索引排序
    sorted_demos = sorted(init_frame_to_idx.keys())
    
    for demo_idx in sorted_demos:
        frames_dict = init_frame_to_idx[demo_idx]
        print_func(f"\nDemo {demo_idx}:")
        print_func(f"  Init Frame数量: {len(frames_dict)}")
        print_func(f"  Replay索引总数: {sum(len(indices) for indices in frames_dict.values())}")
        
        # 显示每个init_frame的映射
        sorted_frames = sorted(frames_dict.keys())
        print_func(f"  映射详情:")
        for init_frame in sorted_frames:
            replay_indices = frames_dict[init_frame]
            print_func(f"    Init Frame {init_frame:>3} -> Replay索引: {replay_indices}")
    
    print_func()


def print_replay_info_statistics(replay_info: np.ndarray, output_file=None):
    """
    打印replay_info的统计信息
    
    参数:
        replay_info: 终止状态数组（0或1）
        output_file: 输出文件对象，如果为None则打印到stdout
    """
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, file=output_file, **kwargs)
        else:
            print(*args, **kwargs)
    
    print_func("=" * 80)
    print_func("3. Replay Info 终止状态统计")
    print_func("=" * 80)
    
    if replay_info is None:
        print_func("未找到 replay_info 数据")
        return
    
    total_samples = len(replay_info)
    terminal_count = int(np.sum(replay_info == 1))
    non_terminal_count = total_samples - terminal_count
    terminal_ratio = terminal_count / total_samples if total_samples > 0 else 0.0
    
    print_func(f"\n总样本数量: {total_samples}")
    print_func(f"终止状态数量 (terminal=1): {terminal_count}")
    print_func(f"非终止状态数量 (terminal=0): {non_terminal_count}")
    print_func(f"终止状态比例: {terminal_ratio:.2%}")
    
    # 统计连续的终止状态分布（可选）
    print_func(f"\n终止状态分布:")
    terminal_indices = np.where(replay_info == 1)[0]
    if len(terminal_indices) > 0:
        print_func(f"  第一个终止状态索引: {terminal_indices[0]}")
        print_func(f"  最后一个终止状态索引: {terminal_indices[-1]}")
        print_func(f"  终止状态索引范围: {min(terminal_indices)} - {max(terminal_indices)}")
        
        # 显示前10个终止状态的索引
        print_func(f"\n  前10个终止状态索引:")
        for idx in terminal_indices[:10]:
            print_func(f"    索引 {idx}")
        if len(terminal_indices) > 10:
            print_func(f"    ... 还有 {len(terminal_indices) - 10} 个终止状态")
    else:
        print_func("  未找到终止状态")
    
    print_func()


def view_replay_file(replay_dir: str, index: int, max_keys: int = 20):
    """
    查看特定replay文件的详细内容
    
    参数:
        replay_dir: replay buffer目录路径
        index: replay文件索引
        max_keys: 最大显示键的数量
    """
    replay_file_path = os.path.join(replay_dir, f'{index}.replay')
    
    if not os.path.exists(replay_file_path):
        print(f"错误: 文件不存在: {replay_file_path}")
        return
    
    try:
        with open(replay_file_path, 'rb') as f:
            sample = pickle.load(f)
        
        print("=" * 80)
        print(f"Replay文件 #{index} 详细内容")
        print("=" * 80)
        
        print(f"\n文件路径: {replay_file_path}")
        print(f"文件大小: {format_size(os.path.getsize(replay_file_path))}")
        print(f"数据类型: {type(sample).__name__}")
        
        if isinstance(sample, dict):
            print(f"\n键数量: {len(sample)}")
            print(f"\n所有键:")
            keys = list(sample.keys())
            for i, key in enumerate(keys[:max_keys]):
                value = sample[key]
                print(f"\n  [{i+1}] {key}:")
                print(f"      类型: {type(value).__name__}")
                
                if isinstance(value, np.ndarray):
                    print(f"      形状: {value.shape}")
                    print(f"      数据类型: {value.dtype}")
                    if value.size < 20:  # 如果数组很小，显示值
                        print(f"      值: {value}")
                    else:
                        print(f"      值范围: [{value.min():.4f}, {value.max():.4f}]")
                elif isinstance(value, (list, tuple)):
                    print(f"      长度: {len(value)}")
                    if len(value) > 0:
                        print(f"      第一个元素类型: {type(value[0]).__name__}")
                        if len(value) <= 5:
                            print(f"      值: {value}")
                elif isinstance(value, (int, float, bool)):
                    print(f"      值: {value}")
                elif isinstance(value, str):
                    print(f"      值: {value[:100]}")  # 只显示前100个字符
                else:
                    print(f"      值: {str(value)[:100]}")
            
            if len(keys) > max_keys:
                print(f"\n  ... 还有 {len(keys) - max_keys} 个键未显示")
        else:
            print(f"\n值: {sample}")
        
    except Exception as e:
        print(f"错误: 无法加载replay文件: {e}")
        import traceback
        traceback.print_exc()


def print_all_statistics(replay_dir: str, output_file=None):
    """
    打印所有统计信息
    
    参数:
        replay_dir: replay buffer目录路径
        output_file: 输出文件对象，如果为None则打印到stdout
    """
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, file=output_file, **kwargs)
        else:
            print(*args, **kwargs)
    
    print_func("\n" + "=" * 80)
    print_func(f"Replay Buffer 数据统计报告")
    print_func(f"目录: {replay_dir}")
    print_func("=" * 80 + "\n")
    
    # 1. Replay文件统计
    replay_files = get_replay_files_info(replay_dir)
    print_replay_files_statistics(replay_files, output_file)
    
    # 2. Init Frame to Index 映射
    init_frame_to_idx = load_init_frame_to_idx(replay_dir)
    if init_frame_to_idx:
        print_init_frame_to_idx_statistics(init_frame_to_idx, output_file)
    
    # 3. Replay Info 统计
    replay_info = load_replay_info(replay_dir)
    if replay_info is not None:
        print_replay_info_statistics(replay_info, output_file)
    
    print_func("=" * 80)
    print_func("统计报告完成")
    print_func("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='可视化Replay Buffer数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看所有统计信息
  python visualize_replay_data.py /path/to/replay_buffer/BinFill
  
  # 查看特定replay文件的详细内容
  python visualize_replay_data.py /path/to/replay_buffer/BinFill --view-replay 0
  
  # 保存统计信息到文件
  python visualize_replay_data.py /path/to/replay_buffer/BinFill --save-output report.txt
        """
    )
    
    parser.add_argument(
        'replay_dir',
        type=str,
        nargs='?',
        help='Replay buffer目录路径',
        default="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/dataset_generate/sam2act/test_buffer/BinFill"
    )
    
    parser.add_argument(
        '--view-replay',
        type=int,
        metavar='INDEX',
        help='查看指定索引的replay文件详细内容',
        default=None
    )
    
    parser.add_argument(
        '--save-output',
        type=str,
        metavar='FILE',
        help='将统计信息保存到文件'
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(args.replay_dir):
        print(f"错误: 目录不存在: {args.replay_dir}")
        sys.exit(1)
    
    # 如果指定了--view-replay，只查看特定文件
    if args.view_replay is not None:
        view_replay_file(args.replay_dir, args.view_replay)
        return
    
    # 打印所有统计信息
    if args.save_output:
        with open(args.save_output, 'w', encoding='utf-8') as f:
            print_all_statistics(args.replay_dir, output_file=f)
        print(f"\n统计信息已保存到: {args.save_output}")
    else:
        print_all_statistics(args.replay_dir)


if __name__ == "__main__":
    main()