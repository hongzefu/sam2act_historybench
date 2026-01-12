#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行解压指定目录下的所有 zip 文件
使用 20 个并行进程
"""

import os
import zipfile
import multiprocessing
from pathlib import Path
import argparse
import threading
import time

# 尝试导入 tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度显示类
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.current = 0
            self.desc = kwargs.get('desc', '')
        
        def update(self, n=1):
            self.current += n
            if self.total > 0:
                percent = (self.current / self.total) * 100
                print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%)", end='', flush=True)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            print()  # 换行


def unzip_file(zip_path, progress_dict=None):
    """
    解压单个 zip 文件
    
    Args:
        zip_path: zip 文件的路径
        progress_dict: 共享的进度字典，用于存储解压进度
        
    Returns:
        (zip_path, success, message, file_size): 元组，包含文件路径、是否成功、消息、文件大小
    """
    try:
        zip_path = Path(zip_path)
        zip_path_str = str(zip_path)
        
        if not zip_path.exists():
            return (zip_path_str, False, f"文件不存在", 0)
        
        file_size = zip_path.stat().st_size / (1024 * 1024)  # MB
        
        # 解压到 zip 文件所在的目录
        extract_dir = zip_path.parent
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 检查是否已经解压过（检查是否有同名目录，去掉 .zip 后缀）
            expected_dir = extract_dir / zip_path.stem
            if expected_dir.exists() and expected_dir.is_dir():
                # 检查目录是否非空
                if any(expected_dir.iterdir()):
                    if progress_dict is not None:
                        progress_dict[zip_path_str] = {
                            'total': 0,
                            'current': 0,
                            'status': '已跳过'
                        }
                    return (zip_path_str, True, f"已存在，跳过", file_size)
            
            # 获取文件列表
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # 初始化进度信息
            if progress_dict is not None:
                progress_dict[zip_path_str] = {
                    'total': total_files,
                    'current': 0,
                    'status': '解压中'
                }
            
            # 逐个解压文件以跟踪进度
            for i, member in enumerate(file_list):
                zip_ref.extract(member, extract_dir)
                if progress_dict is not None:
                    progress_dict[zip_path_str]['current'] = i + 1
            
            # 标记为完成
            if progress_dict is not None:
                progress_dict[zip_path_str]['status'] = '完成'
            
            return (zip_path_str, True, f"解压成功 ({total_files} 个文件)", file_size)
            
    except zipfile.BadZipFile:
        zip_path_str = str(zip_path) if isinstance(zip_path, Path) else zip_path
        if progress_dict is not None:
            progress_dict[zip_path_str] = {
                'total': 0,
                'current': 0,
                'status': '错误: 不是有效的 zip 文件'
            }
        return (zip_path_str, False, f"不是有效的 zip 文件", 0)
    except Exception as e:
        zip_path_str = str(zip_path) if isinstance(zip_path, Path) else zip_path
        if progress_dict is not None:
            progress_dict[zip_path_str] = {
                'total': 0,
                'current': 0,
                'status': f'错误: {str(e)}'
            }
        return (zip_path_str, False, f"错误: {str(e)}", 0)


def find_all_zip_files(root_dir):
    """
    递归查找所有 zip 文件
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        list: zip 文件路径列表
    """
    root_path = Path(root_dir)
    zip_files = list(root_path.rglob("*.zip"))
    return [str(f) for f in zip_files]


def main():
    parser = argparse.ArgumentParser(description="并行解压指定目录下的所有 zip 文件")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/dataset/rlbench-18-tasks",
        help="要解压的根目录路径"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="并行进程数（默认：20）"
    )
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    num_workers = args.num_workers
    
    print("=" * 80)
    print("并行解压 zip 文件")
    print("=" * 80)
    print(f"根目录: {root_dir}")
    print(f"并行进程数: {num_workers}")
    print("=" * 80)
    
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 目录不存在: {root_dir}")
        return
    
    # 查找所有 zip 文件
    print("\n正在查找所有 zip 文件...")
    zip_files = find_all_zip_files(root_dir)
    
    if not zip_files:
        print("未找到任何 zip 文件")
        return
    
    print(f"找到 {len(zip_files)} 个 zip 文件")
    print("\n开始并行解压...\n")
    
    # 创建共享的进度字典
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    
    # 用于控制进度显示线程的标志
    display_running = threading.Event()
    display_running.set()
    
    def display_progress():
        """定期显示每个文件的解压进度"""
        while display_running.is_set():
            time.sleep(0.5)  # 每0.5秒更新一次
            if not progress_dict:
                continue
            
            # 清屏并显示进度（使用 ANSI 转义码）
            print("\033[2J\033[H", end='')  # 清屏并移动光标到顶部
            print("=" * 80)
            print("并行解压 zip 文件 - 实时进度")
            print("=" * 80)
            print(f"总文件数: {len(zip_files)} | 已完成: {len([k for k, v in progress_dict.items() if v.get('status') in ['完成', '已跳过']])}")
            print("=" * 80)
            print()
            
            # 显示正在解压的文件（最多显示10个）
            active_files = [(k, v) for k, v in progress_dict.items() 
                          if v.get('status') == '解压中']
            active_files.sort(key=lambda x: x[0])
            
            if active_files:
                print("正在解压的文件:")
                for zip_path, progress_info in active_files[:10]:
                    try:
                        rel_path = Path(zip_path).relative_to(root_dir)
                        file_name = str(rel_path)
                    except:
                        file_name = Path(zip_path).name
                    
                    total = progress_info.get('total', 0)
                    current = progress_info.get('current', 0)
                    if total > 0:
                        percent = (current / total) * 100
                        bar_length = 30
                        filled = int(bar_length * current / total)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"  [{bar}] {current}/{total} ({percent:.1f}%) - {file_name[:60]}")
                    else:
                        print(f"  准备中... - {file_name[:60]}")
                
                if len(active_files) > 10:
                    print(f"  ... 还有 {len(active_files) - 10} 个文件正在解压")
                print()
            
            # 显示错误状态的文件（如果有）
            error_files = [(k, v) for k, v in progress_dict.items() 
                          if v.get('status', '').startswith('错误:')]
            if error_files:
                print("错误文件:")
                for zip_path, progress_info in error_files[:5]:
                    try:
                        rel_path = Path(zip_path).relative_to(root_dir)
                        file_name = str(rel_path)
                    except:
                        file_name = Path(zip_path).name
                    status = progress_info.get('status', '')
                    print(f"  ✗ {status} - {file_name[:60]}")
                if len(error_files) > 5:
                    print(f"  ... 还有 {len(error_files) - 5} 个错误文件")
                print()
            
            # 显示最近完成的文件（最多显示5个）
            completed_files = [(k, v) for k, v in progress_dict.items() 
                             if v.get('status') in ['完成', '已跳过']]
            if completed_files:
                print("最近完成的文件:")
                for zip_path, progress_info in completed_files[-5:]:
                    try:
                        rel_path = Path(zip_path).relative_to(root_dir)
                        file_name = str(rel_path)
                    except:
                        file_name = Path(zip_path).name
                    
                    status = progress_info.get('status', '')
                    total = progress_info.get('total', 0)
                    if total > 0:
                        print(f"  ✓ {status} ({total} 个文件) - {file_name[:60]}")
                    else:
                        print(f"  ✓ {status} - {file_name[:60]}")
                print()
    
    # 启动进度显示线程
    display_thread = threading.Thread(target=display_progress, daemon=True)
    display_thread.start()
    
    # 使用进程池并行解压
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 准备参数：将进度字典传递给每个任务
        tasks = [(zip_file, progress_dict) for zip_file in zip_files]
        
        # 使用 imap_unordered 可以更快地显示进度
        for zip_file in zip_files:
            result = pool.apply_async(unzip_file, (zip_file, progress_dict))
            results.append(result)
        
        # 等待所有任务完成
        final_results = []
        for result in results:
            final_results.append(result.get())
    
    # 停止进度显示
    display_running.clear()
    time.sleep(0.6)  # 等待最后一次显示更新
    
    # 清屏并显示最终结果
    print("\033[2J\033[H", end='')
    
    # 处理结果
    results = final_results
    
    # 统计结果
    print("\n" + "=" * 80)
    print("解压完成！统计结果：")
    print("=" * 80)
    
    success_count = sum(1 for _, success, message, _ in results if success and "跳过" not in message)
    skipped_count = sum(1 for _, success, message, _ in results if success and "跳过" in message)
    failed_count = sum(1 for _, success, _, _ in results if not success)
    total_size = sum(size for _, _, _, size in results)
    
    print(f"总计: {len(results)} 个文件")
    print(f"成功解压: {success_count} 个")
    print(f"跳过（已存在）: {skipped_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"总大小: {total_size:.2f} MB")
    
    # 显示失败的文件
    if failed_count > 0:
        print("\n失败的文件:")
        for zip_path, success, message, _ in results:
            if not success:
                try:
                    rel_path = Path(zip_path).relative_to(root_dir)
                    display_path = str(rel_path)
                except:
                    display_path = zip_path
                print(f"  - {display_path}: {message}")
    
    # 显示跳过的文件（已存在）- 可选，如果数量不多就显示
    skipped = [r for r in results if r[1] and "跳过" in r[2]]
    if skipped and len(skipped) <= 10:
        print(f"\n跳过的文件（已存在）:")
        for zip_path, _, message, _ in skipped:
            try:
                rel_path = Path(zip_path).relative_to(root_dir)
                display_path = str(rel_path)
            except:
                display_path = zip_path
            print(f"  - {display_path}: {message}")
    elif skipped:
        print(f"\n跳过的文件（已存在）: {len(skipped)} 个（已省略详细列表）")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

