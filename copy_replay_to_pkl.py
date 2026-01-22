#!/usr/bin/env python3
"""
脚本：将指定目录下所有 .replay 文件复制为 .pkl 文件
"""
import os
import shutil
from pathlib import Path

def copy_replay_to_pkl(root_dir):
    """
    递归查找所有 .replay 文件，并复制为 .pkl 文件
    
    Args:
        root_dir: 根目录路径
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误：目录不存在: {root_dir}")
        return
    
    # 查找所有 .replay 文件
    replay_files = list(root_path.rglob("*.replay"))
    
    if not replay_files:
        print(f"未找到任何 .replay 文件在目录: {root_dir}")
        return
    
    print(f"找到 {len(replay_files)} 个 .replay 文件")
    
    success_count = 0
    error_count = 0
    
    for replay_file in replay_files:
        try:
            # 生成对应的 .pkl 文件路径
            pkl_file = replay_file.with_suffix('.pkl')
            
            # 如果 .pkl 文件已存在，跳过
            if pkl_file.exists():
                print(f"跳过（已存在）: {pkl_file}")
                continue
            
            # 复制文件
            shutil.copy2(replay_file, pkl_file)
            print(f"已复制: {replay_file} -> {pkl_file}")
            success_count += 1
            
        except Exception as e:
            print(f"错误：复制 {replay_file} 时出错: {e}")
            error_count += 1
    
    print(f"\n完成！")
    print(f"成功: {success_count} 个文件")
    print(f"错误: {error_count} 个文件")
    print(f"总计: {len(replay_files)} 个文件")

if __name__ == "__main__":
    target_dir = "/home/hongzefu/sam2act_historybench/dataset_generate/sam2act"
    copy_replay_to_pkl(target_dir)
