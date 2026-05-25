#!/usr/bin/env python3
"""
修复 NumPy pickle 兼容性问题的工具脚本

此脚本会：
1. 在用户主目录创建 .pth 文件，自动加载兼容性补丁
2. 或者提供手动修复方法
"""
import os
import sys
from pathlib import Path

def create_site_packages_patch():
    """在 site-packages 中创建自动加载的补丁"""
    import site
    import numpy as np
    
    # 获取 site-packages 目录
    site_packages = site.getsitepackages()
    if not site_packages:
        # 如果没有找到，尝试使用 distutils
        try:
            from distutils.sysconfig import get_python_lib
            site_packages = [get_python_lib()]
        except:
            print("无法找到 site-packages 目录")
            return False
    
    site_packages_dir = site_packages[0]
    patch_file = Path(site_packages_dir) / "numpy_compat_patch.py"
    pth_file = Path(site_packages_dir) / "numpy_compat.pth"
    
    # 创建补丁文件
    patch_content = '''"""NumPy 2.0 兼容性补丁 - 自动加载"""
import sys
import numpy as np

# 检查是否是 numpy 1.x 版本
if np.__version__.startswith('1.'):
    # 创建兼容性模块映射
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    
    # 尝试导入更多可能的子模块
    for submodule in ['arrayprint', 'fromnumeric', 'numeric', 'defchararray']:
        try:
            sys.modules[f'numpy._core.{submodule}'] = getattr(np.core, submodule)
        except AttributeError:
            pass
'''
    
    try:
        # 写入补丁文件
        with open(patch_file, 'w') as f:
            f.write(patch_content)
        print(f"✓ 已创建补丁文件: {patch_file}")
        
        # 创建 .pth 文件以自动加载
        with open(pth_file, 'w') as f:
            f.write("import numpy_compat_patch\n")
        print(f"✓ 已创建自动加载文件: {pth_file}")
        print(f"\n✓ 兼容性补丁已安装！现在可以加载 NumPy 2.0+ 创建的 pickle 文件了。")
        print(f"  当前 NumPy 版本: {np.__version__}")
        return True
    except PermissionError:
        print(f"✗ 权限不足，无法写入 {site_packages_dir}")
        print(f"  请使用 sudo 运行此脚本，或手动安装补丁")
        return False
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        return False

def main():
    print("=" * 60)
    print("NumPy 2.0 Pickle 兼容性修复工具")
    print("=" * 60)
    print()
    
    import numpy as np
    print(f"当前 NumPy 版本: {np.__version__}")
    print()
    
    if not np.__version__.startswith('1.'):
        print("您的 NumPy 版本已经是 2.0+，不需要兼容性补丁。")
        return
    
    print("检测到 NumPy 1.x 版本，需要安装兼容性补丁。")
    print()
    
    choice = input("是否安装系统级补丁（需要管理员权限）？[y/N]: ").strip().lower()
    
    if choice == 'y':
        if create_site_packages_patch():
            print("\n✓ 安装完成！请重启 Python 或 Cursor 以使补丁生效。")
        else:
            print("\n✗ 安装失败。请查看上面的错误信息。")
    else:
        print("\n手动使用方法：")
        print("=" * 60)
        print("在加载 pickle 文件之前，先导入兼容性补丁：")
        print()
        print("  import numpy_compat_patch  # 从项目根目录导入")
        print("  import pickle")
        print("  # 然后正常加载 pickle 文件")
        print()
        print("或者将以下代码添加到您的脚本开头：")
        print("=" * 60)
        print("""
import sys
import numpy as np
if np.__version__.startswith('1.'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
""")
        print("=" * 60)

if __name__ == "__main__":
    main()
