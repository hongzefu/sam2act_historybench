#!/usr/bin/env python3
"""
自动安装 NumPy 2.0 兼容性补丁
"""
import sys
import site
from pathlib import Path

def install_patch():
    """安装兼容性补丁到 site-packages"""
    import numpy as np
    
    if not np.__version__.startswith('1.'):
        print(f"NumPy 版本 {np.__version__} 不需要兼容性补丁")
        return True
    
    # 优先使用用户级 site-packages（不需要 sudo）
    user_site = site.getusersitepackages()
    if user_site:
        site_packages_dir = Path(user_site)
        print(f"使用用户级 site-packages: {site_packages_dir}")
    else:
        # 回退到系统级 site-packages
        site_packages = site.getsitepackages()
        if not site_packages:
            try:
                from distutils.sysconfig import get_python_lib
                site_packages = [get_python_lib()]
            except:
                print("无法找到 site-packages 目录")
                return False
        site_packages_dir = Path(site_packages[0])
        print(f"使用系统级 site-packages: {site_packages_dir}")
    
    # 确保目录存在
    site_packages_dir.mkdir(parents=True, exist_ok=True)
    patch_file = site_packages_dir / "numpy_compat_patch.py"
    pth_file = site_packages_dir / "numpy_compat.pth"
    
    # 补丁内容
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
        patch_file.write_text(patch_content)
        print(f"✓ 已创建补丁文件: {patch_file}")
        
        # 创建 .pth 文件以自动加载
        pth_file.write_text("import numpy_compat_patch\n")
        print(f"✓ 已创建自动加载文件: {pth_file}")
        print(f"\n✓ 兼容性补丁已安装！")
        print(f"  当前 NumPy 版本: {np.__version__}")
        print(f"  请重启 Python 或 Cursor 以使补丁生效。")
        return True
    except PermissionError:
        print(f"✗ 权限不足，无法写入 {site_packages_dir}")
        print(f"  请使用以下命令安装：")
        print(f"  sudo python3 {sys.argv[0]}")
        return False
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        return False

if __name__ == "__main__":
    install_patch()
