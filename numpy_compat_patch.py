#!/usr/bin/env python3
"""
NumPy 2.0 兼容性补丁
用于在 NumPy 1.x 环境中加载 NumPy 2.0+ 创建的 pickle 文件

使用方法：
在加载 pickle 文件之前导入此模块：
    import numpy_compat_patch
    import pickle
    # 然后正常加载 pickle 文件
"""
import sys
import numpy as np

# 检查是否是 numpy 1.x 版本
if np.__version__.startswith('1.'):
    # 创建兼容性模块
    class NumpyCoreCompat:
        """兼容性包装器，将 numpy._core 映射到 numpy.core"""
        def __getattr__(self, name):
            # 将 _core 的访问重定向到 core
            if name == '_core':
                return np.core
            # 对于其他属性，尝试从 numpy 获取
            return getattr(np, name)
    
    # 将兼容性模块注入到 sys.modules
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    
    # 尝试导入更多可能的子模块
    try:
        sys.modules['numpy._core.arrayprint'] = np.core.arrayprint
    except AttributeError:
        pass
    
    try:
        sys.modules['numpy._core.fromnumeric'] = np.core.fromnumeric
    except AttributeError:
        pass
    
    try:
        sys.modules['numpy._core.numeric'] = np.core.numeric
    except AttributeError:
        pass
    
    try:
        sys.modules['numpy._core.defchararray'] = np.core.defchararray
    except AttributeError:
        pass
    
    print(f"NumPy 兼容性补丁已激活 (NumPy {np.__version__} -> 兼容 NumPy 2.0+ pickle 文件)")
else:
    print(f"NumPy 版本 {np.__version__} 不需要兼容性补丁")
