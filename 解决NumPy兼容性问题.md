# 解决 NumPy 2.0 Pickle 兼容性问题

## 问题描述

当使用 Cursor 的 Pickler 扩展查看 pickle 文件时，出现错误：
```
ModuleNotFoundError: No module named 'numpy._core'
```

这是因为 pickle 文件是用 NumPy 2.0+ 创建的，而当前环境使用的是 NumPy 1.x。NumPy 2.0 将 `numpy.core` 重命名为 `numpy._core`，导致版本不兼容。

## 解决方案

### 方案 1：安装自动兼容性补丁（推荐，已完成✅）

运行以下命令安装自动补丁：

```bash
python3 install_numpy_compat.py
```

**注意**：补丁会安装到用户级目录（`~/.local/lib/python3.8/site-packages/`），不需要 sudo 权限。

安装后，**重启 Cursor** 即可。补丁会自动加载，无需修改代码。

✅ **补丁已成功安装并验证！** 现在可以正常使用 Cursor 的 Pickler 扩展查看 pickle 文件了。

### 方案 2：在代码中手动导入补丁

在加载 pickle 文件之前，先导入兼容性补丁：

```python
import numpy_compat_patch  # 从项目根目录导入
import pickle

# 然后正常加载 pickle 文件
with open('file.pkl', 'rb') as f:
    data = pickle.load(f)
```

### 方案 3：在代码开头添加兼容性代码

将以下代码添加到您的脚本开头：

```python
import sys
import numpy as np

# NumPy 2.0 兼容性补丁
if np.__version__.startswith('1.'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
```

### 方案 4：升级 NumPy 到 2.0+（如果项目允许）

如果项目依赖允许，可以升级 NumPy：

```bash
pip install "numpy>=2.0"
```

**注意**：升级 NumPy 可能会影响其他依赖，请先测试。

## 验证补丁是否生效

运行以下命令测试：

```bash
python3 -c "import pickle; f = open('dataset_generate/sam2act/test_buffer/PickXtimes/29.pkl', 'rb'); data = pickle.load(f); print('✓ 成功！补丁已自动生效。')"
```

如果看到 "✓ 成功！补丁已自动生效。"，说明补丁已生效。

**注意**：如果在新启动的 Python 环境中测试，补丁会自动加载。如果是在已运行的 Python 进程中，需要重启进程。

## 文件说明

- `numpy_compat_patch.py` - 兼容性补丁模块
- `install_numpy_compat.py` - 自动安装脚本
- `fix_numpy_pickle.py` - 交互式修复工具

## 技术细节

NumPy 2.0 引入了以下变化：
- `numpy.core` → `numpy._core`
- `numpy.core.multiarray` → `numpy._core.multiarray`
- `numpy.core.umath` → `numpy._core.umath`

兼容性补丁通过 `sys.modules` 将新的模块路径映射到旧的路径，使得 NumPy 1.x 可以加载 NumPy 2.0+ 创建的 pickle 文件。
