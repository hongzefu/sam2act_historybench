# Hbench13 脚本使用说明

本脚本用于运行 `test_load_and_pointcloud_create_Hbench13-fixFirst.py`，支持在 SSH 退出后继续运行，并自动记录日志。

## 文件说明

- `run_Hbench13_fixFirst.sh` - 运行和管理脚本（所有功能集成）
- `logs/` - 日志文件目录（自动创建）

## 快速开始

### 1. 运行脚本（默认行为）

```bash
./run_Hbench13_fixFirst.sh
```

或者明确指定：

```bash
./run_Hbench13_fixFirst.sh run
```

脚本会：
- 自动激活 `micromamba sam2act4` 环境
- 在后台运行 Python 脚本
- 将输出保存到日志文件
- 即使 SSH 断开连接，进程仍会继续运行

### 2. 查看运行状态

```bash
./run_Hbench13_fixFirst.sh status
```

### 3. 查看实时日志

```bash
./run_Hbench13_fixFirst.sh logs
```

### 4. 查看错误日志

```bash
./run_Hbench13_fixFirst.sh errors
```

### 5. 停止进程

```bash
./run_Hbench13_fixFirst.sh stop
```

## 所有可用命令

| 命令 | 说明 |
|------|------|
| `run` 或 无参数 | 运行 Python 脚本（后台运行，支持 SSH 断开） |
| `status` | 显示运行状态、PID、日志文件信息等 |
| `logs` | 实时查看最新的日志输出（类似 `tail -f`） |
| `errors` | 实时查看错误日志输出 |
| `stop` | 停止正在运行的进程 |
| `list` | 列出所有日志文件和 PID 文件 |
| `clean` | 清理7天前的日志文件 |
| `help` | 显示使用说明 |

## 命令详解

### run - 运行脚本（默认）

启动 Python 脚本，在后台运行。

```bash
./run_Hbench13_fixFirst.sh
# 或
./run_Hbench13_fixFirst.sh run
```

### status - 查看状态

显示进程运行状态、PID、日志文件信息等。

```bash
./run_Hbench13_fixFirst.sh status
```

输出示例：
```
进程正在运行
PID: 12345
PID 文件: logs/Hbench13_fixFirst_20240101_120000.pid

进程详细信息:
  PID  PPID CMD                         %MEM %CPU     ELAPSED STAT
12345     1 python test_load_and...     5.2  10.5    01:23:45 S

日志文件: logs/Hbench13_fixFirst_20240101_120000.log
文件大小: 15M
最后更新: 2024-01-01 13:23:45
```

### logs - 实时查看日志

实时查看最新的日志输出（类似 `tail -f`）。

```bash
./run_Hbench13_fixFirst.sh logs
```

按 `Ctrl+C` 退出。

### errors - 实时查看错误日志

实时查看错误日志输出。

```bash
./run_Hbench13_fixFirst.sh errors
```

按 `Ctrl+C` 退出。

### stop - 停止进程

停止正在运行的进程。

```bash
./run_Hbench13_fixFirst.sh stop
```

### list - 列出所有日志文件

列出所有日志文件和 PID 文件。

```bash
./run_Hbench13_fixFirst.sh list
```

### clean - 清理旧日志

删除7天前的日志文件。

```bash
./run_Hbench13_fixFirst.sh clean
```

## 日志文件

日志文件保存在 `logs/` 目录下，命名格式：
- 标准日志: `Hbench13_fixFirst_YYYYMMDD_HHMMSS.log`
- 错误日志: `Hbench13_fixFirst_YYYYMMDD_HHMMSS_error.log`
- PID 文件: `Hbench13_fixFirst_YYYYMMDD_HHMMSS.pid`

## 手动查看日志

如果知道日志文件名，也可以直接查看：

```bash
# 查看完整日志
cat logs/Hbench13_fixFirst_*.log

# 查看最后100行
tail -n 100 logs/Hbench13_fixFirst_*.log

# 实时查看
tail -f logs/Hbench13_fixFirst_*.log
```

## 完整工作流示例

```bash
# 1. 启动脚本
./run_Hbench13_fixFirst.sh

# 2. 检查状态（在另一个终端）
./run_Hbench13_fixFirst.sh status

# 3. 查看日志（在另一个终端）
./run_Hbench13_fixFirst.sh logs

# 4. 如果需要停止
./run_Hbench13_fixFirst.sh stop

# 5. 清理旧日志（定期执行）
./run_Hbench13_fixFirst.sh clean
```

## 注意事项

1. **环境要求**: 确保已安装并配置 `micromamba`，且存在 `sam2act4` 环境
2. **磁盘空间**: 日志文件可能会很大，定期清理旧日志
3. **进程管理**: 如果脚本异常退出，PID 文件可能仍存在，需要手动清理
4. **权限**: 确保脚本有执行权限（已自动设置）

## 故障排除

### 找不到 micromamba

如果提示找不到 `micromamba`，请检查：
1. `micromamba` 是否已安装
2. 是否已初始化（运行 `micromamba shell init`）
3. 是否在 `~/.bashrc` 中配置了初始化代码

### 环境未激活

如果环境未正确激活，可以手动激活：

```bash
micromamba activate sam2act4
python test_load_and_pointcloud_create_Hbench13-fixFirst.py
```

### 进程无法停止

如果正常停止失败，可以强制终止：

```bash
# 查找进程
ps aux | grep test_load_and_pointcloud_create_Hbench13

# 强制终止（替换 PID）
kill -9 <PID>
```

### 查看帮助

```bash
./run_Hbench13_fixFirst.sh help
```
