#!/bin/bash
# Hbench_replay 脚本运行和管理工具
# 使用 micromamba sam2act4 环境，支持 SSH 退出后继续运行，并记录日志
#
# 用法:
#   ./run_Hbench_replay.sh          # 运行脚本（默认）
#   ./run_Hbench_replay.sh status   # 查看运行状态
#   ./run_Hbench_replay.sh logs     # 查看实时日志
#   ./run_Hbench_replay.sh errors   # 查看错误日志
#   ./run_Hbench_replay.sh stop     # 停止进程
#   ./run_Hbench_replay.sh list     # 列出所有日志文件
#   ./run_Hbench_replay.sh clean    # 清理旧日志

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置日志目录
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Python 脚本路径
PYTHON_SCRIPT="${SCRIPT_DIR}/test_load_and_pointcloud_create_Hbench14-fixLangGoal.py"

# ============================================================================
# 辅助函数
# ============================================================================

# 显示使用说明
show_usage() {
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  (无参数)  - 运行 Python 脚本（后台运行，支持 SSH 断开）"
    echo "  status    - 显示运行状态"
    echo "  logs      - 查看最新日志（实时，包含所有输出，无缓冲）"
    echo "  errors    - 从日志中查看错误信息（实时过滤）"
    echo "  stop      - 停止正在运行的进程"
    echo "  list      - 列出所有日志文件"
    echo "  clean     - 清理旧的日志文件（保留最近7天）"
    echo ""
    echo "示例:"
    echo "  $0              # 启动脚本"
    echo "  $0 status       # 查看状态"
    echo "  $0 logs         # 查看日志"
    echo "  $0 stop         # 停止进程"
}

# 初始化 micromamba
init_micromamba() {
    # 初始化 micromamba（如果尚未初始化）
    if [ -f "$HOME/.bashrc" ] && grep -q "micromamba" "$HOME/.bashrc"; then
        source "$HOME/.bashrc"
    fi
    
    # 尝试找到 micromamba
    if ! command -v micromamba &> /dev/null; then
        # 尝试常见的 micromamba 安装路径
        if [ -f "$HOME/micromamba/etc/profile.d/conda.sh" ]; then
            source "$HOME/micromamba/etc/profile.d/conda.sh"
        elif [ -f "$HOME/.micromamba/etc/profile.d/conda.sh" ]; then
            source "$HOME/.micromamba/etc/profile.d/conda.sh"
        else
            echo "错误: 无法找到 micromamba，请确保已安装并配置" >&2
            exit 1
        fi
    fi
}

# 查找最新的 PID 文件
find_latest_pid_file() {
    ls -t "${LOG_DIR}"/*.pid 2>/dev/null | head -1
}

# ============================================================================
# 运行脚本功能
# ============================================================================

run_script() {
    # 检查 Python 脚本是否存在
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "错误: Python 脚本不存在: $PYTHON_SCRIPT" >&2
        exit 1
    fi
    
    # 初始化 micromamba
    init_micromamba
    
    # 设置日志文件路径（使用时间戳避免覆盖）
    # 所有输出（标准输出和标准错误）都保存到同一个日志文件
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/Hbench_replay_${TIMESTAMP}.log"
    PID_FILE="${LOG_DIR}/Hbench_replay_${TIMESTAMP}.pid"
    
    # 激活 sam2act4 环境
    echo "激活 micromamba 环境: sam2act4"
    micromamba activate sam2act4
    
    # 验证环境是否激活成功
    if [ "$CONDA_DEFAULT_ENV" != "sam2act4" ]; then
        echo "警告: 环境可能未正确激活，当前环境: $CONDA_DEFAULT_ENV"
    fi
    
    # 打印运行信息
    echo "=========================================="
    echo "开始运行脚本"
    echo "时间: $(date)"
    echo "工作目录: $SCRIPT_DIR"
    echo "Python 脚本: $PYTHON_SCRIPT"
    echo "日志文件: $LOG_FILE (包含所有输出，无缓冲实时写入)"
    echo "环境: $CONDA_DEFAULT_ENV"
    echo "输出模式: 无缓冲（实时写入）"
    echo "=========================================="
    
    # 使用 nohup 运行 Python 脚本，将所有输出（标准输出和标准错误）保存到同一个日志文件
    # nohup 确保即使 SSH 连接断开，进程仍能继续运行
    # 2>&1 将标准错误重定向到标准输出，这样所有内容都保存到同一个文件
    # -u 参数启用无缓冲输出，确保实时写入日志文件
    # PYTHONUNBUFFERED=1 环境变量也确保无缓冲输出
    PYTHONUNBUFFERED=1 nohup python -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    
    # 获取进程 ID
    PID=$!
    
    # 保存 PID 到文件，方便后续管理
    echo $PID > "$PID_FILE"
    
    echo ""
    echo "脚本已在后台启动"
    echo "进程 ID (PID): $PID"
    echo "PID 文件: $PID_FILE"
    echo "日志文件: $LOG_FILE (包含所有输出，无缓冲实时写入)"
    echo ""
    echo "管理命令:"
    echo "  查看状态: $0 status"
    echo "  查看日志: $0 logs (实时查看所有输出)"
    echo "  停止进程: $0 stop"
    echo ""
    echo "提示: 所有输出（包括标准输出和错误）都保存在日志文件中"
    echo "      输出为无缓冲模式，可以实时查看最新内容"
}

# ============================================================================
# 管理功能
# ============================================================================

# 显示状态
show_status() {
    PID_FILE=$(find_latest_pid_file)
    if [ -z "$PID_FILE" ]; then
        echo "未找到运行中的进程"
        return
    fi
    
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "PID 文件为空"
        return
    fi
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "进程正在运行"
        echo "PID: $PID"
        echo "PID 文件: $PID_FILE"
        echo ""
        
        # 显示进程信息
        echo "进程详细信息:"
        ps -p "$PID" -o pid,ppid,cmd,%mem,%cpu,etime,stat
        echo ""
        
        # 显示日志文件
        TIMESTAMP=$(basename "$PID_FILE" .pid | sed 's/Hbench_replay_//')
        LOG_FILE="${LOG_DIR}/Hbench_replay_${TIMESTAMP}.log"
        
        if [ -f "$LOG_FILE" ]; then
            echo "日志文件: $LOG_FILE (包含所有输出，无缓冲实时写入)"
            echo "文件大小: $(du -h "$LOG_FILE" | cut -f1)"
            echo "最后更新: $(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)"
            echo "最后几行输出:"
            tail -n 5 "$LOG_FILE" | sed 's/^/  /'
        fi
    else
        echo "进程未运行（PID: $PID）"
        echo "PID 文件: $PID_FILE"
    fi
}

# 查看日志
view_logs() {
    PID_FILE=$(find_latest_pid_file)
    if [ -z "$PID_FILE" ]; then
        echo "未找到运行中的进程"
        return
    fi
    
    TIMESTAMP=$(basename "$PID_FILE" .pid | sed 's/Hbench_replay_//')
    LOG_FILE="${LOG_DIR}/Hbench_replay_${TIMESTAMP}.log"
    
    if [ -f "$LOG_FILE" ]; then
        echo "查看日志: $LOG_FILE (实时无缓冲输出)"
        echo "按 Ctrl+C 退出"
        echo "----------------------------------------"
        # 使用 tail -f 实时跟踪文件，--retry 确保即使文件暂时不存在也会重试
        tail -f --retry "$LOG_FILE" 2>/dev/null || tail -f "$LOG_FILE"
    else
        echo "日志文件不存在: $LOG_FILE"
    fi
}

# 查看错误日志（从主日志文件中过滤错误）
view_errors() {
    PID_FILE=$(find_latest_pid_file)
    if [ -z "$PID_FILE" ]; then
        echo "未找到运行中的进程"
        return
    fi
    
    TIMESTAMP=$(basename "$PID_FILE" .pid | sed 's/Hbench_replay_//')
    LOG_FILE="${LOG_DIR}/Hbench_replay_${TIMESTAMP}.log"
    
    if [ -f "$LOG_FILE" ]; then
        echo "从日志文件中查看错误信息: $LOG_FILE (实时无缓冲)"
        echo "按 Ctrl+C 退出"
        echo "----------------------------------------"
        # 使用 grep 过滤包含错误关键词的行，--line-buffered 确保实时输出
        # 如果过滤后没有内容，则显示所有内容
        tail -f --retry "$LOG_FILE" 2>/dev/null | grep --line-buffered -i -E "(error|exception|traceback|failed|warning)" || tail -f "$LOG_FILE"
    else
        echo "日志文件不存在: $LOG_FILE"
    fi
}

# 停止进程
stop_process() {
    PID_FILE=$(find_latest_pid_file)
    if [ -z "$PID_FILE" ]; then
        echo "未找到运行中的进程"
        return
    fi
    
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "PID 文件为空"
        return
    fi
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "正在停止进程 $PID..."
        kill "$PID"
        
        # 等待进程结束
        sleep 2
        
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "进程仍在运行，强制终止..."
            kill -9 "$PID"
            sleep 1
        fi
        
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "错误: 无法停止进程"
        else
            echo "进程已停止"
        fi
    else
        echo "进程未运行"
    fi
}

# 列出所有日志文件
list_logs() {
    echo "日志文件列表:"
    echo "----------------------------------------"
    if [ -d "$LOG_DIR" ]; then
        echo "标准日志:"
        ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -20 || echo "  无日志文件"
        echo ""
        echo "PID 文件:"
        ls -lht "$LOG_DIR"/*.pid 2>/dev/null | head -10 || echo "  无 PID 文件"
    else
        echo "日志目录不存在: $LOG_DIR"
    fi
}

# 清理旧日志
clean_logs() {
    echo "清理7天前的日志文件..."
    if [ -d "$LOG_DIR" ]; then
        DELETED_LOGS=$(find "$LOG_DIR" -name "*.log" -mtime +7 -delete -print | wc -l)
        DELETED_PIDS=$(find "$LOG_DIR" -name "*.pid" -mtime +7 -delete -print | wc -l)
        echo "已删除 $DELETED_LOGS 个日志文件和 $DELETED_PIDS 个 PID 文件"
        echo "清理完成"
    else
        echo "日志目录不存在: $LOG_DIR"
    fi
}

# ============================================================================
# 主逻辑
# ============================================================================

case "${1:-run}" in
    run|"")
        # 默认行为：运行脚本
        run_script
        ;;
    status)
        show_status
        ;;
    logs)
        view_logs
        ;;
    errors)
        view_errors
        ;;
    stop)
        stop_process
        ;;
    list)
        list_logs
        ;;
    clean)
        clean_logs
        ;;
    help|-h|--help)
        show_usage
        ;;
    *)
        echo "错误: 未知命令 '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac
