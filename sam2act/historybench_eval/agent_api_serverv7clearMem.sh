#!/bin/bash

# agent_api_serverv7clearMem 实时查看日志脚本
# 功能：
# 1. 实时查看日志（无缓冲输出）
# 2. SSH关闭后仍能运行（使用screen）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/agent_api_serverv7clearMem.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/agent_api_serverv7clearMem.py"
SCREEN_NAME="agent_api_serverv7clearMem"
# 虚拟环境配置
VENV_PATH="/home/hongzefu/micromamba/envs/sam2act4"
PYTHON_BIN="${VENV_PATH}/bin/python"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start         启动服务并在screen中运行（SSH关闭后仍能运行）"
    echo "  watch         实时查看日志（无缓冲输出）"
    echo "  stop          停止服务"
    echo "  status        查看服务状态"
    echo "  restart       重启服务"
    echo "  tail          直接使用tail -f查看日志（简单模式）"
    echo ""
    echo "示例:"
    echo "  $0 start       # 启动服务"
    echo "  $0 watch       # 在另一个终端实时查看日志"
    echo "  $0 tail        # 简单模式查看日志"
}

# 检查screen是否安装
check_screen() {
    if ! command -v screen &> /dev/null; then
        echo -e "${YELLOW}警告: screen未安装，将使用nohup方式运行${NC}"
        echo "安装screen: sudo apt-get install screen 或 sudo yum install screen"
        return 1
    fi
    # 确保screen的autodetach功能开启（默认开启）
    return 0
}

# 启动服务
start_service() {
    # 检查服务是否已经在运行
    if screen -list | grep -q "${SCREEN_NAME}"; then
        echo -e "${YELLOW}agent_api_serverv7clearMem 服务已在运行中（screen名称: ${SCREEN_NAME}）${NC}"
        echo "使用 '$0 watch' 查看日志"
        echo "使用 '$0 stop' 停止服务"
        return 1
    fi
    
    # 检查Python脚本是否存在
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        echo -e "${RED}错误: 找不到Python脚本: ${PYTHON_SCRIPT}${NC}"
        return 1
    fi
    
    # 检查虚拟环境和Python解释器
    if [ ! -f "${PYTHON_BIN}" ]; then
        echo -e "${RED}错误: 找不到虚拟环境Python解释器: ${PYTHON_BIN}${NC}"
        echo "请检查虚拟环境路径: ${VENV_PATH}"
        return 1
    fi
    
    echo -e "${GREEN}正在启动 agent_api_serverv7clearMem 服务...${NC}"
    echo "虚拟环境: ${VENV_PATH}"
    echo "Python解释器: ${PYTHON_BIN}"
    echo "日志文件: ${LOG_FILE}"
    
    # 使用screen启动服务，确保无缓冲输出
    if check_screen; then
        # 创建启动脚本，确保环境变量和路径正确
        START_SCRIPT="${LOG_DIR}/start_${SCREEN_NAME}.sh"
        cat > "${START_SCRIPT}" << EOF
#!/bin/bash
# 自动生成的启动脚本 - 确保SSH断开后仍能运行
cd "${SCRIPT_DIR}"
export PATH="${VENV_PATH}/bin:\$PATH"
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH}"
# 确保使用虚拟环境的Python，完全脱离终端控制
# 使用exec确保进程替换，避免额外的shell进程
exec stdbuf -oL -eL "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "\$@" 2>&1 | tee -a "${LOG_FILE}"
EOF
        chmod +x "${START_SCRIPT}"
        
        # 使用screen启动，确保完全detached
        # -dmS: d=detached(后台运行), m=multiuser(多用户), S=session name
        # 直接使用启动脚本，不通过bash -c，避免额外的shell层
        screen -dmS "${SCREEN_NAME}" "${START_SCRIPT}" "$@"
        
        # 等待一下确保服务启动
        sleep 3
        
        if screen -list | grep -q "${SCREEN_NAME}"; then
            # 检查进程是否真的在运行
            sleep 1
            if pgrep -f "agent_api_serverv7clearMem.py" > /dev/null; then
                # 验证screen会话状态
                SCREEN_STATUS=$(screen -list | grep "${SCREEN_NAME}" | head -1)
                if echo "$SCREEN_STATUS" | grep -q "Detached"; then
                    echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已启动（在screen中运行，已detached）${NC}"
                else
                    echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已启动（在screen中运行）${NC}"
                fi
                echo "日志文件: ${LOG_FILE}"
                echo "Screen会话: ${SCREEN_NAME}"
                echo "会话状态: ${SCREEN_STATUS}"
                echo ""
                echo "常用命令:"
                echo "  $0 watch       # 实时查看日志"
                echo "  $0 stop        # 停止服务"
                echo "  screen -r ${SCREEN_NAME}  # 进入screen会话（查看后按Ctrl+A+D退出）"
                echo "  screen -list   # 查看所有screen会话"
                echo ""
                echo -e "${GREEN}✓ SSH断开后服务将继续运行（screen会话已detached）${NC}"
            else
                echo -e "${YELLOW}⚠ Screen会话已创建，但进程可能未启动，请检查日志${NC}"
                echo "日志文件: ${LOG_FILE}"
                echo "提示: 可以运行 '$0 watch' 查看实时日志，或 'screen -r ${SCREEN_NAME}' 进入会话查看"
            fi
        else
            echo -e "${RED}✗ agent_api_serverv7clearMem 服务启动失败${NC}"
            echo "请检查日志文件: ${LOG_FILE}"
            return 1
        fi
    else
        # 使用nohup作为备选方案，确保完全脱离终端
        echo -e "${YELLOW}使用nohup方式启动 agent_api_serverv7clearMem 服务...${NC}"
        cd "${SCRIPT_DIR}"
        # 使用setsid确保进程完全独立，不依赖于终端
        setsid nohup stdbuf -oL -eL "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "$@" > "${LOG_FILE}" 2>&1 &
        PID=$!
        echo $PID > "${LOG_DIR}/agent_api_serverv7clearMem.pid"
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已启动（PID: $PID）${NC}"
            echo "日志文件: ${LOG_FILE}"
            echo -e "${YELLOW}提示: SSH断开后服务将继续运行${NC}"
        else
            echo -e "${RED}✗ 服务启动失败，请检查日志${NC}"
            echo "日志文件: ${LOG_FILE}"
            rm -f "${LOG_DIR}/agent_api_serverv7clearMem.pid"
            return 1
        fi
    fi
}

# 实时查看日志（无缓冲）
watch_log() {
    if [ ! -f "${LOG_FILE}" ]; then
        echo -e "${YELLOW}日志文件不存在: ${LOG_FILE}${NC}"
        echo "agent_api_serverv7clearMem 服务可能还未启动，使用 '$0 start' 启动服务"
        return 1
    fi
    
    echo -e "${GREEN}实时查看 agent_api_serverv7clearMem 日志（无缓冲输出）...${NC}"
    echo "日志文件: ${LOG_FILE}"
    echo "按 Ctrl+C 退出"
    echo "----------------------------------------"
    
    # 使用tail -f实时查看，确保无缓冲
    tail -f -n +1 "${LOG_FILE}" 2>/dev/null || {
        # 如果tail失败，尝试使用less
        echo "使用less查看日志..."
        less +F "${LOG_FILE}"
    }
}

# 简单模式查看日志
tail_log() {
    if [ ! -f "${LOG_FILE}" ]; then
        echo -e "${YELLOW}日志文件不存在: ${LOG_FILE}${NC}"
        echo "agent_api_serverv7clearMem 服务可能还未启动，使用 '$0 start' 启动服务"
        return 1
    fi
    
    echo -e "${GREEN}实时查看 agent_api_serverv7clearMem 日志（简单模式）...${NC}"
    echo "日志文件: ${LOG_FILE}"
    echo "按 Ctrl+C 退出"
    echo "----------------------------------------"
    
    tail -f "${LOG_FILE}"
}

# 停止服务
stop_service() {
    # 尝试停止screen会话
    if screen -list | grep -q "${SCREEN_NAME}"; then
        echo -e "${YELLOW}正在停止 agent_api_serverv7clearMem screen会话...${NC}"
        screen -S "${SCREEN_NAME}" -X quit
        sleep 1
        if ! screen -list | grep -q "${SCREEN_NAME}"; then
            echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已停止${NC}"
        else
            echo -e "${RED}✗ 停止screen会话失败${NC}"
        fi
    fi
    
    # 尝试通过PID文件停止
    if [ -f "${LOG_DIR}/agent_api_serverv7clearMem.pid" ]; then
        PID=$(cat "${LOG_DIR}/agent_api_serverv7clearMem.pid")
        if ps -p "${PID}" > /dev/null 2>&1; then
            echo -e "${YELLOW}正在停止 agent_api_serverv7clearMem 进程 (PID: ${PID})...${NC}"
            kill "${PID}"
            sleep 1
            if ! ps -p "${PID}" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已停止${NC}"
                rm -f "${LOG_DIR}/agent_api_serverv7clearMem.pid"
            else
                echo -e "${RED}✗ 停止进程失败，尝试强制停止...${NC}"
                kill -9 "${PID}" 2>/dev/null
                rm -f "${LOG_DIR}/agent_api_serverv7clearMem.pid"
            fi
        else
            echo -e "${YELLOW}PID文件存在但进程不存在，清理PID文件${NC}"
            rm -f "${LOG_DIR}/agent_api_serverv7clearMem.pid"
        fi
    fi
    
    # 尝试通过进程名停止
    PIDS=$(pgrep -f "agent_api_serverv7clearMem.py")
    if [ -n "${PIDS}" ]; then
        echo -e "${YELLOW}发现运行中的 agent_api_serverv7clearMem 进程，正在停止...${NC}"
        echo "${PIDS}" | xargs kill
        sleep 1
        REMAINING=$(pgrep -f "agent_api_serverv7clearMem.py")
        if [ -z "${REMAINING}" ]; then
            echo -e "${GREEN}✓ 所有相关进程已停止${NC}"
        else
            echo -e "${YELLOW}仍有进程运行，尝试强制停止...${NC}"
            echo "${REMAINING}" | xargs kill -9
        fi
    fi
    
    if ! screen -list | grep -q "${SCREEN_NAME}" && [ -z "$(pgrep -f "agent_api_serverv7clearMem.py")" ]; then
        echo -e "${GREEN}✓ agent_api_serverv7clearMem 服务已完全停止${NC}"
    else
        echo -e "${YELLOW}⚠ 请手动检查服务状态${NC}"
    fi
}

# 查看服务状态
check_status() {
    echo "agent_api_serverv7clearMem 服务状态检查:"
    echo "----------------------------------------"
    
    # 检查screen会话
    if screen -list | grep -q "${SCREEN_NAME}"; then
        echo -e "${GREEN}✓ Screen会话运行中: ${SCREEN_NAME}${NC}"
        screen -list | grep "${SCREEN_NAME}"
    else
        echo -e "${YELLOW}✗ Screen会话未运行${NC}"
    fi
    
    echo ""
    
    # 检查进程
    PIDS=$(pgrep -f "agent_api_serverv7clearMem.py")
    if [ -n "${PIDS}" ]; then
        echo -e "${GREEN}✓ agent_api_serverv7clearMem Python进程运行中:${NC}"
        ps -p "${PIDS}" -o pid,cmd --no-headers
    else
        echo -e "${YELLOW}✗ agent_api_serverv7clearMem Python进程未运行${NC}"
    fi
    
    echo ""
    
    # 检查日志文件
    if [ -f "${LOG_FILE}" ]; then
        SIZE=$(du -h "${LOG_FILE}" | cut -f1)
        LINES=$(wc -l < "${LOG_FILE}")
        echo -e "${GREEN}✓ 日志文件存在:${NC}"
        echo "  路径: ${LOG_FILE}"
        echo "  大小: ${SIZE}"
        echo "  行数: ${LINES}"
        echo "  最后修改: $(stat -c %y "${LOG_FILE}" 2>/dev/null || stat -f %Sm "${LOG_FILE}" 2>/dev/null)"
    else
        echo -e "${YELLOW}✗ 日志文件不存在${NC}"
    fi
}

# 重启服务
restart_service() {
    echo -e "${YELLOW}正在重启 agent_api_serverv7clearMem 服务...${NC}"
    stop_service
    sleep 2
    start_service "$@"
}

# 主逻辑
case "${1}" in
    start)
        shift
        start_service "$@"
        ;;
    watch)
        watch_log
        ;;
    tail)
        tail_log
        ;;
    stop)
        stop_service
        ;;
    status)
        check_status
        ;;
    restart)
        shift
        restart_service "$@"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
