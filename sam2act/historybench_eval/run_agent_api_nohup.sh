#!/usr/bin/env bash
# =============================================================================
# 后台运行 agent_api_serverv7.5clearMem-parallel2gpu-seed.py
# - 使用 micromamba 环境 sam2act4
# - 无缓冲输出 (nobuffer)
# - SSH 断开后仍继续运行 (nohup)
#
# 用法:
#   ./run_agent_api_nohup.sh         # 启动
#   ./run_agent_api_nohup.sh log     # 实时查看日志输出
#   ./run_agent_api_nohup.sh stop    # 关闭所有相关进程
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.agent_api_state"
PY_SCRIPT="${SCRIPT_DIR}/agent_api_serverv7.5clearMem-parallel2gpu-seed.py"
PKILL_PATTERN="agent_api_serverv7.5clearMem-parallel2gpu-seed.py"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_BIN="/home/hongzefu/micromamba/envs/sam2act4/bin/python"

# ---------- 子命令: log ----------
if [[ "${1:-}" == "log" ]]; then
  if [[ -f "$STATE_FILE" ]]; then
    LOG_FILE=$(sed -n '2p' "$STATE_FILE")
    if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
      echo ">>> 实时输出: $LOG_FILE (Ctrl+C 退出)"
      exec tail -f "$LOG_FILE"
    fi
  fi
  # 无状态文件时，用 logs 下最新日志
  LATEST=$(ls -t "$LOG_DIR"/agent_api_*.log 2>/dev/null | head -1)
  if [[ -n "$LATEST" ]]; then
    echo ">>> 实时输出: $LATEST (Ctrl+C 退出)"
    exec tail -f "$LATEST"
  fi
  echo "未找到日志文件。请先运行 ./run_agent_api_nohup.sh 启动服务。" >&2
  exit 1
fi

# ---------- 子命令: stop ----------
if [[ "${1:-}" == "stop" ]]; then
  KILLED=0
  # 1) 若有状态文件，先对主进程发 SIGTERM
  if [[ -f "$STATE_FILE" ]]; then
    PID=$(sed -n '1p' "$STATE_FILE")
    if [[ -n "$PID" && "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
      echo "已对主进程发送结束信号 (PID: $PID)"
      KILLED=1
      sleep 2
    fi
    rm -f "$STATE_FILE"
  fi
  # 2) 按脚本名杀光所有相关进程（含子进程、多 GPU 等）
  while pkill -f "$PKILL_PATTERN" 2>/dev/null; do
    KILLED=1
    echo "已结束匹配 $PKILL_PATTERN 的进程"
    sleep 1
  done
  if [[ "$KILLED" -eq 0 ]]; then
    echo "未找到运行中的 agent_api 进程。"
  else
    echo "所有相关进程已结束。"
  fi
  exit 0
fi

# ---------- 默认: 启动 ----------
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/agent_api_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo " 启动 Agent API (nohup, unbuffered)"
echo " 环境: /home/hongzefu/micromamba/envs/sam2act4"
echo " 日志: $LOG_FILE"
echo "=============================================="
echo " 查看实时输出: ./run_agent_api_nohup.sh log"
echo " 关闭进程:     ./run_agent_api_nohup.sh stop"
echo "=============================================="

nohup stdbuf -oL -eL "$PYTHON_BIN" -u "$PY_SCRIPT" "$@" >> "$LOG_FILE" 2>&1 &
PID=$!
disown -h 2>/dev/null || true

# 保存 PID 与日志路径，供 log/stop 使用
printf "%s\n%s\n" "$PID" "$LOG_FILE" > "$STATE_FILE"

echo " 已启动 PID: $PID"
echo " 查看输出: ./run_agent_api_nohup.sh log"
echo " 关闭进程: ./run_agent_api_nohup.sh stop"
