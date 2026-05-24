#!/bin/bash

# 初始化micromamba（如果未在PATH中）
if ! command -v micromamba &> /dev/null; then
    export MAMBA_EXE='/home/hongzefu/micromamba/micromamba'
    export MAMBA_ROOT_PREFIX='/home/hongzefu/micromamba'
    eval "$($MAMBA_EXE shell hook --shell bash)"
fi

# 设置无缓冲输出
export PYTHONUNBUFFERED=1
export WANDB_MODE="online"

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行训练命令，使用micromamba环境，stdbuf确保无缓冲输出，并使用tee同时显示和保存日志
# 使用nohup防止关闭终端窗口时进程被终止
LOG_FILE="train_plus_$(date +%Y%m%d_%H%M%S).log"

# 获取micromamba的绝对路径
MAMBA_EXE="${MAMBA_EXE:-/home/hongzefu/micromamba/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/hongzefu/micromamba}"

# 使用nohup运行，在bash -c中初始化micromamba环境
nohup bash -c "
  # 初始化micromamba环境
  export MAMBA_EXE='$MAMBA_EXE'
  export MAMBA_ROOT_PREFIX='$MAMBA_ROOT_PREFIX'
  eval \"\$(\$MAMBA_EXE shell hook --shell bash)\"
  
  # 设置环境变量
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=\"online\"
  
  # 切换到脚本目录
  cd \"$(dirname "$0")\"
  
  # 运行训练命令
  micromamba run -n sam2act_3 bash -c \"
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=\\\"online\\\"
    stdbuf -oL -eL torchrun --nproc_per_node=\\\"2\\\" --nnodes=\\\"1\\\" train.py \\
      --exp_cfg_path configs/sam2act.yaml \\
      --mvt_cfg_path mvt/configs/sam2act.yaml \\
      --exp_cfg_opts \\\"tasks close_jar train_iter 16000 epochs 10 demo 100\\\" \\
      --mvt_cfg_opts \\\"depth 4\\\"
  \"
" > "$LOG_FILE" 2>&1 &

# 显示进程ID和日志文件位置
echo "训练进程已在后台启动，PID: $!"
echo "日志文件: $LOG_FILE"
echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
