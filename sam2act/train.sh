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
micromamba run -n sam2act_3 bash -c "
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=\"online\"
  stdbuf -oL -eL torchrun --nproc_per_node=\"2\" --nnodes=\"1\" train.py \
    --exp_cfg_path configs/sam2act.yaml \
    --mvt_cfg_path mvt/configs/sam2act.yaml \
    --exp_cfg_opts \"tasks close_jar train_iter 80000 epochs 10\" \
    --mvt_cfg_opts \"depth 4\"
" 2>&1 | tee -a train_plus_$(date +%Y%m%d_%H%M%S).log
