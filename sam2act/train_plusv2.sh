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

# 默认模型路径
DEFAULT_MODEL_PATH="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench/sam2act/runs/sam2act_binfill4/model_last.pth"

# 如果提供了参数，使用参数作为模型路径；否则使用默认路径
if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    MODEL_PATH="$DEFAULT_MODEL_PATH"
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 运行训练命令，使用micromamba环境，stdbuf确保无缓冲输出，并使用tee同时显示和保存日志
micromamba run -n sam2act4 bash -c "
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=\"online\"
  stdbuf -oL -eL torchrun --nproc_per_node=\"2\" --nnodes=\"1\" train_plus.py \
    --exp_cfg_path configs/sam2act.yaml \
    --mvt_cfg_path mvt/configs/sam2act.yaml \
    --mvt_cfg_opts \"use_memory True\" \
    --exp_cfg_opts \"tasks BinFill train_iter 800000 epochs 10 bs 10 resume $MODEL_PATH exp_id sam2act_plus_binfill exp_name v1\" \

" 2>&1 | tee -a train_plus_$(date +%Y%m%d_%H%M%S).log
