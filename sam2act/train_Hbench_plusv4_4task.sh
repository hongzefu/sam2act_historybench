#!/bin/bash
#SBATCH --job-name=sam2act-plus-binfill
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=gl1527

source ~/.bashrc
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench/sam2act

# 激活环境
micromamba activate /home/hongzefu/micromamba/envs/sam2act

# 默认模型路径
DEFAULT_MODEL_PATH="//nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench/sam2act/runs/sam2act_all_v1/model_last.pth"

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

export PYTHONUNBUFFERED=1
export WANDB_MODE="online"

# 准备实验目录
# 根据 exp_id 和 exp_name 确定目录 (sam2act_plus_binfill + v3)
EXP_DIR="runs/sam2act_plus_binfill_v3"

echo "准备实验目录: $EXP_DIR"
mkdir -p "$EXP_DIR"

# 复制模型文件到实验目录，命名为 model_last.pth
# 这样 train_plus.py 会自动加载它作为预训练权重，并重置 epoch 从 0 开始
echo "复制模型文件 $MODEL_PATH 到 $EXP_DIR/model_last.pth ..."
cp "$MODEL_PATH" "$EXP_DIR/model_last.pth"

# 运行训练命令
# 注意：已移除 resume 参数，并更新了 config 路径为 _plus.yaml
srun --jobid "$SLURM_JOBID" bash -c "PYTHONUNBUFFERED=1 WANDB_MODE=online \
  torchrun --nproc_per_node=4 --nnodes=1 train_plus.py \
    --exp_cfg_path configs/sam2act_plus.yaml \
    --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
    --mvt_cfg_opts 'use_memory True' \
    --exp_cfg_opts 'tasks VideoRepick,VideoPlaceButton,VideoPlaceOrder,PickHighlight train_iter 80000 epochs 10 bs 20 exp_id sam2act exp_name plus_4task'"