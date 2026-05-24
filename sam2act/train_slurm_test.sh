#!/bin/bash
#SBATCH --job-name=sam2act-train
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=gl1527

source ~/.bashrc
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench/sam2act
micromamba activate /home/hongzefu/micromamba/envs/sam2act
unset LEROBOT_HOME
unset TRANSFORMERS_CACHE

srun --jobid "$SLURM_JOBID" bash -c 'PYTHONUNBUFFERED=1 WANDB_MODE=online \
  /home/hongzefu/micromamba/envs/sam2act/bin/torchrun \
  --nproc_per_node=1 --nnodes=1 train.py \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml \
  --exp_cfg_opts "tasks reopen_drawer train_iter 80000 epochs 10 demo 100" \
  --mvt_cfg_opts "depth 4"'
