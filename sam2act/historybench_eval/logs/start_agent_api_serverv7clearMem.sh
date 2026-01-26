#!/bin/bash
# 自动生成的启动脚本 - 确保SSH断开后仍能运行
cd "/home/hongzefu/sam2act_historybench/sam2act/historybench_eval"
export PATH="/home/hongzefu/micromamba/envs/sam2act4/bin:$PATH"
export PYTHONPATH="/home/hongzefu/sam2act_historybench/sam2act/historybench_eval:${PYTHONPATH}"
# 确保使用虚拟环境的Python，完全脱离终端控制
# 使用exec确保进程替换，避免额外的shell进程
exec stdbuf -oL -eL "/home/hongzefu/micromamba/envs/sam2act4/bin/python" "/home/hongzefu/sam2act_historybench/sam2act/historybench_eval/agent_api_serverv7clearMem.py" "$@" 2>&1 | tee -a "/home/hongzefu/sam2act_historybench/sam2act/historybench_eval/logs/agent_api_serverv7clearMem.log"
