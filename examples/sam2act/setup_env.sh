#!/usr/bin/env bash
# =============================================================================
# 重建 SAM2Act robomme **评测客户端** 环境（micromamba `sam2act-robomme-eval`）。
#
# 与服务端 sam2act/historybench_eval/infer_env/setup_venv.sh 对称：
# 两套环境现在都在本仓库内管理 —— uv .venv（推理服务端） + 本 micromamba env（仿真+评测客户端）。
#
# 装：torch 2.9.1 + ManiSkill(RoboMME fork) + websockets/msgpack，并以 editable 方式
#     安装本仓库内的 robomme benchmark（third_party 的 submodule）与 openpi-client（packages）。
# SAM2Act 用 Null 子目标，省去 robomme readme 里的 VLM 子目标依赖
# （ms_swift/deepspeed/google-*/flash-attn）。
#
# 用法： bash examples/sam2act/setup_env.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

ENV=sam2act-robomme-eval
MM=/home/hongzefu/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/hongzefu/micromamba

# robomme benchmark 是 submodule —— 确保已初始化（纯远端，从 canonical 拉取）。
if [ ! -e "$REPO/third_party/robomme_benchmark/pyproject.toml" ]; then
  echo ">>> 初始化 submodule third_party/robomme_benchmark"
  git -C "$REPO" submodule update --init third_party/robomme_benchmark
fi

echo ">>> 创建 micromamba env: $ENV (python 3.11)"
"$MM" create -n "$ENV" python=3.11 -y

echo ">>> 装 torch 2.9.1 + ManiSkill(RoboMME fork) + ws 依赖"
"$MM" run -n "$ENV" pip install torch==2.9.1 torchvision==0.24.1 moviepy==2.2.1 ninja==1.13.0 setuptools==80.9.0 \
  websockets msgpack "git+https://github.com/YinpeiDai/ManiSkill.git@dev"

echo ">>> editable 安装本仓库内 robomme benchmark + openpi-client"
"$MM" run -n "$ENV" pip install -e "$REPO/third_party/robomme_benchmark"
"$MM" run -n "$ENV" pip install -e "$REPO/packages/openpi-client"

echo ">>> 校验 import 指向本仓库"
"$MM" run -n "$ENV" python -c "import robomme, openpi_client, os; print('robomme ->', os.path.dirname(robomme.__file__)); print('openpi  ->', os.path.dirname(openpi_client.__file__))"
echo ">>> 客户端环境就绪：micromamba env '$ENV'（robomme/openpi 来自 $REPO）"
