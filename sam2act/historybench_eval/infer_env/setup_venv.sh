#!/usr/bin/env bash
# =============================================================================
# 重建 SAM2Act/SAM2Act+ **纯推理** uv 环境 (.venv) —— HistoryBench 评测用。
#
# 必须用 uv（不要用 pip/conda 临时装）：本脚本跑 `uv lock`(若无锁) + `uv sync`，
# 然后把推理专用的 venv 内挂件装进 .venv/site-packages：
#   * _sam2act_stubs.py  —— meta-path shim，把训练/仿真依赖
#     (rlbench/pyrep/pytorch3d/pyrender/trimesh/tensorflow/transformers) 伪装成
#     dummy、并屏蔽 torch.utils.tensorboard，同时注入推理真正要用的
#     Observation 与 VisionSensor.pointcloud_from_depth_and_camera_params。
#   * _sam2act_stubs.pth  —— 启动时 `import _sam2act_stubs`。
#   * _sam2act_paths.pth  —— 把 repo 根 + 内置 YARR / peract_colab / point-renderer
#     + sam2act 目录加进 sys.path（这样 sam2act / yarr / peract_colab / point_renderer
#     / mvt 都能 import；仓库源码保持原样，不打 import 守卫）。
#
# 用法： bash sam2act/historybench_eval/infer_env/setup_venv.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"

command -v uv >/dev/null 2>&1 || { echo "未找到 uv，请先安装 uv。"; exit 1; }

echo ">>> uv sync (依据 pyproject.toml / uv.lock 建 .venv)"
uv sync

PY="$REPO/.venv/bin/python"
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo ">>> site-packages: $SP"

cp "$HERE/_sam2act_stubs.py" "$SP/_sam2act_stubs.py"
printf 'import _sam2act_stubs\n' > "$SP/_sam2act_stubs.pth"
printf '%s\n%s\n%s\n%s\n%s\n' \
  "$REPO" \
  "$REPO/sam2act/libs/YARR" \
  "$REPO/sam2act/libs/peract_colab" \
  "$REPO/sam2act/libs/point-renderer" \
  "$REPO/sam2act" \
  > "$SP/_sam2act_paths.pth"

echo ">>> 校验 import 链"
"$PY" -c "from sam2act.eval import load_agent; print('OK: from sam2act.eval import load_agent')"
echo ">>> 推理环境就绪：$REPO/.venv"
