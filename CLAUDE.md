# CLAUDE.md — SAM2Act / SAM2Act+ 评测指南

> 本文件只讲**评测（HistoryBench 仿真）**：硬性约定、评测链路、推理环境怎么建、最新模型在哪、怎么跑。
> 训练、RLBench 原始评测等不在此文档范围内。

---

## 0. 硬性约定（务必遵守）

- **必须用中文和用户沟通。** 所有回复、解释、总结一律用中文。
- **推理环境必须用 `uv` 管理**（`uv lock` + `uv sync`，见第 3 节），不要用 `pip install` / `conda` 临时往环境里塞包。依赖锁在 `pyproject.toml` + `uv.lock`，要改依赖就改 `pyproject.toml` 后重新 `uv lock`。
- 仿真/客户端那一侧用现成的 micromamba 环境 `maniskillenv1028`（这一侧不归 uv 管，别动）。

---

## 1. 评测链路（三方拓扑，现已全在 sled-vail 本地）

一次 HistoryBench 评测由**三部分**组成：

| 角色 | 位置 | 职责 |
| --- | --- | --- |
| **模型 / 权重 + 推理服务** | 本仓库 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench`，用 repo 的 **uv `.venv`** 跑 | 加载 checkpoint，暴露 `POST /reset_memory` + `POST /act`（Flask，默认 :8001） |
| **仿真 + 评测客户端** | `/data/hongzefu/robomme-sam2act`，用 micromamba `maniskillenv1028` 跑 | ManiSkill 3 + 16 个 HistoryBench 任务；评测脚本是 **HTTP 客户端** |

数据流：客户端读数据集 → 在 ManiSkill 里复现 episode → 每步观测 POST 给 `/act` → 拿动作执行 → 统计成功率。

> ⚠ 历史背景：CLAUDE.md 旧版说推理服务在远程 `141.212.48.176`、环境 `sam2act4`。那台主机/那个环境**已不存在**；推理服务端代码（`sam2act/historybench_eval/agent_api_serverv7.*.py`）其实**已在仓库里**。现在服务端和客户端**都在本机 sled-vail（141.212.115.116，2× RTX 6000 Ada）跑**。

---

## 2. 最新模型 checkpoint 在哪

均在本仓库 `sam2act/runs/` 下。

### SAM2Act+（带记忆，`use_memory: True`，`model_plus_*.pth`，约 610MB）
| 路径 | 说明 |
| --- | --- |
| `sam2act/runs/sam2act_plus_all_v4/model_plus_last.pth` | **推荐**：10 epoch 完整 run |
| `sam2act/runs/sam2act_plus_all_v5/model_plus_last.pth` | 时间最新，但只训到 epoch 3，**未训完** |
| `sam2act/runs/sam2act_plus_binfill_v3/model_plus_last.pth` | BinFill 单任务专项 |

### SAM2Act（基础，`use_memory: False`，`model_*.pth`，约 1.2GB）
| 路径 | 说明 |
| --- | --- |
| `sam2act/runs/sam2act_all_v1/model_last.pth` | 最新基础模型 |

> `load_agent` 用 `model_path` 是否含 `_plus_` 自动选配置：含 `_plus_` → 用同目录 `exp_cfg_plus.yaml` / `mvt_cfg_plus.yaml`；否则用 `exp_cfg.yaml` / `mvt_cfg.yaml`。
> SAM2 底座权重：`sam2act/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt`。

---

## 3. 建推理环境（uv，仅推理）

环境定义在 `pyproject.toml` + `uv.lock`：**纯推理依赖**，torch 2.5.1+cu121 / py3.10，**不含** tensorflow/rlbench/pyrep/pytorch3d 等训练-仿真栈。

仓库 `sam2act` 源码**保持原样**（不打 import 守卫）。推理适配只放在 venv 内的挂件文件里：
- `.venv/.../site-packages/_sam2act_stubs.py`：meta-path shim，把训练/仿真依赖伪装成 dummy、屏蔽 `torch.utils.tensorboard`，并注入推理**真正要用**的 `Observation` + `VisionSensor.pointcloud_from_depth_and_camera_params`。
- `_sam2act_stubs.pth` / `_sam2act_paths.pth`：启动时加载 shim，并把 repo 根、内置 `YARR`/`peract_colab`/`point-renderer`、`sam2act` 目录加进 `sys.path`。

这些挂件 `uv sync` **不会自动生成**（`.venv` 被 gitignore）。一条命令搞定（含 `uv sync` + 装挂件 + 校验）：

```bash
bash sam2act/historybench_eval/infer_env/setup_venv.sh
```

跑完应打印 `OK: from sam2act.eval import load_agent`。（挂件源码与该脚本是 git-tracked 的，所以可复现。）

---

## 4. 怎么跑评测

### Step 1 — 起推理服务（本机，repo 目录，用 `.venv`）

```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  sam2act/historybench_eval/agent_api_serverv7.5clearMem-parallel2gpu-seed.py \
  --num_servers 1 --device 0 --port 8001 \
  --model_folder "$PWD/sam2act/runs/sam2act_plus_all_v4" \
  --model_name model_plus_last.pth
# 评基础 SAM2Act → --model_folder .../sam2act_all_v1 --model_name model_last.pth
```

- ⚠ **GPU 写法必须如此**：`CUDA_VISIBLE_DEVICES=<空闲GPU>` + `--device 0`。CLIP 默认落在 `cuda:0`，若用 `--device 1` 之类会在 `/act` 报 `cuda:0 vs cuda:1` 设备冲突。先用 `nvidia-smi` 挑一张空闲卡。
- 就绪判据：`curl -s http://127.0.0.1:8001/health` 返回 `{"message":"Agent is ready",...}`（模型加载约 15s）。
- 后台跑：`setsid nohup CUDA_VISIBLE_DEVICES=1 .venv/bin/python <上面那串> > /tmp/sam2act_server.log 2>&1 < /dev/null &`，停止用 `kill <PID>`。
  - 注意别用 `pkill -f agent_api_serverv7.5...`：模式会匹配到你自己的命令行，把当前 shell 也杀了。按 PID 杀。

### Step 2 — 跑评测客户端（sim 侧，`maniskillenv1028`）

```bash
cd /data/hongzefu/robomme-sam2act
/home/hongzefu/micromamba/envs/maniskillenv1028/bin/python scripts/eval_binfill_test_client.py \
  --api_url http://127.0.0.1:8001 \
  --env_id BinFill --split test \
  --env_metadata_root /data/hongzefu/robomme-sam2act/env_metadata \
  --max_episodes 10 --max_steps 40
```

- 数据集：`/data/hongzefu/robomme-sam2act/env_metadata/{test,train,val}/record_dataset_{env_id}_metadata.json`（每条含 `seed`、`difficulty`）。
- `--max_episodes 0` = 全部；结果写到 `scripts/results/<env>_<split>_<时间戳>.json`（逐条增量写）。
- ⚠ **128×128 必须项**：HistoryBench 相机默认渲染 256×256，但 SAM2Act 要 `IMAGE_SIZE=128`。客户端已在 `gym.make` 传 `sensor_configs=dict(width=128,height=128)`；少了它 `/act` 会报点云/mask 形状不匹配（131072 vs 32768）。新写客户端务必保留这一项。

### Step 3 — 评测语义
1. 从 metadata 读每个 episode。
2. `env.reset()` 内部自动跑完**演示段**，返回稠密轨迹；客户端 `POST /reset_memory` 清记忆，再把采样的演示帧喂给 `/act` 让模型积累记忆。
3. 评测段：逐步 `raw obs → /act → ee 位姿动作 → 运动规划器执行`，直到 success / fail / max_steps。
4. 统计每任务成功率（总体 + 按难度）。

> SAM2Act 与 SAM2Act+ 共用同一套客户端，区别只在服务端加载了哪个 checkpoint。同时评两个 → 起两个服务（不同端口/不同空闲 GPU）。

### 已验证基线
`sam2act_plus_all_v4/model_plus_last.pth` 在 **BinFill test 10 episodes = 4/10（easy 3/6, medium 1/2, hard 0/2）**。

### 16 个 HistoryBench 任务
`PickXtimes, StopCube, SwingXtimes, BinFill, VideoUnmaskSwap, VideoUnmask, ButtonUnmaskSwap, ButtonUnmask, VideoRepick, VideoPlaceButton, VideoPlaceOrder, PickHighlight, InsertPeg, MoveCube, PatternLock, RouteStick`
