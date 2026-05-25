# CLAUDE.md — SAM2Act / SAM2Act+ 评测指南

> 本文件只讲**评测（HistoryBench / RoboMME 仿真）**：硬性约定、评测链路、两个环境怎么建、最新模型在哪、怎么跑。
> 训练、RLBench 原始评测等不在此文档范围内。

---

## 0. 硬性约定（务必遵守）

- **必须用中文和用户沟通。** 所有回复、解释、总结一律用中文。
- **服务端（推理）环境必须用 `uv` 管理**（`uv lock` + `uv sync`，见第 3 节），不要 `pip install` 临时塞包。依赖锁在 `pyproject.toml` + `uv.lock`。
- **客户端（仿真+评测）用全新独立 micromamba 环境 `sam2act-robomme-eval`**，按 robomme_policy_learning 的 `examples/robomme/readme.md` 装（见第 3 节）。**不再用旧的 `maniskillenv1028`**。
- **SAM2Act 是离散 waypoint（关键帧）策略：必须用 robomme 的 `action_space="waypoint"`，由 benchmark 内置 planner 执行，绝不自己写运动规划器。**

---

## 1. 评测链路（两环境 + WebSocket，对齐 robomme_policy_learning）

参考 `https://github.com/RoboMME/robomme_policy_learning`（尤其 `examples/robomme/`）。一次评测两部分，通过 **WebSocket + msgpack** 通信：

| 角色 | 位置 / 环境 | 职责 |
| --- | --- | --- |
| **WS 推理服务端**（SAM2Act 权重） | 本仓库 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench`，repo **uv `.venv`**（torch 2.5.1） | `serve_policy.py` 加载 checkpoint，WS 暴露 `reset` / `add_buffer` / `infer`（默认 :8001，被占就换端口如 8011） |
| **评测客户端**（sim + 编排） | `/data/hongzefu/robomme_policy_learning-vqa-test/examples/sam2act/`，micromamba **`sam2act-robomme-eval`**（torch 2.9.1 + ManiSkill + robomme + openpi-client） | `robomme.BenchmarkEnvBuilder(action_space="waypoint")` 跑 16 任务；`env_runner` 适配器是 WS 客户端 |

数据流：`env.reset()` 跑完**演示段** → 采样 demo 帧 `add_buffer` 喂服务端积累记忆 → 评测段每步 `env.unwrapped.get_obs()` → `infer` 拿 **9 维动作** `[x,y,z, qw,qx,qy,qz, grip, coll]` → 客户端转 **7 维 waypoint** `[x,y,z, rpy, grip(-1/+1)]` → `env.step()`（**内置 screw→RRT\* planner 执行**）→ 读 `info["status"]` 判成败。

> **协议**（`serving/websocket_policy_server.py` 的 `_handler` 按标志位分发）：
> - `client.reset()` → `{"reset":True}` → 清 `mvt1/mvt2` 记忆库；
> - `client.add_buffer({"add_buffer":True,"frames":[...]})` → 逐帧 `agent.act` 仅积累记忆；
> - `client.infer({obs_obj, curr_idx, lang_goal, episode_length})` → `{"actions":[9维]}`（一次一个关键帧）。
> 两端用同一份 `msgpack_numpy.py`（服务端 vendored 在 `serving/`，客户端来自 openpi-client）。

> ⚠ 历史：旧链路是 Flask `POST /act` + 客户端**自写 planner**（`/data/hongzefu/robomme-sam2act/scripts/eval_binfill_test_client.py` + `planner_fail_safe.py`）。已被本 WS + waypoint 链路取代；旧的 `agent_api_serverv7.*.py` 仍在仓库但不再是主路径。

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

> `load_agent` 用 `model_path` 是否含 `_plus_` 自动选配置：含 `_plus_` → 同目录 `exp_cfg_plus.yaml` / `mvt_cfg_plus.yaml`；否则 `exp_cfg.yaml` / `mvt_cfg.yaml`。
> SAM2 底座权重：`sam2act/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt`。

---

## 3. 建两个环境

### 3a. 服务端环境（uv，仅推理）——本仓库 `.venv`
`pyproject.toml` + `uv.lock`：纯推理依赖，torch 2.5.1+cu121 / py3.10，**已含 `websockets` + `msgpack`**（WS 服务用）。源码保持原样，推理适配在 venv 挂件：
- `_sam2act_stubs.py`：把训练/仿真依赖伪装成 dummy、屏蔽 tensorboard，注入 `Observation` + `VisionSensor.pointcloud_from_depth_and_camera_params`。
- `_sam2act_stubs.pth` / `_sam2act_paths.pth`：加载 shim，并把 repo 根、内置 `YARR`/`peract_colab`/`point-renderer`、`sam2act` 加进 `sys.path`。

挂件 `uv sync` 不自动生成，一条命令（含 `uv sync` + 装挂件 + 校验）：
```bash
bash sam2act/historybench_eval/infer_env/setup_venv.sh   # 打印 OK: from sam2act.eval import load_agent
# 若改过依赖（如加 ws/msgpack）：uv lock && uv sync
```

### 3b. 客户端环境（micromamba `sam2act-robomme-eval`）——按 robomme_policy_learning readme
```bash
ENV=sam2act-robomme-eval
REPO=/data/hongzefu/robomme_policy_learning-vqa-test
micromamba create -n $ENV python=3.11 -y
micromamba run -n $ENV pip install torch==2.9.1 torchvision==0.24.1 moviepy==2.2.1 ninja==1.13.0 setuptools==80.9.0 \
  websockets msgpack "git+https://github.com/YinpeiDai/ManiSkill.git@dev"
micromamba run -n $ENV pip install -e $REPO/third_party/robomme_benchmark
micromamba run -n $ENV pip install -e $REPO/packages/openpi-client
```
- SAM2Act 用 Null 子目标，**精简掉**了 readme 里 VLM 子目标依赖（ms_swift/deepspeed/google-*/flash-attn）。
- robomme 包即 `$REPO/third_party/robomme_benchmark`（submodule 已初始化），test split 元数据在包内 `env_metadata/test/`。
- micromamba 二进制：`/home/hongzefu/.local/bin/micromamba`（`MAMBA_ROOT_PREFIX=/home/hongzefu/micromamba`）。

---

## 4. 怎么跑评测

一键编排（tmux 双窗口，镜像 robomme `scripts/eval.sh`）：
```bash
bash /data/hongzefu/robomme_policy_learning-vqa-test/examples/sam2act/run_eval.sh
# 内含参数：MODEL=sam2act_plus_all_v4/model_plus_last.pth，GPU_server/GPU_client，ONLY_TASKS=BinFill，MAX_EPISODES=5
```

或手动两步：

### Step 1 — 起 WS 服务端（repo `.venv`）
```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  sam2act/historybench_eval/serve_policy.py \
  --model_folder "$PWD/sam2act/runs/sam2act_plus_all_v4" \
  --model_name model_plus_last.pth --device 0 --port 8011 --seed 0
```
- ⚠ **GPU 写法**：`CUDA_VISIBLE_DEVICES=<空闲卡>` + `--device 0`（CLIP 默认 cuda:0；用 `--device 1` 会设备冲突）。先 `nvidia-smi` 挑空闲卡；显存吃紧加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 就绪判据：日志出现 `server listening on 0.0.0.0:8011`（模型加载约 20s）。
- 后台：`setsid ... > /tmp/sam2act_ws_server.log 2>&1 < /dev/null &`，停止按 PID `kill`（**别 pkill -f serve_policy**，会误杀自己的 shell）。

### Step 2 — 跑评测客户端（`sam2act-robomme-eval`）
```bash
cd /data/hongzefu/robomme_policy_learning-vqa-test
CUDA_VISIBLE_DEVICES=1 /home/hongzefu/.local/bin/micromamba run -n sam2act-robomme-eval \
  python examples/sam2act/eval.py --host 127.0.0.1 --port 8011 \
  --only_tasks BinFill --max_episodes 5 --max_steps 40 --history_frames 16
```
- 结果写 `examples/sam2act/runs/eval_<时间戳>/log.json`（+ `progress.json` 增量）。`--only_tasks` 留空 = 全 16 任务；`--max_episodes 0` = 该任务全部。
- ⚠ **128×128**：robomme env 默认渲 256×256，但 SAM2Act 要 128。客户端 `utils.build_obs_obj` 已 **resize RGB/depth 到 128 并同步缩放相机内参**（否则点云错位）。`BenchmarkEnvBuilder` 不暴露 sensor_configs，故在客户端 resize，不改第三方包。

### Step 3 — 评测语义（客户端 `examples/sam2act/{eval,env_runner,utils}.py`）
1. `runner.make_env(ep)` 用 `BenchmarkEnvBuilder(dataset="test", action_space="waypoint", include_maniskill_obs=True)`。
2. `client.reset()` 清记忆；`env.reset()` 自动跑演示段返回 `maniskill_obs` 稠密帧；采样后 `client.add_buffer` 喂服务端积累记忆。
3. 评测段：`build_obs_obj(get_obs()) → client.infer → 9维动作 → env_runner 转 7维 waypoint → env.step()`（内置 planner），直到 `status∈{success,fail}` 或 `max_steps`(timeout)。
4. 统计每任务成功率（总体 + 按难度），写 `log.json`。
- **9→7 转换**经旧 `action_to_pose` 中转保序（保留已验证四元数约定）；grip `<=0.5→-1关`、else `+1开`。MultiStep 内部 rpy→quat 还原同一 Pose。

> SAM2Act 与 SAM2Act+ 共用同一客户端，区别只在服务端加载哪个 checkpoint。

### 已验证基线
- 旧 historybench + 自写 planner（已弃用）：`sam2act_plus_all_v4` 在 BinFill test 10ep = 4/10。
- 新 robomme + waypoint 链路（`sam2act_plus_all_v4/model_plus_last.pth`，2026-05-24 验证）：**BinFill test 5ep = 1/5（easy 1/3, medium 0/1, hard 0/1）**。三种判定（success/fail/timeout）均正常，每集执行 10–40 个 waypoint 经内置 planner。
  > ⚠ 换成第三方 robomme 包后 env 定义/seed/成功判定与旧 historybench 不同，**与旧 4/10 基线不可直接比较**。
  > ⚠ robomme 的 `env.reset()` 每个 list 只返回 1 帧（非 historybench 的稠密 demo 轨迹），故 demo 记忆只喂 1 帧种子；SAM2Act+ 记忆主要靠评测循环里每次 `infer→act` 逐步累积。若要给记忆型任务喂更多 demo 帧，需另查 robomme demo 录制粒度。

### 16 个 HistoryBench / RoboMME 任务
`PickXtimes, StopCube, SwingXtimes, BinFill, VideoUnmaskSwap, VideoUnmask, ButtonUnmaskSwap, ButtonUnmask, VideoRepick, VideoPlaceButton, VideoPlaceOrder, PickHighlight, InsertPeg, MoveCube, PatternLock, RouteStick`
