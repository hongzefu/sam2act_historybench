# CLAUDE.md — SAM2Act / SAM2Act+ 评测指南

> 本文件只讲**评测（HistoryBench 仿真）**：评测链路是什么、**现在还缺哪些文件**、最新模型在哪、补齐后怎么跑。
> 训练、RLBench 原始评测等不在此文档范围内。

---

## 1. 评测链路（三方拓扑）

一次 HistoryBench 评测由**三部分**组成，分别在不同位置：

| 角色 | 位置 | 职责 |
| --- | --- | --- |
| **模型 / 权重** | 本仓库 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench`（`sam2act/runs/`） | 存放 SAM2Act / SAM2Act+ 的 checkpoint 和模型代码 |
| **仿真 + 评测客户端** | `/data/hongzefu/robomme-sam2act` | ManiSkill 3.0 + 16 个 HistoryBench 任务；评测脚本是 **HTTP 客户端** |
| **推理服务（远程）** | `http://141.212.48.176:<port>` | 加载 checkpoint，暴露 `POST /reset_memory` 和 `POST /act` |

数据流：评测客户端读数据集 → 在 ManiSkill 里复现 episode → 把每步观测 POST 给推理服务的 `/act` → 拿到动作并执行 → 统计成功率。

---

## 2. ⚠ 评测现在还缺什么（跑不起来的根因）

> 以下是当前文件系统的真实状况。**三项都需补齐，评测才能跑通。**

### ✗ 缺口 1：推理服务端代码（最关键）
- **期望**：模型主机 `141.212.48.176` 上有一个 HTTP 服务，定义 `/reset_memory`（重置记忆）和 `/act`（收观测、返回动作）两个路由，并加载某个 checkpoint。
- **实际（已全盘扫描确认）**：对 `/data/hongzefu/` 与 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 下所有 `*.py` / `*.ipynb`（含 cursor 编辑历史）做穷尽扫描——共 **2845** 个含 Web 服务框架的文件，与「SAM2Act 推理 token（`load_agent` / `reset_memory_bank` / `SAM2Act_Agent` / `sam2act_agent`）＋ HTTP 路由 token」取交集，**命中 0 个**。所有出现 `/reset_memory`、`/act` 的文件都是**客户端**评测脚本（含 `/data/hongzefu/legacy/` 下的多份历史备份）。⇒ **服务端代码确实不在这两处磁盘上**，只存在于 `141.212.48.176` 本地，**未纳入版本控制**。
- **需要补**：一个把 agent 包成 HTTP 服务的脚本。仓库内现成的集成点是 `sam2act/real/sam2act_agent.py`：
  - `SAM2Act.__init__(model_path)` → `load_agent(model_path)`（`model_path` 含 `_plus_` 时自动加载 `*_plus.yaml` 配置）
  - `SAM2Act.get_action(step, observation)` → `agent.act(...)`
  - 用 Flask/FastAPI 暴露：`POST /reset_memory` 调 `agent._network.mvt*.reset_memory_bank()`，`POST /act` 调 `get_action(...)`。
  - **可直接套用的最小模板**：`/data/hongzefu/EvalAI-Starters-minigrid/mock_agent.py`（FastAPI + `@app.post("/act")` + `uvicorn.run(..., port=...)`，路由结构完全同构；把其中返回随机动作的部分换成 `SAM2Act.get_action(...)`，再补一个 `@app.post("/reset_memory")` 即可）。
  - **建议把这个服务端脚本提交进仓库**，否则评测无法复现。

### ✗ 缺口 2：评测用 conda/micromamba 环境失效
- **期望（脚本写死）**：`/data/hongzefu/maniskillenv1114`
- **实际**：该路径**不存在**。当前实际存在的 maniskill 环境只有 `/home/hongzefu/micromamba/envs/maniskillenv1028`（另有 `.../robomme`、`.../sam2act-1`）。
  - ⚠ 注意：`run_oraclev4.sh` 引用的 `/home/hongzefu/micromamba/envs/maniskillenv1228` **同样不存在**——又一处失效引用。
- **需要补**：把评测脚本里的 `/data/hongzefu/maniskillenv1114` 改成现存的 `maniskillenv1028`，或按其依赖重建 `maniskillenv1114`。涉及文件：`scripts/eval_sam2actV8.3multiprocess_manager.sh`、`public_scripts/run_evaluate_dataset_replay_parallel.sh` 中的 `CONDA_ENV` / `MICROMAMBA_ENV`。

### ✗ 缺口 3：评测数据集缺失 / 改名
- **期望（脚本写死）**：`/data/hongzefu/historybench-v5.7.6-sam2act7-full-dataset2-annotate/dataset_json`，每个任务一个 `record_dataset_{env_id}_metadata.json`
- **实际**：该目录**不存在**；`/data/hongzefu` 下只有 `dataset-distribution-0329`、`dataset_replay-0306`、`dataset-test-tokendrop` 等其它数据集。
- **需要补**：补回该标注数据集，或把 `eval_sam2actV8.3multiprocess.py` / `eval_sam2actV9-sample.py` 里的 `dataset_root` 指向现存且含 `record_dataset_*_metadata.json` 的数据集。

### ⚠ 顺带的小问题
- `historybench/HistoryBench_env/RouteStick copy.py` 与 `RouteStick.py` **都 `@register_env("RouteStick")`**，是重复注册的冗余拷贝，建议删掉 `RouteStick copy.py` 以免 import 时冲突。

---

## 3. 最新模型 checkpoint 在哪

均在本仓库 `sam2act/runs/` 下。

### SAM2Act+（带记忆，`use_memory: True`，`model_plus_*.pth`，约 610MB）
| 路径 | 时间 | 说明 |
| --- | --- | --- |
| `sam2act/runs/sam2act_plus_all_v4/model_plus_last.pth` | 2026-01-28 | **推荐**：10 epoch 完整 run（`model_plus_0..9`） |
| `sam2act/runs/sam2act_plus_all_v5/model_plus_last.pth` | 2026-01-29 | 时间最新，但只训到 epoch 3（`model_plus_0..3`），**未训完** |
| `sam2act/runs/sam2act_plus_binfill_v3/model_plus_last.pth` | 2026-01-23 | BinFill 单任务专项 |

### SAM2Act（基础，`use_memory: False`，`model_*.pth`，约 1.2GB）
| 路径 | 时间 | 说明 |
| --- | --- | --- |
| `sam2act/runs/sam2act_all_v1/model_last.pth` | 2026-01-24 | 最新基础模型（plus 各目录里的 `model_last.pth` 是它的副本，作训练初始化） |

### SAM2 底座权重
- `sam2act/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt`

> 注：`sam2act/runs/legacy-directRun/` 下是更早的实验 run，已被上面的版本取代。

---

## 4. 评测怎么跑（补齐第 2 节缺口后）

### Step 1 — 在模型主机 `141.212.48.176` 起推理服务
> ⚠ 该服务端脚本目前**不在仓库里**（见缺口 1）。补齐后大致形如：
```bash
# 伪命令：用所选 checkpoint 起服务，暴露 /reset_memory + /act
# 评 SAM2Act+ → 指向 model_plus_*.pth（路径含 _plus_，会自动用 plus 配置）
# 评 SAM2Act  → 指向 model_*.pth
python <推理服务脚本> \
    --model-path .../sam2act/runs/sam2act_plus_all_v4/model_plus_last.pth \
    --port 8001
```
集成逻辑参考 `sam2act/real/sam2act_agent.py`。

### Step 2 — 在 sim 主机激活环境后跑评测客户端
工作目录 `/data/hongzefu/robomme-sam2act`，先激活可用环境（见缺口 2）。

**并行批量评测（推荐，nohup 后台 + 日志监控）：**
```bash
# 管理器支持 start | monitor | stop | status | restart，日志在 scripts/logs/
bash scripts/eval_sam2actV8.3multiprocess_manager.sh start
# 等价于内部运行（默认值）：
#   python eval_sam2actV8.3multiprocess.py \
#     --base_url http://141.212.48.176 --start_port 8001 --num_ports 8 \
#     --max_steps 40 --max_episodes_per_env 10
```

**单进程 / 抽样评测：**
```bash
python scripts/eval_sam2actV9-sample.py --api_url http://141.212.48.176:8001 --max_steps 40
```

### Step 3 — 评测语义
1. 从 `dataset_root/record_dataset_{env_id}_metadata.json` 读取每个 episode（含 `seed`、难度）。
2. 先回放**历史 / 演示段**（视频演示），让模型积累记忆。
3. 再用 `/act` 逐步查询策略执行**评测段**，直到 episode 结束。
4. 统计每任务成功率。

> **SAM2Act 与 SAM2Act+ 用的是同一套客户端脚本**，区别只在「推理服务加载了哪个 checkpoint / 监听哪个端口」。要同时评两个模型 → 起两个服务（不同端口）或换 checkpoint 重启服务。

### 16 个 HistoryBench 任务
`PickXtimes, StopCube, SwingXtimes, BinFill, VideoUnmaskSwap, VideoUnmask, ButtonUnmaskSwap, ButtonUnmask, VideoRepick, VideoPlaceButton, VideoPlaceOrder, PickHighlight, InsertPeg, MoveCube, PatternLock, RouteStick`
