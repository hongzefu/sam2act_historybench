# `_add_keypoints_to_replay_temporal` 函数逻辑详解

## 概述

`_add_keypoints_to_replay_temporal` 是时序 Replay Buffer 数据生成过程中的核心函数。它的主要功能是：**将从某个起始帧（sample_frame）开始的一系列关键点转换作为样本序列加入 Replay Buffer**。

## 函数签名

```python
def _add_keypoints_to_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
)
```

### 关键参数说明

- `sample_frame`: 起始帧索引（演示中的某一帧）
- `inital_obs`: 起始帧的观测数据
- `episode_keypoints`: 演示中所有关键点的索引列表（已排序）
- `next_keypoint_idx`: 在 `episode_keypoints` 中，从哪个关键点开始处理（即第一个大于等于 `sample_frame` 的关键点索引）
- `demo`: 完整的演示数据（包含所有帧的观测）

## 核心逻辑流程

### 1. 初始化状态

```python
prev_action = None
obs = inital_obs  # 当前观测 = 起始帧观测
initial_frame = sample_frame  # 记录起始帧，用于元数据
```

### 2. 遍历关键点序列

```python
for k in range(next_keypoint_idx, len(episode_keypoints)):
```

从 `next_keypoint_idx` 开始，遍历后续的所有关键点。对于每个关键点 `k`：

#### 2.1 获取目标观测

```python
keypoint = episode_keypoints[k]  # 关键点在演示中的索引
obs_tp1 = demo[keypoint]         # 目标观测（关键点帧的观测）
obs_tm1 = demo[max(0, keypoint - 1)]  # 前一帧（用于获取 ignore_collisions）
```

#### 2.2 计算动作

```python
(
    trans_indicies,        # 平移动作的离散索引（体素坐标）
    rot_grip_indicies,     # 旋转+夹爪动作的离散索引
    ignore_collisions,     # 是否忽略碰撞
    action,                # 连续动作向量 [x, y, z, qx, qy, qz, qw, grip]
    attention_coordinates, # 注意力坐标（多尺度）
) = _get_action(
    obs_tp1,               # 目标观测
    obs_tm1,               # 前一帧观测
    rlbench_scene_bounds,  # 场景边界
    voxel_sizes,           # 体素大小（多尺度）
    rotation_resolution,   # 旋转分辨率
    crop_augmentation,     # 是否启用裁剪增强
)
```

**动作含义**：
- 动作表示从当前观测 `obs` 到目标观测 `obs_tp1` 所需的机器人操作
- `trans_indicies`: 目标位置的体素索引（离散空间坐标）
- `rot_grip_indicies`: 目标姿态的离散欧拉角索引 + 夹爪开闭状态

#### 2.3 计算奖励和终止标志

```python
terminal = k == len(episode_keypoints) - 1  # 最后一个关键点才是 terminal
reward = float(terminal) * 1.0 if terminal else 0  # 仅在最后一个关键点给奖励
```

**设计理念**：
- 只有到达最后一个关键点（即任务完成）时才给予奖励 1.0
- 其他关键点奖励为 0（稀疏奖励）
- `terminal=True` 表示这是一个 episode 的结束

#### 2.4 提取当前观测特征

```python
obs_dict = extract_obs(
    obs,                        # 当前观测（起始帧或上一个关键点的观测）
    CAMERAS,                    # 使用的相机列表
    t=k - next_keypoint_idx,    # 相对时间步（从起始帧算起）
    prev_action=prev_action,    # 上一个动作
    episode_length=25,          # 最大 episode 长度（用于归一化时间）
)
```

**提取的内容**：
- 多视角 RGB 图像（`front_rgb`, `left_shoulder_rgb`, ...）
- 多视角深度图（`front_depth`, ...）
- 多视角点云（`front_point_cloud`, ...）
- 低维状态向量（`low_dim_state`）：包含夹爪状态、关节位置、归一化时间等
- 相机内外参矩阵

#### 2.5 编码语言目标

```python
tokens = clip.tokenize([description]).numpy()
token_tensor = torch.from_numpy(tokens).to(device)
with torch.no_grad():
    lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
```

**语言编码**：
- 使用 CLIP 模型将任务描述文本编码为特征向量
- 存储在 `lang_goal_embs` 中，形状为 `(77, 512)`（CLIP 的最大序列长度 × 嵌入维度）

#### 2.6 构建元数据

```python
if k == 0:
    keypoint_frame = -1  # 第一个关键点没有前置关键点
else:
    keypoint_frame = episode_keypoints[k - 1]  # 上一个关键点的索引

others = {
    "demo": True,                    # 标记为演示数据
    "keypoint_idx": k,               # 当前关键点在序列中的索引
    "episode_idx": episode_idx,      # 演示的索引 ID
    "keypoint_frame": keypoint_frame,        # 上一个关键点帧索引
    "next_keypoint_frame": keypoint,         # 当前关键点帧索引
    "sample_frame": sample_frame,            # 起始帧索引（不变）
    "initial_frame": initial_frame,          # 起始帧索引（不变，用于恢复）
}

final_obs = {
    "trans_action_indicies": trans_indicies,
    "rot_grip_action_indicies": rot_grip_indicies,
    "gripper_pose": obs_tp1.gripper_pose,    # 目标姿态
    "lang_goal": np.array([description], dtype=object),
}
```

#### 2.7 添加到 Replay Buffer

```python
others.update(final_obs)
others.update(obs_dict)

replay.add(
    task,
    task_replay_storage_folder,
    action,      # 连续动作向量
    reward,      # 奖励
    terminal,    # 是否终止
    timeout,     # 是否超时（False）
    **others     # 所有其他数据
)
```

#### 2.8 更新状态

```python
prev_action = np.copy(action)  # 保存当前动作作为下一个 transition 的 prev_action
obs = obs_tp1                   # 当前观测更新为目标观测
sample_frame = keypoint         # 更新 sample_frame（用于元数据）
```

**重要**：更新 `obs` 使得下一个循环中，当前观测就是上一个关键点的观测，从而形成连续的序列。

### 3. 添加最终状态（Terminal State）

```python
if next_keypoint_idx < len(episode_keypoints):
    obs_dict_tp1 = extract_obs(
        obs_tp1,  # 最后一个关键点的观测
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
    obs_dict_tp1.pop("wrist_world_to_cam", None)  # 移除不需要的字段
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)
```

**为什么需要 `add_final`**：
- Replay Buffer 需要存储 `s_t` 和 `s_{t+1}` 用于训练
- 当最后一个 transition 是 terminal 时，需要显式添加 `s_{t+1}`（即最终状态）

## 数据流示意图

```
初始状态: obs = inital_obs (sample_frame 的观测)

循环 1 (k = next_keypoint_idx):
  obs (当前) -> [计算动作] -> obs_tp1 (关键点 k 的观测)
  添加到 Buffer: (obs, action, reward, terminal)
  更新: obs = obs_tp1

循环 2 (k = next_keypoint_idx + 1):
  obs (关键点 k) -> [计算动作] -> obs_tp1 (关键点 k+1 的观测)
  添加到 Buffer: (obs, action, reward, terminal)
  更新: obs = obs_tp1

...

循环 N (k = len(episode_keypoints) - 1):
  obs (倒数第二个关键点) -> [计算动作] -> obs_tp1 (最后一个关键点)
  添加到 Buffer: (obs, action, reward=1.0, terminal=True)
  更新: obs = obs_tp1

最后: 调用 add_final 添加最终状态
```

## Replay 切分逻辑详解

### 整体流程（在 `fill_replay_temporal` 中）

```python
for d_idx in range(start_idx, start_idx + num_demos):
    # 1. 加载演示数据
    demo = get_stored_demo(data_path=data_path, index=d_idx)
    
    # 2. 提取关键点（例如：夹爪状态变化、机器人静止等时刻）
    episode_keypoints = keypoint_discovery(demo)
    # 示例: episode_keypoints = [5, 12, 25, 38, 50]  # 5 个关键点，索引为演示中的帧号
    
    next_keypoint_idx = 0
    
    # 3. 数据增强循环：遍历演示中的每一帧作为可能的起始帧
    for i in range(len(demo) - 1):
        # 3.1 如果不启用数据增强，只使用第一帧
        if not demo_augmentation and i > 0:
            break
        
        # 3.2 按间隔采样起始帧（例如每 5 帧采样一次）
        if i % demo_augmentation_every_n != 0:
            continue
        
        obs = demo[i]  # 当前帧作为起始帧
        
        # 3.3 找到当前帧之后的第一个关键点索引
        while (
            next_keypoint_idx < len(episode_keypoints)
            and i >= episode_keypoints[next_keypoint_idx]
        ):
            next_keypoint_idx += 1
        
        # 3.4 如果没有后续关键点，停止当前演示的处理
        if next_keypoint_idx == len(episode_keypoints):
            break
        
        # 3.5 将从当前帧开始到最后一个关键点的序列添加到 Buffer
        _add_keypoints_to_replay_temporal(
            replay,
            task,
            task_replay_storage_folder,
            d_idx,              # episode_idx
            i,                  # sample_frame (起始帧)
            obs,                # inital_obs (起始帧观测)
            demo,
            episode_keypoints,
            cameras,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
            next_keypoint_idx=next_keypoint_idx,  # 从哪个关键点开始
            description=desc,
            clip_model=clip_model,
            device=device,
        )
```

### 切分示例

假设：
- 演示总长度为 60 帧
- 关键点索引：`episode_keypoints = [5, 12, 25, 38, 50]`
- `demo_augmentation_every_n = 10`（每 10 帧采样一次）

#### 切分结果：

**切分 1**（起始帧 `i=0`）：
- `next_keypoint_idx = 0`（因为 `0 < 5`，第一个关键点在帧 5）
- 添加序列：`[0 -> 5], [5 -> 12], [12 -> 25], [25 -> 38], [38 -> 50]`
- 共 5 个 transitions

**切分 2**（起始帧 `i=10`）：
- `next_keypoint_idx = 1`（因为 `10 >= 5` 且 `10 < 12`，跳过第一个关键点）
- 添加序列：`[10 -> 12], [12 -> 25], [25 -> 38], [38 -> 50]`
- 共 4 个 transitions

**切分 3**（起始帧 `i=20`）：
- `next_keypoint_idx = 2`（因为 `20 >= 12` 且 `20 < 25`）
- 添加序列：`[20 -> 25], [25 -> 38], [38 -> 50]`
- 共 3 个 transitions

**切分 4**（起始帧 `i=30`）：
- `next_keypoint_idx = 3`（因为 `30 >= 25` 且 `30 < 38`）
- 添加序列：`[30 -> 38], [38 -> 50]`
- 共 2 个 transitions

**切分 5**（起始帧 `i=40`）：
- `next_keypoint_idx = 4`（因为 `40 >= 38` 且 `40 < 50`）
- 添加序列：`[40 -> 50]`
- 共 1 个 transition

**切分 6**（起始帧 `i=50`）：
- `next_keypoint_idx = 5`（因为 `50 >= 50`，所有关键点都被跳过）
- 跳过（没有后续关键点）

### 切分的意义

1. **数据增强（Data Augmentation）**：
   - 从同一个演示中生成多个训练样本
   - 每个样本从不同的起始帧开始
   - 增加数据的多样性和鲁棒性

2. **部分轨迹学习（Partial Trajectory Learning）**：
   - 模型可以学习从任意中间状态完成任务
   - 提高模型的泛化能力
   - 支持错误恢复（从失败点重新开始）

3. **多长度序列**：
   - 不同切分产生不同长度的序列
   - 模型可以学习处理不同复杂度的子任务

## 关键数据结构

### Transition 结构

每个添加到 Buffer 的 transition 包含：

```python
{
    # 动作
    "action": np.array([x, y, z, qx, qy, qz, qw, grip]),  # 连续动作
    "trans_action_indicies": [ix1, iy1, iz1, ix2, ...],   # 离散平移索引（多尺度）
    "rot_grip_action_indicies": [rx, ry, rz, grip],       # 离散旋转+夹爪索引
    
    # 观测（当前状态）
    "low_dim_state": np.array([gripper_open, left_finger, right_finger, time]),  # 低维状态 (4,)
    "front_rgb": np.array([3, H, W]),                     # RGB 图像
    "front_depth": np.array([1, H, W]),                   # 深度图
    "front_point_cloud": np.array([3, H, W]),             # 点云
    "front_camera_extrinsics": np.array([4, 4]),          # 相机外参
    "front_camera_intrinsics": np.array([3, 3]),          # 相机内参
    # ... 其他相机的数据
    
    # 语言目标
    "lang_goal_embs": np.array([77, 512]),                # CLIP 编码的语言特征
    "lang_goal": np.array(["描述文本"]),                   # 原始文本描述
    
    # 元数据
    "reward": float,                                       # 奖励（0 或 1.0）
    "terminal": bool,                                      # 是否终止
    "demo": True,                                          # 是否为演示数据
    "episode_idx": int,                                    # 演示 ID
    "keypoint_idx": int,                                   # 关键点序列索引
    "sample_frame": int,                                   # 当前观测对应的帧索引（会更新）
    "initial_frame": int,                                  # 起始帧索引（不变，用于恢复）
    "keypoint_frame": int,                                 # 上一个关键点帧索引
    "next_keypoint_frame": int,                            # 当前关键点帧索引（目标帧）
    "gripper_pose": np.array([x, y, z, qx, qy, qz, qw]),  # 目标姿态
}
```

## Frame 字段详解

### 关键概念

在理解这些字段之前，需要明确：

1. **当前观测（obs）对应的 frame**：transition 中状态 `s_t` 来自演示的哪一帧
2. **目标动作（action）对应的 frame**：transition 中动作 `a_t` 要将机器人移动到演示的哪一帧的状态

### Frame 字段说明

#### 1. `initial_frame` - 初始起始帧（不变）

```python
initial_frame = sample_frame  # 在函数开始时设置，之后不再改变
```

**含义**：
- 表示**整个序列的起始帧索引**（在演示中的帧号）
- **在整个循环中保持不变**
- 用于标识这个序列是从哪个起始帧开始的

**用途**：
- 用于恢复整个序列（如果需要从某个起始帧重新开始）
- 用于索引和查找（如 `_init_frame_to_idx_per_task_per_demo` 数据结构）

---

#### 2. `sample_frame` - 当前观测对应的帧（会更新）

```python
# 初始值
sample_frame = 初始传入的起始帧

# 每次循环后更新
sample_frame = keypoint  # 更新为当前关键点帧
```

**含义**：
- 表示**当前观测 `obs` 对应的演示帧索引**
- **在每次循环后会更新**（更新为上一个关键点帧）
- 第一个 transition：`sample_frame` = 初始起始帧
- 后续 transition：`sample_frame` = 上一个关键点帧

**用途**：
- 标识当前状态 `s_t` 来自演示的哪一帧
- 用于追踪序列中的当前位置

**注意**：虽然 `sample_frame` 在代码中会被更新，但在每个 transition 添加到 buffer 时，`sample_frame` 的值已经被写入到 `others` 字典中，所以 buffer 中保存的值是正确的。

---

#### 3. `keypoint_frame` - 上一个关键点帧

```python
if k == 0:
    keypoint_frame = -1  # 第一个关键点没有前置关键点
else:
    keypoint_frame = episode_keypoints[k - 1]  # 上一个关键点的帧索引
```

**含义**：
- 表示**上一个关键点的帧索引**
- 如果是第一个关键点（`k == next_keypoint_idx` 且该关键点是序列的第一个），则为 `-1`
- 否则为 `episode_keypoints[k - 1]`

**用途**：
- 标识当前观测来自哪个关键点帧
- 用于追踪序列中的关键点位置

---

#### 4. `next_keypoint_frame` - 当前关键点帧（目标帧）

```python
next_keypoint_frame = keypoint  # episode_keypoints[k]
```

**含义**：
- 表示**当前关键点的帧索引**（目标帧）
- 这是**目标动作对应的帧**
- 动作 `action` 的目标是将机器人从当前状态移动到 `next_keypoint_frame` 对应的状态

**用途**：
- 标识目标动作对应的帧
- **这是理解动作含义的关键字段**：动作的目标是达到 `next_keypoint_frame` 的状态

---

### Frame 关系图解

假设演示有 100 帧，关键点为 `[5, 12, 25, 38, 50]`，起始帧为 `0`：

```
演示帧序列: 0, 1, 2, ..., 4, 5, ..., 11, 12, ..., 24, 25, ..., 37, 38, ..., 49, 50, ...
           |<--起始-->|  |<--关键点-->|  |<--关键点-->|  |<--关键点-->|  |<--关键点-->|  |<--关键点-->|
```

**Transition 1**（k=0，第一个关键点）：
```
当前观测 (obs): demo[0]（初始起始帧）
目标动作 (action): 将机器人从 demo[0] 移动到 demo[5] 的状态

字段值：
- initial_frame = 0（起始帧，不变）
- sample_frame = 0（当前观测对应的帧）
- keypoint_frame = -1（没有上一个关键点）
- next_keypoint_frame = 5（目标帧，关键点1）
```

**Transition 2**（k=1，第二个关键点）：
```
当前观测 (obs): demo[5]（上一个关键点的状态）
目标动作 (action): 将机器人从 demo[5] 移动到 demo[12] 的状态

字段值：
- initial_frame = 0（起始帧，不变）
- sample_frame = 5（当前观测对应的帧，已更新）
- keypoint_frame = 5（上一个关键点帧）
- next_keypoint_frame = 12（目标帧，关键点2）
```

**Transition 3**（k=2，第三个关键点）：
```
当前观测 (obs): demo[12]（上一个关键点的状态）
目标动作 (action): 将机器人从 demo[12] 移动到 demo[25] 的状态

字段值：
- initial_frame = 0（起始帧，不变）
- sample_frame = 12（当前观测对应的帧，已更新）
- keypoint_frame = 12（上一个关键点帧）
- next_keypoint_frame = 25（目标帧，关键点3）
```

### 核心问题解答

#### Q1: 每个 transition 当前的 frame 是什么？

**答案**：当前观测 `obs` 对应的帧是 `sample_frame` 或 `keypoint_frame`（如果 `keypoint_frame != -1`）。

- **第一个 transition**：
  - 当前观测：`demo[sample_frame]` = `demo[initial_frame]`（初始起始帧）
  - `sample_frame` = 初始起始帧
  - `keypoint_frame` = -1（没有上一个关键点）

- **后续 transition**：
  - 当前观测：`demo[sample_frame]` = `demo[keypoint_frame]`（上一个关键点帧）
  - `sample_frame` = 上一个关键点帧（已更新）
  - `keypoint_frame` = 上一个关键点帧

**总结**：当前帧 = `sample_frame`（第一个 transition 除外，第一个 transition 的当前帧是 `initial_frame`）

---

#### Q2: 目标 action 对应的 frame 应该看什么？

**答案**：目标动作对应的帧是 **`next_keypoint_frame`**。

**关键理解**：
- 动作 `action` 的目标是将机器人从当前状态（`demo[sample_frame]`）移动到目标状态（`demo[next_keypoint_frame]`）
- `action` 是根据 `obs_tp1 = demo[next_keypoint_frame]` 计算得到的
- `gripper_pose` = `obs_tp1.gripper_pose` = `demo[next_keypoint_frame].gripper_pose`

**代码证据**：
```python
keypoint = episode_keypoints[k]           # 当前关键点索引
obs_tp1 = demo[keypoint]                  # 目标观测（目标帧的观测）
action = compute_action_from(obs_tp1)     # 根据目标观测计算动作
next_keypoint_frame = keypoint            # 目标帧索引
```

**总结**：
- **当前观测的帧**：看 `sample_frame`（第一个 transition 看 `initial_frame`）
- **目标动作的帧**：看 **`next_keypoint_frame`**

---

### 实际示例

假设：
- 演示：100 帧（0-99）
- 关键点：`[5, 12, 25, 38, 50]`
- 起始帧：`10`
- `next_keypoint_idx = 1`（起始帧10之后的第一个关键点是索引1，即帧12）

**Transition 1**（k=1，对应关键点12）：
```python
# 当前状态
obs = demo[10]  # 起始帧的观测
sample_frame = 10
initial_frame = 10

# 目标状态
obs_tp1 = demo[12]  # 关键点12的观测
next_keypoint_frame = 12
keypoint_frame = 5  # 上一个关键点（虽然当前观测不是来自它）

# 动作
action = 从 demo[10] 到 demo[12] 的动作
```

**Transition 2**（k=2，对应关键点25）：
```python
# 当前状态
obs = demo[12]  # 上一个关键点的观测（已更新）
sample_frame = 12  # 已更新
initial_frame = 10  # 不变

# 目标状态
obs_tp1 = demo[25]  # 关键点25的观测
next_keypoint_frame = 25
keypoint_frame = 12  # 上一个关键点

# 动作
action = 从 demo[12] 到 demo[25] 的动作
```

---

### 字段对比总结

| 字段 | 含义 | 是否变化 | 用途 |
|------|------|----------|------|
| `initial_frame` | 序列的起始帧 | **不变** | 标识序列起点，用于恢复 |
| `sample_frame` | 当前观测对应的帧 | **会更新** | 标识当前状态来自哪一帧 |
| `keypoint_frame` | 上一个关键点帧 | 每步不同 | 标识当前观测来自哪个关键点 |
| `next_keypoint_frame` | 目标关键点帧 | 每步不同 | **标识目标动作对应的帧** |

### 关键理解

1. **当前观测的帧** = `sample_frame`（第一个 transition 是 `initial_frame`）
2. **目标动作的帧** = **`next_keypoint_frame`**（这是最重要的字段，用于理解动作的目标）
3. **序列的起始帧** = `initial_frame`（用于追踪和恢复整个序列）
4. **上一个关键点帧** = `keypoint_frame`（用于追踪关键点序列）
```

### 动作相关字段详解

这些字段都表示"目标动作"，但用途和格式不同：

#### 1. `action` - 连续动作向量（主要动作字段）

```python
action = np.array([x, y, z, qx, qy, qz, qw, grip])
# 形状: (8,)
# 类型: float32
```

**内容**：
- `[x, y, z]`: 目标位置（连续坐标，单位：米）
- `[qx, qy, qz, qw]`: 目标姿态（四元数，连续旋转）
- `grip`: 夹爪状态（0.0 = 闭合，1.0 = 张开，连续值）

**用途**：
- 这是传递给 `replay.add()` 的**主要动作参数**
- 用于**行为克隆（Behavior Cloning）**训练，直接监督学习连续动作
- 包含完整的连续动作信息（位置、姿态、夹爪）

**来源**：
```python
action = np.concatenate([obs_tp1.gripper_pose, np.array([grip])])
# gripper_pose = [x, y, z, qx, qy, qz, qw] (7维)
# 加上 grip (1维) = 8维
```

---

#### 2. `trans_action_indicies` - 离散平移索引（体素坐标）

```python
trans_action_indicies = [ix1, iy1, iz1, ix2, iy2, iz2, ...]
# 形状: (3 * len(voxel_sizes),)  # 例如 voxel_sizes=[100] 则形状为 (3,)
# 类型: int32
```

**内容**：
- 将连续位置 `[x, y, z]` 离散化到体素网格索引
- **多尺度支持**：如果 `voxel_sizes=[100, 200]`，则包含两个尺度的索引
  - `[ix1, iy1, iz1]`: 尺度1的体素索引（细粒度）
  - `[ix2, iy2, iz2]`: 尺度2的体素索引（粗粒度）

**转换过程**：
```python
# 连续位置 -> 体素索引
point = obs_tp1.gripper_pose[:3]  # [x, y, z] (连续)
voxel_index = point_to_voxel_index(point, voxel_size, scene_bounds)
# 例如: [0.5, 0.3, 0.2] -> [50, 30, 20] (离散索引)
```

**用途**：
- 用于 **PerAct** 等需要离散动作空间的模型
- 允许模型在离散的体素空间中预测目标位置
- 多尺度索引支持多分辨率的注意力机制

---

#### 3. `rot_grip_action_indicies` - 离散旋转+夹爪索引

```python
rot_grip_action_indicies = [rx, ry, rz, grip_idx]
# 形状: (4,)
# 类型: int32
```

**内容**：
- `[rx, ry, rz]`: 离散旋转索引（欧拉角的离散化）
  - 将连续四元数转换为离散的 XYZ 欧拉角索引
  - 例如：如果 `rotation_resolution=72`，每个轴有 72 个离散值（每 5 度一个）
  - 取值范围：`[0, rotation_resolution-1]` 每个轴
- `grip_idx`: 离散夹爪索引（0 或 1）

**转换过程**：
```python
# 连续四元数 -> 离散欧拉角索引
quat = obs_tp1.gripper_pose[3:]  # [qx, qy, qz, qw]
euler_disc = quaternion_to_discrete_euler(quat, rotation_resolution)
# 例如: 四元数 -> [36, 42, 18] (离散索引，每个轴 0-71)

# 夹爪状态
grip_idx = int(obs_tp1.gripper_open)  # 0 或 1
```

**用途**：
- 用于需要离散旋转动作空间的模型
- 与 `trans_action_indicies` 配合使用，形成完整的离散动作空间
- 支持离散动作空间的强化学习或模仿学习

---

#### 4. `gripper_pose` - 目标姿态（位置+旋转）

```python
gripper_pose = np.array([x, y, z, qx, qy, qz, qw])
# 形状: (7,)
# 类型: float32
```

**内容**：
- `[x, y, z]`: 目标位置（连续坐标，单位：米）
- `[qx, qy, qz, qw]`: 目标旋转（四元数）

**注意**：
- **不包含夹爪状态**（只有位置和姿态）
- 这与 `action` 的前 7 维相同，但不包括第 8 维的 `grip`

**用途**：
- 作为**目标姿态的参考**，方便模型理解期望的末端执行器位置和姿态
- 可能用于：
  - 计算目标点（用于注意力机制）
  - 辅助训练或验证
  - 其他需要目标姿态但不关心夹爪状态的场景

**来源**：
```python
gripper_pose = obs_tp1.gripper_pose  # 直接来自目标观测
```

---

### 字段对比总结

| 字段 | 维度 | 类型 | 包含内容 | 主要用途 |
|------|------|------|----------|----------|
| `action` | 8 | float32 | 位置(3) + 四元数(4) + 夹爪(1) | **主要动作字段**，用于行为克隆 |
| `trans_action_indicies` | 3×N | int32 | 体素索引（多尺度） | 离散平移动作，用于 PerAct 等 |
| `rot_grip_action_indicies` | 4 | int32 | 欧拉角索引(3) + 夹爪索引(1) | 离散旋转动作，配合平移使用 |
| `gripper_pose` | 7 | float32 | 位置(3) + 四元数(4) | 目标姿态参考（不含夹爪状态） |

### 使用场景示例

**场景 1：连续动作空间模型（如 BC-RNN）**
```python
# 使用连续动作
predicted_action = model(observation)
loss = MSE(predicted_action, transition["action"])  # 使用 action
```

**场景 2：离散动作空间模型（如 PerAct）**
```python
# 预测离散动作
predicted_trans_idx = model_predict_transition(observation)
predicted_rot_grip_idx = model_predict_rotation_grip(observation)

loss = CrossEntropy(predicted_trans_idx, transition["trans_action_indicies"])
loss += CrossEntropy(predicted_rot_grip_idx, transition["rot_grip_action_indicies"])
```

**场景 3：混合使用**
```python
# 从离散预测转换回连续动作用于执行
discrete_trans = model_predict_transition(observation)
discrete_rot_grip = model_predict_rotation_grip(observation)

# 转换回连续空间
continuous_action = discrete_to_continuous(discrete_trans, discrete_rot_grip)
robot.execute(continuous_action)
```

## 时序关系

### 时间步编码

在 `extract_obs` 中，时间步被归一化到 `[-1, 1]`：

```python
time = (1. - (t / float(episode_length - 1))) * 2. - 1.
```

- `t=0`（起始帧）→ `time=1.0`
- `t=episode_length-1`（最后）→ `time=-1.0`

这个时间编码帮助模型理解当前处于任务的哪个阶段。

### 序列连续性

通过在每个循环中更新 `obs = obs_tp1`，确保：
- 第 k 个 transition 的状态是 `obs_tp1`（来自第 k-1 个关键点）
- 第 k 个 transition 的目标是 `obs_tp1`（来自第 k 个关键点）
- 第 k+1 个 transition 的状态是 `obs_tp1`（来自第 k 个关键点）

因此，序列是连续的：`s0 -> s1 -> s2 -> ... -> sN`

## `low_dim_state` 详细解释

### 概述

`low_dim_state` 是一个低维状态向量，用于表示机器人的**本体感觉（proprioceptive）状态**。它包含了夹爪的状态信息和时间步信息，用于补充视觉观测（RGB、深度图、点云）。

### 数据结构

```python
low_dim_state = np.array([gripper_open, left_finger_joint, right_finger_joint, time])
# 形状: (4,)
# 类型: float32
```

### 各维度详解

#### 1. `gripper_open` - 夹爪开合状态（索引 0）

```python
gripper_open = obs.gripper_open  # 0.0 或 1.0
```

**含义**：
- 夹爪的二进制开合状态
- `0.0`：夹爪闭合（closed）
- `1.0`：夹爪张开（open）
- 这是一个二值化后的状态（根据夹爪开合程度阈值化）

**来源**：
```python
# 在 RLBench 中，gripper_open 是根据夹爪开合程度计算的
gripper_open = 1.0 if gripper.get_open_amount()[0] > 0.95 else 0.0
```

**用途**：
- 快速判断夹爪是否张开
- 用于决策是否需要抓取或释放物体

---

#### 2. `left_finger_joint` - 左手指关节位置（索引 1）

```python
left_finger_joint = obs.gripper_joint_positions[0]
```

**含义**：
- 左手指关节的**实际位置**（单位：米）
- 取值范围：`[0.0, 0.04]`（会被裁剪到这个范围）
- 表示左手指相对于闭合位置的位移

**来源**：
```python
gripper_joint_positions = obs.gripper_joint_positions  # (2,) 数组
left_finger_joint = gripper_joint_positions[0]
```

**预处理**：
```python
# 在 extract_obs 中，会对关节位置进行裁剪
if obs.gripper_joint_positions is not None:
    obs.gripper_joint_positions = np.clip(
        obs.gripper_joint_positions, 0., 0.04)
```

**用途**：
- 提供夹爪的**连续状态信息**（比二值化的 `gripper_open` 更精确）
- 用于精确控制夹爪的抓取力度

---

#### 3. `right_finger_joint` - 右手指关节位置（索引 2）

```python
right_finger_joint = obs.gripper_joint_positions[1]
```

**含义**：
- 右手指关节的**实际位置**（单位：米）
- 取值范围：`[0.0, 0.04]`（会被裁剪到这个范围）
- 表示右手指相对于闭合位置的位移

**说明**：
- 与 `left_finger_joint` 类似，但表示右手指的状态
- 通常左右手指的位置是对称的（但在抓取不规则物体时可能不同）

---

#### 4. `time` - 归一化时间步（索引 3）

```python
time = (1. - (t / float(episode_length - 1))) * 2. - 1.
# 取值范围: [-1.0, 1.0]
```

**含义**：
- 当前时间步在 episode 中的**归一化位置**
- 用于表示任务执行的进度

**计算公式**：
```python
# t: 当前时间步（相对于起始关键点的索引）
# episode_length: 最大 episode 长度（通常为 25）

normalized_t = t / float(episode_length - 1)  # [0, 1]
time = (1.0 - normalized_t) * 2.0 - 1.0       # [-1, 1]
```

**取值范围**：
- `t = 0`（起始）→ `time = 1.0`
- `t = episode_length - 1`（结束）→ `time = -1.0`
- 中间值线性插值

**示例**（假设 `episode_length = 25`）：
```
t = 0   → time = (1 - 0/24) * 2 - 1 = 1.0    # 开始
t = 6   → time = (1 - 6/24) * 2 - 1 = 0.5    # 1/4 处
t = 12  → time = (1 - 12/24) * 2 - 1 = 0.0   # 中间
t = 18  → time = (1 - 18/24) * 2 - 1 = -0.5  # 3/4 处
t = 24  → time = (1 - 24/24) * 2 - 1 = -1.0  # 结束
```

**用途**：
- 帮助模型理解任务执行的**时间进度**
- 对于需要在不同阶段执行不同动作的任务很重要
- 例如：抓取任务中，开始需要张开夹爪，结束时需要闭合

---

### 构建过程

#### 步骤 1：构建机器人状态

```python
# 构造机器人状态向量
robot_state = np.array([
    obs.gripper_open,              # 标量：0.0 或 1.0
    *obs.gripper_joint_positions   # 展开 (2,) 数组：[left, right]
])
# robot_state 形状: (3,)
# 内容: [gripper_open, left_finger_joint, right_finger_joint]
```

#### 步骤 2：计算归一化时间步

```python
# 计算归一化时间步
time = (1. - (t / float(episode_length - 1))) * 2. - 1.
# time 是标量，取值范围: [-1.0, 1.0]
```

#### 步骤 3：拼接得到完整状态

```python
# 将时间步添加到机器人状态中
obs_dict['low_dim_state'] = np.concatenate(
    [robot_state, [time]]  # 将 time 包装成列表以便拼接
).astype(np.float32)
# low_dim_state 形状: (4,)
# 内容: [gripper_open, left_finger_joint, right_finger_joint, time]
```

**完整代码流程**：
```python
# 1. 提取夹爪状态
robot_state = np.array([
    obs.gripper_open,              # 0.0 或 1.0
    *obs.gripper_joint_positions   # [left_finger, right_finger]
])
# robot_state = [gripper_open, left_finger, right_finger]  (3,)

# 2. 计算时间步
t = k - next_keypoint_idx  # 相对时间步
time = (1. - (t / float(episode_length - 1))) * 2. - 1.

# 3. 拼接
low_dim_state = np.concatenate([robot_state, [time]]).astype(np.float32)
# low_dim_state = [gripper_open, left_finger, right_finger, time]  (4,)
```

---

### 常量定义

```python
# 在 sam2act/utils/peract_utils.py 中定义
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}
```

**注意**：注释中的顺序与代码中的实际顺序不同：
- **注释顺序**：`[left_finger_joint, right_finger_joint, gripper_open, timestep]`
- **实际顺序**：`[gripper_open, left_finger_joint, right_finger_joint, timestep]`

**实际代码顺序**是正确的（以代码为准）。

---

### 数值范围总结

| 维度 | 名称 | 取值范围 | 单位 | 说明 |
|------|------|----------|------|------|
| 0 | `gripper_open` | `{0.0, 1.0}` | 无 | 二值化开合状态 |
| 1 | `left_finger_joint` | `[0.0, 0.04]` | 米 | 左手指位置 |
| 2 | `right_finger_joint` | `[0.0, 0.04]` | 米 | 右手指位置 |
| 3 | `time` | `[-1.0, 1.0]` | 无 | 归一化时间步 |

---

### 使用场景

#### 1. 模型输入

```python
# 在模型中使用 low_dim_state
obs_dict = extract_obs(obs, cameras, t=0, episode_length=25)
low_dim_state = obs_dict['low_dim_state']  # (4,) float32

# 输入到模型
model_output = model(rgb_images, point_clouds, low_dim_state)
```

#### 2. 状态表示

- **视觉信息**（RGB、深度、点云）：提供环境的**外部感知**
- **低维状态**（`low_dim_state`）：提供机器人的**内部状态**

两者结合，形成完整的观察表示。

#### 3. 时间建模

时间步信息帮助模型：
- 理解任务进度
- 在不同阶段执行不同策略
- 处理时序依赖关系

---

### 与高维观测的区别

| 类型 | 维度 | 信息类型 | 更新频率 | 用途 |
|------|------|----------|----------|------|
| **高维观测**（RGB、深度、点云） | 高（128×128×3 等） | 外部环境 | 每帧 | 视觉感知、物体识别 |
| **低维状态**（`low_dim_state`） | 低（4） | 内部状态 | 每帧 | 本体感觉、时间信息 |

**互补关系**：
- 高维观测：回答"环境中有什么？"
- 低维状态：回答"机器人当前状态如何？任务进行到哪一步？"

---

### 示例数据

```python
# 示例 1：夹爪闭合，任务开始
low_dim_state = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
# gripper_open=0.0 (闭合)
# left_finger=0.0 (闭合位置)
# right_finger=0.0 (闭合位置)
# time=1.0 (开始)

# 示例 2：夹爪张开，任务进行中
low_dim_state = np.array([1.0, 0.04, 0.04, 0.0], dtype=np.float32)
# gripper_open=1.0 (张开)
# left_finger=0.04 (最大张开)
# right_finger=0.04 (最大张开)
# time=0.0 (中间)

# 示例 3：夹爪部分闭合，任务接近结束
low_dim_state = np.array([0.0, 0.02, 0.02, -0.5], dtype=np.float32)
# gripper_open=0.0 (闭合)
# left_finger=0.02 (部分闭合)
# right_finger=0.02 (部分闭合)
# time=-0.5 (接近结束)
```

---

### 注意事项

1. **数值裁剪**：
   - `gripper_joint_positions` 会被裁剪到 `[0.0, 0.04]` 范围
   - 防止异常值影响训练

2. **时间步计算**：
   - 时间步 `t` 是相对于起始关键点的索引
   - `episode_length` 通常设置为 25（最大关键点数量）

3. **数据类型**：
   - 所有值都是 `float32` 类型
   - 确保数值精度和内存效率

4. **顺序一致性**：
   - 虽然注释中的顺序不同，但代码中的顺序是固定的
   - 使用时应以代码实现为准

---

## 总结

`_add_keypoints_to_replay_temporal` 函数的核心思想是：

1. **从起始帧开始**，遍历后续所有关键点
2. **为每个关键点创建 transition**：(当前状态, 动作, 奖励, 下一状态)
3. **状态更新**：每次循环后将目标状态作为下一次的当前状态
4. **形成连续序列**：所有 transitions 构成一个完整的轨迹

Replay 切分的核心思想是：

1. **多起始点采样**：从演示中采样多个起始帧
2. **部分轨迹生成**：从每个起始帧生成到最后一个关键点的子序列
3. **数据增强**：大幅增加训练数据的数量和多样性

这种设计使得模型能够：
- 学习完整的任务执行（从第一个关键点开始）
- 学习部分任务完成（从中间状态开始）
- 具备错误恢复能力（从任意状态重新开始任务）