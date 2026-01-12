# 没有Replay Buffer的完整数据生成流程

本文档详细说明在没有replay buffer的情况下，从原始demo数据到训练样本的完整生成流程。

## 1. 整体流程概览

```
原始Demo文件 → Keypoint提取 → 数据增强采样 → 样本生成 → 数据处理 → 训练样本
```

## 2. 详细步骤

### 2.1 加载原始Demo数据

**位置**: `sam2act/utils/dataset.py` - `fill_replay_temporal()` 函数

```python
# 从磁盘加载demo
demo = get_stored_demo(data_path=data_path, index=d_idx)

# 加载语言目标描述
varation_descs_pkl_file = os.path.join(
    data_path, episode_folder % d_idx, variation_desriptions_pkl
)
with open(varation_descs_pkl_file, "rb") as f:
    descs = pickle.load(f)
```

**输入**:
- `data_path`: demo数据存储路径
- `d_idx`: demo索引
- `episode_folder`: episode文件夹格式（如 `"episode%d"`）

**输出**:
- `demo`: Demo对象，包含一系列Observation
- `descs`: 语言目标描述列表

---

### 2.2 Keypoint提取

**位置**: `sam2act/libs/peract/helpers/demo_loading_utils.py` - `keypoint_discovery()`

```python
episode_keypoints = keypoint_discovery(demo)  # 返回关键帧索引列表 [id0, id1, id2, ...]
```

**Keypoint提取逻辑**（heuristic方法）:
1. 遍历demo中的所有帧
2. 检测以下情况作为keypoint:
   - **Gripper状态变化**: 夹爪开合状态改变
   - **机器人停止**: 关节速度接近0且持续一段时间
   - **Episode结束**: 最后一帧
3. 返回所有keypoint的帧索引列表

**示例**:
```python
# 假设demo有100帧，keypoints可能是:
episode_keypoints = [0, 15, 32, 48, 65, 99]
```

---

### 2.3 数据增强采样

**位置**: `sam2act/utils/dataset.py` - `fill_replay_temporal()` 函数

```python
for i in range(len(demo) - 1):
    if not demo_augmentation and i > 0:
        break
    if i % demo_augmentation_every_n != 0:  # 只选择每n帧
        continue
    
    obs = demo[i]  # 起始观察帧
    desc = descs[0]  # 语言目标
    
    # 如果起始点已经超过某个keypoint，则移除该keypoint
    while (
        next_keypoint_idx < len(episode_keypoints)
        and i >= episode_keypoints[next_keypoint_idx]
    ):
        next_keypoint_idx += 1
    if next_keypoint_idx == len(episode_keypoints):
        break
```

**参数**:
- `demo_augmentation`: 是否启用数据增强（True/False）
- `demo_augmentation_every_n`: 每隔n帧采样一次起始帧（如 `DEMO_AUGMENTATION_EVERY_N=10`）

**逻辑**:
- 如果 `demo_augmentation=False`，只使用第一帧作为起始帧
- 如果 `demo_augmentation=True`，每隔 `demo_augmentation_every_n` 帧采样一个起始帧
- 对于每个起始帧，生成从该帧到后续所有keypoints的训练样本

**示例**:
```python
# demo有100帧，demo_augmentation_every_n=10
# 起始帧: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# 对于起始帧0，生成样本: (0→15), (0→32), (0→48), (0→65), (0→99)
# 对于起始帧10，生成样本: (10→32), (10→48), (10→65), (10→99)
# ...
```

---

### 2.4 样本生成（核心流程）

**位置**: `sam2act/utils/dataset.py` - `_add_keypoints_to_replay_temporal()` 函数

对于每个起始帧，生成从该帧到后续所有keypoints的训练样本：

```python
def _add_keypoints_to_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,  # 起始帧索引
    inital_obs: Observation,  # 起始观察
    demo: Demo,
    episode_keypoints: List[int],  # 所有keypoints
    ...
    next_keypoint_idx: int,  # 从哪个keypoint开始
):
    prev_action = None
    obs = inital_obs  # 当前观察（初始为起始帧）
    initial_frame = sample_frame
    
    # 遍历从next_keypoint_idx开始的所有keypoints
    for k in range(next_keypoint_idx, len(episode_keypoints)):
        keypoint = episode_keypoints[k]  # keypoint帧索引
        obs_tp1 = demo[keypoint]  # keypoint帧的观察
        obs_tm1 = demo[max(0, keypoint - 1)]  # keypoint前一帧
        
        # 2.4.1 计算动作
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds,
            voxel_sizes, rotation_resolution, crop_augmentation,
        )
        
        # 2.4.2 提取观察
        obs_dict = extract_obs(
            obs,  # 当前观察（不是keypoint观察）
            CAMERAS,
            t=k - next_keypoint_idx,  # 时间步（相对于起始keypoint）
            prev_action=prev_action,
            episode_length=25,
        )
        
        # 2.4.3 编码语言目标
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
        
        # 2.4.4 构建样本
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
            "initial_frame": initial_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }
        
        # 2.4.5 如果没有replay buffer，这里应该直接返回样本
        # 而不是调用 replay.add()
        
        # 更新状态
        prev_action = np.copy(action)
        obs = obs_tp1  # 更新当前观察为keypoint观察
        sample_frame = keypoint
```

---

### 2.5 动作计算

**位置**: `sam2act/utils/dataset.py` - `_get_action()` 函数

```python
def _get_action(
    obs_tp1: Observation,  # keypoint帧观察
    obs_tm1: Observation,  # keypoint前一帧观察
    rlbench_scene_bounds: List[float],  # 场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
    voxel_sizes: List[int],  # 体素大小列表
    rotation_resolution: int,  # 旋转离散化分辨率
    crop_augmentation: bool,
):
    # 2.5.1 计算旋转动作（离散化）
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    # disc_rot: [rot_x_idx, rot_y_idx, rot_z_idx]
    
    # 2.5.2 计算平移动作（离散化到体素空间）
    attention_coordinate = obs_tp1.gripper_pose[:3]  # 目标位置 [x, y, z]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    
    for depth, vox_size in enumerate(voxel_sizes):
        # 将3D点转换为体素索引
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds
        )
        trans_indicies.extend(index.tolist())  # [vox_x, vox_y, vox_z]
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)
    
    # 2.5.3 计算夹爪动作
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])  # [rot_x, rot_y, rot_z, grip]
    
    # 2.5.4 碰撞忽略标志
    ignore_collisions = int(obs_tm1.ignore_collisions)
    
    # 2.5.5 完整动作（连续值，用于某些用途）
    action = np.concatenate([
        obs_tp1.gripper_pose,  # [x, y, z, qx, qy, qz, qw]
        np.array([grip])
    ])  # shape: (8,)
    
    return (
        trans_indicies,  # 平移体素索引 [vox_x1, vox_y1, vox_z1, vox_x2, ...]
        rot_and_grip_indicies,  # [rot_x_idx, rot_y_idx, rot_z_idx, grip]
        ignore_collisions,  # 0 or 1
        action,  # 连续动作 [x, y, z, qx, qy, qz, qw, grip]
        attention_coordinates,  # 注意力坐标列表
    )
```

**输出说明**:
- `trans_indicies`: 平移动作的离散化索引，长度为 `3 * len(voxel_sizes)`
- `rot_and_grip_indicies`: 旋转和夹爪动作索引，长度为 `3 + 1 = 4`
- `action`: 连续动作值，shape为 `(8,)`

---

### 2.6 观察提取

**位置**: `sam2act/libs/peract/helpers/utils.py` - `extract_obs()` 函数

```python
def extract_obs(
    obs: Observation,
    cameras: List[str],  # ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    t: int = 0,  # 时间步（相对于起始keypoint）
    prev_action=None,
    episode_length: int = 10,
):
    # 2.6.1 提取低维状态
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions
    ])
    
    # 2.6.2 提取图像数据（RGB, depth, point cloud）
    obs_dict = {}
    for camera_name in cameras:
        obs_dict[f'{camera_name}_rgb'] = obs.get_image(camera_name, 'rgb')  # (H, W, 3)
        obs_dict[f'{camera_name}_depth'] = obs.get_image(camera_name, 'depth')  # (H, W, 1)
        obs_dict[f'{camera_name}_point_cloud'] = obs.get_image(camera_name, 'point_cloud')  # (3, H, W)
        obs_dict[f'{camera_name}_camera_extrinsics'] = obs.misc[f'{camera_name}_camera_extrinsics']  # (4, 4)
        obs_dict[f'{camera_name}_camera_intrinsics'] = obs.misc[f'{camera_name}_camera_intrinsics']  # (3, 3)
    
    # 2.6.3 添加时间步信息
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.  # 归一化到[-1, 1]
    obs_dict['low_dim_state'] = np.concatenate([
        robot_state, [time]
    ]).astype(np.float32)
    
    # 2.6.4 添加碰撞忽略标志
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    
    # 2.6.5 转换数据格式（channels first）
    for k, v in obs_dict.items():
        if type(v) == np.ndarray and v.ndim == 3:
            obs_dict[k] = np.transpose(v, [2, 0, 1])  # (H, W, C) -> (C, H, W)
    
    return obs_dict
```

**输出观察字典包含**:
- `low_dim_state`: 低维状态 (robot_state + time)
- `{camera}_rgb`: RGB图像 (3, H, W)
- `{camera}_depth`: 深度图像 (1, H, W)
- `{camera}_point_cloud`: 点云 (3, H, W)
- `{camera}_camera_extrinsics`: 相机外参 (4, 4)
- `{camera}_camera_intrinsics`: 相机内参 (3, 3)
- `ignore_collisions`: 碰撞忽略标志

---

### 2.7 语言目标编码

**位置**: `sam2act/utils/dataset.py` - `_clip_encode_text()` 函数

```python
def _clip_encode_text(clip_model, text):
    # 使用CLIP模型编码文本
    tokens = clip.tokenize([description]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device)
    
    with torch.no_grad():
        # 获取token embeddings
        x = clip_model.token_embedding(text).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        
        emb = x.clone()  # token embeddings: (1, 77, 512)
        # 全局特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    
    return x, emb  # 返回全局特征和token embeddings
```

**输出**:
- `lang_goal_embs`: Token embeddings，shape为 `(77, 512)`（最大序列长度77，嵌入维度512）

---

### 2.8 构建最终训练样本

**位置**: `sam2act/utils/dataset.py` - `_add_keypoints_to_replay_temporal()` 函数

每个训练样本包含以下字段：

```python
sample = {
    # === 观察数据 ===
    'low_dim_state': np.array,  # 低维状态
    'front_rgb': np.array,  # (3, 128, 128)
    'front_depth': np.array,  # (1, 128, 128)
    'front_point_cloud': np.array,  # (3, 128, 128)
    'front_camera_extrinsics': np.array,  # (4, 4)
    'front_camera_intrinsics': np.array,  # (3, 3)
    # ... 其他相机类似
    
    # === 动作数据 ===
    'trans_action_indicies': np.array,  # 平移动作索引
    'rot_grip_action_indicies': np.array,  # 旋转和夹爪动作索引
    'gripper_pose': np.array,  # (7,) 夹爪位姿
    'ignore_collisions': np.array,  # 碰撞忽略标志
    
    # === 语言数据 ===
    'lang_goal_embs': np.array,  # (77, 512) CLIP token embeddings
    'lang_goal': np.array,  # 语言目标字符串
    
    # === 元数据 ===
    'demo': bool,  # 是否为demo数据
    'keypoint_idx': int,  # keypoint索引
    'episode_idx': int,  # episode索引
    'keypoint_frame': int,  # 上一个keypoint帧
    'next_keypoint_frame': int,  # 当前keypoint帧
    'sample_frame': int,  # 采样帧
    'initial_frame': int,  # 起始帧
    
    # === 奖励和终止标志 ===
    'reward': float,  # 奖励（最后一个keypoint为1.0，其他为0.0）
    'terminal': bool,  # 是否为终止状态
}
```

---

## 3. 完整数据生成流程示例

假设有一个demo包含100帧，keypoints为 `[0, 15, 32, 48, 65, 99]`，`demo_augmentation_every_n=10`：

### 3.1 起始帧采样
```
起始帧: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
```

### 3.2 对于起始帧0，生成样本：
```
样本1: 起始帧=0, 目标keypoint=15
  - obs: demo[0]的观察
  - action: 从demo[0]到demo[15]的动作
  - t: 0 (第一个keypoint)
  
样本2: 起始帧=0, 目标keypoint=32
  - obs: demo[0]的观察
  - action: 从demo[0]到demo[32]的动作
  - t: 1 (第二个keypoint)
  
样本3: 起始帧=0, 目标keypoint=48
  - obs: demo[0]的观察
  - action: 从demo[0]到demo[48]的动作
  - t: 2 (第三个keypoint)
  
... 以此类推
```

### 3.3 对于起始帧10，生成样本：
```
样本1: 起始帧=10, 目标keypoint=32
  - obs: demo[10]的观察
  - action: 从demo[10]到demo[32]的动作
  - t: 0
  
样本2: 起始帧=10, 目标keypoint=48
  - obs: demo[10]的观察
  - action: 从demo[10]到demo[48]的动作
  - t: 1
  
... 以此类推
```

**注意**: 起始帧10不会生成到keypoint 15的样本，因为10 > 15，已经超过了该keypoint。

---

## 4. 没有Replay Buffer的实现方式

如果没有replay buffer，应该：

1. **直接生成样本列表**: 在 `fill_replay_temporal()` 中，不调用 `replay.add()`，而是将样本添加到列表中
2. **返回样本列表**: 返回所有生成的样本，而不是replay buffer
3. **创建PyTorch Dataset**: 使用生成的样本列表创建自定义PyTorch Dataset
4. **直接用于训练**: 使用DataLoader加载样本进行训练

**伪代码**:
```python
def generate_samples_without_replay_buffer(...):
    all_samples = []
    
    for task in tasks:
        for d_idx in range(num_demos):
            demo = get_stored_demo(...)
            episode_keypoints = keypoint_discovery(demo)
            
            for i in range(len(demo) - 1):
                if i % demo_augmentation_every_n != 0:
                    continue
                
                # 生成从起始帧i到后续keypoints的所有样本
                samples = generate_samples_from_start_frame(
                    demo, i, episode_keypoints, ...
                )
                all_samples.extend(samples)
    
    return all_samples
```

---

## 5. 关键参数总结

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `demo_augmentation` | 是否启用数据增强 | `True` |
| `demo_augmentation_every_n` | 每隔n帧采样起始帧 | `10` |
| `voxel_sizes` | 体素大小列表 | `[100]` |
| `rotation_resolution` | 旋转离散化分辨率 | `5` (度) |
| `episode_length` | Episode长度（用于时间步计算） | `25` |
| `IMAGE_SIZE` | 图像尺寸 | `128` |
| `CAMERAS` | 相机列表 | `['front', 'left_shoulder', 'right_shoulder', 'wrist']` |

---

## 6. 数据流图

```
┌─────────────────┐
│  原始Demo文件    │
│  (RLBench格式)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Keypoint提取   │
│  (关键帧检测)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  起始帧采样     │
│  (每n帧采样)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  样本生成循环   │
│  (起始帧→keypoint)│
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  提取观察    │  │  计算动作    │  │  编码语言    │  │  添加元数据  │
│  (RGB/Depth/ │  │  (离散化)    │  │  (CLIP)      │  │  (索引/帧号) │
│   PointCloud)│  │              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   训练样本      │
                        │  (字典格式)     │
                        └─────────────────┘
```

---

## 7. 注意事项

1. **内存占用**: 没有replay buffer时，所有样本都在内存中，需要确保有足够内存
2. **数据加载速度**: 每次训练都需要重新生成样本，可能较慢
3. **数据增强**: 可以通过调整 `demo_augmentation_every_n` 控制样本数量
4. **Keypoint质量**: Keypoint提取的质量直接影响训练效果
5. **时间步信息**: 每个样本包含时间步信息 `t`，用于模型理解当前进度

---

## 8. 相关文件位置

- **数据集创建**: `sam2act/utils/get_dataset.py`
- **Replay填充**: `sam2act/utils/dataset.py` - `fill_replay_temporal()`
- **样本生成**: `sam2act/utils/dataset.py` - `_add_keypoints_to_replay_temporal()`
- **Keypoint提取**: `sam2act/libs/peract/helpers/demo_loading_utils.py` - `keypoint_discovery()`
- **观察提取**: `sam2act/libs/peract/helpers/utils.py` - `extract_obs()`
- **动作计算**: `sam2act/utils/dataset.py` - `_get_action()`
