# 函数定义对比报告

## 测试文件: test_get_dataset_temporalv5.py
## 库路径: sam2act/

---

## 1. normalize_quaternion

**测试文件 (99-107行):**
```python
def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
```

**库文件 (sam2act/libs/peract/helpers/utils.py:43-44):**
```python
def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
```

**✅ 一致**

---

## 2. quaternion_to_discrete_euler

**测试文件 (109-123行):**
```python
def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc
```

**库文件 (sam2act/libs/peract/helpers/utils.py:72-77):**
```python
def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc
```

**✅ 一致**

---

## 3. point_to_voxel_index

**测试文件 (125-146行):**
```python
def point_to_voxel_index(
    point: np.ndarray,
    voxel_size: np.ndarray,
    coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy
```

**库文件 (sam2act/libs/peract/helpers/utils.py:84-96):**
```python
def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy
```

**✅ 一致**

---

## 4. extract_obs

**测试文件 (148-232行):**
- 包含详细注释
- 处理逻辑相同

**库文件 (sam2act/libs/peract/helpers/utils.py:322-378):**
- 实现逻辑相同，但测试文件有更详细的注释

**✅ 功能一致** (测试文件有更详细的注释)

---

## 5. _is_stopped

**测试文件 (238-252行):**
```python
def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped
```

**库文件 (sam2act/libs/peract/helpers/demo_loading_utils.py:8-18):**
```python
def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped
```

**✅ 一致**

---

## 6. keypoint_discovery

**测试文件 (254-308行):**
- 包含 'heuristic', 'random', 'fixed_interval' 三种方法
- 有详细注释

**库文件 (sam2act/libs/peract/helpers/demo_loading_utils.py:21-62):**
- 实现逻辑相同，三种方法都支持

**✅ 功能一致** (测试文件有更详细的注释)

---

## 7. get_stored_demo

**测试文件 (314-374行):**
- 加载 RGB、深度图、点云
- 处理深度图的反归一化

**库文件 (sam2act/libs/peract_colab/peract_colab/rlbench/utils.py:31-87):**
- 实现逻辑相同

**✅ 功能一致**

---

## 8. _get_action

**测试文件 (376-424行):**
```python
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes):
        index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)
    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )
```

**库文件 (sam2act/utils/dataset.py:292-326):**
```python
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes):
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)
    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )
```

**⚠️ 差异:** 库文件使用 `utils.normalize_quaternion` 等，测试文件直接调用函数（因为函数在同一个文件中定义）

**✅ 功能一致** (只是导入方式不同)

---

## 9. _clip_encode_text

**测试文件 (426-440行):**
```python
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return x, emb
```

**库文件 (sam2act/utils/dataset.py:330-344):**
```python
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return x, emb
```

**✅ 一致** (库文件有额外注释)

---

## 10. create_replay_temporal

**测试文件 (442-528行):**
- 定义 observation_elements 和 extra_replay_elements
- 创建 UniformReplayBuffer_temporal

**库文件 (sam2act/utils/dataset.py:158-287):**
- 实现逻辑相同

**✅ 功能一致**

---

## 11. _add_keypoints_to_replay_temporal

**测试文件 (530-664行):**
- 包含 `initial_frame` 参数处理
- 在 others 字典中包含 `initial_frame`

**库文件 (sam2act/utils/dataset.py:566-675):**
- 实现逻辑相同，也包含 `initial_frame` 处理

**✅ 功能一致**

---

## 12. fill_replay_temporal

**测试文件 (666-777行):**
- 包含 `rank` 参数，用于控制打印信息
- 在 rank == 0 时打印信息

**库文件 (sam2act/utils/dataset.py:678-783):**
- 实现逻辑相同，也包含 `rank` 参数处理

**✅ 功能一致**

---

## 13. get_dataset_temporal

**测试文件 (780-931行):**
- 创建 replay buffer
- 加载 CLIP 模型
- 遍历任务并填充 buffer
- 包装为 PyTorch Dataset

**库文件 (sam2act/utils/get_dataset.py:156-291):**
- 实现逻辑相同

**✅ 功能一致**

---

## 总结

### ✅ 所有函数功能一致

所有函数的核心逻辑都与库中的实现一致。主要差异在于：

1. **导入方式**: 测试文件中直接定义了辅助函数（如 `normalize_quaternion`），而库中通过 `utils.` 前缀调用
2. **注释详细程度**: 测试文件包含更详细的中文注释，便于理解
3. **代码组织**: 测试文件将所有相关函数集中在一个文件中，便于测试和调试

### 建议

测试文件中的函数定义与库中的实现保持一致，可以放心使用。测试文件作为一个独立的测试脚本，包含了完整的数据处理流程，适合用于调试和验证。
