# `get_stored_demo` 函数数据格式详解

本文档详细解释 `get_stored_demo` 函数加载的数据格式和结构。

## 1. 函数概述

**位置**: `sam2act/libs/peract_colab/peract_colab/rlbench/utils.py`

**功能**: 从磁盘加载存储的RLBench演示数据（demo），包括低维观察数据和所有相机的高维图像数据。

---

## 2. 文件结构

### 2.1 目录结构

```
data_path/
└── episode{index}/              # 例如: episode0, episode1, ...
    ├── low_dim_obs.pkl         # 低维观察数据（pickle格式）
    ├── variation_number.pkl    # 变体编号
    ├── front_rgb/              # 前相机RGB图像文件夹
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    ├── front_depth/            # 前相机深度图像文件夹
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    ├── left_shoulder_rgb/      # 左肩相机RGB图像文件夹
    │   └── ...
    ├── left_shoulder_depth/    # 左肩相机深度图像文件夹
    │   └── ...
    ├── right_shoulder_rgb/     # 右肩相机RGB图像文件夹
    │   └── ...
    ├── right_shoulder_depth/   # 右肩相机深度图像文件夹
    │   └── ...
    └── wrist_rgb/             # 腕部相机RGB图像文件夹
        └── ...
```

### 2.2 常量定义

```python
EPISODE_FOLDER = 'episode%d'           # Episode文件夹格式
LOW_DIM_PICKLE = 'low_dim_obs.pkl'     # 低维观察pickle文件
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'  # 变体编号文件
IMAGE_FORMAT = '%d.png'                 # 图像文件名格式（0.png, 1.png, ...）
DEPTH_SCALE = 2**24 - 1                # 深度图像缩放因子 (16777215)

# 相机名称
CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
```

---

## 3. 数据加载流程

### 3.1 步骤1: 加载低维观察数据

```python
episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
# 例如: data_path/episode0

with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
    obs = pickle.load(f)
```

**说明**:
- `obs` 是一个 `Demo` 对象（实际上是 `List[Observation]`）
- 每个 `Observation` 包含除图像外的所有观察数据
- 图像数据在保存时被设置为 `None`，以节省存储空间

**低维观察数据包含**:
- 关节位置 (`joint_positions`)
- 关节速度 (`joint_velocities`)
- 关节力 (`joint_forces`)
- 夹爪状态 (`gripper_open`, `gripper_pose`, `gripper_matrix`)
- 夹爪关节位置 (`gripper_joint_positions`)
- 任务低维状态 (`task_low_dim_state`)
- 相机参数 (`misc` 字典中的相机内外参、near/far平面等)

### 3.2 步骤2: 加载变体编号

```python
with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
    obs.variation_number = pickle.load(f)
```

**说明**:
- 变体编号表示任务的变体版本（同一任务可能有多个变体）

### 3.3 步骤3: 加载图像数据

对每个时间步 `i`，加载4个相机的RGB和深度图像：

#### 3.3.1 RGB图像加载

```python
obs[i].front_rgb = np.array(Image.open(
    os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), 
                 IMAGE_FORMAT % i)
))
# 例如: episode0/front_rgb/0.png
```

**数据格式**:
- **类型**: `numpy.ndarray`
- **形状**: `(H, W, 3)` - 例如 `(128, 128, 3)` 或 `(256, 256, 3)`
- **数据类型**: `uint8`
- **值范围**: `[0, 255]`
- **通道顺序**: RGB

**所有RGB图像**:
- `obs[i].front_rgb`: 前相机RGB图像
- `obs[i].left_shoulder_rgb`: 左肩相机RGB图像
- `obs[i].right_shoulder_rgb`: 右肩相机RGB图像
- `obs[i].wrist_rgb`: 腕部相机RGB图像

#### 3.3.2 深度图像加载和转换

```python
# 1. 从PNG图像加载深度数据（24位RGB编码）
obs[i].front_depth = image_to_float_array(
    Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), 
                           IMAGE_FORMAT % i)),
    DEPTH_SCALE  # 2**24 - 1 = 16777215
)

# 2. 从归一化深度值转换为实际深度值（米）
near = obs[i].misc['front_camera_near']  # 近平面距离（米）
far = obs[i].misc['front_camera_far']   # 远平面距离（米）
obs[i].front_depth = near + obs[i].front_depth * (far - near)
```

**深度图像编码说明**:
- 深度值以24位RGB格式编码在PNG图像中
- R通道 = 高字节，G通道 = 中字节，B通道 = 低字节
- `image_to_float_array()` 将RGB编码转换为归一化的浮点深度值 `[0.0, 1.0]`
- 然后使用相机的 `near` 和 `far` 平面将归一化值转换为实际深度值（米）

**数据格式**:
- **类型**: `numpy.ndarray`
- **形状**: `(H, W)` - 例如 `(128, 128)` 或 `(256, 256)`
- **数据类型**: `float32`
- **值范围**: `[near, far]` (米) - 例如 `[0.1, 2.0]`
- **单位**: 米 (meters)

**所有深度图像**:
- `obs[i].front_depth`: 前相机深度图像
- `obs[i].left_shoulder_depth`: 左肩相机深度图像
- `obs[i].right_shoulder_depth`: 右肩相机深度图像
- `obs[i].wrist_depth`: 腕部相机深度图像

#### 3.3.3 点云生成

```python
obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
    obs[i].front_depth,                                    # 深度图像 (H, W)
    obs[i].misc['front_camera_extrinsics'],               # 相机外参 (4, 4)
    obs[i].misc['front_camera_intrinsics']                # 相机内参 (3, 3)
)
```

**点云生成原理**:
1. 使用深度图像和相机内参，将每个像素的 `(u, v, depth)` 转换为相机坐标系下的3D点
2. 使用相机外参，将相机坐标系下的点转换到世界坐标系

**数据格式**:
- **类型**: `numpy.ndarray`
- **形状**: `(3, H, W)` - 例如 `(3, 128, 128)`
- **数据类型**: `float32`
- **通道含义**:
  - `[0, :, :]`: X坐标（米）
  - `[1, :, :]`: Y坐标（米）
  - `[2, :, :]`: Z坐标（米）
- **坐标系**: 世界坐标系（world frame）

**所有点云**:
- `obs[i].front_point_cloud`: 前相机点云
- `obs[i].left_shoulder_point_cloud`: 左肩相机点云
- `obs[i].right_shoulder_point_cloud`: 右肩相机点云
- `obs[i].wrist_point_cloud`: 腕部相机点云

---

## 4. Observation对象完整数据结构

### 4.1 视觉观察数据（高维）

#### RGB图像
```python
obs[i].front_rgb              # (H, W, 3) uint8, [0, 255]
obs[i].left_shoulder_rgb      # (H, W, 3) uint8, [0, 255]
obs[i].right_shoulder_rgb     # (H, W, 3) uint8, [0, 255]
obs[i].wrist_rgb              # (H, W, 3) uint8, [0, 255]
```

#### 深度图像
```python
obs[i].front_depth             # (H, W) float32, [near, far] 米
obs[i].left_shoulder_depth     # (H, W) float32, [near, far] 米
obs[i].right_shoulder_depth    # (H, W) float32, [near, far] 米
obs[i].wrist_depth             # (H, W) float32, [near, far] 米
```

#### 点云
```python
obs[i].front_point_cloud              # (3, H, W) float32, 世界坐标系
obs[i].left_shoulder_point_cloud     # (3, H, W) float32, 世界坐标系
obs[i].right_shoulder_point_cloud    # (3, H, W) float32, 世界坐标系
obs[i].wrist_point_cloud             # (3, H, W) float32, 世界坐标系
```

### 4.2 机器人状态数据（低维）

#### 关节信息
```python
obs[i].joint_positions    # (7,) float32 - 7个关节的位置（弧度）
obs[i].joint_velocities   # (7,) float32 - 7个关节的速度（弧度/秒）
obs[i].joint_forces       # (7,) float32 - 7个关节的力（牛顿）
```

#### 夹爪信息
```python
obs[i].gripper_open              # float - 夹爪开合状态 (0.0=关闭, 1.0=打开)
obs[i].gripper_pose              # (7,) float32 - 夹爪位姿 [x, y, z, qx, qy, qz, qw]
                                  # 前3个是位置（米），后4个是四元数旋转
obs[i].gripper_matrix            # (4, 4) float32 - 夹爪变换矩阵（齐次变换矩阵）
obs[i].gripper_joint_positions   # (2,) float32 - 左右手指关节位置（米）
obs[i].gripper_touch_forces      # (2,) float32 - 左右手指接触力（牛顿）
```

#### 任务状态
```python
obs[i].task_low_dim_state   # (N,) float32 - 任务特定的低维状态（取决于任务）
obs[i].ignore_collisions    # bool - 是否忽略碰撞检测
```

### 4.3 相机参数（存储在misc字典中）

```python
obs[i].misc = {
    # 相机外参（世界坐标系到相机坐标系的变换矩阵）
    'front_camera_extrinsics': np.array,      # (4, 4) float32
    'left_shoulder_camera_extrinsics': np.array,  # (4, 4) float32
    'right_shoulder_camera_extrinsics': np.array, # (4, 4) float32
    'wrist_camera_extrinsics': np.array,      # (4, 4) float32
    
    # 相机内参（针孔相机模型）
    'front_camera_intrinsics': np.array,      # (3, 3) float32
    'left_shoulder_camera_intrinsics': np.array,  # (3, 3) float32
    'right_shoulder_camera_intrinsics': np.array,   # (3, 3) float32
    'wrist_camera_intrinsics': np.array,     # (3, 3) float32
    
    # 深度范围
    'front_camera_near': float,              # 近平面距离（米）
    'front_camera_far': float,               # 远平面距离（米）
    'left_shoulder_camera_near': float,
    'left_shoulder_camera_far': float,
    'right_shoulder_camera_near': float,
    'right_shoulder_camera_far': float,
    'wrist_camera_near': float,
    'wrist_camera_far': float,
    
    # 其他任务特定信息
    # ...
}
```

#### 相机内参格式
```python
# 典型的相机内参矩阵 (3, 3)
intrinsics = np.array([
    [fx,  0,  cx],  # fx, fy: 焦距（像素）
    [ 0, fy,  cy],  # cx, cy: 主点坐标（像素）
    [ 0,  0,   1]
])
```

#### 相机外参格式
```python
# 齐次变换矩阵 (4, 4)
extrinsics = np.array([
    [R00, R01, R02, tx],  # R: 旋转矩阵 (3x3)
    [R10, R11, R12, ty],  # t: 平移向量 (3x1)
    [R20, R21, R22, tz],
    [  0,   0,   0,  1]
])
# 将世界坐标系下的点转换为相机坐标系下的点
```

### 4.4 Demo对象属性

```python
obs.variation_number  # int - 任务变体编号
len(obs)              # int - Demo中的时间步数量
```

---

## 5. 数据转换细节

### 5.1 深度图像转换流程

```
PNG图像 (24位RGB编码)
    ↓
image_to_float_array()  # 解码RGB为归一化深度值
    ↓
归一化深度值 [0.0, 1.0]
    ↓
near + depth * (far - near)  # 线性映射到实际深度范围
    ↓
实际深度值 [near, far] (米)
```

**示例**:
```python
# 假设 near=0.1米, far=2.0米
# PNG图像中的RGB值 → 归一化深度值 0.5
# → 实际深度 = 0.1 + 0.5 * (2.0 - 0.1) = 1.05米
```

### 5.2 点云生成流程

```
深度图像 (H, W) + 相机内参 (3, 3) + 相机外参 (4, 4)
    ↓
对每个像素 (u, v):
    1. 获取深度值 d = depth[u, v]
    2. 使用内参将像素坐标转换为相机坐标系下的3D点:
       x_cam = (u - cx) * d / fx
       y_cam = (v - cy) * d / fy
       z_cam = d
    3. 使用外参将相机坐标系下的点转换为世界坐标系:
       P_world = extrinsics @ [x_cam, y_cam, z_cam, 1]^T
    ↓
点云 (3, H, W) - 世界坐标系下的3D点
```

---

## 6. 数据使用示例

### 6.1 加载Demo

```python
from peract_colab.rlbench.utils import get_stored_demo

data_path = "/path/to/data/train/task_name/all_variations/episodes"
demo = get_stored_demo(data_path, index=0)

print(f"Demo包含 {len(demo)} 个时间步")
print(f"变体编号: {demo.variation_number}")
```

### 6.2 访问观察数据

```python
# 访问第10个时间步的观察
obs_10 = demo[10]

# RGB图像
rgb = obs_10.front_rgb  # (H, W, 3) uint8

# 深度图像
depth = obs_10.front_depth  # (H, W) float32, 单位: 米

# 点云
point_cloud = obs_10.front_point_cloud  # (3, H, W) float32

# 夹爪位姿
gripper_pose = obs_10.gripper_pose  # [x, y, z, qx, qy, qz, qw]
position = gripper_pose[:3]  # [x, y, z] 位置（米）
quaternion = gripper_pose[3:]  # [qx, qy, qz, qw] 四元数旋转

# 相机参数
intrinsics = obs_10.misc['front_camera_intrinsics']  # (3, 3)
extrinsics = obs_10.misc['front_camera_extrinsics']  # (4, 4)
```

### 6.3 遍历所有时间步

```python
for i, observation in enumerate(demo):
    print(f"时间步 {i}:")
    print(f"  夹爪位置: {observation.gripper_pose[:3]}")
    print(f"  夹爪开合: {observation.gripper_open}")
    print(f"  RGB图像形状: {observation.front_rgb.shape}")
    print(f"  深度图像形状: {observation.front_depth.shape}")
```

---

## 7. 数据格式总结表

| 数据类型 | 属性名 | 形状 | 数据类型 | 值范围/单位 | 说明 |
|---------|--------|------|---------|------------|------|
| RGB图像 | `{camera}_rgb` | (H, W, 3) | uint8 | [0, 255] | RGB颜色图像 |
| 深度图像 | `{camera}_depth` | (H, W) | float32 | [near, far] 米 | 实际深度值 |
| 点云 | `{camera}_point_cloud` | (3, H, W) | float32 | 米 | 世界坐标系3D点 |
| 关节位置 | `joint_positions` | (7,) | float32 | 弧度 | 7个关节角度 |
| 关节速度 | `joint_velocities` | (7,) | float32 | 弧度/秒 | 关节角速度 |
| 夹爪位姿 | `gripper_pose` | (7,) | float32 | 米, 四元数 | [x,y,z,qx,qy,qz,qw] |
| 夹爪开合 | `gripper_open` | () | float | [0.0, 1.0] | 0=关闭, 1=打开 |
| 相机内参 | `misc['{camera}_camera_intrinsics']` | (3, 3) | float32 | - | 针孔相机模型 |
| 相机外参 | `misc['{camera}_camera_extrinsics']` | (4, 4) | float32 | - | 齐次变换矩阵 |

**注**: `{camera}` 可以是: `front`, `left_shoulder`, `right_shoulder`, `wrist`

---

## 8. 注意事项

1. **图像尺寸**: 图像尺寸取决于数据采集时的配置，常见的有 `128x128` 或 `256x256`
2. **坐标系**: 
   - 点云使用世界坐标系
   - 夹爪位姿使用世界坐标系
   - 相机外参定义从世界坐标系到相机坐标系的变换
3. **深度精度**: 深度图像使用24位RGB编码，精度约为 1/256 mm
4. **内存占用**: 加载完整demo会占用大量内存，特别是高分辨率图像
5. **Mask图像**: 代码中mask图像的加载被注释掉了，但文件结构中可能包含mask数据

---

## 9. 相关文件

- **Observation类定义**: `sam2act/libs/RLBench/rlbench/backend/observation.py`
- **深度图像转换**: `sam2act/libs/RLBench/rlbench/backend/utils.py` - `image_to_float_array()`
- **点云生成**: `pyrep.objects.VisionSensor.pointcloud_from_depth_and_camera_params()`
