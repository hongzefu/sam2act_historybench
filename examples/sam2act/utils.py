"""Utilities for the SAM2Act robomme eval client.

Mirrors examples/robomme/utils.py in spirit, but provides ``build_obs_obj`` that
turns a raw ManiSkill observation into the dict SAM2Act's WebSocket server expects
(the same contract as the legacy ``get_model_input`` in
``robomme-sam2act/scripts/eval_binfill_test_client.py``), additionally resizing
RGB/depth to 128x128 and scaling camera intrinsics, since the robomme env renders
at its default resolution (256) while SAM2Act needs 128 (IMAGE_SIZE=128).
"""

import numpy as np
import cv2

TASK_NAME_LIST = [
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]

IMAGE_SIZE = 128


def sample_indices(n, k):
    """Evenly pick k indices from range(n). k<=0 or k>=n means every index."""
    if n <= 0:
        return []
    if k <= 0 or k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _drop_batch(x):
    """Drop a leading singleton batch dim if present."""
    x = _to_np(x)
    if x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    return x


def _resize_rgb(rgb):
    if rgb.shape[0] == IMAGE_SIZE and rgb.shape[1] == IMAGE_SIZE:
        return rgb
    return cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _resize_depth(depth):
    if depth.shape[0] == IMAGE_SIZE and depth.shape[1] == IMAGE_SIZE:
        return depth
    return cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def _scale_intrinsic(K, src_w, src_h):
    """Scale a 3x3 OpenCV intrinsic matrix to the IMAGE_SIZE resolution."""
    K = np.asarray(K, dtype=np.float64).copy()
    if K.shape != (3, 3):
        K = K.reshape(3, 3)
    K[0, :] *= IMAGE_SIZE / float(src_w)   # fx, skew, cx
    K[1, :] *= IMAGE_SIZE / float(src_h)   # fy, cy
    return K


def build_obs_obj(env, maniskill_obs):
    """Raw ManiSkill obs -> SAM2Act obs_obj (128x128, intrinsics scaled).

    Replicates the validated ``get_model_input`` field-by-field (end-effector pose
    is read from the live agent, exactly like the legacy client), then resizes the
    images/depth to 128 and rescales the intrinsics so the server's point cloud is
    geometrically correct.
    """
    sensor_data = maniskill_obs["sensor_data"]
    sensor_param = maniskill_obs["sensor_param"]
    agent = env.unwrapped.agent

    obs_obj = {}

    def _cam(cam):
        rgb = _drop_batch(sensor_data[cam]["rgb"])
        depth = _drop_batch(sensor_data[cam]["depth"])
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth / 1000.0
        return np.asarray(rgb).astype(np.uint8), depth

    base_src = None
    if "base_camera" in sensor_data:
        rgb, depth = _cam("base_camera")
        base_src = (rgb.shape[1], rgb.shape[0])  # (w, h)
        obs_obj["image"] = _resize_rgb(rgb)
        obs_obj["base_camera_depth"] = _resize_depth(depth)
    hand_src = None
    if "hand_camera" in sensor_data:
        rgb, depth = _cam("hand_camera")
        hand_src = (rgb.shape[1], rgb.shape[0])
        obs_obj["wrist_image"] = _resize_rgb(rgb)
        obs_obj["wrist_camera_depth"] = _resize_depth(depth)

    qpos = _drop_batch(maniskill_obs["agent"]["qpos"])
    qpos = np.asarray(qpos).flatten()
    if len(qpos) >= 7:
        obs_obj["joint_positions"] = qpos[:7]
    if len(qpos) >= 9:
        obs_obj["gripper_joint_positions"] = qpos[7:9]
    gripper_width = float(np.sum(obs_obj.get("gripper_joint_positions", [0.0, 0.0])))
    obs_obj["gripper_open"] = 1.0 if gripper_width > 0.002 else 0.0

    position = _to_np(agent.tcp.pose.p).flatten()
    q_wxyz = _to_np(agent.tcp.pose.q).flatten()  # SAPIEN order [w, x, y, z]
    obs_obj["robot_endeffector_p"] = position
    obs_obj["robot_endeffector_q"] = np.array(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # -> [x, y, z, w]
    obs_obj["ignore_collisions"] = 1

    if "base_camera" in sensor_param and base_src is not None:
        K = _drop_batch(sensor_param["base_camera"]["intrinsic_cv"])
        E = _drop_batch(sensor_param["base_camera"]["extrinsic_cv"])
        obs_obj["base_camera_intrinsic_opencv"] = _scale_intrinsic(K, base_src[0], base_src[1])
        obs_obj["base_camera_extrinsic_opencv"] = np.asarray(E, dtype=np.float64)
    if "hand_camera" in sensor_param and hand_src is not None:
        K = _drop_batch(sensor_param["hand_camera"]["intrinsic_cv"])
        E = _drop_batch(sensor_param["hand_camera"]["extrinsic_cv"])
        obs_obj["wrist_camera_intrinsic_opencv"] = _scale_intrinsic(K, hand_src[0], hand_src[1])
        obs_obj["wrist_camera_extrinsic_opencv"] = np.asarray(E, dtype=np.float64)

    obs_obj["misc"] = {}
    return obs_obj
