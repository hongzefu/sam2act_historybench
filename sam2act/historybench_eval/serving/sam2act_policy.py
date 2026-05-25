"""SAM2Act / SAM2Act+ policy wrapper for the WebSocket serving protocol.

Ports the heavy inference logic from
``sam2act/historybench_eval/agent_api_serverv7.5clearMem-parallel2gpu-seed.py``
(``extract_obs`` + agent loading + the ``/act`` body + ``/reset_memory``) into a
policy object exposing ``reset`` / ``add_buffer`` / ``infer`` so it can be served
by ``websocket_policy_server.WebsocketPolicyServer`` and driven by the robomme
``examples/sam2act`` eval client.

The model I/O contract is preserved verbatim:
- input  ``obs_obj``: image/base_camera_depth/wrist_image/wrist_camera_depth (128x128),
  robot_endeffector_p/q, gripper_joint_positions, base/wrist_camera_intrinsic/extrinsic_opencv, misc;
- output ``action``: 9-dim ``[x, y, z, qw, qx, qy, qz, gripper, collision]`` (see agent).
"""

import os
import sys
import time
import random
from typing import Any, Dict as DictType

import numpy as np
import torch
import torch.nn.functional as F

# --- path setup: mirror v7.5 so `mvt` / `sam2act` modules import correctly ---
# This file lives at sam2act/historybench_eval/serving/sam2act_policy.py;
# the `sam2act` package dir is three levels up from this file's dir.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sam2act_dir = os.path.dirname(os.path.dirname(_this_dir))  # .../sam2act
if _sam2act_dir not in sys.path:
    sys.path.insert(0, _sam2act_dir)

from sam2act.eval import load_agent
from rlbench.backend.observation import Observation
import clip
from pyrep.objects import VisionSensor


def set_seed(seed):
    """Set all random seeds (mirrors v7.5)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[SAM2ActPolicy] random seed set to: {seed}")


def convert_camera_matrix_maniskill_to_coppeliasim(extrinsics_opencv, intrinsics_opencv):
    """Maniskill/OpenCV camera params -> CoppeliaSim convention.

    OpenCV extrinsics are world->camera; CoppeliaSim wants camera->world (inverse).
    Intrinsics use the same pinhole model.
    """
    extrinsics_coppeliasim = np.linalg.inv(extrinsics_opencv)
    intrinsics_coppeliasim = intrinsics_opencv.copy()
    return extrinsics_coppeliasim, intrinsics_coppeliasim


def extract_obs(obs_dict: DictType[str, Any], curr_idx: int, lang_goal, episode_length: int = 25,
                time_in_state: bool = True):
    """Build the feature dict consumed by ``agent.act`` (ported verbatim from v7.5)."""
    obs = Observation(
        left_shoulder_rgb=None, left_shoulder_depth=None, left_shoulder_mask=None, left_shoulder_point_cloud=None,
        right_shoulder_rgb=None, right_shoulder_depth=None, right_shoulder_mask=None, right_shoulder_point_cloud=None,
        overhead_rgb=None, overhead_depth=None, overhead_mask=None, overhead_point_cloud=None,
        wrist_rgb=None, wrist_depth=None, wrist_mask=None, wrist_point_cloud=None,
        front_rgb=None, front_depth=None, front_mask=None, front_point_cloud=None,
        joint_velocities=None, joint_positions=None, joint_forces=None,
        gripper_open=None, gripper_pose=None, gripper_matrix=None, gripper_joint_positions=None, gripper_touch_forces=None,
        task_low_dim_state=None, ignore_collisions=None, misc=None
    )

    for key, val in obs_dict.items():
        if hasattr(obs, key):
            if isinstance(val, list):
                val = np.array(val)
            setattr(obs, key, val)

    if 'misc' in obs_dict:
        obs.misc = obs_dict['misc']
    else:
        obs.misc = {}

    # --- 1. RGB ---
    if 'image' in obs_dict:
        obs.front_rgb = np.array(obs_dict['image'])
    if 'wrist_image' in obs_dict:
        obs.wrist_rgb = np.array(obs_dict['wrist_image'])

    # --- 2. Depth (resampled to 128x128) ---
    if 'base_camera_depth' in obs_dict:
        depth = np.array(obs_dict['base_camera_depth']).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        obs.front_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

    if 'wrist_camera_depth' in obs_dict:
        depth = np.array(obs_dict['wrist_camera_depth']).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(depth_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        obs.wrist_depth = depth_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

    # --- 3. Robot state ---
    if 'robot_endeffector_p' in obs_dict and 'robot_endeffector_q' in obs_dict:
        gripper_pos = np.array(obs_dict['robot_endeffector_p']).flatten()
        gripper_quat = np.array(obs_dict['robot_endeffector_q']).flatten()
        obs.gripper_pose = np.concatenate([gripper_pos, gripper_quat])

    if obs.ignore_collisions is None:
        obs.ignore_collisions = 0

    # --- 4. Camera params ---
    if 'base_camera_extrinsic_opencv' in obs_dict and 'base_camera_intrinsic_opencv' in obs_dict:
        ext_opencv = np.array(obs_dict['base_camera_extrinsic_opencv'])
        intr_opencv = np.array(obs_dict['base_camera_intrinsic_opencv'])
        if ext_opencv.ndim == 3:
            ext_opencv = ext_opencv[0]
        if intr_opencv.ndim == 3:
            intr_opencv = intr_opencv[0]
        if ext_opencv.shape == (3, 4):
            ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
        ext_cs, intr_cs = convert_camera_matrix_maniskill_to_coppeliasim(ext_opencv, intr_opencv)
        obs.misc['front_camera_extrinsics'] = ext_cs
        obs.misc['front_camera_intrinsics'] = intr_cs

    if 'wrist_camera_extrinsic_opencv' in obs_dict and 'wrist_camera_intrinsic_opencv' in obs_dict:
        ext_opencv = np.array(obs_dict['wrist_camera_extrinsic_opencv'])
        intr_opencv = np.array(obs_dict['wrist_camera_intrinsic_opencv'])
        if ext_opencv.ndim == 3:
            ext_opencv = ext_opencv[0]
        if intr_opencv.ndim == 3:
            intr_opencv = intr_opencv[0]
        if ext_opencv.shape == (3, 4):
            ext_opencv = np.vstack([ext_opencv, [0, 0, 0, 1]])
        ext_cs, intr_cs = convert_camera_matrix_maniskill_to_coppeliasim(ext_opencv, intr_opencv)
        obs.misc['wrist_camera_extrinsics'] = ext_cs
        obs.misc['wrist_camera_intrinsics'] = intr_cs

    # --- 5. Point clouds from depth + camera params ---
    if obs.front_point_cloud is None and 'front_camera_extrinsics' in obs.misc and obs.front_depth is not None:
        obs.front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs.front_depth, obs.misc['front_camera_extrinsics'], obs.misc['front_camera_intrinsics'])

    if obs.wrist_point_cloud is None and 'wrist_camera_extrinsics' in obs.misc and obs.wrist_depth is not None:
        obs.wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs.wrist_depth, obs.misc['wrist_camera_extrinsics'], obs.misc['wrist_camera_intrinsics'])

    if obs.front_point_cloud is None:
        obs.front_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)
    if obs.wrist_point_cloud is None:
        obs.wrist_point_cloud = np.zeros((128, 128, 3), dtype=np.float32)

    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions

    obs.joint_velocities = None
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None

    channels_last = False
    obs_dict_extracted = vars(obs)
    obs_dict_extracted = {k: v for k, v in obs_dict_extracted.items() if v is not None}

    ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']
    obs_dict_extracted = {k: v for k, v in obs_dict_extracted.items() if k not in ROBOT_STATE_KEYS}

    if not channels_last:
        new_obs_dict = {}
        for k, v in obs_dict_extracted.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 3:        # (H, W, 3) RGB or point cloud
                    new_obs_dict[k] = np.transpose(v, [2, 0, 1])
                elif v.ndim == 2:      # (H, W) depth
                    new_obs_dict[k] = np.expand_dims(v, 0)
                else:
                    new_obs_dict[k] = np.expand_dims(v, 0) if v.ndim == 0 else v
            else:
                new_obs_dict[k] = v
        obs_dict_extracted = new_obs_dict
    else:
        new_obs_dict = {}
        for k, v in obs_dict_extracted.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 2:
                    new_obs_dict[k] = np.expand_dims(v, -1)
                else:
                    new_obs_dict[k] = v
            else:
                new_obs_dict[k] = v
        obs_dict_extracted = new_obs_dict

    obs_dict_extracted['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)

    for k, v in obs_dict_extracted.items():
        if 'point_cloud' in k and isinstance(v, np.ndarray):
            obs_dict_extracted[k] = v.astype(np.float32)

    camera_names = ['left_shoulder', 'right_shoulder', 'front', 'wrist', 'overhead']
    for name in camera_names:
        if '%s_camera_extrinsics' % name in obs.misc:
            obs_dict_extracted['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
        if '%s_camera_intrinsics' % name in obs.misc:
            obs_dict_extracted['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]

    # --- low_dim_state: [gripper_open, left_finger, right_finger, (time)] ---
    left_finger = obs.gripper_joint_positions[0] if obs.gripper_joint_positions is not None else 0.0
    right_finger = obs.gripper_joint_positions[1] if obs.gripper_joint_positions is not None else 0.0

    if left_finger <= 0.035 or right_finger <= 0.035:
        gripper_open_val = 0   # strictly: even slightly closed counts as closed
    else:
        gripper_open_val = 1

    print(f"[SAM2ActPolicy] current gripper_open: {'open' if gripper_open_val == 1 else 'close'}")
    if time_in_state:
        time_feat = (1. - (curr_idx / float(episode_length - 1))) * 2. - 1.
        obs_dict_extracted['low_dim_state'] = np.array(
            [gripper_open_val, left_finger, right_finger, time_feat], dtype=np.float32)
    else:
        obs_dict_extracted['low_dim_state'] = np.array(
            [gripper_open_val, left_finger, right_finger], dtype=np.float32)

    if lang_goal is not None:
        tokens = clip.tokenize([lang_goal]).numpy()
        obs_dict_extracted['lang_goal_tokens'] = tokens

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict_extracted


class SAM2ActPolicy:
    """SAM2Act policy served over WebSocket (reset / add_buffer / infer)."""

    def __init__(self, model_folder, model_name, device=0, exp_cfg_path=None,
                 mvt_cfg_path=None, use_input_place_with_mean=False, seed=0):
        self.device = device
        if seed is not None:
            set_seed(seed)

        model_path = os.path.join(model_folder, model_name) if model_name else None
        print(f"[SAM2ActPolicy] loading agent: {model_path} on cuda:{device}")
        agent = load_agent(
            model_path=model_path,
            peract_official=False,
            peract_model_dir=None,
            exp_cfg_path=exp_cfg_path,
            mvt_cfg_path=mvt_cfg_path,
            eval_log_dir="",
            device=device,
            use_input_place_with_mean=use_input_place_with_mean,
        )
        agent.eval()
        if hasattr(agent, 'load_clip'):
            agent.load_clip()
        self.agent = agent
        self.metadata = {
            "policy": "sam2act",
            "model_folder": str(model_folder),
            "model_name": str(model_name),
        }
        print("[SAM2ActPolicy] agent ready")

    # -- protocol: reset --------------------------------------------------
    def reset(self) -> None:
        """Clear SAM2Act+ memory banks (episode boundary)."""
        net = getattr(self.agent, '_network', None)
        if net is not None:
            if hasattr(net, 'mvt1'):
                net.mvt1.reset_memory_bank()
            if hasattr(net, 'mvt2'):
                net.mvt2.reset_memory_bank()
        print("[SAM2ActPolicy] memory banks reset")

    # -- shared act (also accumulates memory) -----------------------------
    def _act_once(self, obs_obj, curr_idx, lang_goal, episode_length, deterministic=True):
        observation = extract_obs(obs_obj, curr_idx, lang_goal, episode_length)
        with torch.cuda.device(self.device):
            prepped_data = {}
            for key, val in observation.items():
                if isinstance(val, list):
                    val = np.array(val)
                elif not isinstance(val, np.ndarray):
                    val = np.array(val)

                if key == 'lang_goal_tokens':
                    if val.dtype in [np.float64, np.float32]:
                        val = val.astype(np.int64)
                    elif val.dtype not in [np.int64, np.int32]:
                        val = val.astype(np.int64)
                else:
                    if val.dtype == np.float64:
                        val = val.astype(np.float32)
                    elif val.dtype in [np.int64, np.int32, np.int16, np.int8]:
                        val = val.astype(np.float32)

                if key == 'lang_goal_tokens':
                    val_tensor = torch.tensor(np.array([val]), device=f"cuda:{self.device}", dtype=torch.long)
                else:
                    val_tensor = torch.tensor(np.array([val]), device=f"cuda:{self.device}", dtype=torch.float32)

                if key != 'lang_goal_tokens':
                    val_tensor = val_tensor.unsqueeze(1)  # add time dimension

                prepped_data[key] = val_tensor

            with torch.no_grad():
                act_result = self.agent.act(curr_idx, prepped_data, deterministic=deterministic)

        pred_action = act_result.action
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.cpu().numpy()
        if isinstance(pred_action, np.ndarray):
            pred_action = pred_action.tolist()
        if isinstance(pred_action, list) and len(pred_action) > 0 and isinstance(pred_action[0], (torch.Tensor, np.ndarray)):
            pred_action = [float(x.item() if hasattr(x, 'item') else x) for x in pred_action]
        else:
            pred_action = [float(x) for x in pred_action]
        return pred_action

    # -- protocol: add_buffer (feed demo/history frames to build memory) --
    def add_buffer(self, obs: dict) -> None:
        frames = obs.get("frames", [])
        episode_length = int(obs.get("episode_length", 25))
        lang_goal = obs.get("lang_goal", None)
        for i, fr in enumerate(frames):
            f_obs = fr["obs_obj"]
            f_idx = int(fr.get("curr_idx", i))
            f_lang = fr.get("lang_goal", lang_goal)
            f_len = int(fr.get("episode_length", episode_length))
            # run act() only to accumulate memory; discard the action
            self._act_once(f_obs, f_idx, f_lang, f_len)
        print(f"[SAM2ActPolicy] add_buffer: fed {len(frames)} demo frames into memory")

    # -- protocol: infer --------------------------------------------------
    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()
        action = self._act_once(
            obs["obs_obj"],
            int(obs.get("curr_idx", 0)),
            obs.get("lang_goal", None),
            int(obs.get("episode_length", 25)),
            deterministic=bool(obs.get("deterministic", True)),
        )
        return {
            "actions": [action],  # single 9-dim waypoint (keyframe); chunk length 1
            "infer_time_ms": (time.monotonic() - t0) * 1000,
        }
