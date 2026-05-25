"""RoboMME environment wrapper for SAM2Act (waypoint action space).

Mirrors examples/robomme/env_runner.py, but:
- uses ``action_space="waypoint"`` so the benchmark's built-in screw->RRT* planner
  executes each end-effector keyframe (NO hand-written planner here);
- enables ``include_maniskill_obs`` so we get raw per-frame observations (RGB+depth+
  camera params) for SAM2Act's point-cloud input, both for demo replay and eval;
- converts SAM2Act's 9-dim action ``[x,y,z, qw,qx,qy,qz, grip, coll]`` into the
  benchmark's 7-dim waypoint ``[x,y,z, roll,pitch,yaw, grip(-1/+1)]``.

This is the ONLY file that imports the third-party ``robomme`` package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R

from robomme.robomme_env import *  # noqa: F401, F403 - registers gym envs
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from utils import TASK_NAME_LIST


def _scalar(x) -> bool:
    """Coerce a possibly-tensor terminated/truncated flag to bool."""
    if hasattr(x, "item"):
        try:
            return bool(x.item())
        except Exception:
            return bool(np.asarray(x).reshape(-1)[0])
    if isinstance(x, np.ndarray):
        return bool(x.reshape(-1)[0]) if x.size else False
    return bool(x)


def action9_to_waypoint7(action) -> np.ndarray:
    """SAM2Act 9-dim action -> benchmark 7-dim waypoint.

    Goes through the EXACT same pose interpretation as the validated legacy client
    (``action_to_pose``), then expresses it as [x,y,z, rpy(xyz euler), gripper].
    The MultiStepDemonstrationWrapper converts rpy->quat (scipy 'xyz') internally,
    so this round-trips to the same sapien.Pose the legacy planner received.
    """
    action = np.asarray(action, dtype=np.float64).flatten()
    pos = action[:3]
    qx, qy, qz, qw = action[3:7]
    pose = sapien.Pose(p=pos, q=[qw, qx, qy, qz])  # SAPIEN wants [w, x, y, z]
    q = np.asarray(pose.q, dtype=np.float64).flatten()  # [w, x, y, z]
    rpy = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")  # scipy wants [x,y,z,w]
    grip = -1.0 if float(action[7]) <= 0.5 else 1.0  # -1 close / +1 open
    p = np.asarray(pose.p, dtype=np.float64).flatten()
    return np.concatenate([p, rpy, [grip]]).astype(np.float64)


class SAM2ActEnvRunner:
    """Wraps robomme BenchmarkEnvBuilder(waypoint) for a single task."""

    def __init__(self, env_id: str, sim_step_budget: int = 100000) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        # IMPORTANT: this is the SIM-step budget (max_steps_without_demonstration),
        # NOT the number of waypoints. One waypoint = a full screw/RRT* motion of
        # many sim substeps, so this must be large (matches the legacy client's
        # max_steps_without_demo=100000). The waypoint-count cap lives in eval.py.
        self.sim_step_budget = sim_step_budget
        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space="waypoint",
            gui_render=False,
            max_steps=sim_step_budget,
        )
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""
        self.info: dict = {}

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        # include_maniskill_obs gives raw per-frame obs (sensor_data/param) needed
        # to build SAM2Act point-cloud inputs for both demo replay and eval steps.
        self.env = self.env_builder.make_env_for_episode(
            episode_id,
            max_steps=self.sim_step_budget,
            include_maniskill_obs=True,
        )
        self.episode_id = episode_id

    def get_init_obs(self) -> dict:
        """Reset (auto-runs the demo) and return demo frames + goal + current obs."""
        obs_batch, self.info = self.env.reset()
        tg = self.info.get("task_goal", "")
        self.task_goal = tg[0] if isinstance(tg, list) else tg
        demo_frames = []
        if isinstance(obs_batch, dict):
            demo_frames = [f for f in obs_batch.get("maniskill_obs", []) if isinstance(f, dict)]
        cur_obs = self.env.unwrapped.get_obs()
        return {"demo_frames": demo_frames, "task_goal": self.task_goal, "cur_obs": cur_obs}

    def step(self, action9) -> tuple[Any, bool, str]:
        """Execute one SAM2Act waypoint via the benchmark's built-in planner.

        Returns (current_raw_obs, stop_flag, status). status in
        {success, fail, timeout, ongoing, unknown, error}.
        """
        waypoint = action9_to_waypoint7(action9)
        try:
            _, _, terminated, truncated, self.info = self.env.step(waypoint)
        except Exception as e:  # planner exhausted, etc.
            print(f"[env_runner] step error: {e}")
            return None, True, "error"
        status = self.info.get("status", "unknown")
        stop = _scalar(terminated) or _scalar(truncated)
        cur_obs = self.env.unwrapped.get_obs()
        return cur_obs, stop, status

    def close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.episode_id = None
