"""SAM2Act / SAM2Act+ evaluation on RoboMME (waypoint action space, WebSocket).

Mirrors examples/robomme/eval.py structure, adapted for SAM2Act:
- one keyframe (waypoint) per inference (chunk length 1), no action-chunk queue;
- demo frames are replayed via ``client.add_buffer`` to populate SAM2Act+ memory;
- actions are fed to the benchmark's built-in waypoint planner (no custom planner).

Per-episode flow:
  client.reset()                      -> server clears memory banks
  env.reset() (runs demo)             -> demo frames (maniskill_obs) + task goal
  client.add_buffer({frames...})      -> server replays frames to accumulate memory
  loop: build_obs_obj -> client.infer -> env_runner.step(waypoint) -> status

Run (in the robomme micromamba env), with the SAM2Act server already up:
  CUDA_VISIBLE_DEVICES=1 python examples/sam2act/eval.py \
      --host 127.0.0.1 --port 8001 --only_tasks BinFill --max_episodes 5
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy

from env_runner import SAM2ActEnvRunner
from utils import TASK_NAME_LIST, build_obs_obj, sample_indices


def eval_episode(client, runner, episode_id, max_steps, history_frames):
    """Run a single episode end-to-end; return a result dict."""
    result = {"episode": episode_id, "status": "error", "steps": 0, "difficulty": None}

    runner.make_env(episode_id)
    try:
        result["difficulty"] = runner.env.unwrapped.difficulty
    except Exception:
        pass

    # 1. clear server memory
    client.reset()

    # 2. reset env -> demo segment + current obs
    init = runner.get_init_obs()
    lang_goal = init["task_goal"]
    episode_length = max_steps  # matches the legacy client's curr_idx/time semantics

    # 3. replay sampled demo frames to accumulate SAM2Act+ memory
    demo_frames = init["demo_frames"]
    idxs = sample_indices(len(demo_frames), history_frames)
    frames_payload = []
    for ci, fi in enumerate(idxs):
        try:
            obs_obj = build_obs_obj(runner.env, demo_frames[fi])
        except Exception as e:
            print(f"[demo] frame {fi} build error: {e}")
            continue
        frames_payload.append({
            "obs_obj": obs_obj,
            "curr_idx": ci,
            "lang_goal": lang_goal,
            "episode_length": episode_length,
        })
    print(f"[ep{episode_id}] goal='{lang_goal}'  feeding {len(frames_payload)}/{len(demo_frames)} demo frames")
    if frames_payload:
        client.add_buffer({
            "add_buffer": True,
            "frames": frames_payload,
            "lang_goal": lang_goal,
            "episode_length": episode_length,
        })

    # 4. eval loop
    obs = init["cur_obs"]
    status = "timeout"
    for step in range(max_steps):
        try:
            obs_obj = build_obs_obj(runner.env, obs)
        except Exception as e:
            print(f"[ep{episode_id}] obs build error at step {step}: {e}")
            status = "error"
            break
        resp = client.infer({
            "obs_obj": obs_obj,
            "curr_idx": step,
            "lang_goal": lang_goal,
            "episode_length": episode_length,
        })
        actions = resp.get("actions") if isinstance(resp, dict) else None
        if not actions:
            print(f"[ep{episode_id}] no action at step {step}; stopping")
            break
        action9 = actions[0]

        obs, stop, outcome = runner.step(action9)
        result["steps"] = step + 1
        if outcome in ("success", "fail"):
            status = outcome
            print(f"[ep{episode_id}] {outcome.upper()} at step {step}")
            break
        if stop:
            status = outcome if outcome in ("success", "fail", "timeout") else "timeout"
            print(f"[ep{episode_id}] env stopped at step {step} (status={outcome})")
            break

    result["status"] = status
    print(f"[ep{episode_id}] final status={status} (difficulty={result['difficulty']}, steps={result['steps']})")
    runner.close_env()
    return result


def summarize(results):
    total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    by_diff = {}
    for r in results:
        d = r.get("difficulty") or "unknown"
        b = by_diff.setdefault(d, {"total": 0, "success": 0})
        b["total"] += 1
        if r["status"] == "success":
            b["success"] += 1
    for b in by_diff.values():
        b["success_rate"] = (b["success"] / b["total"]) if b["total"] else 0.0
    return {
        "total": total,
        "success": n_success,
        "success_rate": (n_success / total) if total else 0.0,
        "by_difficulty": by_diff,
    }


def evaluate(args):
    if args.only_tasks:
        task_names = [t.strip() for t in args.only_tasks.split(",") if t.strip()]
    else:
        task_names = list(TASK_NAME_LIST)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] connecting to SAM2Act server ws://{args.host}:{args.port} ...")
    client = MMEVLAWebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[eval] connected. server metadata: {client.get_server_metadata()}")

    log = {}
    t0 = time.time()
    for task in task_names:
        runner = SAM2ActEnvRunner(task, sim_step_budget=args.sim_budget)
        n = runner.num_episodes
        if args.max_episodes > 0:
            n = min(n, args.max_episodes)
        print(f"\n==== task {task}: {n} episodes ====")
        task_results = []
        for ep in range(n):
            res = eval_episode(client, runner, ep, args.max_steps, args.history_frames)
            task_results.append(res)
            # incremental persistence
            with (save_dir / "progress.json").open("w") as f:
                json.dump({task: task_results}, f, indent=2)
        log[task] = {
            "results": task_results,
            "summary": summarize(task_results),
        }

    elapsed = time.time() - t0
    with (save_dir / "log.json").open("w") as f:
        json.dump(log, f, indent=2)

    print("\n==================== SUMMARY ====================")
    for task, entry in log.items():
        s = entry["summary"]
        print(f"{task}: success {s['success']}/{s['total']} = {s['success_rate']:.1%}")
        for d, b in sorted(s["by_difficulty"].items()):
            print(f"    {d:8s}: {b['success']}/{b['total']} = {b['success_rate']:.1%}")
    print(f"({elapsed:.0f}s)  results -> {save_dir/'log.json'}")


def main():
    parser = argparse.ArgumentParser(description="SAM2Act robomme eval (WebSocket, waypoint)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--only_tasks", default="BinFill", help="comma-separated task ids (empty = all 16)")
    parser.add_argument("--max_episodes", type=int, default=5, help="cap episodes per task (0 = all)")
    parser.add_argument("--max_steps", type=int, default=40, help="max keyframe (waypoint) steps per episode")
    parser.add_argument("--sim_budget", type=int, default=100000, help="benchmark sim-step budget (max_steps_without_demonstration); one waypoint = many sim substeps")
    parser.add_argument("--history_frames", type=int, default=16, help="demo frames replayed into memory (0 = all)")
    parser.add_argument("--save_dir", default=f"examples/sam2act/runs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
