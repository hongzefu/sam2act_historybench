"""One-shot probe: reset one BinFill episode and dump obs_batch key shapes.

Goal: determine whether the dense demo trajectory survives in obs_batch under
keys OTHER than `maniskill_obs` (client-side fix) or is genuinely collapsed to
1 frame everywhere (robomme-side fix). Read-only; needs no SAM2Act WS server.
"""

import sys

from env_runner import SAM2ActEnvRunner


def describe(d, name):
    print(f"\n=== {name}: dict with {len(d)} keys ===")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, list):
            n = len(v)
            n_dict = sum(1 for x in v if isinstance(x, dict))
            n_none = sum(1 for x in v if x is None)
            n_arr = sum(1 for x in v if hasattr(x, "shape"))
            sample_shape = ""
            for x in v:
                if hasattr(x, "shape"):
                    sample_shape = f" ex_shape={tuple(x.shape)}"
                    break
            print(f"  {k:34s} list len={n:<4d} dict={n_dict} None={n_none} ndarray={n_arr}{sample_shape}")
        elif hasattr(v, "shape"):
            print(f"  {k:34s} ndarray shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            s = repr(v)
            print(f"  {k:34s} {type(v).__name__}: {s[:60]}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "BinFill"
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"[probe] task={task} episode={ep}")
    runner = SAM2ActEnvRunner(task, sim_step_budget=100000)
    runner.make_env(ep)
    obs_batch, info = runner.env.reset()

    print(f"\nobs_batch type: {type(obs_batch)}")
    if isinstance(obs_batch, dict):
        describe(obs_batch, "obs_batch")

    if isinstance(info, dict):
        # info is flattened (last values), just show keys + scalar-ish values
        print(f"\n=== info: {len(info)} keys ===")
        for k in sorted(info.keys()):
            v = info[k]
            if isinstance(v, list):
                print(f"  {k:34s} list len={len(v)}")
            elif hasattr(v, "shape"):
                print(f"  {k:34s} ndarray shape={tuple(v.shape)}")
            else:
                print(f"  {k:34s} {type(v).__name__}: {repr(v)[:60]}")

    mk = obs_batch.get("maniskill_obs", []) if isinstance(obs_batch, dict) else []
    real = [f for f in mk if isinstance(f, dict)]
    print(f"\n[probe] maniskill_obs: total={len(mk)} dict-frames={len(real)}")
    if real:
        print(f"[probe] one maniskill_obs frame top-level keys: {list(real[0].keys())}")
        sd = real[0].get("sensor_data")
        if isinstance(sd, dict):
            print(f"[probe] maniskill_obs.sensor_data cameras: {list(sd.keys())}")

    runner.close_env()
    print("\n[probe] done")


if __name__ == "__main__":
    main()
