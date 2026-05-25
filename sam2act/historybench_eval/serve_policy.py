"""WebSocket policy server entry point for SAM2Act / SAM2Act+.

Mirrors robomme_policy_learning's ``scripts/serve_policy.py`` role: load a
checkpoint and serve it over the WebSocket protocol (reset / add_buffer / infer).
Run inside this repo's uv ``.venv`` (torch 2.5.1 inference env).

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        sam2act/historybench_eval/serve_policy.py \
        --model_folder "$PWD/sam2act/runs/sam2act_plus_all_v4" \
        --model_name model_plus_last.pth --device 0 --port 8001

Note (GPU): always use CUDA_VISIBLE_DEVICES=<free gpu> together with --device 0,
so CLIP (which defaults to cuda:0) and the model land on the same device.
"""

import argparse
import logging
import os
import socket
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from serving.sam2act_policy import SAM2ActPolicy
from serving.websocket_policy_server import WebsocketPolicyServer


def main():
    parser = argparse.ArgumentParser(description="SAM2Act WebSocket policy server")
    parser.add_argument("--model_folder", type=str, required=True, help="checkpoint folder")
    parser.add_argument("--model_name", type=str, required=True, help="checkpoint filename, e.g. model_plus_last.pth")
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument("--use_input_place_with_mean", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="GPU index (use with CUDA_VISIBLE_DEVICES); keep 0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = SAM2ActPolicy(
        model_folder=args.model_folder,
        model_name=args.model_name,
        device=args.device,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        use_input_place_with_mean=args.use_input_place_with_mean,
        seed=args.seed,
    )

    hostname = socket.gethostname()
    print("=" * 60)
    print(f"[serve_policy] SAM2Act WebSocket policy server")
    print(f"[serve_policy] host={hostname}  bind={args.host}:{args.port}  cuda:{args.device}")
    print(f"[serve_policy] healthz: GET ws://{args.host}:{args.port}/healthz")
    print("=" * 60)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
