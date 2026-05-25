#!/usr/bin/env bash
# Orchestrate the SAM2Act / SAM2Act+ robomme eval (mirrors scripts/eval.sh).
#
#   Window 1 (serve_policy): SAM2Act WebSocket policy server, in this repo's uv
#       .venv (torch 2.5.1 inference env), on GPU_server.
#   Window 2 (eval):         robomme eval client, in the `sam2act-robomme-eval`
#       micromamba env (torch 2.9.1 + ManiSkill + robomme), on GPU_client.
#
# The server loads a SAM2Act+ checkpoint and exposes reset/add_buffer/infer over
# WebSocket; the client drives BinFill with action_space="waypoint" (built-in
# planner, no hand-written planner).
set -u

#### set your own parameters ####
# Single repo now holds both envs: uv .venv (server) + micromamba sam2act-robomme-eval (client).
REPO="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/sam2act_historybench"
ENV_CLIENT="sam2act-robomme-eval"
MODEL_FOLDER="$REPO/sam2act/runs/sam2act_plus_all_v4"   # SAM2Act+ recommended
MODEL_NAME="model_plus_last.pth"
GPU_server=1      # GPU for the SAM2Act inference server
GPU_client=0      # GPU for the ManiSkill sim + eval client
ONLY_TASKS="BinFill"
MAX_EPISODES=5
MAX_STEPS=40
HISTORY_FRAMES=16
#--------------------------------#

MM=/home/hongzefu/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/hongzefu/micromamba

find_free_port() {
  local min=${1:-8000} max=${2:-30000} port
  for ((i=0; i<5000; i++)); do
    port=$(shuf -i "${min}"-"${max}" -n1)
    if ! ss -ltn 2>/dev/null | grep -q ":${port}\b"; then echo "${port}"; return 0; fi
  done
  echo "ERROR: no free port in ${min}-${max}" >&2; return 1
}
PORT=$(find_free_port)
SESSION="sam2act_eval_${ONLY_TASKS}_port${PORT}"
echo "Eval ${ONLY_TASKS} (${MAX_EPISODES} ep) on port ${PORT}  [server GPU${GPU_server} / client GPU${GPU_client}]"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists. Change the port or use a different session."
  exit 1
fi

# Window 1: SAM2Act WebSocket policy server (repo uv .venv)
tmux new-session -d -s "$SESSION" -n serve_policy
tmux send-keys -t "$SESSION:serve_policy" \
  "cd ${REPO} && CUDA_VISIBLE_DEVICES=${GPU_server} .venv/bin/python sam2act/historybench_eval/serve_policy.py --model_folder '${MODEL_FOLDER}' --model_name ${MODEL_NAME} --device 0 --port ${PORT} --seed 0" Enter

sleep 30   # wait for the model to load (~20s)

# Window 2: robomme eval client (new micromamba env)
tmux new-window -t "$SESSION" -n eval
tmux send-keys -t "$SESSION:eval" \
  "cd ${REPO} && CUDA_VISIBLE_DEVICES=${GPU_client} ${MM} run -n ${ENV_CLIENT} python examples/sam2act/eval.py --host 127.0.0.1 --port ${PORT} --only_tasks ${ONLY_TASKS} --max_episodes ${MAX_EPISODES} --max_steps ${MAX_STEPS} --history_frames ${HISTORY_FRAMES}; tmux wait-for -S eval-done" Enter

# Wait for eval to finish (or session to be killed)
tmux wait-for eval-done &
wait_pid=$!
while kill -0 $wait_pid 2>/dev/null; do
  tmux has-session -t "$SESSION" 2>/dev/null || { kill $wait_pid 2>/dev/null; echo "Tmux session killed, exiting."; exit 1; }
  sleep 2
done

echo "eval done. The server is still running in tmux window 'serve_policy' of session ${SESSION}."
echo "Attach with:  tmux attach -t ${SESSION}    (Ctrl-C in serve_policy window to stop the server)"
echo "Kill all:     tmux kill-session -t ${SESSION}"
