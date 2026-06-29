#!/usr/bin/env bash
# Box bootstrap — safe setup steps for the A6000 box. Read before running.
# Does the cheap/safe things automatically; prints the big BOVText pull command
# for you to start in tmux (it's ~1.75M frames — don't block the shell on it).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root
ROOT="$(pwd)"
echo "== repo: $ROOT =="

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "!! nvidia-smi failed"

echo "== system =="
echo "cpus: $(nproc 2>/dev/null)"; echo "python: $(python --version 2>&1)"
echo "disk (home):"; df -h "$HOME" | tail -1
echo "conda envs:"; conda env list 2>/dev/null || echo "(conda not found)"

echo "== modal =="
if command -v modal >/dev/null 2>&1; then
  modal volume list >/dev/null 2>&1 && echo "modal OK" || echo "!! modal not authed — run: modal token set ..."
else
  echo "!! modal not installed — pip install modal && modal token set ..."
fi

echo "== dirs =="
mkdir -p code/GoMatching_v7/datasets code/GoMatching_v7/pretrained_models checkpoints
echo "ok"

echo "== pull weights (small) =="
if command -v modal >/dev/null 2>&1; then
  modal volume get mevl-vts-weights /pretrained_models code/GoMatching_v7/pretrained_models || echo "!! weights pull failed"
  modal volume get mevl-vts-v7-checkpoints /gomatching_iter30k checkpoints/gomatching_iter30k || echo "!! gomatching warm-start pull failed"
  ls -lh code/GoMatching_v7/pretrained_models/ 2>/dev/null
fi

cat <<'EOF'

== NEXT (do these yourself; see docs/RUNBOOK.md) ==
1) Start the BIG BOVText download in tmux (risk #1):
     tmux new -s bov
     modal volume get mevl-vts-datasets /BOVText code/GoMatching_v7/datasets/BOVText
     # Ctrl-b d to detach; tmux attach -t bov to check
2) Build the `gom` conda env (python 3.10, torch cu121/cu124, detectron2 0.6,
   build AdelaiDet from code/GoMatching_v7/third_party/adet, pip install requirements.txt).
3) Follow ORCHESTRATION/INSTRUCTIONS_FOR_BOX.md from T5.
EOF
echo "== bootstrap done =="
