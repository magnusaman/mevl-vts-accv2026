#!/bin/bash
# Launch v7 DenseTrack training on E2E L4 in background (detached from ssh).
# Run AFTER setup_e2e_training.sh has succeeded.
# Pass --resume to pick up from /root/outputs/DenseTrack_IC15/last_checkpoint
# (useful if the previous launch crashed or was killed).
set -e
GM_DIR=/root/GoMatching
OUTPUTS=/root/outputs/DenseTrack_IC15
LOG=/root/data/densetrack_train.log
PIDFILE=/root/data/densetrack_train.pid

mkdir -p "$OUTPUTS"

# allow resume via env var (RESUME=1 bash launch_e2e_training.sh)
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ] && [ -f "$OUTPUTS/last_checkpoint" ]; then
    echo "[v7-train] RESUME=1: continuing from last_checkpoint"
    RESUME_FLAG="--resume"
fi

# don't start two trainings at once
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "[v7-train] training already running (pid $(cat $PIDFILE)). exiting."
    exit 0
fi

cd "$GM_DIR"
echo "[v7-train] launching at $(date -u +%FT%TZ)"
echo "[v7-train] cwd=$(pwd)  output=$OUTPUTS  resume=${RESUME:-0}"

setsid nohup /root/venv/bin/python train_net.py \
    --num-gpus 1 \
    --config-file configs/DenseTrack_PP_ICDAR15.yaml \
    $RESUME_FLAG \
    --opts \
      SOLVER.MAX_ITER 12000 \
      SOLVER.CHECKPOINT_PERIOD 1000 \
      OUTPUT_DIR "$OUTPUTS" \
      MODEL.WEIGHTS "$GM_DIR/pretrained_models/model_final.pth" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PIDFILE"
echo "[v7-train] pid=$PID  log=$LOG"
echo ""
echo "Tail log:"
echo "  tail -f $LOG"
echo "Status check:"
echo "  ps -p $PID -o pid,etime,pcpu,pmem,comm"
echo "Resume after kill:"
echo "  RESUME=1 bash launch_e2e_training.sh"
echo "Kill if needed:"
echo "  kill $PID"
