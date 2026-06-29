#!/bin/bash
# Run AFTER v7 training finishes (or as soon as a checkpoint you want to eval lands).
# Produces an RRC IC15-V Task 4 submission zip using the trained model
# + Qwen-VL LoRA recognition.
#
# Defaults: latest v7_full checkpoint; flip with --ckpt to eval an earlier one.
set -e

CKPT="${1:-/root/outputs/v7_full/model_final.pth}"
LABEL="${2:-v7_full}"  # used to name output dirs
OUT_BASE=/root/outputs
JSONS_RAW="$OUT_BASE/${LABEL}_jsons"
JSONS_SWAPPED="$OUT_BASE/${LABEL}_swapped"
ZIP_OUT="$OUT_BASE/${LABEL}_rrc.zip"

LORA_ADAPTER=/root/old_l4_backup/outputs/full/20260526_061407/best_adapter
RECOG_DIR=/root/old_l4_backup/recognition_lora
FRAMES_TEST=/root/old_l4_backup/data/frames/ic15v_test

echo "============================================================"
echo "  post-training pipeline   $(date -u +%FT%TZ)"
echo "============================================================"
echo "  CKPT          = $CKPT"
echo "  LABEL         = $LABEL"
echo "  RAW JSON DIR  = $JSONS_RAW"
echo "  SWAP JSON DIR = $JSONS_SWAPPED"
echo "  OUT ZIP       = $ZIP_OUT"
echo "  LORA          = $LORA_ADAPTER"
echo "  TEST FRAMES   = $FRAMES_TEST"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi
mkdir -p "$JSONS_RAW" "$JSONS_SWAPPED"

# 1. Inference on IC15-V test using the trained v7 model
echo "[1/3] running v7 inference on IC15-V test..."
cd /root/GoMatching
/root/venv/bin/python eval.py \
    --config-file configs/DenseTrack_full.yaml \
    --input "$FRAMES_TEST/Video_*" \
    --output "$JSONS_RAW" \
    --opts MODEL.WEIGHTS "$CKPT" 2>&1 | tee /root/post_train_eval.log

n_json=$(ls "$JSONS_RAW"/*.json 2>/dev/null | wc -l)
echo "  produced $n_json video JSONs"
if [ "$n_json" -lt 5 ]; then
    echo "  WARNING: only $n_json JSONs produced — eval may have failed"
fi

# 2. Qwen-VL LoRA transcription swap (track-deduped)
echo "[2/3] running Qwen-VL LoRA transcription swap..."
/root/venv/bin/python "$RECOG_DIR/infer_swap_track_dedup.py" \
    --jsons-dir "$JSONS_RAW" \
    --frames-root "$FRAMES_TEST" \
    --adapter "$LORA_ADAPTER" \
    --out-dir "$JSONS_SWAPPED" \
    --batch-size 32 --target-size 224 --min-track-len 5 2>&1 | tee /root/post_train_swap.log

# 3. Build RRC IC15-V Task 4 zip
echo "[3/3] building RRC Task 4 zip..."
/root/venv/bin/python "$RECOG_DIR/build_rrc_zip_task4.py" \
    --jsons-dir "$JSONS_SWAPPED" \
    --out "$ZIP_OUT"

echo
echo "============================================================"
echo "  DONE"
echo "  Final zip: $ZIP_OUT  ($(du -h "$ZIP_OUT" | awk '{print $1}'))"
echo "  Upload to: https://rrc.cvc.uab.es/?ch=3&com=mymethods&task=4"
echo "============================================================"
