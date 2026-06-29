#!/bin/bash
# One-shot bootstrap script for v7 DenseTrack training on E2E L4 box.
# Run AFTER:
#   1. Modal cache transfer landed at /root/data/v7_cache/ (175 GB)
#   2. Recognition LoRA training finished and freed the GPU
#   3. iter-30k checkpoint scp'd to /root/data/model_final.pth (298 MB)
#
# This script:
#   - clones GoMatching repo at /root/GoMatching
#   - applies our forked v7 changes (the gomatching/ subpackage + configs + tools)
#   - builds AdelaiDet (third_party)
#   - sets up training environment + symlinks
#
# Run as: bash setup_e2e_training.sh
# After it finishes, kick training with: bash launch_e2e_training.sh
set -e
set -o pipefail

GOMATCHING_REPO=https://github.com/Hxyz-123/GoMatching.git
GM_DIR=/root/GoMatching
V7_SRC=/root/v7_src              # where we scp our forked gomatching/, configs/, tools/
CACHE_DIR=/root/data/v7_cache    # where modal_v7_transfer.py landed the caches
CKPT_SRC=/root/data/model_final.pth   # iter-30k baseline
OUTPUTS=/root/outputs/DenseTrack_IC15

echo "============================================================"
echo "  v7 DenseTrack E2E setup  $(date -u +%FT%TZ)"
echo "============================================================"

# 0. sanity checks
for d in "$CACHE_DIR/ic15v/clip-l-336" "$CACHE_DIR/ic15v/dinov2-l" \
         "$CACHE_DIR/ic15v/convnext-l" "$CACHE_DIR/ic15v/sam_proposals"; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    echo "  cache check $d: $n files"
    if [ "$n" -lt 49 ]; then
        echo "ERROR: $d has $n files, expected 49. Cache transfer incomplete."
        exit 1
    fi
done
[ -f "$CKPT_SRC" ] || { echo "ERROR: $CKPT_SRC missing"; exit 1; }
[ -d "$V7_SRC/gomatching" ] || { echo "ERROR: $V7_SRC/gomatching/ missing - scp the v7 source first"; exit 1; }
echo "  all sanity checks PASS"

# 0.5 disk space check (need ≥ 210 GB free)
free_gb=$(df -BG /root | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "  /root free space: ${free_gb} GB"
if [ "$free_gb" -lt 25 ]; then
    echo "ERROR: only ${free_gb} GB free on /root; need ≥25 GB beyond cache (cache should already be at $CACHE_DIR)."
    exit 1
fi

# 0.7 venv check / build
if [ ! -x /root/venv/bin/python ]; then
    echo "[0.7] no /root/venv — installing Python venv + torch + CUDA stack"
    apt-get update -qq && apt-get install -y -qq python3-venv build-essential ninja-build clang \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 git wget rsync
    python3 -m venv /root/venv
    /root/venv/bin/pip install --upgrade pip wheel "setuptools>=68" ninja
    /root/venv/bin/pip install \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121
    /root/venv/bin/pip install \
        opencv-python-headless==4.10.0.84 Pillow "numpy<2" scipy shapely pyclipper \
        timm==1.0.11 einops Polygon3 rapidfuzz editdistance fvcore iopath yacs \
        tabulate tqdm termcolor open_clip_torch gdown h5py huggingface_hub \
        safetensors "transformers>=4.45,<5.0" "accelerate>=0.34,<1.2"
    /root/venv/bin/pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git"
    echo "  venv built"
else
    echo "  /root/venv present"
fi

# 1. clone GoMatching repo (fresh)
if [ ! -d "$GM_DIR" ]; then
    echo "[1/4] cloning $GOMATCHING_REPO -> $GM_DIR"
    git clone --depth 1 "$GOMATCHING_REPO" "$GM_DIR"
fi

# 2. overlay our v7 changes (overwrite cloned files)
echo "[2/4] overlaying v7 source onto $GM_DIR"
cp -r "$V7_SRC/gomatching/"* "$GM_DIR/gomatching/"
cp -r "$V7_SRC/configs/"*    "$GM_DIR/configs/"
mkdir -p "$GM_DIR/tools_v7"
cp -r "$V7_SRC/tools/"*      "$GM_DIR/tools_v7/"

# 3. build AdelaiDet
echo "[3/4] installing AdelaiDet (third_party)"
cd "$GM_DIR/third_party"
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" \
    /root/venv/bin/pip install -e . --no-build-isolation 2>&1 | tail -10

# 4. install extra deps the existing /root/venv/ may lack
echo "[4/4] checking extra deps"
/root/venv/bin/pip install rapidfuzz h5py editdistance --quiet 2>&1 | tail -3

# 5. symlink baseline checkpoint where the config expects it
mkdir -p "$GM_DIR/pretrained_models"
ln -sf "$CKPT_SRC" "$GM_DIR/pretrained_models/model_final.pth"

# 5.5 datasets layout — GoMatching's vts.py auto-registers
#     'icdar15_train' at datasets/ICDAR15/{train.json, frame/}
#     (relative to cwd when train_net.py runs). Symlink real paths.
mkdir -p "$GM_DIR/datasets/ICDAR15"
# train.json — must be scp'd to /root/data/ICDAR15_train.json beforehand
if [ -f /root/data/ICDAR15_train.json ]; then
    ln -sf /root/data/ICDAR15_train.json "$GM_DIR/datasets/ICDAR15/train.json"
    echo "  symlinked train.json"
else
    echo "  WARN: /root/data/ICDAR15_train.json missing — scp it before launching training"
fi
# frames — IC15-V test frames need to exist somewhere too
if [ -d /root/data/frames/ic15v_train ]; then
    ln -sf /root/data/frames/ic15v_train "$GM_DIR/datasets/ICDAR15/frame"
    echo "  symlinked frame -> /root/data/frames/ic15v_train"
elif [ -d /root/data/ICDAR15_Video/frames ]; then
    ln -sf /root/data/ICDAR15_Video/frames "$GM_DIR/datasets/ICDAR15/frame"
    echo "  symlinked frame -> /root/data/ICDAR15_Video/frames"
else
    echo "  WARN: no IC15-V frames directory found — scp or download before launching training"
fi

# 6. set cache root paths in the config (overwrite the defaults)
sed -i "s|/data/encoder_cache|$CACHE_DIR/ic15v|g" "$GM_DIR/configs/DenseTrack_PP_ICDAR15.yaml"
sed -i "s|/data/sam_cache|$CACHE_DIR/ic15v/sam_proposals|g" "$GM_DIR/configs/DenseTrack_PP_ICDAR15.yaml"
echo "  config patched with cache paths"

mkdir -p "$OUTPUTS"

echo ""
echo "============================================================"
echo "  Setup OK. Run training with:"
echo "    bash /root/launch_e2e_training.sh"
echo "============================================================"
