# MEVL-VTS v7 — Modal setup

Three-stage verification before any v7 code runs. If all 3 stages pass on Modal, the foundation is solid and we move on to (a) caching encoder features and (b) writing the bridge module.

---

## Prerequisites (one-time, on your laptop)

```bash
# install modal client
pip install --upgrade modal

# log in to your Modal workspace
modal token new
# (this opens a browser; pick the workspace you want this app to live in)
```

## Step 1 — Deploy the app

From this directory (`Project/MEVL-VTS/GoMatching_v7/modal/`):

```bash
modal deploy modal_v7_setup.py
```

This will:
- build the Docker image (CUDA 12.1, torch 2.5.1, detectron2 from source, AdelaiDet compiled for L4 sm_89)
- clone `https://github.com/Hxyz-123/GoMatching` into the image at `/opt/GoMatching`
- create 4 volumes if they don't exist: `mevl-vts-v7-{checkpoints,data,outputs,encoder-cache}`

**First build takes ~10-15 minutes.** Subsequent runs reuse the cached image.

If the image build fails on AdelaiDet (most likely culprit), the error will tell us whether to bump CUDA arch or pin a different torch.

## Step 2 — Upload the reproduced iter-30k checkpoint

The 298 MB `.pth` lives locally at:
```
Project/MEVL-VTS/E2E_FINAL_BACKUP/v6_checkpoints/GoMPP_IC15/model_final.pth
```

Push to Modal volume:
```bash
modal volume put mevl-vts-v7-checkpoints \
    "../E2E_FINAL_BACKUP/v6_checkpoints/GoMPP_IC15/model_final.pth" \
    gomatching_iter30k/model_final.pth
```

(verify with `modal volume ls mevl-vts-v7-checkpoints`)

## Step 3 — Run the 3 verifications

```bash
# Stage 1: image + imports + GPU
modal run modal_v7_setup.py::verify_image

# Stage 2: build model from config + load checkpoint
modal run modal_v7_setup.py::verify_baseline_load

# Stage 3: forward + backward on synthetic 6-frame clip
modal run modal_v7_setup.py::verify_smoke_forward
```

Or all three:
```bash
modal run modal_v7_setup.py --stage all
```

## What you should see

### Stage 1 ✓
```
python: 3.10...
torch: 2.5.1+cu121  cuda_available=True
  device: NVIDIA L4
  capability: (8, 9)
  mem: 23.0 GB
detectron2: 0.6+...
adet:       imported OK from ...
gomatching: imported OK from ...
```

### Stage 2 ✓
```
ckpt keys (top-level): ['model', 'optimizer', 'scheduler', 'iteration']
total params in state_dict: ~85,000,000
iter recorded: 29999
META_ARCH: GoMatching
ROI_HEADS: SHA_FFN_CRSATTN
total params: ~85,000,000
trainable params: ~11,800,000  (~13.9%)
```

(The 11.8M trainable matches the paper's GoMatching++ matcher.)

### Stage 3 ✓
```
forward: ~2-4s
losses: ['loss_long_asso', 'loss_short_asso', 'loss_res']
total_loss: 0.5–2.0  (depends on synthetic input)
backward: ~3-5s
GPU mem peak: ~8-12 GB
```

## What to do if a stage fails

Paste the **full traceback** here. Common failures:

- **AdelaiDet import error**: usually means `TORCH_CUDA_ARCH_LIST` doesn't match your GPU. We're set to `8.9` (L4). If you're using H100 add `9.0`, A100 add `8.0`.
- **No checkpoint at /ckpts/...**: you skipped Step 2. Re-run the `modal volume put`.
- **OOM on Stage 3**: synthetic input might be larger than your GPU. Tell me which GPU you set.

## What comes after the 3 verifications

Once Stage 3 returns `{"ok": True}`, the next Modal apps to add:

1. `modal_v7_encoder_cache.py` — Phase 1, run 6 frozen encoders over IC15-V + DSText, save HDF5 cache to `mevl-vts-v7-encoder-cache` volume (~3h on L4, one-time)
2. `modal_v7_bridge_train.py` — Phase 2/3, fine-tune from iter-30k with the encoder bridge (~24h, the real training, expected on E2E not Modal)
3. `modal_v7_eval.py` — Phase 4, run inference on IC15-V test + compute MOTA, generate RRC submission zip

I'll write those after Stage 3 is green.

## Workspace / billing notes

- Image build: ~$0 (caches), build time billed minimally
- Stage 1: ~30 seconds on L4 ≈ $0.01
- Stage 2: ~1 minute on L4 ≈ $0.02
- Stage 3: ~3-5 minutes on L4 ≈ $0.10

Total verification budget: well under $1.

---

When you've run the 3 stages, paste the output back and I'll either green-light the next phase or fix whatever broke.
