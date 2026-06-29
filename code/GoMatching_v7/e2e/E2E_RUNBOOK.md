# E2E training runbook — v7 DenseTrack

Sequence for taking the v7 caches from Modal → E2E → trained checkpoint.
Assumes you're done with the recognition LoRA (GPU free).

## Prereqs
- Modal cache transfer (`modal_v7_transfer.py`) ran successfully → `/root/data/v7_cache/` on E2E has the 175 GB
- Recognition LoRA finished → `/root/outputs/full/<ts>/best_adapter/` exists
- iter-30k checkpoint copied to E2E at `/root/data/model_final.pth` (298 MB)
- This `GoMatching_v7/` directory scp'd to E2E at `/root/v7_src/`

## Quick check before starting

```bash
ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68
df -h /root            # need ≥ 200 GB free
ls /root/data/v7_cache/ic15v/clip-l-336/ | wc -l      # 49
ls /root/data/v7_cache/ic15v/dinov2-l/ | wc -l        # 49
ls /root/data/v7_cache/ic15v/convnext-l/ | wc -l      # 49
ls /root/data/v7_cache/ic15v/sam_proposals/ | wc -l   # 49
ls -lh /root/data/model_final.pth                     # 298 MB
ls /root/v7_src/gomatching/ /root/v7_src/configs/ /root/v7_src/tools/  # populated
nvidia-smi                                            # GPU idle
```

## Push v7 source from local laptop to E2E

```bash
# from local (Windows shell):
scp -i ~/.ssh/id_ed25519_e2e_new -r \
    "Project/MEVL-VTS/GoMatching_v7/gomatching" \
    "Project/MEVL-VTS/GoMatching_v7/configs" \
    "Project/MEVL-VTS/GoMatching_v7/tools" \
    "Project/MEVL-VTS/GoMatching_v7/e2e" \
    root@164.52.193.68:/root/v7_src/

scp -i ~/.ssh/id_ed25519_e2e_new \
    "Project/MEVL-VTS/E2E_FINAL_BACKUP/v6_checkpoints/GoMPP_IC15/model_final.pth" \
    root@164.52.193.68:/root/data/model_final.pth

ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 \
    "cp /root/v7_src/e2e/*.sh /root/ && chmod +x /root/*.sh"
```

## Setup + launch

```bash
ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 "bash /root/setup_e2e_training.sh"
# ~5-10 min (AdelaiDet compile, deps install)

ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 "bash /root/launch_e2e_training.sh"
# returns immediately; training runs in background

# follow logs:
ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 "tail -f /root/data/densetrack_train.log"
```

## Expected timeline on L4

- AdelaiDet build: 5-10 min
- Training: 12k iters × ~5 sec/iter ≈ 16-18h
- First val ckpt: at SOLVER.CHECKPOINT_PERIOD=1000 iters (~80 min in)

## When training finishes

Checkpoint: `/root/outputs/DenseTrack_IC15/model_final.pth`

### Eval on IC15-V test:
```bash
ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 \
    "cd /root/GoMatching && /root/venv/bin/python eval.py \
       --config-file configs/DenseTrack_PP_ICDAR15.yaml \
       --input /root/data/frames/ic15v_test/ \
       --output /root/outputs/DenseTrack_IC15/eval/ \
       --opts MODEL.WEIGHTS /root/outputs/DenseTrack_IC15/model_final.pth"
```

### Build RRC zips:
The existing scripts in `/root/recognition_lora/build_rrc_zip.py` work directly on the eval output JSONs.

## Disk budget check
- 175 GB cache + 0.3 GB checkpoint + ~5 GB training output = ~180 GB
- E2E free was 207 GB last check → tight but should fit
- If tight: drop ConvNeXt cache (75 GB) and exclude it from COMP2_ENCODERS → ~100 GB total

## What's NOT included yet
- Component 3 (VLM text cache) — needs the LoRA's `best_adapter` + a separate cache run on E2E (write `tools/cache_vlm_text.py`)
- DSText support — encoder cache + frames for DSText would need to be built too

## Killing the run
```bash
ssh -i ~/.ssh/id_ed25519_e2e_new root@164.52.193.68 "kill \$(cat /root/data/densetrack_train.pid)"
```
