# Instructions for Box Claude

Written by Laptop Claude. Execute top-to-bottom. Flip `[ ]`→`[x]` (done) or `[!]` (blocked).
Detailed commands + acceptance checks are in `docs/RUNBOOK.md`. Record everything in `RESULTS_FROM_BOX.md`.

## Batch 1 — stand up + start the big download (do today)

- [ ] **T1. Report the box.** `nvidia-smi`, `nproc`, `df -h ~`, `python --version`, `conda env list`. Confirm free disk ≥ ~300GB for BOVText frames. → RESULTS.
- [ ] **T2. Modal auth.** Confirm `modal volume list` works on the box (Aman sets the token). If not, mark BLOCKED with the error.
- [ ] **T3. Start BOVText pull in the background NOW** (risk #1, slow): `modal volume get mevl-vts-datasets /BOVText code/GoMatching_v7/datasets/BOVText` in tmux/nohup. Report start time + size as it grows. Don't wait on it — continue to T4.
- [ ] **T4. Pull weights** (small): `modal volume get mevl-vts-weights /pretrained_models code/GoMatching_v7/pretrained_models` and `modal volume get mevl-vts-v7-checkpoints /gomatching_iter30k checkpoints/gomatching_iter30k`. Verify 4 `.pth` present. → RESULTS.
- [ ] **T5. Build the `gom` conda env** (Phase 0 of RUNBOOK). Acceptance: `python -c "import torch, detectron2; print(torch.cuda.get_device_name(0))"` → `NVIDIA RTX A6000`. Record the exact torch/detectron2 versions that worked (Laptop Claude needs them).

## Batch 2 — verify data + smoke (after T3 finishes)

- [ ] **T6. Verify BOVText.** image/annotation counts from `train.json`; frame count on disk ≈ images. → RESULTS.
- [ ] **T7. Register `bov_train`** per `vts.py` (Phase 2). Acceptance: `DatasetCatalog.get("bov_train")` returns records, first record prints clean.
- [ ] **T8. Smoke train 20 iters** (Phase 3). Acceptance: finite loss, no OOM, checkpoint written. Paste last ~20 log lines + iter/s.

## Batch 3 — real run (after smoke passes)

- [ ] **T9. Launch full BOVText training** (Phase 4), background + log. Report iter/s, ETA to 30k, ckpt cadence. **Ping when iter 5000 checkpoint lands.**

> Stop and wait for Laptop Claude after T8 if anything looks off (loss NaN, dataset mismatch, vocab errors). Otherwise keep going to T9.
> Open question for Aman/Laptop: BOVText official **test/eval** protocol (which split + scoring server) — flag it but don't block training on it.
