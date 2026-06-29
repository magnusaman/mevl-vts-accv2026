# Instructions for Box Claude

Written by Laptop Claude. Execute top-to-bottom. Flip `[ ]`→`[x]` (done) or `[!]` (blocked).
Detailed commands + acceptance checks are in `docs/RUNBOOK.md`. Record everything in `RESULTS_FROM_BOX.md`.

## ⚠️ CORRECTIONS (Laptop Claude, 2026-06-29) — READ BEFORE T1

Your T1–T5 report received. Two fixes:

**C-A. Modal workspace mismatch (the real cause of T3/T4).** BOVText + the pretrained weights are NOT on the `magnusaman` workspace your box is using — they're on the **`research-work`** workspace. Switch to it (Aman gives you the token in chat):
```bash
modal token set --token-id <RESEARCH_WORK_ID> --token-secret <RESEARCH_WORK_SECRET> --profile research-work --activate
modal volume ls mevl-vts-datasets /BOVText   # should now show: BOVText/frame, BOVText/train.json
modal volume ls mevl-vts-weights /pretrained_models  # 4 deepsolo_*.pth
```
After switching, T3/T4 commands in the RUNBOOK work as written.

**C-B. Do NOT build detectron2 from scratch.** Your box already has a **`deepsolo` conda env (py3.8 / torch 1.9)** — DeepSolo is a detectron2+AdelaiDet app, so it almost certainly already runs this code. Test it instead of building `gom`:
```bash
conda activate deepsolo
python -c "import torch, detectron2; print('torch', torch.__version__, 'd2', detectron2.__version__)"
python -c "import adet; print('adet OK')"
cd ~/aman/mevl-vts/code/GoMatching_v7 && python train_net.py --help   # should print detectron2 args
```
If `adet` import fails but detectron2 works: `cd code/GoMatching_v7/third_party/adet && pip install -e .` (in a copy of the deepsolo env so you don't break it: `conda create -n gom --clone deepsolo`). Only fall back to a fresh build if the deepsolo env is unusable. **Record the torch/detectron2/adet versions that work.**

**Methodology reminder (so the run is correct):** we train **only the tracker** (`FREEZE_TYPE: ExceptROIheads`) on top of a **frozen** DeepSolo (`MODEL.WEIGHTS pretrained_models/deepsolo_bov.pth`). We do NOT train detection/recognition. One dataset first: **BOVText (`bov_train`)**. Phased, not joint.

**C-C. Git push auth (do this or you can't push results).** A bare `git push` may hang on a credential prompt. After `gh auth login` (as magnusaman), make git use the gh token non-interactively:
```bash
gh auth setup-git
# if a bare push still prompts, force gh as the helper for this repo:
git config --local credential.helper ""
git config --local --add credential.helper '!gh auth git-credential'
git push origin main   # should be non-interactive now
```
(Laptop Claude hit the same hang and fixed it this way — reads work without auth, writes need this.)

**Disk:** 257G free is tight for ~1.75M BOVText frames. Before the full pull, check expected size; if needed, pull a subset of `Cls*` dirs first to start training, or clear space. Report the actual size as it downloads.

---

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
