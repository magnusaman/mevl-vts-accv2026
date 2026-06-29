# Results from Box Claude

Append-only. One entry per task. Format:

```
## [Tn] <short title> — STATUS: DONE | BLOCKED | NEEDS-DECISION
- UTC: <timestamp>
- ran: <commands actually executed>
- output: <key lines; full log path on the box>
- metrics: <iter/s, loss, counts, etc. if any>
- notes / blocker: <what happened, what you tried>
```

---

<!-- Box Claude: start appending below this line -->

## [T1] Report the box — STATUS: DONE
- UTC: 2026-06-29T06:50:00Z
- ran: `nvidia-smi`, `nproc`, `df -h ~`, `python --version`, `conda env list`
- output:
  ```
  GPU:  NVIDIA RTX A6000, 48GB (49140MiB), idle (15MiB used), Driver 580.159.03, CUDA 13.0 max
  CPU:  20 cores
  Disk: 457G total, 177G used, 257G free (mount: /dev/nvme0n1p2)
  Python: 3.12.13 (active env: py312)
  Conda envs: base, deepsolo (py3.8, torch 1.9.0+cu111, detectron2 0.6), nvcc11, py312
  ```
- metrics: n/a
- notes: ⚠️ Disk is 257G free — BOVText (~1.75M frames) may be tight. Existing `deepsolo` env has wrong Python version (3.8) and old torch — creating fresh `gom` env.

## [T2] Modal auth — STATUS: DONE
- UTC: 2026-06-29T06:52:00Z
- ran: `pip install modal`, `modal token info`, `modal volume list`
- output:
  ```
  modal client version: 1.5.1
  Token: ak-TuBLfisb40FjfXhQX3ZCdj (workspace: magnusaman)
  Volumes: mevl-vts-outputs, mevl-vts-datasets, model-weights, curve-aware-datasets-v2,
           hf-model-cache, vlm-outputs, icdar2015-dataset
  ```
- notes: Token was pre-configured in ~/.modal.toml. Auth works.

## [T3] Start BOVText pull — STATUS: BLOCKED
- UTC: 2026-06-29T06:53:00Z
- ran: `modal volume ls mevl-vts-datasets /`
- output:
  ```
  mevl-vts-datasets contents: ICDAR15_Video, DSText_v2, ArTVideo, unified, prepared, _bundles
  — NO /BOVText directory
  ```
- notes / blocker: Volume `mevl-vts-datasets` exists but contains no BOVText data.
  Volumes `mevl-vts-weights` and `mevl-vts-v7-checkpoints` (referenced in RUNBOOK/INSTRUCTIONS)
  do NOT exist in the Modal workspace. Available volumes that might contain weights: `model-weights`
  (listed empty on `ls /`).
  **Cannot pull BOVText or pretrained weights as instructed. NEEDS LAPTOP CLAUDE to confirm
  correct volume names and/or upload BOVText + weights to Modal.**

## [T4] Pull weights — STATUS: BLOCKED
- UTC: 2026-06-29T06:53:00Z
- ran: `modal volume list`
- output: See T3 — neither `mevl-vts-weights` nor `mevl-vts-v7-checkpoints` volumes exist.
- notes / blocker: Same blocker as T3. `model-weights` volume exists but appears empty.
  Laptop Claude needs to confirm volume names and upload weights.

## [C-A] Modal workspace switch — STATUS: DONE
- UTC: 2026-06-29T07:40:00Z
- ran: `modal token set --profile research-work --activate`
- output: Token verified. Now seeing: mevl-vts-datasets, mevl-vts-weights, mevl-vts-v7-checkpoints ✓
  `modal volume ls mevl-vts-datasets /BOVText` → BOVText/frame, BOVText/train.json ✓

## [C-B] deepsolo env test — STATUS: DONE
- UTC: 2026-06-29T07:40:00Z
- ran: conda activate deepsolo && python -c "import torch,detectron2,adet; ..."
- output: `torch 1.9.0+cu111 | d2 0.6 | adet 0.1.1 | GPU NVIDIA RTX A6000`
- notes: adet required building from third_party/. Needed cuda-libraries-dev+nvcc in deepsolo env
  (CUDA 11.8 from nvidia channel). CUDA_HOME=/home/isiuts/miniconda3/envs/deepsolo python setup.py build develop ✓

## [T3] BOVText pull (background) — STATUS: IN PROGRESS
- UTC: 2026-06-29T07:49:49Z
- ran: `/tmp/download_bov.sh` (nohup background, class-by-class with retry, PID=854165)
- output: 07:49:49 Starting Cls7_Game → 07:51:07 DONE (745MB) → 07:51:07 Starting Cls10_Program
- metrics: ~1 min per class × 11 classes → ETA ~11 min total from 07:49Z
- notes: train.json (751MB) downloaded first, fixed: replaced 5.4M null bytes, removed 1 invalid annotation.
  Full modal-volume-get on /BOVText/frame fails with "stream reset" → per-class loop workaround.
  Monitoring disk: 240G free (was 257G, weights used 15GB, safe).

## [T4] Pull weights — STATUS: DONE (retry)
- UTC: 2026-06-29T07:40:00Z
- ran: `modal volume get mevl-vts-weights /pretrained_models/deepsolo_bov.pth ...` (×4 files individually)
       `modal volume get mevl-vts-v7-checkpoints /gomatching_iter30k/model_final.pth ...`
- output:
  ```
  deepsolo_bov.pth          169M ✓
  deepsolo_dstext.pth       168M ✓
  deepsolo_icdar15.pth      164M ✓
  deepsolo_icdar15_rescore.pth 164M ✓
  checkpoints/gomatching_iter30k/model_final.pth 299M ✓
  ```
- notes: `modal volume get /pretrained_models` downloads as a zip archive — had to download each .pth individually.

## [T5] Build gom env / deepsolo acceptance — STATUS: DONE
- UTC: 2026-06-29T07:40:00Z
- ran: python -c "import torch,detectron2,adet; print(torch.cuda.get_device_name(0))"
- output: `NVIDIA RTX A6000`
- metrics: torch 1.9.0+cu111, detectron2 0.6, adet 0.1.1, Python 3.8
- notes: Using existing deepsolo env (not new gom env). adet built with CUDA 11.8 dev headers from nvidia channel.
  train_net.py --help: confirmed --config-file, --num-gpus flags work ✓
  gomatching/data/ was gitignored (data/ rule too broad) — fixed: added !**/gomatching/data/**/*.py negation.
  Copied gomatching/data/ module from upstream Hxyz-123/GoMatching.

## [T7] Register bov_train — STATUS: DONE
- UTC: 2026-06-29T07:55:00Z
- ran: `DatasetCatalog.get("bov_train")`
- output:
  ```
  bov_train records: 430147
  first file_name: datasets/BOVText/frame/Cls10_Program/Cls10_Program_video1/1.jpg
  first record annotations: 4
  ```
- notes: vts.py expects datasets/BOVText/{frame/,train.json} relative to GoMatching_v7 cwd ✓

---

=== MESSAGE TO LAPTOP CLAUDE (2026-06-29 07:55 UTC) ===

**DONE this tick:**
- C-A: Switched to research-work Modal profile. All volumes visible. ✓
- C-B: deepsolo env confirmed: torch 1.9.0+cu111, d2 0.6, adet 0.1.1 (built from third_party/), A6000. ✓
- T3: BOVText download running class-by-class (Cls7_Game done 745MB, Cls10_Program in progress). ETA ~10 min total.
- T4: All 4 deepsolo weights + gomatching_iter30k/model_final.pth pulled individually ✓
- T5: Acceptance check passes (train_net.py --help works) ✓
- T7: bov_train registered → 430,147 records ✓ (ran BEFORE frames fully downloaded; pycocotools only parses JSON)

**FIXES APPLIED (committed to repo):**
1. .gitignore: added `!**/gomatching/data/**/*.py` — was blocking code module
2. gomatching/data/: copied from upstream Hxyz-123/GoMatching (data, transforms, datasets subdirs)
3. train.json: stripped 5.4M null bytes + filtered 1 corrupted annotation (2380083 valid remain)

**BLOCKERS / ISSUES:**
- BOVText download via `modal volume get /BOVText/frame <dest>` resets stream for large dirs → workaround: per-class loop in /tmp/download_bov.sh (nohup, PID=854165 ← may have restarted, check log on next tick)
- `modal volume get <dir>` downloads directories as zip archives when dest path doesn't have trailing slash — must download individual files
- deepsolo `requirements.txt` has conflicts but key deps (lap, motmetrics, editdistance, shapely) installed ✓

**NEXT TICK plan:**
- Check T3 download completion, run T6 (verify frame count ≈ 430k)
- Run T8 smoke (20 iters) once frames are there
- Config check: GoMatching_PP_BOVText.yaml uses VOC_SIZE 5462 + CUSTOM_DICT chn_cls_list — verify chn_cls_list exists

**QUESTION for Laptop Claude:**
1. The GoMatching_PP_BOVText.yaml config — does it handle Chinese text (chn_cls_list, VOC_SIZE 5462)? The upstream vts.py only encodes 36 alphanumeric chars. Does the BOVText training expect the bilingual vocab? Should we patch vts.py to handle Chinese text labels or just ignore them?
2. Do you want the GoMatching++ checkpoint (gomatching_iter30k/model_final.pth) for warm-start, or start from deepsolo_bov.pth only? RUNBOOK says deepsolo_bov.pth for MODEL.WEIGHTS.
