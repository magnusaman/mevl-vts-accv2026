# RUNBOOK — BOVText training on the A6000 (ACCV sprint)

Target: get **BOVText tracker training running today**, then ablation + recognition. IC15-V is the safety net.
Box: **RTX A6000 48GB, Linux, conda**. Everything below runs **on the box** (Box Claude executes; Aman relays).

> Box Claude: treat each phase as a gate. Don't start a phase until the previous one's acceptance check passes. Adapt commands as needed and record what you actually ran in `RESULTS_FROM_BOX.md`.

---

## Known facts (verified from the laptop)
- Data on Modal volume **`mevl-vts-datasets`**: `/BOVText/frame/` (extracted, classes `Cls7…Cls19`) + `/BOVText/train.json`; also `/ICDAR15_Video`, `/DSText`.
- Warm-start weights on Modal **`mevl-vts-weights:/pretrained_models/`**: `deepsolo_bov.pth`, `deepsolo_dstext.pth`, `deepsolo_icdar15.pth`, `deepsolo_icdar15_rescore.pth` (uploaded from laptop).
- GoMatching++ warm-start: **`mevl-vts-v7-checkpoints:/gomatching_iter30k/model_final.pth`**.
- BOVText config: `code/GoMatching_v7/configs/GoMatching_PP_BOVText.yaml` → `META_ARCHITECTURE GoMatching`, `FREEZE_TYPE ExceptROIheads` (only ROI heads train), `WEIGHTS pretrained_models/deepsolo_bov_ft.pth` **(file is actually `deepsolo_bov.pth` → override on CLI)**, `MAX_ITER 30000`, `IMS_PER_BATCH 1`, `DATASETS.TRAIN ("bov_train",)`, vocab `VOC_SIZE 5462` + `CUSTOM_DICT ./chn_cls_list` (bilingual).
- Dataset registration: `code/GoMatching_v7/gomatching/data/datasets/vts.py` registers `bov_train`, `icdar15_train`, `dstext_train` and expects them under `datasets/<DS>/{train.json, frame/}` relative to the GoMatching_v7 cwd. **Verify exact expected paths in vts.py before linking.**

---

## Phase 0 — Environment (gate: smoke import works)
Two conda envs (the tracker stack and the VLM stack have conflicting deps):

**`gom`** (tracker — detectron2 + AdelaiDet):
- Python 3.10 (NOT 3.12 — the old numpy/detectron2 pins won't build on 3.12).
- PyTorch with CUDA (cu121 or cu124 wheels work with driver 580/A6000).
- detectron2 0.6 + build AdelaiDet from `code/GoMatching_v7/third_party/adet` (`python setup.py build develop`).
- `pip install -r code/GoMatching_v7/requirements.txt`.
- Follow `code/GoMatching_v7/README.md` / `DENSETRACK_README.md` for the exact original GoMatching versions; adapt if a build fails.

**`rec`** (recognition — Qwen3-VL, do later, Phase 5):
- Python 3.10/3.11, recent torch, **`transformers==4.57.0`** (4.56 fails on `qwen3_vl`), `peft`, `bitsandbytes`, `accelerate`, `qwen-vl-utils`.

Acceptance: `python -c "import torch, detectron2; print(torch.cuda.get_device_name(0))"` prints `NVIDIA RTX A6000`.

## Phase 1 — Pull data + weights from Modal (gate: counts verified)
```bash
cd ~/aman/mevl-vts
mkdir -p code/GoMatching_v7/datasets code/GoMatching_v7/pretrained_models checkpoints
# weights (small, do first)
modal volume get mevl-vts-weights /pretrained_models code/GoMatching_v7/pretrained_models
modal volume get mevl-vts-v7-checkpoints /gomatching_iter30k checkpoints/gomatching_iter30k
# BOVText (BIG ~1.75M frames — run in the background / tmux, this is risk #1)
modal volume get mevl-vts-datasets /BOVText code/GoMatching_v7/datasets/BOVText
```
Acceptance:
- `ls code/GoMatching_v7/pretrained_models/` shows the 4 `.pth`.
- `python -c "import json;d=json.load(open('code/GoMatching_v7/datasets/BOVText/train.json'));print(len(d['images']),len(d['annotations']))"` returns sane counts.
- frame count on disk ≈ `len(images)` (spot-check a couple of `Cls*` dirs).

## Phase 2 — Register dataset (gate: detectron2 sees `bov_train`)
- Open `gomatching/data/datasets/vts.py`; confirm the path `bov_train` expects, then symlink to match:
  `ln -s $(pwd)/datasets/BOVText/train.json datasets/<expected>` etc. (mirror what vts.py builds).
- Quick check: a tiny python snippet that calls `DatasetCatalog.get("bov_train")` and prints the first record without error.

## Phase 3 — Smoke train (gate: 20 iters, no NaN, no OOM, ckpt saved)
```bash
cd code/GoMatching_v7
python train_net.py --config-file configs/GoMatching_PP_BOVText.yaml --num-gpus 1 \
  MODEL.WEIGHTS pretrained_models/deepsolo_bov.pth \
  SOLVER.MAX_ITER 20 SOLVER.CHECKPOINT_PERIOD 20 \
  OUTPUT_DIR ./trained_models/GoMPP_BOVText_smoke
```
(If `--config-file`/`--num-gpus` differ, check `python train_net.py --help`.) Watch loss is finite and ~48GB not exceeded (it won't be — IMS_PER_BATCH 1, frozen backbone).

## Phase 4 — Full BOVText train (gate: loss converging, ckpts every 5k)
```bash
cd code/GoMatching_v7
nohup python train_net.py --config-file configs/GoMatching_PP_BOVText.yaml --num-gpus 1 \
  MODEL.WEIGHTS pretrained_models/deepsolo_bov.pth \
  OUTPUT_DIR ./trained_models/GoMPP_BOVText > ../../train_bov.log 2>&1 &
```
- MAX_ITER 30000, ckpt every 5000. Report iter/s + ETA. Tune MAX_ITER to fit the deadline if needed.
- **In parallel, kick off the IC15-V safety-net run** the same way with `configs/GoMatching_PP_ICDAR15.yaml` + `deepsolo_icdar15_rescore.pth` (only if VRAM/time allows; otherwise queue after BOVText).

## Phase 5 — Eval + recognition (after a checkpoint exists)
- Tracker eval: `python eval.py ...` → produces RRC-style output (check `eval.py --help`; mirror `GoMatching_PP_BOVText.yaml` TEST). **Open item:** BOVText official test scoring protocol — resolve which split/server we report on.
- Recognition C1: in `rec` env, build the multiframe manifest (`code/recognition_lora/build_manifest_multiframe.py`) on the pulled data, then `train_qwen_lora_multiimage.py`. Compare whole-track fusion vs best-frame vs vote (the gate ablation).

## Phase 6 — C2 (text-as-identity track repair)
- Wire recognized text into the LST-Matcher association cost (laptop will ship this code via the repo). Ablation: baseline → +C1 → +C2 → full.

---
**Risk #1 = BOVText download time** (largest single dependency). Start Phase 1's BOVText pull FIRST, in the background, and do Phase 0 env setup while it downloads.
