# CLAUDE.md — MEVL-VTS (read this first)

You may be **Box Claude** (running on the college GPU box) or **Laptop Claude** (Aman's Windows laptop). Figure out which by your environment (`nvidia-smi` shows RTX A6000 → you're Box Claude) and act accordingly.

## Project
ACCV 2026 video text spotting. **Text-as-Identity:** read each tracked word once from its whole track (C1), use that text to repair tracking (C2). Datasets: **BOVText (first/headline)**, ICDAR15-Video, DSText. Deadline **5 July 2026** — submit by July 4. Full context: `docs/ACCV_2026_PLAN.md`.

## If you are BOX CLAUDE — your job
You are the hands on the A6000. Get **BOVText training running**, then ablation + recognition.
**Read order:** this file → `ORCHESTRATION/INSTRUCTIONS_FOR_BOX.md` → `docs/RUNBOOK.md`. Execute the next unchecked task, write outcomes to `ORCHESTRATION/RESULTS_FROM_BOX.md`, follow `ORCHESTRATION/PROTOCOL.md`.

- You may freely install packages, create conda envs, and **edit code to fix the environment** — that is why you exist on the box. Commit fixes with clear messages.
- Modal is **only for pulling data/weights** to the box (`modal volume get`). **All training runs on the A6000**, never on Modal.
- When you hit a wall after two honest tries: mark the task `BLOCKED` in INSTRUCTIONS, write the exact error + what you tried in RESULTS, and move to the next independent task. Aman relays blockers to Laptop Claude.

## Environment gotchas (save yourself hours)
- **Two conda envs.** `gom` (py3.10, detectron2 0.6 + AdelaiDet from `code/GoMatching_v7/third_party/adet`, `requirements.txt`) for the tracker; `rec` (py3.10/3.11, `transformers==4.57.0`, `peft`, `bitsandbytes`) for Qwen3-VL. They conflict — keep separate.
- **Do NOT use py3.12** for the tracker stack (old numpy/detectron2 pins fail to build).
- PyTorch cu121/cu124 wheels work with driver 580 on the A6000.
- BOVText config references `deepsolo_bov_ft.pth` but the real file is `deepsolo_bov.pth` → override `MODEL.WEIGHTS pretrained_models/deepsolo_bov.pth` on the CLI.
- `transformers==4.57.0` is required for the `qwen3_vl` model_type (4.56 errors).

## Hard rules
- **Public repo.** NEVER commit weights, datasets, frames, logs, or secrets. `.gitignore` enforces it — if you add a new artifact type, gitignore it first.
- Keep `RESULTS_FROM_BOX.md` append-only with UTC timestamps + task numbers.
- Don't edit `INSTRUCTIONS_FOR_BOX.md` except to flip a task's checkbox (`[ ]`→`[x]`/`[!]`).
