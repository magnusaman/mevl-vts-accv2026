# MEVL-VTS — Video Text Spotting (ACCV 2026)

**Text-as-Identity for Video Text Spotting.** Read each tracked word **once from its whole track** with a vision-language model, then use that recognized text to **repair the tracking** (merge tracklets, fix ID-switches). Recognized text becomes an active temporal cue, not just the output.

- **Datasets:** BOVText (headline), ICDAR15-Video, DSText.
- **Base:** frozen DeepSolo image spotter + GoMatching++ LST-Matcher tracker (only ROI heads train, ~12M params).
- **New:** C1 = whole-track VLM recognition fusion · C2 = text-as-identity track repair.
- **Plan:** `docs/ACCV_2026_PLAN.md` · **Step-by-step:** `docs/RUNBOOK.md`

## This is a two-machine setup

| | Where | Role |
|---|---|---|
| **Laptop Claude** | Aman's Windows laptop | lead / architect / writes code + runbook + the paper |
| **Box Claude** | college GPU box (RTX A6000, Linux), via AnyDesk | hands on the GPU: env, data, training, eval |

They talk through this repo. See `ORCHESTRATION/PROTOCOL.md`.
**Box Claude: read `CLAUDE.md`, then `ORCHESTRATION/INSTRUCTIONS_FOR_BOX.md`, then `docs/RUNBOOK.md` — in that order.**

## Quickstart (on the GPU box)
```bash
git clone https://github.com/magnusaman/mevl-vts-accv2026.git ~/aman/mevl-vts
cd ~/aman/mevl-vts
bash scripts/box_bootstrap.sh        # env + modal auth check (read it first)
# then follow docs/RUNBOOK.md
```

## Layout
```
code/GoMatching_v7/      tracker training/eval (detectron2 + AdelaiDet based)
code/recognition_lora/   Qwen3-VL LoRA recognition (C1 whole-track fusion)
docs/                    ACCV plan + RUNBOOK
ORCHESTRATION/           the laptop<->box message bus
scripts/                 box bootstrap helpers
```

⚠️ **Public repo.** No weights, data, or secrets — see `.gitignore`. Data + weights live on Modal and are pulled to the box.
