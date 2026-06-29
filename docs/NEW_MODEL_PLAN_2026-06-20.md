# NEW MODEL PLAN — Video Text Spotting (MEVL-VTS / TIM-VTS)

**Date:** 2026-06-20 · **Target:** Pattern Recognition (Elsevier) / IEEE-T (TMM, TCSVT)
**Advisors:** Prof. Shivakumara (Salford), Prof. Partha Pratim Roy (IITR)
**Supersedes** the model framing in `MASTER_PLAN_2026-06-12.md` (kept for phase/runbook detail).

This plan is the result of a 4-agent literature sweep + a 4-agent repo audit (architecture,
SOTA, novelty/publishability, competitors; old-results, asset inventory, prior-submission rebuttal).

---

## 0. The premises (evidence-grounded, honest)

1. **Field SOTA = GoMatching++ (2025)** on IC15-V / DSText / BOVText, recipe = *frozen image
   spotter (DeepSolo) + lightweight trainable LST-Matcher + rescoring*. This is the right base.
2. **GoMatching++ admits, in its own paper, three open gaps:** (a) recognition is **never
   refined** — frozen, per-frame — and they call it *"the bottleneck"*; (b) **no joint /
   multi-dataset training** (naive joint optimisation "conflicts"); (c) **tiny text + reflections
   fail** (= DSText, where spotting MOTA is only 23.23 → most headroom).
3. **What we actually have:** a completed **IC15-V** GoMatching++ run (`GoMPP_IC15/model_final.pth`,
   ~62.5 MOTA tracking, RRC provenance unverified). **DSText/BOVText were never trained** (smoke only).
4. **Prior ECCV 2026 paper (MEVL-STP #11026, the IMAGE project) is leaning reject** (scores 3/2/4).
   The killer criticism, from two reviewers: **"novelty = integration of pretrained models, not a
   new methodology" / "brute force."** Plus: efficiency, low precision/texture-FPs, hand-wavy theory,
   thin failure analysis. **Every one of these must be designed out of the new (video) model.**
5. **VLM recognition in video is nascent** (EVE = subtitles only). Closest competitor =
   **Gather-and-Trace (ACM MM'25)**: instance-level cross-frame *feature* fusion for Video-TextVQA —
   NOT a generative VLM reading multiple crops, NOT spotting. That's our differentiation seam.

---

## 1. The new model — one conceptual shift

> **The track — not the frame — is the unit of recognition, and the recognised text is itself a
> tracking signal.**

In all current VTS, recognition is a frozen, per-frame, *passive* output. We make it
**track-level, trained, and an active tracking cue.** This hits all three GoMatching++ gaps and is a
genuine *mechanism* (not integration), directly answering the ECCV reviewers.

### Architecture

```
            ┌─────────────────────────── FROZEN ───────────────────────────┐
  frames →  │  ResNet-50  →  DeepSolo spotter (per-dataset weights)         │
            │     → per-frame polygons + per-frame transcription + queries  │
            └───────────────────────────────────────────────────────────────┘
                         │ queries/boxes                 │ per-frame text
                         ▼                               ▼
        ┌─ TRAINABLE (warm-start GoMatching++ iter-30k) ──────────────────────┐
        │  LST-Matcher (ST + LT cross-attn) + rescoring head   ~11.8M params   │
        │     + [M2] Text-as-Identity term in the association cost (gated)     │  ← NEW mechanism
        └──────────────────────────────────────────────────────────────────────┘
                         │ tracks (crops across frames)
                         ▼
        ┌─ [M1] Track-Level Recognition Fusion ──────────────────────────────┐
        │  Qwen3-VL-8B + LoRA, FINE-TUNED ON MULTI-IMAGE input                │  ← NEW mechanism
        │  K legibility-selected crops of one track → ONE transcription       │
        └──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼  official spotting MOTA / IDF1 (RRC)
```

### The two NEW mechanisms (both trained/wired — this is the fix for the audit)

- **M1 — Track-Level Recognition Fusion (headline).** Fine-tune the Qwen3-VL-8B LoRA on
  **multi-image** inputs: K temporal crops of one tracked word → one transcription target. Currently
  the adapter is trained on *single* crops and only *prompted* with multi-image at inference (a
  train/inference mismatch). Fixing this turns a prompt trick into a learned fusion module — the
  recognition refinement GoMatching++ lacks. Improves spotting directly (correct read = true positive).
- **M2 — Text-as-Identity Association (co-contribution that moves the tracking metric).** Add a
  recognised-text similarity term to the LST-Matcher association cost, **gated by motion/appearance**
  (naive string-match alone was proven insufficient in the 06-10 experiments). Merges fragmented
  tracks, fixes ID-switches → IDF1 ↑, IDsw ↓. Uses DeepSolo's per-frame text (no giant VLM cache).
  This is the load-bearing *new layer/term* reviewers demanded.

### Training regime (co-contribution, not headline)

- **M3 — Joint multi-dataset training** on **BOVText + IC15-V + DSText** (`MultiDatasetSampler`,
  ratio 1:1:1). The thing GoMatching++ can't do; targets tiny text (DSText).

### Explicitly DROPPED / demoted (to dodge the ECCV rejection)

- **Multi-Encoder Consensus (MEC) / DenseTrack-v7 Comp1/Comp2** — this is the *exact* "stack frozen
  foundation models" idea two ECCV reviewers called brute-force. **Not the headline.** At most a
  one-line ablation. Bonus: this removes our biggest infra blocker (the v7 encoder/SAM caches are
  empty and infeasible to build over BOVText's ~1.75M frames).
- **"Orthogonal Energy" / metaphorical theory** — banned. Only claim what a loss actually optimises.

---

## 2. How the new model answers every prior-reviewer criticism

| ECCV criticism (image paper) | Fix in the new video model |
|---|---|
| "Novelty = integration, not method" | M1 (learned multi-image fusion) + M2 (new gated text term in tracker) are real mechanisms with losses, not stacked encoders. |
| Efficiency (2.2B params, ~1 FPS) | Trainable params ~12M (tracker) + ~16M (LoRA); report params/FPS/FLOPs **in the paper from day one**; frame as "robustness," not real-time. Drop MEC = far lighter. |
| Low precision / texture FPs | Video: report ID-switches/MT/ML + failure taxonomy; recognition gate rejects empty/"###" crops. |
| Hand-wavy theory | No metaphor math; state the association cost + loss exactly. |
| Weak generalisation / small data | Multi-dataset (BOVText+IC15+DSText) + cross-dataset row built in. |
| No failure analysis | Build error-taxonomy harness from the start (blur/small/occlusion slices — gate already shows these). |

---

## 3. Reuse vs build (asset-grounded)

**REUSE (already trained / present):**
- Base: `E2E_FINAL_BACKUP/v6_checkpoints/GoMPP_IC15/model_final.pth` (warm-start), per-dataset
  `pretrained_models/deepsolo_{icdar15,dstext,bov}.pth`.
- Recognizer: `recognition_lora_outputs/final_adapter` — **Qwen3-VL-8B** LoRA (pin this base).
- Configs: `GoMatching_v7/configs/GoMatching_PP_MultiTrain.yaml` (+ per-dataset).
- Data prep: `misc/modal_data_setup.py`; converters `tools/convert_gom_label/{icdar15,dstext,bovtext}.py`.
- `unified_annotations.json` (267 MB) for manifest building.

**BUILD (net-new work):**
1. **Track-grouped multi-frame manifest** (group the 317k single-crop rows by track-ID across frames)
   — derive from `unified_annotations.json` (+ DSText/BOVText anns).
2. **Multi-image LoRA trainer** — adapt `train_qwen_lora_l4_v9.py` to K-image input + the manifest.
3. **M2 text-identity term + gate** in `roi_heads/shared_ffn_crsattn.py` association cost (the 83-param
   `content_matcher.py` is a usable skeleton; feed DeepSolo per-frame text, NOT a missing VLM cache).
4. **Efficiency + failure-analysis harness** (params/FPS/FLOPs table; blur/small/occlusion slices).

**VERIFY before GPU spend:** live Modal state of `mevl-vts-datasets` — runbook says DSText-90 +
BOVText-485 `train.json` were built; asset scan (local-only) couldn't confirm. Confirm on Modal;
extract DSText frames / finish BOVText if needed.

---

## 4. Training recipe (BOVText + IC15-V + DSText)

- **Stage 0 — Data + sanity.** Verify Modal volume; ensure frames + COCO `train.json` for all 3;
  build the track-grouped manifest. (No GPU.)
- **Stage 1 — Multi-dataset base tracker.** Train LST-Matcher + rescoring on the 3 datasets,
  warm-start iter-30k, 60k iters, `MultiDatasetSampler`. → stronger base **+ our first real
  DSText/BOVText baselines.** Eval → RRC (IC15 ch3, DSText ch22). ~heads-only, ~4 GB, fits L4.
- **Stage 2 — Multi-image recognition LoRA (M1).** Fine-tune Qwen3-VL-8B on the track-grouped
  multi-crop manifest. Ablate vs A1 best-frame / A1.5 vote / Gather-and-Trace-style feature fusion;
  report gains on blur/small/long-track.
- **Stage 3 — Text-as-Identity association (M2).** Wire the gated text term, train jointly
  (warm-start Stage 1), gate init ≈ 0 so baseline reproduces, unfreeze on warmup. Eval ID-switch/IDF1.
- **Stage 4 — Full system + ablations.** A0 base · +M1 · +M2 · +both · multi-dataset vs single ·
  efficiency table · failure taxonomy. Build the paper tables.

---

## 5. Targets & honest odds

| Benchmark | GoMatching++ (to beat), spotting MOTA/IDF1 | Our realistic aim |
|---|---|---|
| IC15-V | 72.20 / 80.11 | ≥ baseline (tight; hard to clearly beat) |
| **DSText** | 23.23 / 46.24 | **best headroom — primary SOTA target** |
| BOVText | 52.9 / 62.8 | report; competitive |

- Pipeline runs + produces official numbers: **~100%**.
- Beat GoMatching++ on **DSText** (the soft target): **likely (~65%)**.
- Clear new SOTA on **IC15-V**: **~40%** (tight).
- **Defensible PR / IEEE-T paper** (2 real mechanisms + multi-dataset + ablations + efficiency +
  failure analysis): **~70%**.
- Recognition-fusion's edge over simple voting is **modest on exact-match, real on blur/1-NED**
  (our gate, n=418) — that's *why* M2 exists: a second, independent axis of gain.

---

## 6. The one-line pitch for Dr. Shiva

A new video-text-spotting model in which a VLM reads each word once from its **whole track** (learned
multi-frame fusion) and the **recognised text repairs the tracking** (gated text-identity association),
trained **jointly across BOVText + ICDAR15-Video + DSText** — directly targeting the recognition and
joint-training gaps GoMatching++ leaves open, evaluated by official spotting MOTA/IDF1.
