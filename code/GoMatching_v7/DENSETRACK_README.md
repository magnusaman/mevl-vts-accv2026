# DenseTrack v7 — runbook

Three-component improvement over the reproduced GoMatching++ baseline, targeting
RRC IC15-V (and DSText) leaderboards.

## What's been built so far

### Components (each opt-in via config flag, gate=0 ⇒ baseline behavior)

| # | Name | File | Trainable | Failure mode it attacks |
|---|------|------|-----------|--------------------------|
| 1 | SAMProposalAugmenter | `gomatching/modeling/v7_densetrack/proposal_augmenter.py` | 262,658 | tiny-text recall (DSText) |
| 2 | MultiEncoderConsensusMatcher | `gomatching/modeling/v7_densetrack/consensus_matcher.py` | 1,246,213 | visually-confusable text |
| 3 | VLMContentMatcher | `gomatching/modeling/v7_densetrack/content_matcher.py` | 83 | ID switches across occlusion |
| | **Total v7 trainable** | | **1,508,954** | |
| | GoMatching++ base trainable | | 11,800,000 | |
| | **Grand total** | | **~13.3M** | (fits L4 24 GB) |

### Wire-up patches (touched original GoMatching files in our fork)

- `gomatching/config.py` — added `MODEL.DENSETRACK.*` config keys
- `gomatching/modeling/freeze_layers.py` — `ExceptROIheads` also unfreezes v7 components
- `gomatching/modeling/roi_heads/shared_ffn_crsattn.py` — instantiates Comp2/Comp3; calls them in `_forward_transformer`
- `gomatching/modeling/meta_arch/gom_lstmatcher.py` — instantiates Comp1; calls it in `forward`; `_v7_set_roi_ctx()` plumbs encoder features / texts / SAM data down to `roi_heads._v7_ctx`

### Config

- `configs/DenseTrack_PP_ICDAR15.yaml` — inherits GoMatching++ base, enables all 3 v7 components, warm-starts from iter-30k, 12k iters at lr=2e-5.

### Modal apps

| App / function | Purpose | GPU | Approx cost |
|---|---|---|---|
| `modal_v7_setup.py::verify_image` | smoke 1: env + deps | L4 | < $0.01 |
| `modal_v7_setup.py::verify_baseline_load` | smoke 2: iter-30k checkpoint loads | L4 | $0.02 |
| `modal_v7_setup.py::verify_smoke_forward` | smoke 3: baseline forward+backward on synthetic input | L4 | $0.10 |
| `modal_v7_setup.py::verify_components` | smoke 4: 3 v7 unit tests | L4 | $0.10 |
| `modal_v7_setup.py::verify_deep_integration` | smoke 5: full pipeline with all 3 v7 active | L4 | $0.10 |
| `modal_v7_setup.py::cache_encoders_ic15v --encoder X` | Phase 1: cache one encoder over IC15-V | L4 | $0.30 / encoder |
| `modal_v7_setup.py::cache_sam_ic15v` | Phase 1: cache SAM-H proposals over IC15-V | L4 | $5-10 (slow) |
| `modal_v7_setup.py::cache_status` | progress check on cache volumes | CPU | $0 |
| `modal_v7_train.py::train --max-iter 12000` | Phase 2: fine-tune from iter-30k | L4 | $13-20 |
| `modal_v7_train.py::eval_ic15v` | Phase 3: eval + RRC submission gen | L4 | $1-2 |

**Total budget for full DenseTrack pipeline ≈ $25-35.**

### Modal volumes

| Volume | Contents | Status |
|---|---|---|
| `mevl-vts-v7-checkpoints` | iter-30k baseline `model_final.pth` | ✓ uploaded (298 MB) |
| `mevl-vts-v7-encoder-cache` | per-encoder per-video HDF5 feature maps | empty — Phase 1 fills it |
| `mevl-vts-v7-data` | (future) DSText extras if needed | empty |
| `mevl-vts-v7-outputs` | training output dirs | empty |
| `mevl-vts-datasets` | IC15-V frames (existing) | ✓ has `/ICDAR15_Video/frames/Video_*/` |
| `mevl-vts-weights` | (existing) other weights | unused for v7 |

## Verified so far (each on Modal L4)

- ✓ Stage 1: image builds (took 3 deploys; final one with `clang` + `wheel` + `setuptools`)
- ✓ Stage 2: GoMatching++ iter-30k checkpoint loads with 11.8M trainable
- ✓ Stage 3: synthetic 6-frame forward+backward through baseline works
- ✓ Stage 4: all 3 v7 component unit tests pass
- ✓ Stage 5: deep integration (all 3 v7 components active, mock context) — model runs end-to-end

## Phase 1: cache builds (the next thing to run)

```bash
# Smoke (2 videos, ~5 min):
modal run modal/modal_v7_setup.py::cache_encoders_ic15v --encoder clip-l-336 --limit-videos 2

# Full (all 49 videos, in parallel for the fast encoders):
modal run modal/modal_v7_setup.py::cache_encoders_ic15v --encoder clip-l-336 --detach  # ~15 min
modal run modal/modal_v7_setup.py::cache_encoders_ic15v --encoder dinov2-l --detach    # ~20 min
modal run modal/modal_v7_setup.py::cache_encoders_ic15v --encoder convnext-l --detach  # ~10 min
modal run modal/modal_v7_setup.py::cache_encoders_ic15v --encoder sam-h --detach       # ~60 min

# SAM-H 'segment everything' proposals (slow, ~4-8h):
modal run modal/modal_v7_setup.py::cache_sam_ic15v --detach

# progress check:
modal run modal/modal_v7_setup.py::cache_status
```

## Phase 2: training (after Phase 1 caches land)

```bash
modal run modal/modal_v7_train.py::train --max-iter 12000 --detach
# ~14-18h on L4. Saves checkpoints every 1000 iters to /outputs/DenseTrack_IC15/
```

## Phase 3: eval + submission

```bash
modal run modal/modal_v7_train.py::eval_ic15v --use-iter 12000
# Output: /outputs/DenseTrack_IC15/eval/
# Then build RRC zip locally and submit to https://rrc.cvc.uab.es/?ch=3
```

## Still to build

- `tools/cache_vlm_text.py` — Component 3's transcription cache (needs the Qwen-VL LoRA that's training on E2E box, ETA ~21h from this writing)
- DSText support: encoder cache for DSText frames (will need to upload DSText frames to a Modal volume first; ~5 GB)
- Joint IC15-V + DSText training config + sampler (RepeatFactorTrainingSampler with 2:1 weighting)
- RRC submission zip generator (currently inherits GoMatching's existing pipeline)
