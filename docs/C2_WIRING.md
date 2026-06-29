# C2 Wiring Plan — Recognition-Driven Association

Goal: make recognized text a **gated identity term** in the LST-Matcher association cost, sourced from the **in-pipeline DeepSolo recognition head** (no separate offline VLM cache). Module already exists: `gomatching/modeling/v7_densetrack/content_matcher.py` (`VLMContentMatcher`). This doc is the integration recipe — validate against code on the box before training.

## What exists (reviewed)
`VLMContentMatcher.forward(base_sim, query_texts, key_track_consensus)`:
- adds `content_w * sim_temperature * fuse([edit_ratio, len_diff, conf])` to `base_sim`.
- `content_w = sigmoid(content_logit)`, init `content_logit=-6` → ≈0 so the baseline is exactly reproduced at start; unfreezes via warmup.
- `consensus_text(votes)` = per-track majority vote → (text, confidence).
- self-test passes; ~5K trainable params.

## What to change (the "wiring", not a rebuild)
1. **Source per-detection text from DeepSolo, not a cache.** DeepSolo's head already decodes text per query (VOC_SIZE 5462, CUSTOM_DICT `./chn_cls_list`). In the matcher's forward, decode the current frame's detections to strings (`query_texts`) from the spotter output already in the pipeline. Drop any dependency on `tools/cache_vlm_text.py`.
2. **Maintain per-track text consensus online.** As tracks are extended each frame, push the matched detection's decoded string into the track's vote list; `key_track_consensus[j] = consensus_text(track_j.votes)`. Store the small vote list on the track/memory object.
3. **Inject into the association similarity.** In the LST-Matcher association step (`shared_ffn_crsattn.py`, the v7 components are instantiated ~L219-234 under `DENSETRACK.ENABLED`), wrap the base similarity `S` with `S = content_matcher(S, query_texts, key_track_consensus)` before the matching/softmax. Confirm the exact tensor that is the (M=current dets, N=tracks) similarity and that text lists are aligned to the same M/N order.
4. **Loss/gradient.** No new loss needed — the association loss (`ASSO_WEIGHT`) already supervises the matcher; the gate + fuse MLP learn through it. Keep `content_logit` warmup so early training matches baseline.

## Config knobs to add (under MODEL.DENSETRACK or similar)
- `CONTENT_ON: True`, `CONTENT_WARMUP_ITERS: 1000` (unfreeze gate after), `CONTENT_USE_MLP: True`.

## Ablation hooks
- `CONTENT_ON False` → baseline tracker (C2 off).
- `CONTENT_ON True` → +C2. Log `sigmoid(content_logit)` over training (should rise above ~0 if text helps) and ΔID-switches / ΔIDF1.

## Risks / checks before the real run
- Confirm M/N ordering of `query_texts` vs `key_track_consensus` matches `base_sim` rows/cols (a transpose bug silently hurts).
- Chinese text: ensure decoded strings are comparable (normalize width/case for Latin; keep CJK as-is). `rapidfuzz` handles unicode.
- Cost: per-frame decode is already done by DeepSolo; consensus vote is O(track len). Negligible vs the spotter.

## Status
- [ ] Box: locate exact similarity tensor + M/N order in `shared_ffn_crsattn.py`.
- [ ] Laptop: provide the edited association hook once the box confirms the tensor shapes/order.
- [ ] Smoke: 20-iter run with `CONTENT_ON True` must equal baseline at init (gate≈0), no shape errors.
