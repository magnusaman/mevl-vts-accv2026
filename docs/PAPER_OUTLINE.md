# Paper Outline — ACCV 2026

**Title (working):** Text-as-Identity: Whole-Track Recognition and Recognition-Driven Tracking for Video Text Spotting
**Venue:** ACCV 2026 (Springer LNCS format — confirm page limit on submission page). **Deadline:** July 5.

## Abstract (draft)
Video text spotting systems still read each frame independently and link detections with geometry/appearance cues, so a word read correctly in one frame is often misread the next, and recognition never feeds back to help tracking. We propose **Text-as-Identity**, built on a frozen image spotter and a lightweight tracker (GoMatching-style). Two contributions: (C1) **whole-track recognition** — gather a tracked word's crops across frames, legibility-gate them, and decode the word **once** with a vision-language model via a multi-image prompt; (C2) **recognition-driven association** — feed recognized text as a gated identity term into the tracker's matching cost to merge fragmented tracklets and suppress ID-switches. On BOVText / ICDAR15-Video / DSText we improve [MOTA/IDF1/word-acc], with the largest gains on blurred/degraded frames. (Fill numbers from runs.)

## Contributions (claim exactly these)
1. **Whole-track recognition (C1):** in-pipeline VLM that reads a tracked word from all its frames at once — not per-frame + vote. Robust to per-frame blur/occlusion.
2. **Recognition-driven association (C2):** recognized text as a learned, gated identity cue in the tracker → fewer ID-switches, recovered tracklets. Closes the recognition→tracking feedback loop that decoupled pipelines lack.
3. **Evidence:** multi-dataset eval + a degraded-frame (blur) protocol where recognition robustness matters most; ablation isolating each component.

## How this differs from our ECCV image paper (MEVL-STP #11026)
| | ECCV (MEVL-STP) | This (ACCV) |
|---|---|---|
| Task | image spotting | **video** spotting (detect+track+recognize over time) |
| Core | combine frozen encoders + VLM (integration) | **trained** mechanisms: whole-track read + text→tracking feedback |
| Data | CTW1500/Total-Text/IC15 | BOVText/IC15-Video/DSText |
| Metrics | P/R/H-mean | MOTA/IDF1/ID-sw + word-acc/1-NED |
> Explicitly drop multi-encoder fusion (the ECCV "integration not novelty" idea).

## Method
1. **Preliminaries / base.** Frozen DeepSolo image spotter (det+rec per frame, ResNet-50, 6enc/6dec, 100 queries, 25 Bézier pts, char head VOC=5462 bilingual). GoMatching++ LST-Matcher tracker (cross-attn-only decoder, rescoring head); only ROI heads train (`FREEZE_TYPE ExceptROIheads`, ~12M params).
2. **C1 — Whole-track recognition.** Per track: collect crops across frames → legibility gate (Laplacian-variance × log-area) → top-K → single multi-image VLM (Qwen3-VL-8B + QLoRA) decode. Trained on track-grouped crops (multi-image manifest). Contrast vs best-frame (YORO) and per-frame+vote.
3. **C2 — Recognition-driven association.** `VLMContentMatcher`: gated term added to the LST-Matcher similarity; per-track text consensus (majority vote) vs current-frame text; learned fuzzy-similarity MLP on (edit-ratio, len-diff, conf). Gate init ≈0 so the baseline is recovered; unfreezes after warmup. **Text sourced from the in-pipeline DeepSolo head (no separate cache).**
4. **Training.** Phased, not joint (joint optimization conflicts — GoMatching finding): Stage A tracker on BOVText (frozen spotter) → Stage B C1 LoRA → Stage C C2 fine-tune. One dataset at a time, BOVText first.

## Experiments
- **Datasets:** BOVText (headline), ICDAR15-Video, DSText.
- **Metrics:** MOTA, IDF1, ID-switches (tracking); word-acc + 1-NED (recognition); + a **blurred-subset** breakdown.
- **Scoring:** bundled `Evaluation_Protocol_BOV_Text/Task2_VideoTextSpotting/evaluation.py` (motmetrics); IC15-V/DSText via RRC protocol.
- **Ablation:** baseline tracker → +C1 → +C2 → full. Plus: K-sweep for C1; gate-weight trajectory for C2; per-frame-vote vs whole-track.

### Table skeletons
- **Tab.1 Main (per dataset):** Method | MOTA | IDF1 | ID-sw | word-acc — rows: GoMatching++ (repro), +C1, +C2, Full.
- **Tab.2 Recognition robustness:** best-frame | per-frame+vote | whole-track(C1) × {overall, blurred-quartile} on exact/1-NED. (Have preliminary IC15-V: 25.6 / 30.9 / 31.8 exact; 47.6 / 52.4 / 58.2 1-NED; +10.8pp 1-NED on blur.)
- **Tab.3 Ablation:** components on/off → ΔMOTA/ΔIDF1.

## Related work (cite, differentiate)
GoMatching / GoMatching++, TransDETR, LOGO, CoText, VimTS, DeepSolo (base), YORO (best-frame select), Gather-and-Trace (nearest prior — VideoTextVQA not spotting), LSTrack-SV (per-frame CTC + edit-dist; we = multi-frame VLM consensus + learned similarity), MME-VideoOCR (evidence MLLMs fail cross-frame text integration → motivates the gap). BOVText/IC15-V/DSText benchmark papers.

## Open / to confirm
- ACCV page limit + LaTeX template (fetch from submission site).
- BOVText train/test split usage for the reported numbers.
- Final headline confirm with Aman: C1+C2 together (recommended) vs single lead.
