# ACCV 2026 — Research Plan (for Prof. Shivakumara & Prof. Partha Pratim Roy)

**Author:** Aman Anand
**Target:** ACCV 2026 (paper deadline **5 July 2026**)
**Working title:** *Text-as-Identity: Reading and Tracking Scene Text Across Video by Whole-Track Recognition*
**GPU:** provided by Prof. Pal

---

## 1. One-line idea

> In video, every method still **reads each frame independently and votes**. We instead treat a tracked word as **one object with one identity**: we read it **once from its whole track** with a vision-language model, and we then **use that recognized text to fix the tracking itself** — merging broken tracklets and repairing ID switches. Recognized text stops being only the *output* and becomes an **active cue that improves spotting**.

**The new challenge we put forward:** *recognition robustness in degraded video* — a word read correctly in one frame is misread the next (blur, motion, scale change), and in end-to-end spotting a single wrong read counts as a miss. This is the field's open problem, and no existing video spotter attacks it with cross-frame reading.

---

## 2. How this is different from our ECCV work (the key question)

Our ECCV submission (**MEVL-STP #11026**) was an **image** text-spotting paper: multi-encoder fusion + a VLM recognizer on still images (CTW1500 / Total-Text / IC15). Its main reviewer criticism was that *"the novelty lies in integration rather than a new method."*

This ACCV paper is a **different problem and a different contribution**, and it is built specifically to avoid that criticism:

| | ECCV work (MEVL-STP) | ACCV work (this paper) |
|---|---|---|
| **Task** | Image text spotting (single frame) | **Video** text spotting (detect + **track** + recognize over time) |
| **Datasets** | CTW1500, Total-Text, IC15 (images) | **ICDAR15-Video, DSText, BOVText** (video) |
| **Metrics** | Precision/Recall/H-mean | **MOTA, IDF1, ID-switches** + track-level word accuracy |
| **Core novelty** | Combining frozen encoders + VLM (an *integration*) | **A trained mechanism:** whole-track recognition + **text-driven track repair** |
| **What's new scientifically** | — | Recognized text used as a **temporal identity signal that corrects tracking** (recognition → tracking feedback) |
| **Answer to ECCV critique** | "integration, not method" | Leads with a learned module + a measurable new effect, **not** an encoder stack |

**In one sentence for the professors:** *the ECCV paper recognizes text in a still image; this paper recognizes a word from its entire trajectory in a video and feeds that identity back to repair the trajectory — a video-only mechanism that does not exist in the image paper.*

This is a focused **conference instalment of our journal plan** (Text-Instance Memory). ACCV takes the two most exciting, deliverable pieces; the full memory framework (incl. missed-detection recovery) stays for the journal.

---

## 3. Proposed method (scoped for ACCV)

We keep the proven GoMatching++ recipe (freeze a strong image spotter, train a light video head) and add **two trained contributions**:

```
video frames
   │
   ▼
[FROZEN DeepSolo image spotter]  ─► per-frame detections + features
   │
   ▼
[LST-Matcher tracker]  ─► tracklets (box + ID over time)
   │
   ├──► (C1) WHOLE-TRACK RECOGNITION
   │         gather a track's crops across frames → legibility-gate (drop blur/occlusion)
   │         → fuse surviving views → decode ONCE with a LoRA Qwen-VL (multi-image prompt)
   │
   └──► (C2) TEXT-AS-IDENTITY TRACK REPAIR
             add a gated recognized-text term to the association cost:
             merge tracklets / fix ID-switches when text identity agrees AND
             motion + appearance are compatible; reject merges when text conflicts.
```

- **Frozen:** DeepSolo spotter, VLM backbone (LoRA adapters only) → fits one 24 GB GPU, trainable params ~12M.
- **Trained:** LST-Matcher, LoRA recognition adapters (multi-image), the text-as-identity association gate.
- **Dropped on purpose:** multi-encoder consensus (that was the ECCV "integration" idea — explicitly out, or a one-line ablation only).
- **Two contributions, mutually reinforcing:** C1 makes the read robust; C2 turns that robust read into better tracking (MOTA/IDF1). This recognition→tracking feedback is the scientific novelty.

---

## 4. Preliminary evidence (already in hand)

We have run the recognition gate on **ICDAR15-Video** ground-truth tracks (n = 418), comparing three ways to read a tracked word with the *same* VLM adapter:

| Method | Exact | Normalized | 1-NED |
|---|---|---|---|
| A1 — best single frame (YORO-style) | 25.6 | 28.7 | 47.6 |
| A1.5 — per-frame read + vote | 30.9 | 34.2 | 52.4 |
| **A2 — whole-track fusion (ours, C1)** | **31.8** | **36.1** | **58.2** |

- On the **blurred quartile** (n = 105), whole-track fusion gives **+10.8 points 1-NED** over voting — i.e. the gain is exactly where the challenge is (degraded frames).
- **Reading:** whole-track recognition is a real, defensible robustness win (clearly beats best-frame and beats voting, especially on character fidelity and blur). C2 (text-fixes-tracking) is what converts this into a MOTA/IDF1 gain — that is the experiment to run on Prof. Pal's GPU.

Base GoMatching++ tracker is already reproduced on IC15-V; official RRC re-scoring is pending.

---

## 5. Datasets & evaluation

| Dataset | Role | Status |
|---|---|---|
| ICDAR15-Video | primary baseline + ablations | ready (frames + annotations) |
| DSText | dense/small-text stress test | prepared (90 vids / 38.7k frames) |
| BOVText | large real spotting benchmark | needs advisor-signed agreement (Dr. Shiva) |

**Metrics:** MOTA, IDF1, ID-switches, track-level word accuracy + **1-NED on a blurred subset** (our degraded-video robustness protocol — a small evaluation novelty).

**Ablation:** baseline GoMatching++ → +C1 whole-track recognition → +C2 text-as-identity repair → full model.

---

## 6. Timeline to 5 July (6 days)

| Day | Work |
|---|---|
| D0 (now) | **Send this plan to professors for review** (the immediate deliverable) |
| D1 | Prof. Pal GPU setup, pull data, reproduce baseline + official RRC numbers |
| D2–D3 | Train C1 (multi-image recognition) + wire C2 (text-as-identity gate) |
| D3–D4 | Full ablation on IC15-V (+DSText if time); blurred-subset eval |
| D4–D5 | Write paper (method + tables + figures) |
| D5–D6 | Polish, internal review, submit |

**Realistic scope:** strong, complete results on **IC15-V** (+ DSText if data/time permit) with the full ablation. BOVText is a stretch goal gated on the data agreement.

---

## 7. What I need from the professors

1. **Confirm the headline framing** (Text-as-Identity / whole-track recognition) — or redirect.
2. **BOVText data agreement** (Dr. Shiva) if we want it in the table.
3. **Prof. Pal GPU access details** to start D1.

---

*This plan is the focused ACCV instalment of our journal (Text-Instance Memory) direction. It is a video paper with a trained, video-only mechanism — distinct from the ECCV image paper in task, data, metrics, and core novelty.*
