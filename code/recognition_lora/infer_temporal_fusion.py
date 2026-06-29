"""Temporal Recognition Aggregator (TRA) — the paper's recognition novelty.

Baseline (infer_swap_track_dedup.py): recognize each track from ONE representative
crop (the middle/best frame), then propagate that read to all frames — i.e.
best-frame *selection* (the YORO-style approach prior VTS uses).

TRA: gather up to K of a track's most *legible* crops across its frames, and decode
the word ONCE from all of them together via a multi-image VLM prompt. A word that is
blurred or occluded in some frames is recovered from the frames where it is clean.
This is the in-network cross-frame recognition *fusion* that prior VTS lacks.

Drop-in: identical JSON I/O to infer_swap_track_dedup.py, so the two are directly
comparable for the ablation (best-frame selection vs multi-frame fusion).

Usage (run yourself; Modal/GPU env):
  python infer_temporal_fusion.py --jsons-dir <tracks> --frames-root <frames> \
         --adapter <lora> --out-dir <out> --k-frames 4
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm

# Reuse the baseline's crop/model helpers (same directory) so the two paths share
# identical pre-processing — the only difference is selection vs fusion.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_swap_track_dedup import crop_and_pad, find_frame, load_model  # noqa: E402

FUSION_PROMPT = (
    "These images are crops of the SAME word, taken from consecutive frames of a "
    "video. Some views may be blurred, partially occluded, or low-resolution. "
    "Using all of the views together, read the word. Output only the text, exactly "
    "as written."
)


def legibility_score(img_bgr, polygon):
    """Higher = cleaner, more readable crop.

    Sharpness (variance of Laplacian) is the main signal — motion blur collapses
    high-frequency content, so a blurred crop scores low. We modulate by log-area
    so larger (more pixels = more recoverable detail) crops are mildly preferred
    without a few huge crops dominating.
    """
    crop, area = crop_and_pad(img_bgr, polygon, target_size=128)
    if crop is None:
        return -1.0
    g = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    return float(sharp) * (1.0 + float(np.log1p(area)))


def select_legible_crops(img_cache, frames_root, video, entries, k, target_size):
    """entries: list of (fid_str, det_idx, polygon). Returns (crops, weights).

    crops:   up to k highest-legibility PIL crops of this track.
    weights: their normalized legibility scores (usable as TCL frame weights).
    """
    scored = []
    for fid_str, _i, poly in entries:
        if fid_str not in img_cache:
            fp = find_frame(frames_root, video, int(fid_str))
            img_cache[fid_str] = cv2.imread(fp) if fp else None
        img = img_cache[fid_str]
        if img is None:
            continue
        s = legibility_score(img, poly)
        if s > 0:
            scored.append((s, fid_str, poly))
    scored.sort(key=lambda t: t[0], reverse=True)

    crops, weights = [], []
    for s, fid_str, poly in scored[:k]:
        crop, _ = crop_and_pad(img_cache[fid_str], poly, target_size=target_size)
        if crop is not None:
            crops.append(crop)
            weights.append(s)
    if weights:
        tot = sum(weights)
        weights = [w / tot for w in weights]
    return crops, weights


@torch.inference_mode()
def fuse_recognize(proc, model, pil_images, max_new_tokens=12):
    """One decode from K images of the same word via a multi-image prompt.

    Qwen2-VL / Qwen3-VL natively accept multiple images in a single message; the
    K crops become K image placeholders, fused by the model's attention before it
    emits a single transcription.
    """
    if not pil_images:
        return ''
    content = [{"type": "image"} for _ in pil_images]
    content.append({"type": "text", "text": FUSION_PROMPT})
    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=pil_images, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out[:, inputs['input_ids'].shape[1]:]
    return proc.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons-dir', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--model-id', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--target-size', type=int, default=224)
    ap.add_argument('--limit-videos', type=int, default=0)
    ap.add_argument('--min-track-len', type=int, default=5)
    ap.add_argument('--k-frames', type=int, default=4,
                    help='max legible crops to FUSE per track (k=1 ≈ the dedup baseline)')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.model_id, args.adapter)

    jsons = sorted(Path(args.jsons_dir).glob('*.json'))
    if args.limit_videos:
        jsons = jsons[:args.limit_videos]
    print(f'Found {len(jsons)} video JSONs  |  TRA fusion k={args.k_frames}')

    for jp in jsons:
        video = jp.stem
        with open(jp) as f:
            data = json.load(f)

        # Group detections by track id (same format as the dedup baseline).
        by_track = defaultdict(list)
        for fid_str, dets in data.items():
            for i, det in enumerate(dets):
                if not det.get('points') or len(det['points']) < 6:
                    continue
                tid = det.get('ID', det.get('id'))
                if tid is None:
                    continue
                by_track[tid].append((fid_str, i, det['points']))

        img_cache, track_texts = {}, {}
        for tid, entries in tqdm(by_track.items(), desc=video):
            if len(entries) < args.min_track_len:
                continue
            crops, _w = select_legible_crops(
                img_cache, args.frames_root, video, entries, args.k_frames, args.target_size)
            if crops:
                track_texts[tid] = fuse_recognize(proc, model, crops)

        n_swapped = 0
        for _fid, dets in data.items():
            for det in dets:
                tid = det.get('ID', det.get('id'))
                if track_texts.get(tid):
                    det['transcription'] = track_texts[tid]
                    n_swapped += 1
        print(f'  {video}: {len(by_track)} tracks → fused {len(track_texts)} reads, '
              f'swapped {n_swapped} transcriptions')

        with open(Path(args.out_dir) / f'{video}.json', 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()
