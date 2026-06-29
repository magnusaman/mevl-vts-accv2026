"""Track-deduplicated transcription swap.

Same text appears in many frames of the same track. Recognize ONCE per track
(the highest-confidence / largest-area crop), propagate the recognition to all
frames of that track. ~10-15x faster than per-detection inference.

Input/output format matches infer_swap_transcriptions.py.
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import cv2
# WORKAROUND: cuDNN 9.x + bf16 Conv3d on Ada (L4) -> CUDNN_STATUS_NOT_INITIALIZED.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
torch.backends.cudnn.enabled = False
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as _AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelForVL
from peft import PeftModel

PROMPT = "Read the text in this image. Output only the text, exactly as written."


def find_frame(frames_root, video, frame_id):
    base = Path(frames_root) / video
    for ext in ('.jpg', '.png', '.jpeg'):
        for fname in (f'{frame_id}{ext}',
                      f'{str(frame_id).zfill(4)}{ext}',
                      f'{str(frame_id).zfill(5)}{ext}',
                      f'{str(frame_id).zfill(6)}{ext}'):
            p = base / fname
            if p.exists():
                return str(p)
    return None


def crop_and_pad(img_bgr, polygon, target_size=384, pad_ratio=0.08):
    h, w = img_bgr.shape[:2]
    xs = polygon[0::2] if not isinstance(polygon[0], (list, tuple)) else [p[0] for p in polygon]
    ys = polygon[1::2] if not isinstance(polygon[0], (list, tuple)) else [p[1] for p in polygon]
    x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
    y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
    bw, bh = x1 - x0, y1 - y0
    px = int(bw * pad_ratio); py = int(bh * pad_ratio)
    x0 = max(0, x0 - px); x1 = min(w, x1 + px)
    y0 = max(0, y0 - py); y1 = min(h, y1 + py)
    if x1 <= x0 or y1 <= y0:
        return None, 0
    crop = img_bgr[y0:y1, x0:x1]
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    cw, ch = pil.size
    scale = target_size / max(cw, ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    pil = pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    canvas.paste(pil, ((target_size - nw) // 2, (target_size - nh) // 2))
    return canvas, bw * bh


def load_model(model_id, adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                         min_pixels=224*224, max_pixels=384*384)
    model = _AutoModelForVL.from_pretrained(
        model_id, quantization_config=bnb, device_map='auto',
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return proc, model


@torch.inference_mode()
def batched_recognize(proc, model, pil_images, max_new_tokens=12):
    if not pil_images:
        return []
    texts = []
    for _ in pil_images:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ]},
        ]
        texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = proc(text=texts, images=pil_images, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=1.0, top_p=1.0)
    in_len = inputs['input_ids'].shape[1]
    new_ids = out[:, in_len:]
    decoded = proc.batch_decode(new_ids, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons-dir', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--model-id', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--target-size', type=int, default=224)
    ap.add_argument('--limit-videos', type=int, default=0)
    ap.add_argument('--min-track-len', type=int, default=5,
                    help='Skip recognizing tracks shorter than this (still kept in JSON)')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.model_id, args.adapter)

    jsons = sorted(Path(args.jsons_dir).glob('*.json'))
    if args.limit_videos:
        jsons = jsons[:args.limit_videos]
    print(f'Found {len(jsons)} video JSONs')

    grand_total_dets = 0
    grand_total_tracks = 0
    for jp in jsons:
        video = jp.stem
        print(f'\n=== {video} ===')
        with open(jp) as f:
            data = json.load(f)

        # --- Group detections by track id, pick best representative per track ---
        # Each detection: (fid_str, det_idx, points, ID). We want to pool by ID across
        # all frames of this video.
        all_dets = []  # (fid_str, det_idx, polygon, track_id)
        for fid_str, dets in data.items():
            for i, det in enumerate(dets):
                if not det.get('points') or len(det['points']) < 6:
                    continue
                tid = det.get('ID', det.get('id', None))
                if tid is None:
                    continue
                all_dets.append((fid_str, i, det['points'], tid))
        grand_total_dets += len(all_dets)

        # Group by track ID
        by_track = defaultdict(list)
        for fid_str, i, poly, tid in all_dets:
            by_track[tid].append((fid_str, i, poly))

        # For each track of length >= min_track_len, pick the MIDDLE entry as
        # representative (most likely to be a clean, well-cropped detection).
        # Short tracks (likely noise) get their transcription kept as-is.
        reps = []  # (track_id, fid_str_rep, det_idx_rep, polygon_rep)
        frame_cache = {}
        n_skipped = 0
        for tid, entries in by_track.items():
            if len(entries) < args.min_track_len:
                n_skipped += 1
                continue
            mid = len(entries) // 2
            fid_str, i, poly = entries[mid]
            reps.append((tid, fid_str, i, poly))
        grand_total_tracks += len(reps)
        print(f'  {len(all_dets)} dets → {len(by_track)} tracks → '
              f'{len(reps)} recognized (skip {n_skipped} short tracks)')

        # --- Recognize representatives in batches ---
        track_texts = {}  # tid -> recognized text
        batch_imgs, batch_tids = [], []
        for tid, fid_str, i, poly in tqdm(reps, desc=video):
            if fid_str not in frame_cache:
                fpath = find_frame(args.frames_root, video, int(fid_str))
                frame_cache[fid_str] = cv2.imread(fpath) if fpath else None
            img = frame_cache[fid_str]
            if img is None:
                track_texts[tid] = ''
                continue
            crop, _ = crop_and_pad(img, poly, target_size=args.target_size)
            if crop is None:
                track_texts[tid] = ''
                continue
            batch_imgs.append(crop); batch_tids.append(tid)
            if len(batch_imgs) >= args.batch_size:
                outs = batched_recognize(proc, model, batch_imgs)
                for t, o in zip(batch_tids, outs):
                    track_texts[t] = o
                batch_imgs, batch_tids = [], []
        if batch_imgs:
            outs = batched_recognize(proc, model, batch_imgs)
            for t, o in zip(batch_tids, outs):
                track_texts[t] = o

        # --- Propagate text to all detections of each track ---
        n_swapped = 0
        for fid_str, dets in data.items():
            for det in dets:
                tid = det.get('ID', det.get('id', None))
                if tid in track_texts and track_texts[tid]:
                    det['transcription'] = track_texts[tid]
                    n_swapped += 1
        print(f'  swapped {n_swapped}/{len(all_dets)} transcriptions via {len(reps)} track reps')

        with open(Path(args.out_dir) / f'{video}.json', 'w') as f:
            json.dump(data, f)

    print(f'\n[summary] total dets={grand_total_dets} unique tracks={grand_total_tracks} '
          f'speedup={grand_total_dets/max(grand_total_tracks,1):.1f}x')


if __name__ == '__main__':
    main()
