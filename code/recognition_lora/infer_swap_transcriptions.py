"""Run Qwen LoRA on v6 detection crops and swap the 'transcription' field.

Input:
  --jsons-dir  : directory of <video>.json with {frame_id: [{points, ID, transcription, ...}, ...]}
  --frames-root: where test frames live, layout <frames_root>/<video>/<frame>.jpg
  --adapter   : path to trained LoRA adapter (saved by train_qwen_lora_l4.py)
  --model-id  : base model id (must match what was trained)
  --out-dir   : write modified JSONs here

After this, run build_rrc_zip.py to package as a submission zip.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import torch
# WORKAROUND: cuDNN 9.x + bf16 Conv3d on Ada (L4) -> CUDNN_STATUS_NOT_INITIALIZED.
# Qwen-VL's patch_embed is a single tiny Conv3d; disabling cuDNN is essentially free.
torch.backends.cudnn.enabled = False
import os as _os
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
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
        return None
    crop = img_bgr[y0:y1, x0:x1]
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    cw, ch = pil.size
    scale = target_size / max(cw, ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    pil = pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    canvas.paste(pil, ((target_size - nw) // 2, (target_size - nh) // 2))
    return canvas


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
def batched_recognize(proc, model, pil_images, max_new_tokens=16):
    """Return list of decoded strings for the given PIL crops."""
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
    # Strip prompt tokens
    in_len = inputs['input_ids'].shape[1]
    new_ids = out[:, in_len:]
    decoded = proc.batch_decode(new_ids, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons-dir', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--model-id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--target-size', type=int, default=384)
    ap.add_argument('--limit-videos', type=int, default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.model_id, args.adapter)

    jsons = sorted(Path(args.jsons_dir).glob('*.json'))
    if args.limit_videos:
        jsons = jsons[:args.limit_videos]
    print(f'Found {len(jsons)} video JSONs to process')

    for jp in jsons:
        video = jp.stem
        print(f'\n=== {video} ===')
        with open(jp) as f:
            data = json.load(f)
        # Pre-decode each frame ONCE
        frame_cache = {}
        # Build list of (frame_id, det_index, polygon) for batching
        pending = []  # (fid_str, det_idx, polygon)
        for fid_str, dets in data.items():
            for i, det in enumerate(dets):
                if not det.get('points') or len(det['points']) < 6:
                    continue
                pending.append((fid_str, i, det['points']))

        # Process in batches
        results = {}  # (fid_str, i) -> new_text
        batch_imgs, batch_keys = [], []
        for fid_str, i, poly in tqdm(pending, desc=video):
            if fid_str not in frame_cache:
                fpath = find_frame(args.frames_root, video, int(fid_str))
                if fpath is None:
                    frame_cache[fid_str] = None
                else:
                    frame_cache[fid_str] = cv2.imread(fpath)
            img = frame_cache[fid_str]
            if img is None:
                continue
            crop = crop_and_pad(img, poly, target_size=args.target_size)
            if crop is None:
                continue
            batch_imgs.append(crop); batch_keys.append((fid_str, i))
            if len(batch_imgs) >= args.batch_size:
                outs = batched_recognize(proc, model, batch_imgs)
                for k, o in zip(batch_keys, outs):
                    results[k] = o
                batch_imgs, batch_keys = [], []
        if batch_imgs:
            outs = batched_recognize(proc, model, batch_imgs)
            for k, o in zip(batch_keys, outs):
                results[k] = o

        # Swap transcriptions
        n_swapped = 0
        for fid_str, dets in data.items():
            for i, det in enumerate(dets):
                key = (fid_str, i)
                if key in results and results[key]:
                    det['transcription'] = results[key]
                    n_swapped += 1
        print(f'  swapped {n_swapped}/{len(pending)} transcriptions')
        with open(Path(args.out_dir) / f'{video}.json', 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()
