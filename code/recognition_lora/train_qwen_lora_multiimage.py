"""M1 — multi-image recognition-fusion LoRA trainer (Qwen3-VL-8B).

This is the train-time counterpart to infer_temporal_fusion.py. Where v9 trains the
LoRA on ONE crop -> one read, this trains on K crops of the SAME tracked word (across
frames) -> one read, via a multi-image prompt. Fixes the train/inference mismatch: the
fusion adapter is now actually trained the way it's used.

Input: the TRACK-GROUPED manifest from build_manifest_multiframe.py, rows:
  {source, video, track, text, frames_dir_remote, crops:[{frame,polygon},...]}

Frame images are resolved per-source from --frames-roots (source:root,source:root),
path = <root>/<video>/<frame>.jpg by default (override pattern with --frame-name-fmt).
Warm-start from the single-crop adapter (--resume-adapter) so the model keeps its
single-image reading skill and only learns to fuse.

Reuses v9's QLoRA recipe verbatim: 4-bit nf4 + double quant, paged_adamw_8bit, bs=1
ga=8, grad checkpointing, LoRA r=16 a=32 on q/k/v/o, label-masked CE on the answer.

Smoke (Modal, cheap):  --max-steps 30  on a few videos.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import cv2
import editdistance
import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False  # cuDNN9.2 + bf16 Conv3d on Ada workaround (v9)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as _AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelForVL
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

FUSION_PROMPT = (
    "These images are crops of the SAME word, taken from consecutive frames of a "
    "video. Some views may be blurred, partially occluded, or low-resolution. "
    "Using all of the views together, read the word. Output only the text, exactly "
    "as written."
)

IM_START_ID = 151644
ASSISTANT_ID = 77091
NL_ID = 198


# ---------------------------------------------------------------- crop helpers
def crop_polygon_masked(img_rgb, polygon, padding=5, background=255):
    h, w = img_rgb.shape[:2]
    poly = np.array(polygon, dtype=np.int32)
    xs, ys = poly[:, 0], poly[:, 1]
    x1 = max(0, int(xs.min()) - padding); y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding); y2 = min(h, int(ys.max()) + padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = img_rgb[y1:y2, x1:x2].copy()
    shifted = poly - np.array([x1, y1])
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    crop[mask == 0] = background
    return crop


def resize_aspect(img, target, background=255):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), background, dtype=np.uint8)
    ox, oy = (target - nw) // 2, (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


# ---------------------------------------------------------------- dataset
class TrackDataset(Dataset):
    """One sample = K crops of a tracked word -> its GT text (multi-image fusion)."""
    def __init__(self, manifest_path, processor, frames_roots: dict, frame_name_fmt: str,
                 crop_size=256, k_min=2, k_max=4, is_train=True, val_ratio=0.03,
                 max_seq_len=768, seed=42, max_train_samples=0):
        self.processor = processor
        self.frames_roots = frames_roots
        self.frame_name_fmt = frame_name_fmt
        self.crop_size = crop_size
        self.k_min, self.k_max = k_min, k_max
        self.is_train = is_train
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)

        rows = []
        with open(manifest_path, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    if r.get('crops') and r.get('text'):
                        rows.append(r)
        rng = random.Random(seed)
        rng.shuffle(rows)
        split = int(len(rows) * (1 - val_ratio))
        self.rows = rows[:split] if is_train else rows[split:]
        if is_train and max_train_samples > 0:
            self.rows = self.rows[:max_train_samples]
        print(f'  TrackDataset[{("train" if is_train else "val")}]: {len(self.rows)} tracks')

    def __len__(self):
        return len(self.rows)

    def _frame_path(self, row, frame):
        root = self.frames_roots.get(row['source'])
        if root is None:
            # fall back to the manifest's recorded dir
            root = row.get('frames_dir_remote', '')
            return str(Path(root) / self.frame_name_fmt.format(frame=frame))
        return str(Path(root) / row['video'] / self.frame_name_fmt.format(frame=frame))

    def _load_k_crops(self, row) -> List[Image.Image]:
        crops_meta = row['crops']
        k = self.rng.randint(self.k_min, min(self.k_max, len(crops_meta))) if self.is_train \
            else min(self.k_max, len(crops_meta))
        chosen = self.rng.sample(crops_meta, k) if self.is_train else crops_meta[:k]
        chosen = sorted(chosen, key=lambda c: c['frame'])
        out = []
        for c in chosen:
            img_bgr = cv2.imread(self._frame_path(row, c['frame']))
            if img_bgr is None:
                continue
            crop = crop_polygon_masked(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), c['polygon'])
            if crop is None:
                continue
            out.append(Image.fromarray(resize_aspect(crop, self.crop_size)))
        return out

    def __getitem__(self, idx):
        for _ in range(5):
            row = self.rows[idx]
            imgs = self._load_k_crops(row)
            if imgs:
                break
            idx = (idx + 1) % len(self.rows)
        else:
            imgs = [Image.new('RGB', (self.crop_size, self.crop_size), (255, 255, 255))]
            row = {'text': ''}
        text = row['text']

        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text": FUSION_PROMPT})
        if self.is_train:
            messages = [{"role": "user", "content": content},
                        {"role": "assistant", "content": [{"type": "text", "text": text}]}]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            messages = [{"role": "user", "content": content}]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[chat], images=imgs, return_tensors='pt',
                                padding=False, truncation=True, max_length=self.max_seq_len)
        sample = {'gt_text': text}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                sample[k] = v.squeeze(0) if v.dim() > 0 else v

        if self.is_train:
            ids = sample['input_ids']
            labels = ids.clone()
            start = -1
            for i in range(len(ids) - 2):
                if ids[i] == IM_START_ID and ids[i + 1] == ASSISTANT_ID:
                    start = i + 3 if ids[i + 2] == NL_ID else i + 2
                    break
            if start != -1:
                labels[:start] = -100
            labels[sample['attention_mask'] == 0] = -100
            sample['labels'] = labels
        return sample


def collate_dynamic(batch, pad_id):
    from torch.nn.utils.rnn import pad_sequence
    out = {}
    seq_ref = batch[0]['input_ids'].shape[0]
    pad_vals = {'input_ids': pad_id, 'attention_mask': 0, 'labels': -100,
                'mm_token_type_ids': 0, 'token_type_ids': 0}
    for k, v in batch[0].items():
        if not torch.is_tensor(v):
            continue
        if v.dim() >= 1 and v.shape[0] == seq_ref:
            out[k] = pad_sequence([s[k] for s in batch], batch_first=True,
                                  padding_value=pad_vals.get(k, 0))
        else:
            try:
                out[k] = torch.stack([s[k] for s in batch])
            except RuntimeError:
                out[k] = [s[k] for s in batch]
    out['gt_texts'] = [s['gt_text'] for s in batch]
    return out


@torch.inference_mode()
def evaluate(model, processor, val_loader, device, max_batches=100):
    model.eval()
    gts, preds = [], []
    tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    n = 0
    for inputs in tqdm(val_loader, desc='eval', leave=False):
        if n >= max_batches:
            break
        gt = inputs.pop('gt_texts')
        tin = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        gen = model.generate(**tin, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = gen[:, tin['input_ids'].shape[1]:]
        for i, d in enumerate(processor.batch_decode(new, skip_special_tokens=True)):
            preds.append(d.strip()); gts.append(gt[i])
        n += 1
    strict = sum(1 for g, p in zip(gts, preds) if p.strip() == g.strip())
    norm = sum(1 for g, p in zip(gts, preds)
               if ' '.join(p.casefold().split()) == ' '.join(g.casefold().split()))
    cer = sum(editdistance.eval(p, g) / max(len(p), len(g), 1) for g, p in zip(gts, preds))
    m = max(len(gts), 1)
    model.train()
    return {'strict': strict / m, 'norm': norm / m, 'cer': cer / m, 'n': m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--frames-roots', required=True,
                    help='source:root comma list, e.g. icdar15v:/ds/ICDAR15_Video/frames,dstext:/ds/DSText/frame')
    ap.add_argument('--frame-name-fmt', default='{frame}.jpg')
    ap.add_argument('--model-id', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--resume-adapter', default='', help='warm-start from the single-crop adapter')
    ap.add_argument('--num-epochs', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=0, help='>0 caps optimizer steps (smoke)')
    ap.add_argument('--lr', type=float, default=5e-6)  # low: continued-training default
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.1)
    ap.add_argument('--crop-size', type=int, default=256)
    ap.add_argument('--k-min', type=int, default=2)
    ap.add_argument('--k-max', type=int, default=4)
    ap.add_argument('--max-seq-len', type=int, default=768)
    ap.add_argument('--min-pixels', type=int, default=128 * 128)
    ap.add_argument('--max-pixels', type=int, default=256 * 256)
    ap.add_argument('--val-ratio', type=float, default=0.03)
    ap.add_argument('--max-val-batches', type=int, default=100)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--max-train-samples', type=int, default=0)
    args = ap.parse_args()

    frames_roots = {}
    for spec in args.frames_roots.split(','):
        if ':' in spec:
            s, r = spec.split(':', 1)
            frames_roots[s.strip()] = r.strip()

    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    print('=' * 70)
    print('M1 multi-image fusion LoRA trainer')
    print(f'  model={args.model_id}  K={args.k_min}..{args.k_max}  lr={args.lr}')
    print(f'  frames_roots={frames_roots}')
    print('=' * 70)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True,
                                              min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    if hasattr(processor, 'image_processor'):
        for a in ('min_pixels', 'max_pixels'):
            if hasattr(processor.image_processor, a):
                setattr(processor.image_processor, a, getattr(args, a))
    tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    pad_id = tok.pad_token_id or 0

    train_ds = TrackDataset(args.manifest, processor, frames_roots, args.frame_name_fmt,
                            crop_size=args.crop_size, k_min=args.k_min, k_max=args.k_max,
                            is_train=True, val_ratio=args.val_ratio, max_seq_len=args.max_seq_len,
                            max_train_samples=args.max_train_samples)
    val_ds = TrackDataset(args.manifest, processor, frames_roots, args.frame_name_fmt,
                          crop_size=args.crop_size, k_min=args.k_max, k_max=args.k_max,
                          is_train=False, val_ratio=args.val_ratio, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=lambda b: collate_dynamic(b, pad_id),
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=lambda b: collate_dynamic(b, pad_id))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = _AutoModelForVL.from_pretrained(args.model_id, quantization_config=bnb,
                                            device_map='auto', torch_dtype=torch.bfloat16,
                                            trust_remote_code=True, attn_implementation='sdpa')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if args.resume_adapter and Path(args.resume_adapter).exists():
        print(f'  warm-start adapter: {args.resume_adapter}')
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        if args.resume_adapter:
            print(f'  [WARN] adapter missing ({args.resume_adapter}); fresh LoRA')
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=args.lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()
    model.train()

    import bitsandbytes as bnb_mod
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('No trainable params — LoRA misconfigured')
    opt = bnb_mod.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.num_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warmup if s < warmup
                                              else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))))

    print(f'[train] total_steps={total_steps}')
    gstep = 0; accum = 0; opt.zero_grad()
    best = 0.0
    for epoch in range(args.num_epochs):
        for batch in tqdm(train_loader, desc=f'epoch {epoch+1}'):
            batch.pop('gt_texts', None)
            labels = batch.pop('labels').to(device, non_blocking=True)
            inp = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(**inp)
            sl, slab = out.logits[:, :-1, :], labels[:, 1:]
            valid = slab != -100
            loss = (out.logits.sum() * 0.0) if valid.sum() == 0 \
                else F.cross_entropy(sl[valid].float(), slab[valid])
            (loss / args.grad_accum).backward()
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); accum = 0; gstep += 1
                if gstep % 10 == 0:
                    print(f'  step {gstep}/{total_steps} loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e}', flush=True)
                if args.max_steps and gstep >= args.max_steps:
                    break
        if args.max_steps and gstep >= args.max_steps:
            break
        m = evaluate(model, processor, val_loader, device, max_batches=args.max_val_batches)
        print(f'  epoch {epoch+1}: val strict={m["strict"]*100:.2f}% norm={m["norm"]*100:.2f}% CER={m["cer"]*100:.2f}% (n={m["n"]})')
        if m['norm'] >= best:
            best = m['norm']; model.save_pretrained(out_dir / 'best_adapter')
        model.save_pretrained(out_dir / f'checkpoint_epoch_{epoch+1}')

    model.save_pretrained(out_dir / 'final_adapter')
    processor.save_pretrained(out_dir / 'final_adapter')
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'best_norm': best, 'steps': gstep, 'out': str(out_dir)}, f, indent=2)
    print(f'DONE. best_norm={best*100:.2f}%  out={out_dir}')


if __name__ == '__main__':
    main()
