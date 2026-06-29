"""L4-tuned Qwen-VL LoRA recognition trainer.

Port of modal_finetune_qwen3_stageA_gt_v9.py to a single-GPU L4 (24 GB):
  - 4-bit nf4 + double quant (BitsAndBytesConfig)
  - paged_adamw_8bit
  - bs=1, ga=8, gradient checkpointing
  - On-the-fly crop + augmentation in Dataset.__getitem__
    (no pre-packed tensor shards: 950k samples * 400KB = 380GB disk -- not viable)
  - LoRA r=16 alpha=32 on q/k/v/o, dropout=0.1
  - Per-dataset normalized accuracy
  - Saves best_adapter + per-epoch checkpoint + history

Reads a JSONL manifest with rows:
  {image_path, polygon, text, dataset, video, frame_id, track_id}

Usage:
    python train_qwen_lora_l4_v9.py \\
        --manifest /root/data/manifest.jsonl \\
        --output-root /root/outputs/qwen_v9_l4 \\
        --num-epochs 5
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Memory + cuDNN setup BEFORE torch import
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import cv2
import editdistance
import numpy as np
import torch
import torch.nn.functional as F
# WORKAROUND: cuDNN 9.2 + bf16 Conv3d on Ada (L4) -> CUDNN_STATUS_NOT_INITIALIZED.
# patch_embed Conv3d is tiny, so disabling cuDNN globally is essentially free.
torch.backends.cudnn.enabled = False
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as _AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelForVL
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

PROMPT = "Read the text in the image. Output ONLY the text exactly as shown. Preserve case, punctuation, and spaces. No extra words."

# Qwen2-VL/Qwen3-VL chat-template anchor tokens for label masking
IM_START_ID = 151644
ASSISTANT_ID = 77091
NL_ID = 198


# ============================================================================
# CROP + AUGMENT
# ============================================================================
def crop_polygon_masked(img_rgb: np.ndarray, polygon: List[List[int]],
                        padding: int = 5, background: int = 255) -> Optional[np.ndarray]:
    h, w = img_rgb.shape[:2]
    poly = np.array(polygon, dtype=np.int32)
    xs, ys = poly[:, 0], poly[:, 1]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding)
    y2 = min(h, int(ys.max()) + padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = img_rgb[y1:y2, x1:x2].copy()
    shifted = poly - np.array([x1, y1])
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    crop[mask == 0] = background
    return crop


def resize_aspect(img: np.ndarray, target: int, background: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), background, dtype=np.uint8)
    ox, oy = (target - nw) // 2, (target - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def aug_geometric(img: np.ndarray) -> np.ndarray:
    """One geometric: rotation or sinusoidal wave distortion (v9 line 610-620)."""
    h, w = img.shape[:2]
    if random.random() < 0.5:
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    A = random.uniform(5, 15)
    f = random.uniform(0.5, 1.5)
    phase = random.uniform(0, 2 * np.pi)
    j_grid, i_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = j_grid
    map_y = i_grid + A * np.sin(2 * np.pi * f * (j_grid / w) + phase)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def aug_appearance(img: np.ndarray) -> np.ndarray:
    """One appearance: blur or noise or brightness (v9 line 622-645)."""
    kind = random.choice(['blur', 'noise', 'bright'])
    if kind == 'blur':
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == 'noise':
        stddev = random.uniform(10, 30)
        noise = np.zeros(img.shape, np.uint8)
        cv2.randn(noise, 0, stddev)
        return cv2.add(img, noise)
    # bright
    alpha = random.uniform(0.6, 1.4)
    beta = random.uniform(-40, 40)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ============================================================================
# DATASET
# ============================================================================
class CropDataset(Dataset):
    """v9-faithful: with augment, each row appears 3x in the epoch -- (clean, geo, appearance).
    Without augment, just the original. Augmentation is keyed off the position in self.rows."""
    def __init__(self, manifest_path: str, processor, crop_size: int = 256,
                 augment: bool = True, is_train: bool = True, val_ratio: float = 0.05,
                 max_seq_len: int = 512, seed: int = 42,
                 max_train_samples: int = 0, subsample_by_dataset: str = ''):
        self.processor = processor
        self.crop_size = crop_size
        self.augment = augment and is_train
        self.is_train = is_train
        self.max_seq_len = max_seq_len

        rows = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))

        if subsample_by_dataset:
            ratios = {}
            for spec in subsample_by_dataset.split(','):
                ds, r = spec.split(':')
                ratios[ds.strip()] = float(r)
            rng = random.Random(seed)
            kept = []
            for r in rows:
                ratio = ratios.get(r.get('dataset', 'unknown'), 1.0)
                if rng.random() < ratio:
                    kept.append(r)
            rows = kept

        rng = random.Random(seed)
        rng.shuffle(rows)
        split = int(len(rows) * (1 - val_ratio))
        base = rows[:split] if is_train else rows[split:]
        if is_train and max_train_samples > 0 and len(base) > max_train_samples:
            base = base[:max_train_samples]

        # v9 augmentation policy: triple the train rows; mark aug type per row
        if self.augment:
            self.rows = []
            for r in base:
                self.rows.append({**r, '_aug': 'clean'})
                self.rows.append({**r, '_aug': 'geo'})
                self.rows.append({**r, '_aug': 'appearance'})
            print(f'  CropDataset[train]: {len(base)} base rows x 3 aug = {len(self.rows)} effective rows')
        else:
            self.rows = [{**r, '_aug': 'clean'} for r in base]
            print(f'  CropDataset[{("train" if is_train else "val")}]: {len(self.rows)} rows')

    def __len__(self):
        return len(self.rows)

    def _load_crop(self, row) -> Optional[Image.Image]:
        img_bgr = cv2.imread(row['image_path'])
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop = crop_polygon_masked(img_rgb, row['polygon'])
        if crop is None:
            return None
        crop = resize_aspect(crop, self.crop_size)
        aug_kind = row.get('_aug', 'clean')
        if aug_kind == 'geo':
            crop = aug_geometric(crop)
        elif aug_kind == 'appearance':
            crop = aug_appearance(crop)
        return Image.fromarray(crop)

    def __getitem__(self, idx):
        for _ in range(5):  # retry up to 5x on bad rows
            row = self.rows[idx]
            pil = self._load_crop(row)
            if pil is not None:
                break
            idx = (idx + 1) % len(self.rows)
        else:
            # last resort: blank white
            pil = Image.new('RGB', (self.crop_size, self.crop_size), (255, 255, 255))
            row = {'text': '', 'dataset': 'unknown'}

        text = row['text']
        if self.is_train:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
            ]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[chat], images=[pil], return_tensors='pt',
                                padding=False, truncation=True, max_length=self.max_seq_len)
        sample = {
            'gt_text': text,
            'dataset_source': row.get('dataset', 'unknown'),
        }
        # Forward EVERY tensor from the processor (Qwen3-VL adds mm_token_type_ids etc.)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                sample[k] = v.squeeze(0) if v.dim() > 0 else v

        if self.is_train:
            # Build labels with -100 mask on prompt
            ids = sample['input_ids']
            labels = ids.clone()
            start = -1
            for i in range(len(ids) - 2):
                if ids[i] == IM_START_ID and ids[i + 1] == ASSISTANT_ID:
                    if ids[i + 2] == NL_ID:
                        start = i + 3
                    else:
                        start = i + 2
                    break
            if start != -1:
                labels[:start] = -100
            labels[sample['attention_mask'] == 0] = -100
            sample['labels'] = labels

        return sample


def collate_dynamic(batch, pad_id: int):
    """Dynamic collate. Sequence tensors (same len as input_ids) are padded,
    other tensors are stacked. Handles Qwen3-VL's mm_token_type_ids and other extras."""
    from torch.nn.utils.rnn import pad_sequence
    out = {}
    seq_len_ref = batch[0]['input_ids'].shape[0]
    # Padding values for known seq tensors
    pad_vals = {'input_ids': pad_id, 'attention_mask': 0, 'labels': -100,
                'mm_token_type_ids': 0, 'token_type_ids': 0}
    seen_tensor_keys = set()
    for k, v in batch[0].items():
        if not torch.is_tensor(v):
            continue
        seen_tensor_keys.add(k)
        if v.dim() >= 1 and v.shape[0] == seq_len_ref:
            # Sequence-length tensor: pad
            pv = pad_vals.get(k, 0)
            out[k] = pad_sequence([s[k] for s in batch], batch_first=True, padding_value=pv)
        else:
            # Fixed-shape tensor: stack
            try:
                out[k] = torch.stack([s[k] for s in batch])
            except RuntimeError:
                # variable-shape (e.g. pixel_values of different sizes): keep as list
                out[k] = [s[k] for s in batch]
    out['gt_texts'] = [s['gt_text'] for s in batch]
    out['dataset_sources'] = [s['dataset_source'] for s in batch]
    return out


# ============================================================================
# EVALUATION
# ============================================================================
@torch.inference_mode()
def evaluate(model, processor, val_loader, device, max_batches: int = 100):
    model.eval()
    gts, preds, sources = [], [], []

    n_done = 0
    for inputs in tqdm(val_loader, desc='eval', leave=False):
        if n_done >= max_batches:
            break
        gt_texts = inputs.pop('gt_texts')
        srcs = inputs.pop('dataset_sources')

        tensor_in = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        gen = model.generate(
            **tensor_in,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        in_len = tensor_in['input_ids'].shape[1]
        new_ids = gen[:, in_len:]
        decoded = processor.batch_decode(new_ids, skip_special_tokens=True)
        for i, d in enumerate(decoded):
            d = d.strip()
            if d.startswith('"') and d.endswith('"'):
                d = d[1:-1]
            preds.append(d)
            gts.append(gt_texts[i])
            sources.append(srcs[i])
        n_done += 1

    strict = norm = 0
    total_cer = 0.0
    per_ds = {}
    for gt, pr, src in zip(gts, preds, sources):
        if pr.strip() == gt.strip():
            strict += 1
        g = ' '.join(gt.casefold().split())
        p = ' '.join(pr.casefold().split())
        if src not in per_ds:
            per_ds[src] = {'c': 0, 't': 0}
        per_ds[src]['t'] += 1
        if g == p:
            norm += 1
            per_ds[src]['c'] += 1
        cer = editdistance.eval(pr, gt) / max(len(pr), len(gt), 1)
        total_cer += cer
    n = max(len(gts), 1)
    metrics = {
        'strict_acc': strict / n,
        'normalized_acc': norm / n,
        'cer': total_cer / n,
        'num_samples': n,
        'per_dataset': {k: v['c'] / max(v['t'], 1) for k, v in per_ds.items()},
        'per_dataset_n': {k: v['t'] for k, v in per_ds.items()},
    }
    samples = list(zip(gts[:30], preds[:30], sources[:30]))
    model.train()
    return metrics, samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--model-id', default='Qwen/Qwen3-VL-8B-Instruct',
                    help='Default Qwen3-VL-8B-Instruct (needs transformers >=4.55 + torch >=2.5)')
    ap.add_argument('--resume-adapter', default='')
    ap.add_argument('--num-epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.1)
    ap.add_argument('--crop-size', type=int, default=256)
    ap.add_argument('--max-seq-len', type=int, default=512)
    ap.add_argument('--min-pixels', type=int, default=128 * 128)
    ap.add_argument('--max-pixels', type=int, default=256 * 256)
    ap.add_argument('--val-ratio', type=float, default=0.03)
    ap.add_argument('--max-val-batches', type=int, default=200)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--warmup-ratio', type=float, default=0.05)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--save-every-n-steps', type=int, default=0,
                    help='If >0, save checkpoint every N optimizer steps (in addition to per-epoch)')
    ap.add_argument('--no-grad-checkpoint', action='store_true',
                    help='Disable gradient checkpointing for ~1.5-2x speedup (uses more GPU mem)')
    ap.add_argument('--max-train-samples', type=int, default=0,
                    help='If >0, cap train set to N samples (after split)')
    ap.add_argument('--subsample-by-dataset', default='',
                    help='Comma list of dataset:keep_ratio e.g. dstext:0.06,ic15v:1.0')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda')
    print('=' * 80)
    print(f'L4 QLoRA Recognition Trainer (v9 port)')
    print(f'  GPU: {torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'  Model: {args.model_id}')
    print(f'  Manifest: {args.manifest}')
    print(f'  Epochs: {args.num_epochs}  bs={args.batch_size} ga={args.grad_accum}  lr={args.lr}')
    print(f'  Crop: {args.crop_size}  Pixels: {args.min_pixels}..{args.max_pixels}')
    print(f'  LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}')
    print(f'  Augment: {not args.no_augment}')
    print('=' * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Processor with pixel limits
    print('[1] Loading processor...')
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True,
                                              min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    if hasattr(processor, 'image_processor'):
        if hasattr(processor.image_processor, 'min_pixels'):
            processor.image_processor.min_pixels = args.min_pixels
        if hasattr(processor.image_processor, 'max_pixels'):
            processor.image_processor.max_pixels = args.max_pixels
    tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    pad_id = tok.pad_token_id or 0

    # Datasets
    print('[2] Building datasets...')
    train_ds = CropDataset(args.manifest, processor, crop_size=args.crop_size,
                           augment=not args.no_augment, is_train=True,
                           val_ratio=args.val_ratio, max_seq_len=args.max_seq_len,
                           max_train_samples=args.max_train_samples,
                           subsample_by_dataset=args.subsample_by_dataset)
    val_ds = CropDataset(args.manifest, processor, crop_size=args.crop_size,
                         augment=False, is_train=False,
                         val_ratio=args.val_ratio, max_seq_len=args.max_seq_len,
                         subsample_by_dataset=args.subsample_by_dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=lambda b: collate_dynamic(b, pad_id),
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=lambda b: collate_dynamic(b, pad_id))
    print(f'  train batches: {len(train_loader)}  val batches: {len(val_loader)}')

    # Model: 4-bit nf4 + LoRA
    print('[3] Loading model (4-bit nf4)...')
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = _AutoModelForVL.from_pretrained(
        args.model_id, quantization_config=bnb, device_map='auto',
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.config.use_cache = False
    use_gc = not args.no_grad_checkpoint
    print(f'  Gradient checkpointing: {"ON" if use_gc else "OFF (faster, more memory)"}')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)

    if args.resume_adapter:
        adapter_path = Path(args.resume_adapter)
        if adapter_path.exists():
            print(f'  Resuming adapter from {adapter_path}')
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
            # v9 convention: continued training uses much lower LR to avoid catastrophic forgetting
            if args.lr > 1e-5:
                old_lr = args.lr
                args.lr = 5e-6
                print(f'  [auto-LR] resuming from checkpoint -> dropping lr {old_lr:.0e} to {args.lr:.0e} (v9 continued-training default)')
        else:
            print(f'  [WARN] adapter path missing, creating fresh LoRA: {adapter_path}')
            args.resume_adapter = ''

    if not args.resume_adapter:
        print('  Creating fresh LoRA adapter...')
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=args.lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    model.train()

    # Optimizer + scheduler
    print('[4] Optimizer = paged_adamw_8bit')
    import bitsandbytes as bnb_mod
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable params — LoRA misconfigured')
    optimizer = bnb_mod.optim.PagedAdamW8bit(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f'  total_steps={total_steps}  warmup_steps={warmup_steps}')

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print('[5] Starting training...')
    history = []
    best_norm = 0.0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_n = 0
        accum_count = 0
        pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{args.num_epochs}')
        for batch in pbar:
            batch.pop('gt_texts', None)
            batch.pop('dataset_sources', None)
            labels = batch.pop('labels').to(device, non_blocking=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}

            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels != -100
            if valid.sum() == 0:
                loss = logits.sum() * 0.0
            else:
                loss = F.cross_entropy(shift_logits[valid].float(), shift_labels[valid], reduction='mean')
            loss = loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            epoch_n += 1
            accum_count += 1
            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1
                pbar.set_postfix(loss=f'{loss.item()*args.grad_accum:.3f}',
                                 lr=f'{scheduler.get_last_lr()[0]:.2e}',
                                 step=global_step)
                if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                    model.save_pretrained(out_dir / f'step_{global_step}')

        avg_loss = epoch_loss / max(epoch_n, 1)
        dt = time.time() - epoch_start
        print(f'  epoch {epoch+1} done: avg_loss={avg_loss:.4f}  time={dt:.0f}s')

        # Eval
        metrics, samples = evaluate(model, processor, val_loader, device,
                                    max_batches=args.max_val_batches)
        print(f'  val strict={metrics["strict_acc"]*100:.2f}%  norm={metrics["normalized_acc"]*100:.2f}%  CER={metrics["cer"]*100:.2f}%')
        for ds, acc in metrics['per_dataset'].items():
            n = metrics['per_dataset_n'].get(ds, 0)
            print(f'    {ds} (n={n}): {acc*100:.2f}%')

        history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'time_s': dt,
            'val_strict': metrics['strict_acc'],
            'val_norm': metrics['normalized_acc'],
            'val_cer': metrics['cer'],
            'per_dataset': metrics['per_dataset'],
        })
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if metrics['normalized_acc'] > best_norm:
            best_norm = metrics['normalized_acc']
            print(f'  *** New best! Saving to best_adapter ***')
            model.save_pretrained(out_dir / 'best_adapter')
            with open(out_dir / 'best_samples.json', 'w', encoding='utf-8') as f:
                json.dump([{'gt': g, 'pred': p, 'src': s} for g, p, s in samples], f,
                          indent=2, ensure_ascii=False)
        model.save_pretrained(out_dir / f'checkpoint_epoch_{epoch+1}')

    # Final
    model.save_pretrained(out_dir / 'final_adapter')
    processor.save_pretrained(out_dir / 'final_adapter')
    summary = {
        'best_normalized_acc': best_norm,
        'total_steps': global_step,
        'output_dir': str(out_dir),
        'history': history,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('=' * 80)
    print(f'DONE.  best_normalized_acc={best_norm*100:.2f}%  out={out_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
