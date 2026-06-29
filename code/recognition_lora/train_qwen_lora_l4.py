"""Qwen-VL QLoRA fine-tune on text crops, tuned for L4 24GB.

Defaults: Qwen2.5-VL-7B (rock solid). Override with --model-id Qwen/Qwen3-VL-8B-Instruct.

Memory budget on L4 24GB with --batch-size 1 --image-size 384:
  4-bit weights (7B):   ~5 GB
  LoRA adapter (r=16):  ~50 MB
  AdamW8bit states:     ~100 MB
  Activations + grads:  ~7 GB
  Framework overhead:   ~2 GB
  ----------------------------
  Total:                ~14 GB  (10 GB headroom)

Effective batch = batch_size * grad_accum.
Default: bs=1 ga=16 → effective bs=16.

Usage:
  HF_TOKEN=hf_xxx python train_qwen_lora_l4.py \
      --dataset-id whoamananand1/ic15v-dstext-text-crops \
      --output-dir /workspace/runs/qwen25vl7b_v1 \
      --epochs 2

  # Switch to Qwen3-VL-8B (latest):
  ... --model-id Qwen/Qwen3-VL-8B-Instruct

  # Resume:
  ... --resume-from /workspace/runs/qwen25vl7b_v1/checkpoint-5000
"""
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
try:
    from transformers import AutoModelForImageTextToText as _AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoModelForVL
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


PROMPT = "Read the text in this image. Output only the text, exactly as written."


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--dataset-id', default='whoamananand1/ic15v-dstext-text-crops')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume-from', default=None)
    # training
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--max-train-samples', type=int, default=0, help='0 = all')
    ap.add_argument('--max-val-samples', type=int, default=2000)
    ap.add_argument('--save-steps', type=int, default=2000)
    ap.add_argument('--eval-steps', type=int, default=2000)
    ap.add_argument('--logging-steps', type=int, default=20)
    # LoRA
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target', default='q_proj,k_proj,v_proj,o_proj')
    # data
    ap.add_argument('--image-size', type=int, default=384)
    ap.add_argument('--max-seq-len', type=int, default=128)
    # bookkeeping
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=4)
    return ap.parse_args()


def build_processor(model_id):
    # min/max pixels control auto image resizing (Qwen2.5-VL feature)
    proc = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=224 * 224,
        max_pixels=384 * 384,
    )
    return proc


def build_model(model_id):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = _AutoModelForVL.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='sdpa',  # bump to 'flash_attention_2' if installed
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


@dataclass
class QwenCollator:
    processor: object
    max_seq_len: int = 128
    image_size: int = 384

    def _resize(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        scale = self.image_size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        canvas.paste(img, ((self.image_size - nw) // 2, (self.image_size - nh) // 2))
        return canvas

    def __call__(self, batch):
        # batch: list of {image: PIL, text: str}
        texts = []
        images = []
        for ex in batch:
            img = self._resize(ex['image'])
            images.append(img)
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": str(ex['text'])},
                ]},
            ]
            txt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(txt)

        enc = self.processor(text=texts, images=images, return_tensors='pt',
                             padding=True, truncation=True, max_length=self.max_seq_len)
        labels = enc['input_ids'].clone()
        # Mask pad + user prompt tokens: we want loss only on assistant tokens.
        # Cheap heuristic: mask everything before the last "assistant" marker.
        # The processor's chat template uses '<|im_start|>assistant\n' as marker.
        assistant_token_ids = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        if len(assistant_token_ids) > 0:
            marker = assistant_token_ids[-1]
            for i in range(labels.size(0)):
                row = labels[i]
                hits = (row == marker).nonzero(as_tuple=True)[0]
                if len(hits) > 0:
                    cut = hits[-1].item() + 1
                    labels[i, :cut] = -100
        labels[enc['attention_mask'] == 0] = -100
        enc['labels'] = labels
        return enc


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading dataset {args.dataset_id}...')
    ds = load_dataset(args.dataset_id, token=os.environ.get('HF_TOKEN'))
    train_ds = ds['train']
    val_ds = ds.get('validation') or ds.get('val')
    if val_ds is None:
        # auto-split
        split = train_ds.train_test_split(test_size=0.02, seed=args.seed)
        train_ds, val_ds = split['train'], split['test']
    if args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_val_samples and len(val_ds) > args.max_val_samples:
        val_ds = val_ds.shuffle(seed=args.seed).select(range(args.max_val_samples))
    print(f'  train: {len(train_ds)}  val: {len(val_ds)}')

    print(f'Loading processor + 4-bit model: {args.model_id}')
    processor = build_processor(args.model_id)
    model = build_model(args.model_id)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias='none', task_type='CAUSAL_LM',
        target_modules=[m.strip() for m in args.lora_target.split(',')],
    )
    if args.resume_from:
        print(f'Resuming LoRA adapter from {args.resume_from}')
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = QwenCollator(processor=processor,
                            max_seq_len=args.max_seq_len,
                            image_size=args.image_size)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        save_total_limit=4,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='paged_adamw_8bit',
        report_to=['tensorboard'],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        lr_scheduler_type='cosine',
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    # Save final adapter
    final_dir = Path(args.output_dir) / 'final_adapter'
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f'Saved final adapter to {final_dir}')


if __name__ == '__main__':
    main()
