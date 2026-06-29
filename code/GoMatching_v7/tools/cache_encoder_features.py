"""Phase 1 cache builder: foundation-encoder feature maps for DenseTrack v7.

For each video, for each encoder, runs the frozen encoder over all frames and
saves 32x32 feature maps as fp16 HDF5. Used by Component 2 (consensus matcher)
which roi_aligns these feature maps at proposal boxes during training.

Output layout:
    <out_root>/<encoder_name>/<video_id>.h5
        features:       (N, C, 32, 32) fp16
        frame_indices:  (N,) int32     (1-indexed GT frame_id)

Frame filename convention on disk:
    <frames_root>/<video_id>/<6digit>.jpg
    Filename '000000.jpg' corresponds to GT frame_id = 1 (off-by-one).

Encoders supported (see gomatching/modeling/v7_densetrack/encoder_loaders.py):
    clip-l-336   (1024 d)
    dinov2-l     (1024 d)
    sam-h        (1280 d)
    convnext-l   (1536 d)

Memory: one encoder load + batched forward at 32x32 resample.
    CLIP-L:   ~1.3 GB weights + ~50 MB per 8-frame batch
    DINOv2-L: ~1.1 GB + ~80 MB
    SAM-H:    ~2.5 GB + ~150 MB
    ConvNeXt-L: ~0.8 GB + ~30 MB
Run encoders sequentially (not concurrently) to keep GPU mem usage low.

Speed estimate (L4):
    CLIP-L:   ~50 ms/frame × 14k frames ≈ 12 min/encoder
    DINOv2-L: ~80 ms × 14k ≈ 19 min
    SAM-H:    ~250 ms × 14k ≈ 58 min
    ConvNeXt-L: ~30 ms × 14k ≈ 7 min
    Total for IC15-V (~14k frames × 4 encoders): ~1.5-2 h
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_encoder(name: str, device: str = "cuda"):
    """Light import (no GoMatching dep needed) — copies from encoder_loaders.py."""
    SUPPORTED = {
        "clip-l-336": (1024, 336),
        "dinov2-l": (1024, 518),
        "sam-h": (1280, 1024),
        "convnext-l": (1536, 384),
    }
    if name not in SUPPORTED:
        raise ValueError(f"Unknown encoder: {name}. Supported: {list(SUPPORTED)}")
    out_dim, in_size = SUPPORTED[name]

    if name == "clip-l-336":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336", pretrained="openai")
        model = model.visual.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.inference_mode()
        def forward(batch_imgs):
            # batch_imgs: (B, 3, H, W) uint8
            x = batch_imgs.to(device).float() / 255.0
            x = F.interpolate(x, size=(in_size, in_size), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                               device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
            # open_clip ViT.forward returns CLS token; we need patch tokens.
            # ViT-L/14-336: patches are 24x24 = 576 tokens; we want (B, 1024, 24, 24)
            # then resample to 32x32.
            tokens = _open_clip_patch_tokens(model, x)
            B, N, C = tokens.shape
            HW = int(N ** 0.5)
            feats = tokens.transpose(1, 2).reshape(B, C, HW, HW)
            feats = F.interpolate(feats, size=(32, 32), mode="bilinear", align_corners=False)
            return feats.half().cpu().numpy()  # (B, 1024, 32, 32) fp16
        return forward, out_dim

    elif name == "dinov2-l":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                               verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.inference_mode()
        def forward(batch_imgs):
            x = batch_imgs.to(device).float() / 255.0
            x = F.interpolate(x, size=(in_size, in_size), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
            out = model.forward_features(x)
            tokens = out["x_norm_patchtokens"]  # (B, N, 1024)
            B, N, C = tokens.shape
            HW = int(N ** 0.5)  # 518/14 = 37
            feats = tokens.transpose(1, 2).reshape(B, C, HW, HW)
            feats = F.interpolate(feats, size=(32, 32), mode="bilinear", align_corners=False)
            return feats.half().cpu().numpy()
        return forward, out_dim

    elif name == "sam-h":
        from transformers import SamModel
        sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        vision_encoder = sam.vision_encoder

        @torch.inference_mode()
        def forward(batch_imgs):
            x = batch_imgs.to(device).float() / 255.0
            x = F.interpolate(x, size=(in_size, in_size), mode="bilinear", align_corners=False)
            mean = torch.tensor([123.675, 116.28, 103.53],
                                device=device).view(1, 3, 1, 1) / 255.0
            std = torch.tensor([58.395, 57.12, 57.375],
                               device=device).view(1, 3, 1, 1) / 255.0
            x = (x - mean) / std
            out = vision_encoder(x)
            # SAM vision encoder returns (B, H'*W', C); reshape to (B, C, H', W')
            feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if feats.dim() == 4:
                # Already (B, C, H, W)
                pass
            else:
                B, N, C = feats.shape
                HW = int(N ** 0.5)
                feats = feats.transpose(1, 2).reshape(B, C, HW, HW)
            feats = F.interpolate(feats, size=(32, 32), mode="bilinear", align_corners=False)
            return feats.half().cpu().numpy()
        return forward, out_dim

    elif name == "convnext-l":
        import timm
        model = timm.create_model("convnext_large.fb_in22k_ft_in1k_384",
                                  pretrained=True, num_classes=0,
                                  global_pool="").to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        @torch.inference_mode()
        def forward(batch_imgs):
            x = batch_imgs.to(device).float() / 255.0
            x = F.interpolate(x, size=(in_size, in_size), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
            feats = model.forward_features(x)  # (B, C, H, W)
            feats = F.interpolate(feats, size=(32, 32), mode="bilinear", align_corners=False)
            return feats.half().cpu().numpy()
        return forward, out_dim
    raise NotImplementedError(name)


def _open_clip_patch_tokens(visual_model, x):
    """Run open_clip ViT to get all patch tokens (B, N, C), not just CLS."""
    # open_clip ViT structure: conv1 -> reshape -> cat CLS -> pos_emb -> transformer
    m = visual_model
    x = m.conv1(x)  # (B, C, H', W')
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, N)
    x = x.permute(0, 2, 1)  # (B, N, C)
    cls = m.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                      dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + m.positional_embedding.to(x.dtype)
    x = m.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = m.transformer(x)
    x = x.permute(1, 0, 2)
    x = m.ln_post(x)
    return x[:, 1:, :]  # drop CLS, return patch tokens (B, N-1, C)


def list_videos(frames_root: Path, pattern: str = "Video_*"):
    """Return sorted list of video subdir Paths under frames_root."""
    vids = sorted([p for p in frames_root.glob(pattern) if p.is_dir()])
    return vids


def load_frames_for_video(video_dir: Path, batch_size: int = 8,
                          subsample_stride: int = 1):
    """Yield (frame_indices, batch_tensor) pairs for one video.

    frame_indices: 1-indexed GT frame_ids (so filename '000000.jpg' yields 1).
    subsample_stride: keep every Nth frame after sorting (1=all, 3=1-in-3).
    """
    jpgs = sorted(video_dir.glob("*.jpg"))
    if not jpgs:
        return
    if subsample_stride > 1:
        jpgs = jpgs[::subsample_stride]
    batch_imgs = []
    batch_idx = []
    for jpg in jpgs:
        stem = jpg.stem
        try:
            file_idx = int(stem)
        except ValueError:
            continue
        gt_frame_id = file_idx + 1  # 6-digit filename is 0-indexed; GT is 1-indexed
        img = Image.open(jpg).convert("RGB")
        arr = np.array(img)  # (H, W, 3) uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
        batch_imgs.append(tensor)
        batch_idx.append(gt_frame_id)
        if len(batch_imgs) >= batch_size:
            # Pad to common shape if needed
            max_h = max(t.shape[1] for t in batch_imgs)
            max_w = max(t.shape[2] for t in batch_imgs)
            padded = torch.zeros(len(batch_imgs), 3, max_h, max_w, dtype=torch.uint8)
            for i, t in enumerate(batch_imgs):
                padded[i, :, :t.shape[1], :t.shape[2]] = t
            yield batch_idx, padded
            batch_imgs = []
            batch_idx = []
    if batch_imgs:
        max_h = max(t.shape[1] for t in batch_imgs)
        max_w = max(t.shape[2] for t in batch_imgs)
        padded = torch.zeros(len(batch_imgs), 3, max_h, max_w, dtype=torch.uint8)
        for i, t in enumerate(batch_imgs):
            padded[i, :, :t.shape[1], :t.shape[2]] = t
        yield batch_idx, padded


def cache_one_video(forward_fn, out_dim: int, video_dir: Path, out_h5: Path,
                    batch_size: int = 8, force: bool = False,
                    output_size: int = 32, subsample_stride: int = 1):
    if out_h5.exists() and not force:
        try:
            with h5py.File(out_h5, "r") as f:
                if "features" in f and "frame_indices" in f:
                    return f["features"].shape[0], "skipped"
        except Exception:
            out_h5.unlink()  # corrupt — redo

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    all_feats = []
    all_idx = []
    for indices, batch in load_frames_for_video(video_dir, batch_size,
                                                 subsample_stride=subsample_stride):
        feats = forward_fn(batch)  # (B, C, 32, 32) fp16 numpy
        if output_size != 32:
            # downsample further to save disk; matcher's roi_align rescales boxes
            t = torch.from_numpy(feats).float()
            t = F.interpolate(t, size=(output_size, output_size),
                              mode="area")
            feats = t.half().numpy()
        all_feats.append(feats)
        all_idx.extend(indices)

    if not all_feats:
        return 0, "empty"

    feats_arr = np.concatenate(all_feats, axis=0)  # (N, C, 32, 32)
    idx_arr = np.array(all_idx, dtype=np.int32)

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("features", data=feats_arr, dtype="float16",
                         compression="gzip", compression_opts=4)
        f.create_dataset("frame_indices", data=idx_arr, dtype="int32")
        f.attrs["encoder_out_dim"] = out_dim
        f.attrs["spatial_h"] = feats_arr.shape[2]
        f.attrs["spatial_w"] = feats_arr.shape[3]
    return feats_arr.shape[0], "wrote"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True,
                    help="root with <video_id>/<6digit>.jpg structure")
    ap.add_argument("--out-root", required=True,
                    help="root to write <encoder>/<video_id>.h5")
    ap.add_argument("--encoder", required=True,
                    choices=["clip-l-336", "dinov2-l", "sam-h", "convnext-l"])
    ap.add_argument("--video-pattern", default="Video_*")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--output-size", type=int, default=32,
                    help="Final spatial map size (default 32). "
                         "Smaller = faster + smaller cache. 16 -> 4x, 8 -> 16x.")
    ap.add_argument("--subsample-stride", type=int, default=1,
                    help="Keep every Nth frame (default 1 = all frames). "
                         "3 -> 3x smaller cache, 3x faster training.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder: {args.encoder} ...")
    forward, out_dim = load_encoder(args.encoder, device=device)
    print(f"  out_dim={out_dim}")

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root) / args.encoder
    out_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(frames_root, args.video_pattern)
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    print(f"Found {len(videos)} videos under {frames_root}")

    t_start = time.time()
    total_frames = 0
    for i, vdir in enumerate(videos):
        out_h5 = out_root / f"{vdir.name}.h5"
        t0 = time.time()
        n, status = cache_one_video(forward, out_dim, vdir, out_h5,
                                    batch_size=args.batch_size, force=args.force,
                                    output_size=args.output_size,
                                    subsample_stride=args.subsample_stride)
        total_frames += n
        dt = time.time() - t0
        print(f"  [{i+1:>3}/{len(videos)}] {vdir.name}: {n} frames, "
              f"{status}, {dt:.1f}s ({n/max(dt,0.01):.1f} frames/s)")

    elapsed = time.time() - t_start
    print(f"\nDONE. {total_frames} frames cached across {len(videos)} videos "
          f"in {elapsed/60:.1f} min ({total_frames/max(elapsed,1):.1f} frames/s overall).")
    print(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()
