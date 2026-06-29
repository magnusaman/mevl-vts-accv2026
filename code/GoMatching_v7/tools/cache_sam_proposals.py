"""Phase 1 cache builder: SAM-H mask proposals for DenseTrack Component 1.

For each frame, runs SAM-H "segment everything" (automatic mask generator),
filters to top-K most stable masks, extracts:
  - polygon (mask contour as flat [x1,y1,...,xn,yn])
  - bounding box (xyxy)
  - SAM stability score (predicted_iou)
  - CLIP feature pooled inside the mask (used by Component 1's text classifier)

Saves per-video as numpy .npz with arrays of varying length per frame.

Output layout:
    <out_root>/<video_id>.npz
        frame_indices    : (F,) int32       1-indexed GT frame_ids
        n_masks_per_frame: (F,) int32       number of masks kept per frame
        polygons_flat    : object array of flat polygon arrays
        boxes            : (sum_n, 4) float32  xyxy
        scores           : (sum_n,) float32    SAM stability scores
        clip_features    : (sum_n, 1024) float16  CLIP features of mask region

Speed estimate (L4): SAM-H ~250ms/frame + CLIP ~50ms × ~30 masks/frame ≈ 1-2 s/frame.
IC15-V 14k frames ≈ 4-6 hours. Will need to run videos in parallel batches or
accept the long runtime.

Memory: SAM-H ~2.5 GB + CLIP-L ~1.3 GB = ~4 GB. Fits.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_sam_h(device: str = "cuda"):
    """Returns (mask_generator, sam_model). Uses HF transformers."""
    from transformers import SamModel, SamProcessor
    sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    for p in sam.parameters():
        p.requires_grad_(False)
    return sam, proc


def load_clip(device: str = "cuda"):
    """Returns (forward_fn) for CLIP-L/14-336 image features."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai")
    model = model.visual.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.inference_mode()
    def encode(crops_pil):
        """crops_pil: list of PIL Images. Returns (N, 1024) fp16 on cpu."""
        if not crops_pil:
            return np.zeros((0, 1024), dtype=np.float16)
        ts = []
        for p in crops_pil:
            arr = np.array(p.resize((336, 336)).convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            ts.append(t)
        batch = torch.stack(ts, dim=0).to(device)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=device).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        # forward through ViT and take the CLS token (image feature)
        out = model(batch)  # (N, 1024)
        return out.half().cpu().numpy()
    return encode


@torch.inference_mode()
def sam_segment_everything(sam, proc, image_pil, device, points_per_side=16,
                            min_mask_area=400, max_masks=50):
    """Run SAM-H automatic mask generator.

    HF transformers doesn't ship SAM's full AMG; we mimic it via a regular grid
    of point prompts.
    Returns list of dicts: {mask: np.ndarray HxW bool, score: float, bbox: xyxy}.
    """
    W, H = image_pil.size
    # Build grid of point prompts in image coords
    xs = np.linspace(0, W - 1, points_per_side)
    ys = np.linspace(0, H - 1, points_per_side)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)  # (N, 2)
    point_labels = np.ones(grid.shape[0], dtype=np.int64)

    # Batch points: each prompt is independent (a single positive click).
    inputs = proc(images=image_pil, input_points=[grid.tolist()],
                  input_labels=[point_labels.tolist()],
                  return_tensors="pt").to(device)
    outputs = sam(**inputs, multimask_output=True)
    # outputs.pred_masks: (B=1, N_points, M=3, H', W')
    # outputs.iou_scores: (B=1, N_points, M=3)
    masks = outputs.pred_masks.squeeze(0).cpu().numpy()  # (N_points, M, H', W')
    scores = outputs.iou_scores.squeeze(0).cpu().numpy()  # (N_points, M)

    # For each point, pick best of 3 masks; gather all
    results = []
    # Post-process masks back to original image size
    orig_h, orig_w = H, W
    for p_idx in range(masks.shape[0]):
        for m_idx in range(masks.shape[1]):
            mask = masks[p_idx, m_idx]
            score = float(scores[p_idx, m_idx])
            # Resize mask to original size
            mask_t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
            mask_resized = F.interpolate(mask_t, size=(orig_h, orig_w),
                                          mode="bilinear", align_corners=False)
            mask_bin = (mask_resized > 0.0).squeeze().numpy()
            area = int(mask_bin.sum())
            if area < min_mask_area:
                continue
            # bbox
            ys_, xs_ = np.where(mask_bin)
            if len(xs_) == 0:
                continue
            x1, y1, x2, y2 = int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())
            results.append({"mask": mask_bin, "score": score,
                            "bbox": [x1, y1, x2, y2]})

    # NMS-style dedup: sort by score, drop masks with >0.9 IoU to a higher-scoring one
    results.sort(key=lambda r: r["score"], reverse=True)
    kept = []
    for r in results:
        keep = True
        for k in kept:
            inter = (r["mask"] & k["mask"]).sum()
            union = (r["mask"] | k["mask"]).sum()
            iou = inter / max(union, 1)
            if iou > 0.85:
                keep = False
                break
        if keep:
            kept.append(r)
        if len(kept) >= max_masks:
            break
    return kept


def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 1.5):
    """Convert binary mask to flat polygon [x1,y1,...,xn,yn]."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # take largest contour
    cnt = max(contours, key=cv2.contourArea)
    # simplify with Douglas-Peucker
    cnt = cv2.approxPolyDP(cnt, simplify_eps, True)
    if cnt.shape[0] < 3:
        return None
    return cnt.reshape(-1, 2).flatten().tolist()


def crop_masked_region(image_pil, mask: np.ndarray, bbox: list, pad: int = 8):
    """Return PIL crop of bbox with non-mask pixels set to white."""
    arr = np.array(image_pil.convert("RGB"))
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(arr.shape[1], x2 + pad), min(arr.shape[0], y2 + pad)
    crop = arr[y1:y2, x1:x2].copy()
    cmask = mask[y1:y2, x1:x2]
    crop[~cmask] = 255
    return Image.fromarray(crop)


def cache_one_video(sam, proc, clip_encode, video_dir: Path, out_npz: Path,
                     points_per_side: int, min_mask_area: int, max_masks: int,
                     force: bool = False):
    if out_npz.exists() and not force:
        return 0, "skipped"

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    device = next(sam.parameters()).device

    frame_indices = []
    n_masks_per_frame = []
    polygons_flat = []
    boxes = []
    scores = []
    clip_features = []

    jpgs = sorted(video_dir.glob("*.jpg"))
    for jpg in jpgs:
        try:
            file_idx = int(jpg.stem)
        except ValueError:
            continue
        gt_frame_id = file_idx + 1
        img = Image.open(jpg).convert("RGB")

        masks = sam_segment_everything(
            sam, proc, img, device,
            points_per_side=points_per_side,
            min_mask_area=min_mask_area, max_masks=max_masks)

        # Per-mask: polygon + crop -> CLIP feature
        frame_polys, frame_boxes, frame_scores, frame_crops = [], [], [], []
        for m in masks:
            poly = mask_to_polygon(m["mask"])
            if poly is None:
                continue
            crop = crop_masked_region(img, m["mask"], m["bbox"])
            frame_polys.append(poly)
            frame_boxes.append(m["bbox"])
            frame_scores.append(m["score"])
            frame_crops.append(crop)

        if frame_crops:
            cfeats = clip_encode(frame_crops)  # (n, 1024) fp16
        else:
            cfeats = np.zeros((0, 1024), dtype=np.float16)

        frame_indices.append(gt_frame_id)
        n_masks_per_frame.append(len(frame_polys))
        polygons_flat.extend(frame_polys)
        boxes.extend(frame_boxes)
        scores.extend(frame_scores)
        clip_features.append(cfeats)

    # Save
    np.savez_compressed(
        str(out_npz),
        frame_indices=np.array(frame_indices, dtype=np.int32),
        n_masks_per_frame=np.array(n_masks_per_frame, dtype=np.int32),
        polygons_flat=np.array(polygons_flat, dtype=object),
        boxes=np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32),
        scores=np.array(scores, dtype=np.float32),
        clip_features=np.concatenate(clip_features, axis=0)
                      if clip_features else np.zeros((0, 1024), np.float16),
    )
    return sum(n_masks_per_frame), "wrote"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--video-pattern", default="Video_*")
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--min-mask-area", type=int, default=400)
    ap.add_argument("--max-masks", type=int, default=50)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM-H + CLIP-L ...")
    sam, proc = load_sam_h(device=device)
    clip_encode = load_clip(device=device)

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in frames_root.glob(args.video_pattern) if p.is_dir()])
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    print(f"Found {len(videos)} videos under {frames_root}")

    t_start = time.time()
    total_masks = 0
    for i, vdir in enumerate(videos):
        out_npz = out_root / f"{vdir.name}.npz"
        t0 = time.time()
        n, status = cache_one_video(
            sam, proc, clip_encode, vdir, out_npz,
            points_per_side=args.points_per_side,
            min_mask_area=args.min_mask_area,
            max_masks=args.max_masks, force=args.force)
        total_masks += n
        dt = time.time() - t0
        n_frames = len(list(vdir.glob("*.jpg")))
        print(f"  [{i+1:>3}/{len(videos)}] {vdir.name}: {n} masks, "
              f"{status}, {dt:.1f}s ({n_frames/max(dt,0.01):.2f} frames/s)")

    elapsed = time.time() - t_start
    print(f"\nDONE. {total_masks} total masks cached in {elapsed/60:.1f} min.")
    print(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()
