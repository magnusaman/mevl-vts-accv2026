"""V7 per-instance cache builder — replaces 175 GB spatial-map cache with
~3 GB per-instance pooled-feature cache.

Mechanism
=========
The original `cache_encoder_features.py` saves (T, C, 32, 32) spatial maps per
video per encoder. At training time the consensus matcher does roi_align at
DeepSolo's proposal boxes against those maps. Most of every 32x32 map is empty
space (text instances are tiny relative to frame), so the cache wastes ~99% of
its bytes.

This builder pre-pools at GT boxes (from the COCO-format training JSON), saving
ONLY (sum_M, C_e) per encoder per video. With per-encoder slicing it stays
trivially loadable, and the matcher just gathers + IoU-matches at training time
instead of roi_aligning.

Combined savings
================
  175 GB -> ~3 GB  (60x) from per-instance pooling
  ~3 GB -> ~1 GB  (3x) from --subsample-stride 3 (1-in-3 frames)
  ~1 GB -> ~0.5 GB (2x) from top-3 encoders only (config-level ablation)

Training I/O drops accordingly, so step time drops from ~2s to ~0.3s on L4.

Cache layout (per video):
    <root>/<video_id>.npz
        frame_ids        : (F,) int32         -- 1-indexed GT frame_id
        box_offsets      : (F+1,) int32       -- cumsum of instances per frame
        gt_boxes_xyxy    : (sum_M, 4) float32 -- image-coordinate bboxes
        gt_track_ids     : (sum_M,) int32     -- COCO instance_id
        image_sizes      : (F, 2) int32       -- (H, W) per frame
        <enc_name>       : (sum_M, C_e) float16  -- one array per encoder

Indexing (loader side):
    Given frame_id f in cache:
        i = where(frame_ids == f)[0]
        start, end = box_offsets[i], box_offsets[i+1]
        boxes_t  = gt_boxes_xyxy[start:end]
        tids_t   = gt_track_ids[start:end]
        feat_e_t = <enc_name>[start:end]  # (M_t, C_e)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

# Re-use the encoder loader from the spatial-map builder
sys.path.insert(0, str(Path(__file__).parent))
from cache_encoder_features import load_encoder  # noqa: E402


def coco_bbox_to_xyxy(bbox):
    """COCO bbox is [x, y, w, h]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def group_annotations_by_video(coco_data: dict, subsample_stride: int = 1):
    """Return {video_id: [(frame_id, image_id, file_name, (H, W), [(xyxy, track_id), ...]), ...]}.

    Frame IDs follow the cache_encoder_features.py convention:
    filename '000000.jpg' -> GT frame_id = 1.

    Subsampling keeps every Nth frame after sorting by file_name.
    """
    img_by_id = {img["id"]: img for img in coco_data["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Group images by video (parent directory of file_name)
    by_video = defaultdict(list)
    for img in coco_data["images"]:
        fn = img["file_name"]
        parts = Path(fn).parts
        if len(parts) < 2:
            continue
        video_id = parts[-2]
        file_stem = Path(fn).stem
        try:
            file_idx = int(file_stem)
        except ValueError:
            continue
        gt_frame_id = file_idx + 1
        H, W = img.get("height", 0), img.get("width", 0)
        boxes_tids = []
        for ann in anns_by_img.get(img["id"], []):
            if "bbox" not in ann:
                continue
            xyxy = coco_bbox_to_xyxy(ann["bbox"])
            tid = ann.get("instance_id", ann.get("id", -1))
            boxes_tids.append((xyxy, int(tid)))
        by_video[video_id].append((gt_frame_id, img["id"], fn, (H, W), boxes_tids))

    # Sort each video by frame_id, then subsample
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[0])
        if subsample_stride > 1:
            by_video[vid] = by_video[vid][::subsample_stride]
    return by_video


def pool_at_boxes(feature_maps_fp32: torch.Tensor,
                  boxes_xyxy: torch.Tensor,
                  image_size: tuple) -> torch.Tensor:
    """RoI-align (B, C, Hf, Wf) at (N, 4) boxes (image coords). Returns (N, C).

    Internally we scale boxes to feature-map coords ourselves so spatial_scale=1.0.
    """
    B, C, Hf, Wf = feature_maps_fp32.shape
    assert B == 1, "pool_at_boxes expects single-frame feature maps"
    if boxes_xyxy.shape[0] == 0:
        return torch.zeros(0, C, dtype=feature_maps_fp32.dtype,
                           device=feature_maps_fp32.device)
    H, W = image_size
    sx, sy = Wf / max(W, 1), Hf / max(H, 1)
    bx = boxes_xyxy.to(feature_maps_fp32.device, dtype=feature_maps_fp32.dtype).clone()
    bx[:, [0, 2]] *= sx
    bx[:, [1, 3]] *= sy
    x1 = torch.minimum(bx[:, 0], bx[:, 2]).clamp_(0, Wf - 0.5)
    y1 = torch.minimum(bx[:, 1], bx[:, 3]).clamp_(0, Hf - 0.5)
    x2 = torch.maximum(bx[:, 0], bx[:, 2]).clamp_(0.5, Wf)
    y2 = torch.maximum(bx[:, 1], bx[:, 3]).clamp_(0.5, Hf)
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    bx = torch.stack([x1, y1, x2, y2], dim=1)
    pooled = roi_align(
        feature_maps_fp32, [bx], output_size=(1, 1),
        spatial_scale=1.0, aligned=True
    ).squeeze(-1).squeeze(-1)
    return pooled  # (N, C)


def build_one_video(video_id: str,
                    video_entries,
                    frames_root: Path,
                    encoder_forwards: dict,
                    encoder_dims: dict,
                    out_npz: Path,
                    force: bool):
    if out_npz.exists() and not force:
        return 0, "skipped"

    frame_ids, image_sizes = [], []
    per_enc_feats = {enc: [] for enc in encoder_forwards}
    boxes_xyxy_all, track_ids_all = [], []
    box_offsets = [0]

    for gt_frame_id, _img_id, fn, (H, W), boxes_tids in video_entries:
        # Resolve image path: <frames_root>/<video_id>/<basename>
        img_path = frames_root / Path(fn).name if "/" not in fn else Path(fn)
        # The COCO JSON's file_name may already be relative to dataset root,
        # not frames_root. Try both.
        if not img_path.exists():
            img_path = frames_root / video_id / Path(fn).name
        if not img_path.exists():
            # Skip frame with missing image
            continue

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)  # (H, W, 3) uint8
        H, W = arr.shape[:2]
        batch = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)

        if not boxes_tids:
            frame_ids.append(gt_frame_id)
            image_sizes.append((H, W))
            box_offsets.append(box_offsets[-1])
            continue

        boxes_xyxy = torch.tensor([b for b, _ in boxes_tids], dtype=torch.float32)
        tids = torch.tensor([t for _, t in boxes_tids], dtype=torch.int32)

        for enc_name, forward_fn in encoder_forwards.items():
            # forward_fn returns numpy (B, C, 32, 32) fp16
            feats_np = forward_fn(batch)  # (1, C, 32, 32)
            feats = torch.from_numpy(feats_np.astype(np.float32))
            pooled = pool_at_boxes(feats, boxes_xyxy, (H, W))  # (M, C)
            per_enc_feats[enc_name].append(pooled.half().numpy())

        frame_ids.append(gt_frame_id)
        image_sizes.append((H, W))
        boxes_xyxy_all.append(boxes_xyxy.numpy())
        track_ids_all.append(tids.numpy())
        box_offsets.append(box_offsets[-1] + boxes_xyxy.shape[0])

    if not frame_ids:
        return 0, "empty"

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "box_offsets": np.array(box_offsets, dtype=np.int32),
        "gt_boxes_xyxy": (np.concatenate(boxes_xyxy_all, axis=0)
                          if boxes_xyxy_all else np.zeros((0, 4), dtype=np.float32)),
        "gt_track_ids": (np.concatenate(track_ids_all, axis=0)
                         if track_ids_all else np.zeros((0,), dtype=np.int32)),
        "image_sizes": np.array(image_sizes, dtype=np.int32),
    }
    for enc_name, feat_list in per_enc_feats.items():
        if not feat_list:
            arrays[enc_name] = np.zeros((0, encoder_dims[enc_name]), dtype=np.float16)
        else:
            arrays[enc_name] = np.concatenate(feat_list, axis=0).astype(np.float16)

    np.savez_compressed(out_npz, **arrays)
    return len(frame_ids), "wrote"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", required=True,
                    help="Path to detectron2 COCO-format training JSON")
    ap.add_argument("--frames-root", required=True,
                    help="Root containing <video_id>/<6digit>.jpg")
    ap.add_argument("--out-root", required=True, help="Output dir for per-video NPZ")
    ap.add_argument("--encoders", nargs="+", required=True,
                    choices=["clip-l-336", "dinov2-l", "sam-h", "convnext-l"])
    ap.add_argument("--subsample-stride", type=int, default=1,
                    help="Keep every Nth frame (1=all, 3=1-in-3)")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root)

    print(f"Loading COCO JSON: {args.coco_json}")
    with open(args.coco_json) as f:
        coco_data = json.load(f)
    print(f"  {len(coco_data['images'])} images, "
          f"{len(coco_data['annotations'])} annotations")

    by_video = group_annotations_by_video(coco_data,
                                          subsample_stride=args.subsample_stride)
    print(f"  -> {len(by_video)} videos, "
          f"subsample_stride={args.subsample_stride}")

    print(f"Loading {len(args.encoders)} encoder(s) sequentially")
    encoder_forwards, encoder_dims = {}, {}
    for enc in args.encoders:
        print(f"  loading {enc} ...")
        fn, out_dim = load_encoder(enc, device=device)
        encoder_forwards[enc] = fn
        encoder_dims[enc] = out_dim
    print(f"  loaded encoders: {list(encoder_forwards.keys())}")

    videos = sorted(by_video.keys())
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    t_start = time.time()
    grand_frames = 0
    for i, vid in enumerate(videos):
        out_npz = out_root / f"{vid}.npz"
        t0 = time.time()
        n, status = build_one_video(vid, by_video[vid], frames_root,
                                    encoder_forwards, encoder_dims,
                                    out_npz, force=args.force)
        grand_frames += n
        dt = time.time() - t0
        print(f"  [{i+1:>3}/{len(videos)}] {vid}: {n} frames {status} "
              f"({dt:.1f}s, {n/max(dt,0.01):.1f} fps)")

    elapsed = time.time() - t_start
    print(f"\nDONE. {grand_frames} frames cached across {len(videos)} videos "
          f"in {elapsed/60:.1f} min")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
