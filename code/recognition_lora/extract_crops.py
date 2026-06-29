"""Extract crops from video frames using a manifest.

Inputs:
  --manifest: JSONL with one crop per line (from build_manifest.py)
  --frames-root: local root where frames live, with layout:
        <frames_root>/<source>/<video_id>/<frame_id>.jpg
      For IC15-V: <frames_root>/icdar15v/Video_10_1_1/8.jpg
      For DSText: <frames_root>/dstext/Video_163_6_3/1.jpg
    OR pass --use-frames-dir-remote to use the absolute frames_dir from manifest
    (useful if frames live at /data/... on the GPU box exactly like remote)
  --out-dir: where to write crops + metadata.parquet
  --crop-size: long edge in px after square pad (default 384)

Output layout:
  <out_dir>/
    crops/<source>/<video>_<frame>_<track>_<idx>.jpg
    metadata.parquet  with columns: path, text, source, video, frame, track, w, h
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


FRAME_EXT_CANDIDATES = ['.jpg', '.png', '.jpeg']


def find_frame(frames_root, frames_dir_remote, source, video, frame_id, use_remote):
    if use_remote:
        for ext in FRAME_EXT_CANDIDATES:
            p = Path(frames_dir_remote) / f'{frame_id}{ext}'
            if p.exists():
                return str(p)
            # Some datasets pad: 0001.jpg
            for pad in (4, 5, 6):
                p2 = Path(frames_dir_remote) / f'{str(frame_id).zfill(pad)}{ext}'
                if p2.exists():
                    return str(p2)
        return None
    # Local layout
    base = Path(frames_root) / source / video
    for ext in FRAME_EXT_CANDIDATES:
        p = base / f'{frame_id}{ext}'
        if p.exists():
            return str(p)
        for pad in (4, 5, 6):
            p2 = base / f'{str(frame_id).zfill(pad)}{ext}'
            if p2.exists():
                return str(p2)
    return None


def crop_polygon(img, polygon, pad_ratio=0.08):
    h, w = img.shape[:2]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def resize_pad_square(img, target):
    h, w = img.shape[:2]
    if max(h, w) == 0:
        return None
    scale = target / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target, 3), 255, dtype=np.uint8)
    yoff = (target - nh) // 2
    xoff = (target - nw) // 2
    canvas[yoff:yoff+nh, xoff:xoff+nw] = img
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--frames-root', default=None)
    ap.add_argument('--use-frames-dir-remote', action='store_true',
                    help='use absolute frames_dir from manifest (when local matches remote layout)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--crop-size', type=int, default=384)
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--jpeg-quality', type=int, default=92)
    args = ap.parse_args()

    if not args.use_frames_dir_remote and not args.frames_root:
        print('ERROR: must pass --frames-root or --use-frames-dir-remote')
        sys.exit(1)

    out_dir = Path(args.out_dir)
    (out_dir / 'crops').mkdir(parents=True, exist_ok=True)

    # Read manifest
    rows = []
    with open(args.manifest, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f'Manifest rows: {len(rows)}')

    # Group by frame -> single decode per frame
    by_frame = {}
    for i, r in enumerate(rows):
        key = (r['source'], r['video'], r['frame'])
        by_frame.setdefault(key, []).append((i, r))

    print(f'Unique frames to decode: {len(by_frame)}')

    meta = []
    n_missing_frame = n_bad_crop = n_ok = 0
    for (source, video, frame), items in tqdm(by_frame.items(), desc='frames'):
        fdir_remote = items[0][1].get('frames_dir_remote', '')
        fpath = find_frame(args.frames_root, fdir_remote, source, video, frame,
                           use_remote=args.use_frames_dir_remote)
        if fpath is None:
            n_missing_frame += len(items)
            continue
        img = cv2.imread(fpath)
        if img is None:
            n_missing_frame += len(items)
            continue
        for idx_in_row, (gi, r) in enumerate(items):
            crop = crop_polygon(img, r['polygon'])
            if crop is None or crop.size == 0:
                n_bad_crop += 1
                continue
            sq = resize_pad_square(crop, args.crop_size)
            if sq is None:
                n_bad_crop += 1
                continue
            crop_name = f'{video}_f{frame}_t{r["track"]}_{idx_in_row}.jpg'
            crop_dir = out_dir / 'crops' / source / video
            crop_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_dir / crop_name),
                        sq, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            meta.append({
                'path': f'crops/{source}/{video}/{crop_name}',
                'text': r['text'],
                'source': source,
                'video': video,
                'frame': frame,
                'track': r['track'],
                'orig_w': crop.shape[1],
                'orig_h': crop.shape[0],
            })
            n_ok += 1

    # Write metadata
    import csv
    meta_csv = out_dir / 'metadata.csv'
    with open(meta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    print(f'\n=== DONE ===')
    print(f'  ok={n_ok}  missing_frame={n_missing_frame}  bad_crop={n_bad_crop}')
    print(f'  metadata: {meta_csv}')

    # Also try parquet if pandas+pyarrow available
    try:
        import pandas as pd
        pd.DataFrame(meta).to_parquet(out_dir / 'metadata.parquet', index=False)
        print(f'  metadata.parquet written')
    except Exception as e:
        print(f'  (parquet skipped: {e})')


if __name__ == '__main__':
    main()
