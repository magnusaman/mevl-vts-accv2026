"""Merge N per-video JSON detection sources via per-frame NMS.

Each input source directory has the format produced by GoMatching's eval.py:
    <source>/Video_X.json with {frame_id_str: [{"points": [...], "ID": int,
                                                "transcription": str, ...}, ...]}

Use cases:
  - Multi-scale TTA: two eval runs at different INPUT.MIN_SIZE_TEST, then merge
  - Detector ensemble: iter-30k baseline + v7 model, merge to recover boxes
    each model individually misses

Algorithm per (video, frame):
  1. Pool all detections from all sources (carry source index + original ID)
  2. NMS at IoU >= --nms-iou (default 0.5), keeping highest-score box
  3. For surviving boxes, preserve their original source's track ID, offset
     by 10000 * source_index so IDs across sources don't collide
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def poly_to_xyxy(points):
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def nms_per_frame(dets, iou_thr=0.5):
    """dets: list of (xyxy, score, det_dict). Returns list of det_dict's kept."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d[1])
    kept = []
    for cand in dets:
        keep = True
        for k in kept:
            if iou_xyxy(cand[0], k[0]) >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(cand)
    return [d[2] for d in kept]


def merge_one_video(source_dicts, iou_thr=0.5, score_key='score'):
    """source_dicts: list of {frame_id_str: [det,...]} dicts.
    Returns one merged {frame_id_str: [det,...]} dict."""
    all_frames = set()
    for s in source_dicts:
        all_frames.update(s.keys())
    out = {}
    for fid in sorted(all_frames, key=lambda x: int(x)):
        pool = []
        for si, src in enumerate(source_dicts):
            dets = src.get(fid, []) or []
            for det in dets:
                pts = det.get('points', [])
                xyxy = poly_to_xyxy(pts)
                if xyxy is None:
                    continue
                if score_key in det:
                    score = float(det[score_key])
                elif 'confidence' in det:
                    score = float(det['confidence'])
                else:
                    # Fallback: use box area as tie-breaker (larger box wins).
                    # GoMatching's eval.py JSON outputs don't include scores,
                    # so we need a deterministic NMS preference.
                    w = max(xyxy[2] - xyxy[0], 1.0)
                    h = max(xyxy[3] - xyxy[1], 1.0)
                    score = w * h
                new_det = dict(det)
                tid = new_det.get('ID', new_det.get('id', 0))
                new_det['ID'] = int(tid) + 10000 * si
                new_det['_source'] = si
                pool.append((xyxy, score, new_det))
        kept = nms_per_frame(pool, iou_thr=iou_thr)
        for d in kept:
            d.pop('_source', None)
        out[fid] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True,
                    help='Source dirs containing per-video JSONs')
    ap.add_argument('--out-dir', required=True, help='Merged output dir')
    ap.add_argument('--nms-iou', type=float, default=0.5)
    ap.add_argument('--score-key', default='score',
                    help='JSON key for det score (default: score)')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src_dirs = [Path(s) for s in args.sources]
    print(f'Merging from {len(src_dirs)} sources:')
    for s in src_dirs:
        n = len(list(s.glob('*.json')))
        print(f'  {s}  ({n} videos)')

    all_video_files = {p.name for d in src_dirs for p in d.glob('*.json')}
    print(f'Total unique video JSONs across sources: {len(all_video_files)}')

    grand_kept = 0
    grand_pre = 0
    for vname in sorted(all_video_files):
        src_dicts = []
        for sd in src_dirs:
            p = sd / vname
            if p.exists():
                with open(p) as f:
                    src_dicts.append(json.load(f))
            else:
                src_dicts.append({})
        merged = merge_one_video(src_dicts, iou_thr=args.nms_iou,
                                  score_key=args.score_key)
        n_pre = sum(len(s.get(fid, []) or []) for s in src_dicts
                    for fid in (set().union(*[d.keys() for d in src_dicts])))
        n_post = sum(len(v) for v in merged.values())
        grand_pre += n_pre
        grand_kept += n_post
        with open(out / vname, 'w') as f:
            json.dump(merged, f)
        print(f'  {vname:30s}  pre-NMS={n_pre:6d}  post-NMS={n_post:6d}  '
              f'compression={n_post/max(n_pre,1)*100:5.1f}%')
    print(f'\nDONE total pre={grand_pre} post={grand_kept} '
          f'kept_ratio={grand_kept/max(grand_pre,1)*100:.1f}%')
    print(f'Out: {out}')


if __name__ == '__main__':
    main()
