"""Build a TRACK-GROUPED manifest for multi-image recognition fusion (M1 / TRA).

Where build_manifest.py emits ONE row per crop (single-image training), this emits
ONE row per TRACK = the set of a tracked word's crops across frames + its GT text.
The multi-image trainer (train_qwen_lora_multiimage.py) samples K of these crops per
step and learns to read the word ONCE from all K views (cross-frame fusion) — the
exact train/inference setup the single-crop adapter never saw.

Output JSONL, one line per track:
  {"source": "icdar15v"|"dstext", "video": str, "track": int, "text": str,
   "frames_dir_remote": str,
   "crops": [{"frame": int, "polygon": [[x,y],...]}, ...]}   # 2..MAX_CROPS, frame-sorted

Filters mirror build_manifest.py (ignore / illegible markers / min-area). The track's
GT text is the majority vote over its legible per-frame texts (tracks are one word, but
per-frame labels occasionally disagree on case/segmentation — majority is the safe pick).
"""
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def is_legible(text):
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False
    if '##' in t:
        return False
    if t.startswith('#') and t.endswith('#'):
        return False
    if t in {'###', '#', '*', '?'}:
        return False
    return True


def poly_aabb_area(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))


def _videos(data, splits_to_keep):
    """Normalize to {video_id: video_dict} for flat (ic15v) or split-nested (dstext)."""
    if all(k in ('train', 'test', 'val') for k in list(data.keys())[:5]):
        videos = {}
        for split, vids in data.items():
            if split in splits_to_keep:
                videos.update(vids)
        return videos
    return {vid: v for vid, v in data.items() if v.get('split', 'train') in splits_to_keep}


def iter_tracks(data, source, splits_to_keep, min_area, min_crops, max_crops, seed):
    rng = random.Random(seed)
    videos = _videos(data, splits_to_keep)
    n_tracks = n_kept = n_drop_short = 0
    n_crops_total = 0

    for vid_id, vinfo in videos.items():
        frames_dir = vinfo.get('frames_dir', '')
        frames = vinfo.get('frames', {})

        per_track = defaultdict(list)  # tid -> [(fid, polygon, text)]
        for fid_str, dets in frames.items():
            fid = int(fid_str)
            for det in dets:
                if det.get('ignore', False):
                    continue
                txt = det.get('text', '')
                if not is_legible(txt):
                    continue
                poly = det.get('polygon', [])
                if not poly or poly_aabb_area(poly) < min_area:
                    continue
                per_track[det.get('track_id', -1)].append((fid, poly, txt.strip()))

        for tid, items in per_track.items():
            n_tracks += 1
            if len(items) < min_crops:
                n_drop_short += 1
                continue
            items.sort(key=lambda x: x[0])  # frame order
            # Cap crops/track: sample evenly across the trajectory (keeps temporal spread)
            if len(items) > max_crops:
                idxs = [round(i * (len(items) - 1) / (max_crops - 1)) for i in range(max_crops)]
                items = [items[i] for i in sorted(set(idxs))]
            text = Counter(t for _f, _p, t in items).most_common(1)[0][0]
            crops = [{'frame': f, 'polygon': p} for f, p, _t in items]
            n_kept += 1
            n_crops_total += len(crops)
            yield {
                'source': source,
                'video': vid_id,
                'track': tid,
                'text': text,
                'frames_dir_remote': frames_dir,
                'crops': crops,
            }

    avg = n_crops_total / max(n_kept, 1)
    print(f'  [{source}] tracks={n_tracks} kept={n_kept} '
          f'(dropped <{min_crops}-crop: {n_drop_short}); avg crops/track={avg:.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', default=str(Path(__file__).resolve().parent.parent / 'data' / 'unified_annotations.json'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'manifest_multiframe_train.jsonl'))
    ap.add_argument('--min-area', type=int, default=144)   # 12*12 px
    ap.add_argument('--min-crops', type=int, default=2)    # need >=2 views to be a fusion sample
    ap.add_argument('--max-crops', type=int, default=8)    # cap per track (trainer samples K<=this)
    ap.add_argument('--include-artvideo', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'Loading {args.annotations} ...')
    with open(args.annotations, encoding='utf-8') as f:
        ann = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    by_source = defaultdict(int)
    sections = [('icdar15v', 'icdar15v'), ('dstext', 'dstext')]
    if args.include_artvideo:
        sections.append(('artvideo', 'artvideo'))

    with open(out_path, 'w', encoding='utf-8') as fo:
        for key, source in sections:
            if key not in ann:
                print(f'  [skip] {key} not in annotations')
                continue
            print(f'\nProcessing {key} ...')
            for rec in iter_tracks(ann[key], source, splits_to_keep={'train'},
                                   min_area=args.min_area, min_crops=args.min_crops,
                                   max_crops=args.max_crops, seed=args.seed):
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total += 1
                by_source[source] += 1

    print(f'\n=== DONE ===')
    print(f'Wrote {total} track-records to {out_path}')
    print(f'By source: {dict(by_source)}')
    print(f'File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
