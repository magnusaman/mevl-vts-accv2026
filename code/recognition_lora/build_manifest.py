"""Build a unified crop manifest from unified_annotations.json.

Outputs JSONL where each line is one crop to extract:
  {"source": "icdar15v"|"dstext", "video": str, "frame": int,
   "polygon": [[x,y],...], "text": str, "track": int}

Filters applied:
  - ignore=True dropped
  - text containing '##' / starting+ending with '#' / empty / whitespace dropped
  - bbox area < min_area dropped (computed from polygon AABB)
  - per-track frame sampling: keep at most 1 crop every TRACK_STRIDE frames

Subsampling to balance datasets:
  - IC15v: keep all (small dataset)
  - DSText: stride=4 within tracks (still gives ~250k+ which is plenty)
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def is_legible(text):
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False
    # Common illegible markers across video text datasets
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


def iter_dataset(data, source, splits_to_keep, track_stride, min_area):
    """Yield filtered crops from a dataset section.
    `data` is either the top dict (ic15v) or split-keyed dict (dstext/artvideo).
    """
    # Normalize to {video_id: video_dict} regardless of whether splits are nested
    if all(k in ('train', 'test', 'val') for k in list(data.keys())[:5]):
        # Split-nested form (dstext, artvideo)
        videos = {}
        for split, vids in data.items():
            if split not in splits_to_keep:
                continue
            for vid_id, v in vids.items():
                videos[vid_id] = v
    else:
        # Flat form (ic15v) -- assume all train
        videos = {vid_id: v for vid_id, v in data.items() if v.get('split', 'train') in splits_to_keep}

    n_total = n_kept = n_drop_ignore = n_drop_text = n_drop_area = n_drop_stride = 0

    for vid_id, vinfo in videos.items():
        frames_dir = vinfo.get('frames_dir', '')
        frames = vinfo.get('frames', {})

        # Group by track for per-track frame sampling
        track_frames = defaultdict(list)
        for fid_str, dets in frames.items():
            fid = int(fid_str)
            for det in dets:
                track_frames[det.get('track_id', -1)].append((fid, det))

        for tid, items in track_frames.items():
            items.sort(key=lambda x: x[0])  # by frame id
            # Stride sampling: keep every Nth frame from this track
            kept_indices = set(range(0, len(items), track_stride))
            for idx, (fid, det) in enumerate(items):
                n_total += 1
                if det.get('ignore', False):
                    n_drop_ignore += 1
                    continue
                if not is_legible(det.get('text', '')):
                    n_drop_text += 1
                    continue
                if poly_aabb_area(det.get('polygon', [])) < min_area:
                    n_drop_area += 1
                    continue
                if idx not in kept_indices:
                    n_drop_stride += 1
                    continue
                n_kept += 1
                yield {
                    'source': source,
                    'video': vid_id,
                    'frame': fid,
                    'polygon': det['polygon'],
                    'text': det['text'].strip(),  # drop trailing/leading whitespace (DSText quirk)
                    'track': tid,
                    'frames_dir_remote': frames_dir,
                }

    print(f'  [{source}] total={n_total} kept={n_kept} '
          f'(dropped: ignore={n_drop_ignore} text={n_drop_text} '
          f'area={n_drop_area} stride={n_drop_stride})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', default=str(Path(__file__).resolve().parent.parent / 'data' / 'unified_annotations.json'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'manifest_train.jsonl'))
    ap.add_argument('--ic15v-stride', type=int, default=1)
    ap.add_argument('--dstext-stride', type=int, default=4)
    ap.add_argument('--min-area', type=int, default=144)  # 12*12 px
    ap.add_argument('--include-artvideo', action='store_true')
    ap.add_argument('--artvideo-stride', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    print(f'Loading {args.annotations}...')
    with open(args.annotations) as f:
        ann = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    by_source = defaultdict(int)
    with open(out_path, 'w', encoding='utf-8') as fo:
        print('\nProcessing icdar15v...')
        for rec in iter_dataset(ann['icdar15v'], 'icdar15v',
                                splits_to_keep={'train'},
                                track_stride=args.ic15v_stride,
                                min_area=args.min_area):
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            total_kept += 1
            by_source['icdar15v'] += 1

        print('\nProcessing dstext (train split only)...')
        for rec in iter_dataset(ann['dstext'], 'dstext',
                                splits_to_keep={'train'},
                                track_stride=args.dstext_stride,
                                min_area=args.min_area):
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            total_kept += 1
            by_source['dstext'] += 1

        if args.include_artvideo:
            print('\nProcessing artvideo (train split only)...')
            for rec in iter_dataset(ann['artvideo'], 'artvideo',
                                    splits_to_keep={'train'},
                                    track_stride=args.artvideo_stride,
                                    min_area=args.min_area):
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total_kept += 1
                by_source['artvideo'] += 1

    print(f'\n=== DONE ===')
    print(f'Wrote {total_kept} crops to {out_path}')
    print(f'By source: {dict(by_source)}')
    print(f'File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
