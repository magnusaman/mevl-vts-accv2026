"""Build a unified crop manifest from IC15-V + DSText raw GT XMLs.

GT formats:
  IC-15-V train: Video_*.xml  (siblings of Video_*.mp4 in same dir)
  DSText train: Train/<scene>/Video_*_GT.xml

Both XMLs share the RRC schema:
  <Frames>
    <frame ID="1">
      <object ID="1" Transcription="hello" Language="LATIN" Quality="HIGH" Mirrored="0">
        <Point x="..." y="..."/>  ... (4 corners)
      </object>
      ...
    </frame>
    ...
  </Frames>

Output JSONL rows:
  {image_path, polygon, text, dataset, video, frame_id, track_id}

Filters (matches v9 conventions):
  - skip ### / illegible
  - skip area < 144 px (12x12)
  - skip if frame jpg missing
  - per-track stride (DSText only: stride=4) to cap dataset size
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def parse_xml(xml_path: Path, frames_root: Path, dataset_tag: str, video_stem: str):
    rows = []
    skip_illeg = skip_small = skip_no_frame = 0
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f'  [WARN] parse error {xml_path}: {e}', file=sys.stderr)
        return rows, (skip_illeg, skip_small, skip_no_frame)
    root = tree.getroot()
    for frame_el in root.findall('.//frame'):
        fid = frame_el.get('ID') or frame_el.get('id')
        if fid is None:
            continue
        try:
            fid_int = int(fid)
        except ValueError:
            continue
        # IC15-V & DSText: frames named "1.jpg", "2.jpg", ...
        frame_jpg = frames_root / video_stem / f'{fid_int}.jpg'
        if not frame_jpg.exists():
            skip_no_frame += 1
            continue
        for obj in frame_el.findall('object'):
            text = obj.get('Transcription', '').strip()
            if not text:
                skip_illeg += 1
                continue
            # Reject any illegibility marker variant:
            #   ###, ##, ####, *, _   ->  text strips to '' after removing #/*/_
            #   ##DONT#CARE##         ->  case-insensitive "DONT" present (IC15-V convention)
            #   "do not care", "don't care" variants
            tnorm = text.upper()
            if 'DONT' in tnorm or "DON'T" in tnorm or 'DO NOT CARE' in tnorm:
                skip_illeg += 1
                continue
            if text.replace('#', '').replace('*', '').replace('_', '').strip() == '':
                skip_illeg += 1
                continue
            pts = obj.findall('Point')
            if len(pts) < 3:
                continue
            polygon = []
            ok = True
            for p in pts:
                try:
                    polygon.append([int(float(p.get('x'))), int(float(p.get('y')))])
                except (TypeError, ValueError):
                    ok = False
                    break
            if not ok or len(polygon) < 3:
                continue
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            if bw * bh < 144:
                skip_small += 1
                continue
            track_id = obj.get('ID')
            rows.append({
                'image_path': str(frame_jpg),
                'polygon': polygon,
                'text': text,
                'dataset': dataset_tag,
                'video': video_stem,
                'frame_id': fid_int,
                'track_id': track_id,
            })
    return rows, (skip_illeg, skip_small, skip_no_frame)


def stride_by_track(rows, stride):
    """Keep every Nth row per track_id (sorted by frame_id) to reduce redundancy."""
    if stride <= 1:
        return rows
    by_track = defaultdict(list)
    for r in rows:
        key = (r['video'], r['track_id'])
        by_track[key].append(r)
    kept = []
    for key, group in by_track.items():
        group.sort(key=lambda r: r['frame_id'])
        kept.extend(group[::stride])
    return kept


def collect_ic15v(gt_root: Path, frames_root: Path, dataset_tag='ic15v'):
    """IC-15-V train: Video_*.xml siblings of Video_*.mp4 in flat structure."""
    xmls = sorted(gt_root.glob('Video_*_GT.xml'))
    print(f'  [{dataset_tag}] found {len(xmls)} GT XMLs')
    all_rows = []
    totals = [0, 0, 0]
    for xml_path in tqdm(xmls, desc=f'parse {dataset_tag}'):
        video_stem = xml_path.stem.replace('_GT', '')
        rows, sk = parse_xml(xml_path, frames_root, dataset_tag, video_stem)
        all_rows.extend(rows)
        for i in range(3):
            totals[i] += sk[i]
    print(f'  [{dataset_tag}] kept={len(all_rows)} illeg={totals[0]} small={totals[1]} no_frame={totals[2]}')
    return all_rows


def collect_dstext(gt_root: Path, frames_root: Path, stride: int, dataset_tag='dstext'):
    """DSText train: Train/<scene>/Video_*_GT.xml"""
    xmls = sorted(gt_root.rglob('Video_*_GT.xml'))
    print(f'  [{dataset_tag}] found {len(xmls)} GT XMLs')
    all_rows = []
    totals = [0, 0, 0]
    for xml_path in tqdm(xmls, desc=f'parse {dataset_tag}'):
        video_stem = xml_path.stem.replace('_GT', '')
        rows, sk = parse_xml(xml_path, frames_root, dataset_tag, video_stem)
        all_rows.extend(rows)
        for i in range(3):
            totals[i] += sk[i]
    print(f'  [{dataset_tag}] raw kept={len(all_rows)}')
    if stride > 1:
        all_rows = stride_by_track(all_rows, stride)
        print(f'  [{dataset_tag}] after per-track stride={stride}: {len(all_rows)}')
    print(f'  [{dataset_tag}] illeg={totals[0]} small={totals[1]} no_frame={totals[2]}')
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ic15v-gt-root', help='Dir of IC15-V Video_*_GT.xml files', default='')
    ap.add_argument('--ic15v-frames-root', help='Dir with <video>/<idx>.jpg', default='')
    ap.add_argument('--dstext-gt-root', help='Dir containing Train/<scene>/Video_*_GT.xml', default='')
    ap.add_argument('--dstext-frames-root', help='Dir with <video>/<idx>.jpg', default='')
    ap.add_argument('--dstext-stride', type=int, default=4)
    ap.add_argument('--out', required=True, help='Output JSONL path')
    args = ap.parse_args()

    all_rows = []

    if args.ic15v_gt_root and args.ic15v_frames_root:
        print('=== IC15-V ===')
        all_rows.extend(collect_ic15v(Path(args.ic15v_gt_root), Path(args.ic15v_frames_root)))

    if args.dstext_gt_root and args.dstext_frames_root:
        print('=== DSText ===')
        all_rows.extend(collect_dstext(Path(args.dstext_gt_root), Path(args.dstext_frames_root), args.dstext_stride))

    print(f'\nTOTAL ROWS: {len(all_rows)}')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote manifest: {out_path}')

    # quick stats
    by_ds = defaultdict(int)
    text_lens = []
    for r in all_rows:
        by_ds[r['dataset']] += 1
        text_lens.append(len(r['text']))
    print('\nPer-dataset counts:')
    for k, v in sorted(by_ds.items()):
        print(f'  {k}: {v}')
    if text_lens:
        text_lens.sort()
        print(f'Text length: min={text_lens[0]}  median={text_lens[len(text_lens)//2]}  p99={text_lens[int(len(text_lens)*0.99)]}  max={text_lens[-1]}')


if __name__ == '__main__':
    main()
