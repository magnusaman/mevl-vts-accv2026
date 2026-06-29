"""Build an RRC submission zip from per-video JSONs.

Two output formats supported via --task:
  --task tracking      : Task 2 — Tracking only. <object ID="N"> + <Point>s.
                         Matches existing v6 baseline submission format.
  --task spotting      : Task 3/4 — Spotting / E2E. Adds Transcription="..." attr.
                         REQUIRED for LoRA recognition to actually move the metric.

Same TEST_NUMS / video name mapping / frame counting as scripts/track_merger.py.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# scripts/track_merger.py lives at <repo>/scripts/track_merger.py
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from track_merger import TEST_NUMS, first_num_to_video_name, count_frames  # noqa: E402


def build_xml(frame_dets, num_frames, task='tracking'):
    """frame_dets: {fid_int: [{ID, points, transcription}, ...]}
    points is a flat list [x1, y1, x2, y2, ...] OR list of [x,y] pairs.
    """
    lines = ['<?xml version="1.0" encoding="utf-8" ?>', '\t<Frames>']
    for fid in range(1, num_frames + 1):
        objs = frame_dets.get(fid, [])
        if not objs:
            lines.append(f'\t\t<frame ID="{fid}" />')
            continue
        lines.append(f'\t\t<frame ID="{fid}">')
        for obj in objs:
            obj_id = int(obj['ID'])
            poly = obj['points']
            if task == 'spotting':
                tr = escape(str(obj.get('transcription', '')))
                lines.append(f'\t\t\t<object ID="{obj_id}" Transcription="{tr}">')
            else:
                lines.append(f'\t\t\t<object ID="{obj_id}">')
            # Flat list path
            if poly and not isinstance(poly[0], (list, tuple)):
                for i in range(0, len(poly), 2):
                    lines.append(f'\t\t\t\t<Point x="{int(poly[i])}" y="{int(poly[i+1])}" />')
            else:
                for p in poly:
                    lines.append(f'\t\t\t\t<Point x="{int(p[0])}" y="{int(p[1])}" />')
            lines.append('\t\t\t</object>')
        lines.append('\t\t</frame>')
    lines.append('\t</Frames>')
    return '\n'.join(lines) + '\n'


def load_video_jsons_to_frame_dets(json_path):
    """Convert {fid_str: [det,...]} to {fid_int: [{ID, points, transcription}, ...]}."""
    with open(json_path) as f:
        data = json.load(f)
    out = {}
    for fid_str, dets in data.items():
        fid = int(fid_str)
        kept = []
        for d in dets:
            if not d.get('points') or len(d['points']) < 6:
                continue
            kept.append({
                'ID': d.get('ID', d.get('id', 0)),
                'points': d['points'],
                'transcription': d.get('transcription', ''),
            })
        out[fid] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons-dir', required=True)
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out-zip', required=True)
    ap.add_argument('--task', choices=['tracking', 'spotting'], default='spotting',
                    help='spotting = include Transcription attribute (needed for LoRA to matter)')
    args = ap.parse_args()

    name_map = first_num_to_video_name()
    Path(args.out_zip).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(args.out_zip, 'w', zipfile.ZIP_DEFLATED) as zo:
        for fn in TEST_NUMS:
            vname = name_map[fn]
            jp = Path(args.jsons_dir) / f'{vname}.json'
            n_frames = count_frames(args.frames_root, vname)
            if not jp.exists():
                xml = build_xml({}, max(n_frames, 1), task=args.task)
            else:
                frame_dets = load_video_jsons_to_frame_dets(jp)
                xml = build_xml(frame_dets, max(n_frames, 1), task=args.task)
            zo.writestr(f'res_video_{fn}.xml', xml)
            written += 1
    sz = Path(args.out_zip).stat().st_size / 1024
    print(f'Wrote {args.out_zip}  ({sz:.1f} KB, {written} videos, task={args.task})')


if __name__ == '__main__':
    main()
