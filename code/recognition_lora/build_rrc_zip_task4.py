"""RRC IC15-V Task 4 (end-to-end spotting) submission zip builder.

Matches the official sample format (ch3-T4-sample.zip):
  - One pair per video: res_video_<N>.xml + res_video_<N>.txt
  - XML: <?xml version="1.0" encoding="utf-8" ?> + <Frames> containing every
    frame from 1..max_frame_id, empty frames as <frame ID="N" />
  - Point self-close: <Point x="..." y="..." /> (with space-slash)
  - TXT: per-track consensus transcription, one row "track_id","text"
"""
import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from xml.sax.saxutils import quoteattr


def video_num_from_name(name: str) -> str:
    """Video_1_1_2.json -> '1'"""
    m = re.search(r"Video_(\d+)", name)
    if not m:
        raise ValueError(f"no Video_<num> in {name}")
    return m.group(1)


def build_xml(video_data: dict) -> str:
    """Emit XML matching the Task 4 sample format.

    video_data = {frame_id_str: [det_dict, ...]}  (may include empty frames)
    """
    if not video_data:
        max_fid = 0
    else:
        max_fid = max(int(fid) for fid in video_data.keys())

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8" ?>')
    lines.append('\t<Frames>')
    for fid in range(1, max_fid + 1):
        dets = video_data.get(str(fid), [])
        if not dets:
            lines.append(f'\t\t<frame ID="{fid}" />')
            continue
        lines.append(f'\t\t<frame ID="{fid}">')
        for det in dets:
            obj_id = det.get("ID", det.get("id", 0))
            pts = det.get("points", [])
            if len(pts) < 6:
                continue
            tr = _sanitize_txt_field(str(det.get("transcription", "")))
            tr_attr = quoteattr(tr)  # safely escape for XML attribute
            lines.append(f'\t\t\t<object ID="{int(obj_id)}" Transcription={tr_attr}>')
            for i in range(0, len(pts) - 1, 2):
                x = int(round(float(pts[i])))
                y = int(round(float(pts[i + 1])))
                lines.append(f'\t\t\t\t<Point x="{x}" y="{y}" />')
            lines.append('\t\t\t</object>\t\t')  # sample uses trailing tabs after </object>
        lines.append('\t\t</frame>')
    lines.append('\t</Frames>')
    return "\n".join(lines) + "\n"


def _sanitize_txt_field(s: str) -> str:
    """Make a transcription safe to put inside CSV "..." quotes.

    The RRC IC15-V parser rejects fields containing literal newlines because
    the format spec is one record per line. We:
      - collapse any \\n, \\r, \\t into spaces
      - escape internal " as "" (CSV standard double-up)
      - strip leading/trailing whitespace
    """
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())              # collapse multiple spaces
    s = s.replace('"', '""')             # CSV-escape internal quotes
    return s.strip()


def build_txt(video_data: dict) -> str:
    """Per-track consensus transcription file.

    Output: one row per unique track ID: "track_id","most_common_transcription"
    Sorted by track ID.
    """
    by_track = defaultdict(Counter)  # track_id -> Counter of (sanitized) transcriptions
    for dets in video_data.values():
        for det in dets:
            tid = det.get("ID", det.get("id", None))
            if tid is None:
                continue
            tr = _sanitize_txt_field(str(det.get("transcription", "")))
            if tr:
                by_track[tid][tr] += 1

    rows = []
    for tid in sorted(by_track.keys()):
        counter = by_track[tid]
        # consensus = most common; if tie, longest then alphabetical
        most_common = counter.most_common()
        if not most_common:
            continue
        top_count = most_common[0][1]
        candidates = [t for t, c in most_common if c == top_count]
        # pick longest (more info), then alpha for determinism
        best = sorted(candidates, key=lambda s: (-len(s), s))[0]
        rows.append(f'"{tid}","{best}"')
    return "\n".join(rows) + ("\n" if rows else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons-dir", required=True)
    ap.add_argument("--out", required=True, help="output zip path")
    args = ap.parse_args()

    jsons = sorted(Path(args.jsons_dir).glob("*.json"))
    print(f"found {len(jsons)} video JSONs in {args.jsons_dir}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_obj_total = 0
    n_track_total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for jp in jsons:
            with open(jp) as f:
                data = json.load(f)
            vnum = video_num_from_name(jp.name)
            xml = build_xml(data)
            txt = build_txt(data)
            n_obj = xml.count("<object")
            n_track = txt.count("\n") if txt else 0
            n_obj_total += n_obj
            n_track_total += n_track
            zf.writestr(f"res_video_{vnum}.xml", xml)
            zf.writestr(f"res_video_{vnum}.txt", txt)
            print(f"  {jp.name} -> res_video_{vnum}.{{xml,txt}}  "
                  f"({n_obj} objects, {n_track} tracks)")

    print(f"\nDONE. wrote {args.out}  "
          f"total objects={n_obj_total}  total tracks={n_track_total}")


if __name__ == "__main__":
    main()
