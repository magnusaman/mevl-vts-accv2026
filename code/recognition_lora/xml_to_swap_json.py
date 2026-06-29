"""Convert DSText GT XML (one per video) into per-video JSON in the format
expected by infer_swap_track_dedup.py:

    {frame_id_str: [
        {"points": [x1,y1,x2,y2,...],
         "ID": int,
         "transcription": str},
        ...], ...}

This lets us run the existing track-deduped Qwen-VL recognizer on GT-localized
boxes to measure the recognition-only ceiling on DSText.

Skips entries whose GT transcription is "##DONT#CARE##" by default
(--keep-dontcare keeps them — useful if the evaluator scores all polygons).
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_video_xml(xml_path: Path, keep_dontcare: bool = False) -> dict:
    """Parse one DSText GT XML into the swap-JSON format."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    out = {}  # frame_id_str -> list of dicts
    n_objs = 0
    n_skipped_dontcare = 0
    for frame in root.findall("frame"):
        fid = frame.get("ID")
        if fid is None:
            continue
        dets = []
        for obj in frame.findall("object"):
            tr = obj.get("Transcription", "")
            if not keep_dontcare and tr == "##DONT#CARE##":
                n_skipped_dontcare += 1
                continue
            oid_raw = obj.get("ID", "0")
            try:
                oid = int(oid_raw)
            except ValueError:
                continue
            pts = []
            for pt in obj.findall("Point"):
                try:
                    x = float(pt.get("x", "0"))
                    y = float(pt.get("y", "0"))
                except ValueError:
                    continue
                pts.append(x)
                pts.append(y)
            if len(pts) < 6:
                continue
            dets.append({
                "points": pts,
                "ID": oid,
                "transcription": tr,
            })
        if dets:
            out[fid] = dets
            n_objs += len(dets)
    return out, n_objs, n_skipped_dontcare


def video_id_from_xml_name(name: str) -> str:
    """Video_186_8_6_GT.xml -> Video_186_8_6 (match frames dir name)."""
    m = re.match(r"(Video_\d+_\d+_\d+)(?:_GT)?", name)
    if m:
        return m.group(1)
    return Path(name).stem.replace("_GT", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True,
                    help="Root dir containing <category>/*_GT.xml")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for per-video JSONs")
    ap.add_argument("--keep-dontcare", action="store_true")
    ap.add_argument("--xml-pattern", default="**/*_GT.xml")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xmls = sorted(Path(args.gt_root).glob(args.xml_pattern))
    print(f"Found {len(xmls)} GT XMLs under {args.gt_root}")

    grand_objs = 0
    grand_skip = 0
    for xp in xmls:
        vid = video_id_from_xml_name(xp.name)
        data, n_objs, n_skip = parse_video_xml(xp, keep_dontcare=args.keep_dontcare)
        out_path = out_dir / f"{vid}.json"
        with open(out_path, "w") as f:
            json.dump(data, f)
        grand_objs += n_objs
        grand_skip += n_skip
        print(f"  {xp.name} -> {out_path.name}  "
              f"({n_objs} objs across {len(data)} frames, "
              f"skipped {n_skip} dontcare)")

    print(f"\nDONE. {grand_objs} objects written, {grand_skip} dontcare skipped.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
