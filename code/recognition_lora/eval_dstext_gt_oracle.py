"""Score Qwen-VL LoRA recognition on DSText GT crops.

Given:
  - GT XMLs:        <gt-root>/<category>/<Video_X>_GT.xml
  - Predicted JSONs: <pred-dir>/<Video_X>.json  (output of infer_swap_track_dedup.py)

Pairs each (video, frame_id, object_id) GT entry with its prediction,
computes strict accuracy + normalized edit distance (1-NED).

Handles partial results (only scores videos with predicted JSON present).
Skips ##DONT#CARE## entries from GT.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        pass
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return 1.0 - prev[n] / max(m, n)


def parse_gt(xml_path: Path) -> dict:
    """Return {(frame_id_str, obj_id): gt_transcription}."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    out = {}
    for frame in root.findall("frame"):
        fid = frame.get("ID")
        for obj in frame.findall("object"):
            oid_raw = obj.get("ID", "")
            try:
                oid = int(oid_raw)
            except ValueError:
                continue
            tr = obj.get("Transcription", "")
            out[(fid, oid)] = tr
    return out


def video_id_from_xml_name(name: str) -> str:
    m = re.match(r"(Video_\d+_\d+_\d+)(?:_GT)?", name)
    if m:
        return m.group(1)
    return Path(name).stem.replace("_GT", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pred-dir", required=True,
                    help="Dir of per-video swap-JSONs with predicted transcriptions")
    ap.add_argument("--case-insensitive", action="store_true", default=True)
    ap.add_argument("--keep-dontcare", action="store_true",
                    help="Include ##DONT#CARE## in the score (default: skip)")
    ap.add_argument("--per-video", action="store_true",
                    help="Print per-video numbers too")
    args = ap.parse_args()

    xmls = sorted(Path(args.gt_root).glob("**/*_GT.xml"))
    print(f"Found {len(xmls)} GT XMLs")
    pred_dir = Path(args.pred_dir)

    n_total = 0
    n_correct = 0
    n_ned_sum = 0.0
    n_dontcare_skipped = 0
    n_missing_pred = 0

    per_video_rows = []

    for xp in xmls:
        vid = video_id_from_xml_name(xp.name)
        pred_path = pred_dir / f"{vid}.json"
        if not pred_path.exists():
            print(f"  [skip] no prediction for {vid}")
            continue

        gt_map = parse_gt(xp)
        with open(pred_path) as f:
            pred = json.load(f)

        v_total = 0
        v_correct = 0
        v_ned_sum = 0.0

        for fid, dets in pred.items():
            for det in dets:
                oid = det.get("ID", det.get("id", None))
                if oid is None:
                    continue
                gt_tr = gt_map.get((fid, oid), None)
                if gt_tr is None:
                    n_missing_pred += 1
                    continue
                if not args.keep_dontcare and gt_tr == "##DONT#CARE##":
                    n_dontcare_skipped += 1
                    continue
                pred_tr = det.get("transcription", "")
                a = pred_tr.lower() if args.case_insensitive else pred_tr
                b = gt_tr.lower() if args.case_insensitive else gt_tr
                v_total += 1
                if a == b:
                    v_correct += 1
                v_ned_sum += _ratio(a, b)

        n_total += v_total
        n_correct += v_correct
        n_ned_sum += v_ned_sum

        if args.per_video and v_total > 0:
            per_video_rows.append((
                vid,
                v_total,
                v_correct / v_total,
                v_ned_sum / v_total))

    if args.per_video:
        print("\n--- per-video ---")
        for vid, vt, acc, ned in per_video_rows:
            print(f"  {vid:32s}  N={vt:6d}  strict={acc*100:6.2f}%  1-NED={ned*100:6.2f}%")

    if n_total == 0:
        print("\nNo paired GT/prediction entries scored.")
        return

    strict = n_correct / n_total
    one_ned = n_ned_sum / n_total
    print(f"\n=== AGGREGATE OVER {len(per_video_rows) if args.per_video else 'all-scored'} VIDEOS ===")
    print(f"  paired objects scored : {n_total:,}")
    print(f"  ##DONT#CARE## skipped : {n_dontcare_skipped:,}")
    print(f"  missing-from-GT       : {n_missing_pred:,}")
    print(f"  STRICT ACCURACY       : {strict*100:6.2f}%")
    print(f"  1-NED (avg ratio)     : {one_ned*100:6.2f}%")
    print(f"  (case_insensitive={args.case_insensitive})")


if __name__ == "__main__":
    main()
