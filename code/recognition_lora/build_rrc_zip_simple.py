"""Standalone RRC IC15-V submission zip builder. No external dependencies.

Input JSON layout (per video):
    {frame_id: [{points, ID, transcription, ...}, ...], ...}

Output RRC XML (one per video, zipped together):
    <Frames>
      <frame ID="1">
        <object ID="1" Transcription="hello">
          <Point x="..." y="..."/>  ... (n points)
        </object>
        ...
      </frame>
      ...
    </Frames>

Naming: res_video_<num>.xml where <num> is the first numeric token in the
video filename (e.g. Video_1_1_2.json -> res_video_1.xml).

Two task modes:
    --task spotting: include Transcription attribute (Task 3)
    --task tracking: skip Transcription (Task 2, matches v6 baseline)
"""
import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def video_num_from_name(name: str) -> str:
    """Video_1_1_2.json -> '1'"""
    m = re.search(r"Video_(\d+)", name)
    if not m:
        raise ValueError(f"no Video_<num> in {name}")
    return m.group(1)


def build_xml(video_data: dict, task: str = "spotting") -> str:
    """video_data = {frame_id_str: [det_dict, ...]}"""
    lines = ['<Frames>']
    # Sort frame ids numerically
    for fid in sorted(video_data.keys(), key=lambda x: int(x)):
        dets = video_data[fid]
        if not dets:
            continue
        lines.append(f'\t<frame ID="{int(fid)}">')
        for det in dets:
            obj_id = det.get("ID", det.get("id", 0))
            pts = det.get("points", [])
            if len(pts) < 6:
                continue
            if task == "spotting":
                tr = escape(str(det.get("transcription", "")))
                lines.append(f'\t\t<object ID="{int(obj_id)}" Transcription="{tr}">')
            else:
                lines.append(f'\t\t<object ID="{int(obj_id)}">')
            # points are flat [x1,y1,x2,y2,...]
            for i in range(0, len(pts) - 1, 2):
                x = int(round(float(pts[i])))
                y = int(round(float(pts[i + 1])))
                lines.append(f'\t\t\t<Point x="{x}" y="{y}"/>')
            lines.append('\t\t</object>')
        lines.append('\t</frame>')
    lines.append('</Frames>')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons-dir", required=True)
    ap.add_argument("--out", required=True, help="output zip path")
    ap.add_argument("--task", choices=["tracking", "spotting"], default="spotting")
    args = ap.parse_args()

    jsons = sorted(Path(args.jsons_dir).glob("*.json"))
    print(f"found {len(jsons)} video JSONs in {args.jsons_dir}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_objs_total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for jp in jsons:
            with open(jp) as f:
                data = json.load(f)
            xml = build_xml(data, task=args.task)
            n_objs = xml.count("<object")
            n_objs_total += n_objs
            vnum = video_num_from_name(jp.name)
            arcname = f"res_video_{vnum}.xml"
            zf.writestr(arcname, xml)
            print(f"  {jp.name} -> {arcname}  ({n_objs} objects)")
    print(f"\nDONE. wrote {args.out}  task={args.task}  total objects={n_objs_total}")


if __name__ == "__main__":
    main()
