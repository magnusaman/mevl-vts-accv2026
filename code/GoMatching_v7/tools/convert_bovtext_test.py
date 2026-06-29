"""
Convert BOVText Test annotations (per-video JSON format) to COCO format (test.json).

Input:  --ann_dir   path to Annotation/ dir (Cls*/Cls*_videoN.json)
        --frame_dir path to test frame dir (Cls*/Cls*_videoN/*.jpg)
Output: --out_file  path for output COCO JSON (e.g. datasets/BOVText/test.json)

Frame dirs must exist so we can read actual image dimensions.
Run after extracting the test Video.zip.
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_dir", required=True, help="Annotation/ dir from Test/Annotation.zip")
    p.add_argument("--frame_dir", required=True, help="Test frame dir (Cls*/Cls*_videoN/*.jpg)")
    p.add_argument("--out_file", required=True, help="Output COCO JSON path")
    return p.parse_args()


def points_to_poly(pts):
    """8-float list [x1,y1,...,x4,y4] → list[float] (same format as training poly)."""
    return [float(v) for v in pts]


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    frame_dir = Path(args.frame_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    img_id, ann_id, vid_id = 1, 1, 1

    video_jsons = sorted(ann_dir.rglob("*.json"))
    print(f"Found {len(video_jsons)} video annotation files")

    for vj in tqdm(video_jsons):
        with open(vj) as f:
            vid_data = json.load(f)

        cls_name = vj.parent.name            # e.g. Cls7_Game
        video_name = vj.stem                 # e.g. Cls7_Game_video10
        frame_prefix = frame_dir / cls_name / video_name

        if not frame_prefix.exists():
            print(f"  [skip] no frames at {frame_prefix}")
            continue

        frame_keys = sorted(vid_data.keys(), key=lambda x: int(x))
        for fk in frame_keys:
            frame_path = frame_prefix / f"{fk}.jpg"
            if not frame_path.exists():
                continue
            try:
                with Image.open(frame_path) as im:
                    W, H = im.size
            except Exception:
                continue

            file_name = f"{cls_name}/{video_name}/{fk}.jpg"
            images.append({
                "id": img_id,
                "file_name": file_name,
                "height": H,
                "width": W,
                "video_id": vid_id,
            })

            for ann in vid_data[fk]:
                pts = ann.get("points", [])
                if len(pts) != 8:
                    continue
                xs = pts[0::2]
                ys = pts[1::2]
                x0, y0 = min(xs), min(ys)
                w = max(xs) - x0
                h = max(ys) - y0

                language = ann.get("language", "alphanumeric")
                text_cat = "nonalphanumeric" if language not in ("alphanumeric", "chinese") else "normal"

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x0, y0, w, h],
                    "poly": pts,
                    "segmentation": [pts],
                    "area": w * h,
                    "iscrowd": 0,
                    "transcription": ann.get("transcription", "###"),
                    "text_category": text_cat,
                    "instance_id": int(ann.get("ID", 0)),
                })
                ann_id += 1

            img_id += 1
        vid_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text"}],
    }
    with open(out_file, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {len(images)} images, {len(annotations)} annotations → {out_file}")


if __name__ == "__main__":
    main()
