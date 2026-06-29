"""Extract frames from MP4 videos in a directory tree.

Mirrors TransDETR's tools/DSText/ExtractFrame_FromVideo.py contract:
  output layout = <out_root>/<video_basename>/<frame_index>.jpg  (1-indexed)

Walks all .mp4 under --videos-root recursively; skips __MACOSX junk.

Usage:
    python extract_frames_video.py --videos-root /root/data/raw_extracted/DS-Training --out-root /root/data/frames/dstext_train
"""
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import cv2


def extract_one(video_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  [WARN] cannot open {video_path}', file=sys.stderr)
        return 0
    n = 0
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = out_dir / f'{idx}.jpg'
        if not out_path.exists():
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        idx += 1
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos-root', required=True, help='Root dir containing .mp4s (recursively)')
    ap.add_argument('--out-root', required=True, help='Where to write <video>/<idx>.jpg')
    ap.add_argument('--limit', type=int, default=0, help='Only process N videos (debug)')
    args = ap.parse_args()

    videos = []
    for rt, dirs, files in os.walk(args.videos_root):
        if '__MACOSX' in rt:
            continue
        for f in files:
            if f.startswith('._'):
                continue
            if f.lower().endswith('.mp4'):
                videos.append(Path(rt) / f)
    videos.sort()
    if args.limit:
        videos = videos[:args.limit]
    print(f'Found {len(videos)} videos under {args.videos_root}')

    total_frames = 0
    for vp in tqdm(videos, desc='videos'):
        out_dir = Path(args.out_root) / vp.stem
        if out_dir.exists() and len(list(out_dir.glob('*.jpg'))) > 0:
            # already done
            existing = len(list(out_dir.glob('*.jpg')))
            total_frames += existing
            continue
        n = extract_one(vp, out_dir)
        total_frames += n
    print(f'Done. {total_frames} frames written total.')


if __name__ == '__main__':
    main()
