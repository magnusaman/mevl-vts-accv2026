"""Push extracted crops to HuggingFace as a private dataset.

Reads <out_dir>/metadata.csv and <out_dir>/crops/*/*/*.jpg from extract_crops.py.
Builds a HF Dataset with columns:
  image (Image), text, source, video, frame, track

Then pushes to hub.

Usage:
  HF_TOKEN=hf_xxx python push_to_hf.py \
      --crops-dir /workspace/crops_out \
      --repo whoamananand1/ic15v-dstext-text-crops \
      --split train  # or 'validation'
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Image, Value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops-dir', required=True)
    ap.add_argument('--metadata', default=None, help='default <crops-dir>/metadata.csv')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--private', action='store_true', default=True)
    ap.add_argument('--max-shard-size', default='500MB')
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir)
    meta_path = Path(args.metadata) if args.metadata else (crops_dir / 'metadata.csv')
    print(f'Reading metadata: {meta_path}')
    df = pd.read_csv(meta_path)
    print(f'  {len(df)} rows')

    # Resolve relative crop paths to absolute
    df['image'] = df['path'].apply(lambda p: str(crops_dir / p))

    # Drop rows where file is missing
    before = len(df)
    df = df[df['image'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f'  {len(df)} rows after missing-file filter (dropped {before-len(df)})')

    # Sanitize text -> str
    df['text'] = df['text'].astype(str)

    features = Features({
        'image': Image(),
        'text': Value('string'),
        'source': Value('string'),
        'video': Value('string'),
        'frame': Value('int32'),
        'track': Value('int64'),
        'orig_w': Value('int32'),
        'orig_h': Value('int32'),
    })

    ds_dict = {
        'image': df['image'].tolist(),
        'text': df['text'].tolist(),
        'source': df['source'].tolist(),
        'video': df['video'].tolist(),
        'frame': df['frame'].astype('int32').tolist(),
        'track': df['track'].astype('int64').tolist(),
        'orig_w': df['orig_w'].astype('int32').tolist(),
        'orig_h': df['orig_h'].astype('int32').tolist(),
    }

    ds = Dataset.from_dict(ds_dict, features=features)
    print(f'Built dataset: {ds}')

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('WARNING: HF_TOKEN not set; push will fail unless logged in.')

    print(f'Pushing to {args.repo} (split={args.split}, private={args.private})...')
    ds.push_to_hub(
        args.repo,
        split=args.split,
        private=args.private,
        max_shard_size=args.max_shard_size,
        token=token,
    )
    print('Done.')


if __name__ == '__main__':
    main()
