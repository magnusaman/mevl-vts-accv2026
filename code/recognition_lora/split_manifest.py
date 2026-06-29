"""Split manifest_train.jsonl into train/val (95/5), stratified by source dataset.
Also writes a small dev manifest (1000 rows) for quick smoke tests.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='manifest_train.jsonl')
    ap.add_argument('--out-train', default='manifest_split_train.jsonl')
    ap.add_argument('--out-val', default='manifest_split_val.jsonl')
    ap.add_argument('--out-dev', default='manifest_split_dev.jsonl')
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--dev-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    by_source = defaultdict(list)
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            by_source[r['source']].append(r)

    train, val = [], []
    for src, rows in by_source.items():
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
        print(f'  {src}: train={len(rows)-n_val} val={n_val}')

    random.shuffle(train)
    random.shuffle(val)

    def write(path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  wrote {path}: {len(rows)} rows')

    write(args.out_train, train)
    write(args.out_val, val)
    write(args.out_dev, train[:args.dev_size])


if __name__ == '__main__':
    main()
