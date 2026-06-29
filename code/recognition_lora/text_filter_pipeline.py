"""Foundation-encoder text-vs-non-text post-detection filter.

A small MLP scores each candidate detector box on whether it's really text.
Trained on (CLIP feature inside box, label) pairs where label = max IoU with
GT polygons exceeds 0.5.

At inference time, run the detector at a LOW threshold (e.g. 0.10) to recover
misses, then this MLP filters out the FPs the threshold drop introduced.

Targets the dominant error in the IC15-V per-video failure CSV: misses (5x
FPs, 25x switches). Lowering threshold recovers misses; the MLP recovers
precision.

Two subcommands:

  train   --gt-json  --pred-jsons  --frames-root  --out-mlp  [--limit-videos N]
  apply   --pred-jsons  --frames-root  --mlp  --out-dir  [--score-thresh 0.5]

The MLP input is a 1024-d CLIP-ViT-L/14 visual pooled feature of the
PIL-cropped polygon region (with 10% padding).
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------- CLIP feature extractor ----------------------

class ClipFeatureExtractor:
    def __init__(self, device='cuda'):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14-336', pretrained='openai')
        self.model = self.model.visual.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.in_size = 336

    @torch.inference_mode()
    def features(self, pil_crops):
        if not pil_crops:
            return torch.zeros(0, 1024, device=self.device)
        batch = torch.stack([self.preprocess(c) for c in pil_crops]).to(self.device)
        # open_clip ViT visual returns CLS-pooled (B, 1024) directly
        feats = self.model(batch)
        return feats  # (B, 1024)


def crop_polygon(img_pil, points, pad_ratio=0.10):
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return None
    x0, x1 = max(0, int(min(xs))), int(max(xs))
    y0, y1 = max(0, int(min(ys))), int(max(ys))
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return None
    px = int(bw * pad_ratio); py = int(bh * pad_ratio)
    W, H = img_pil.size
    x0 = max(0, x0 - px); x1 = min(W, x1 + px)
    y0 = max(0, y0 - py); y1 = min(H, y1 + py)
    if x1 <= x0 or y1 <= y0:
        return None
    return img_pil.crop((x0, y0, x1, y1))


def poly_iou(a_pts, b_pts):
    """Bbox-IoU on flat polygons. Good enough for text proxy."""
    if len(a_pts) < 6 or len(b_pts) < 6:
        return 0.0
    ax, ay = a_pts[0::2], a_pts[1::2]
    bx, by = b_pts[0::2], b_pts[1::2]
    ax1, ay1, ax2, ay2 = min(ax), min(ay), max(ax), max(ay)
    bx1, by1, bx2, by2 = min(bx), min(by), max(bx), max(by)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    return inter / max(a + b - inter, 1e-9)


# ---------------------- GT loader (COCO format) ----------------------

def load_gt_per_frame(coco_json):
    """Return {video_id: {frame_id (int): [GT polygon list, ...]}}.
    GT polygons are flat [x1,y1,x2,y2,...]. We use the segmentation field if
    present, else the bbox-as-poly."""
    print(f'loading {coco_json}...')
    with open(coco_json) as f:
        data = json.load(f)
    img_by_id = {im['id']: im for im in data['images']}
    out = defaultdict(lambda: defaultdict(list))
    n = 0
    for ann in data['annotations']:
        img = img_by_id.get(ann['image_id'])
        if img is None:
            continue
        fn = img['file_name']
        parts = Path(fn).parts
        video_id = parts[-2] if len(parts) >= 2 else 'unknown'
        try:
            file_idx = int(Path(fn).stem)
        except ValueError:
            continue
        frame_id = file_idx + 1
        if 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            if isinstance(seg, list) and seg:
                poly = seg[0] if isinstance(seg[0], list) else seg
            else:
                continue
        elif 'bbox' in ann:
            x, y, w, h = ann['bbox']
            poly = [x, y, x+w, y, x+w, y+h, x, y+h]
        else:
            continue
        out[video_id][frame_id].append(poly)
        n += 1
    print(f'loaded {n} GT polygons across {len(out)} videos')
    return out


# ---------------------- Build training data ----------------------

def build_training_data(pred_jsons_dir, gt_per_frame, frames_root, limit_videos=0):
    """Iterate detector predictions, label each by IoU with GT, return
    (PIL_crops_list, labels_list) for batch CLIP feature extraction."""
    pred_dir = Path(pred_jsons_dir)
    frames_root = Path(frames_root)
    jsons = sorted(pred_dir.glob('*.json'))
    if limit_videos:
        jsons = jsons[:limit_videos]
    print(f'iterating {len(jsons)} prediction JSONs...')

    crops, labels = [], []
    n_pos = n_neg = 0
    for jp in jsons:
        video_id = jp.stem
        gt_frames = gt_per_frame.get(video_id, {})
        if not gt_frames:
            # try category-prefixed lookup (for DSText layout)
            for k in gt_per_frame:
                if k.endswith(video_id):
                    gt_frames = gt_per_frame[k]; break
        with open(jp) as f:
            preds = json.load(f)
        for fid_str, dets in preds.items():
            try:
                fid = int(fid_str)
            except ValueError:
                continue
            # locate the frame on disk
            frame_path = None
            for ext in ('.jpg', '.png', '.jpeg'):
                p = frames_root / video_id / f'{fid}{ext}'
                if p.exists():
                    frame_path = p; break
                p = frames_root / video_id / f'{str(fid).zfill(6)}{ext}'
                if p.exists():
                    frame_path = p; break
            if frame_path is None:
                continue
            img = Image.open(frame_path).convert('RGB')
            gt_polys = gt_frames.get(fid, [])
            for det in dets:
                pts = det.get('points', [])
                if len(pts) < 6:
                    continue
                crop = crop_polygon(img, pts)
                if crop is None:
                    continue
                max_iou = 0.0
                for gp in gt_polys:
                    iou = poly_iou(pts, gp)
                    if iou > max_iou:
                        max_iou = iou
                label = 1 if max_iou > 0.5 else 0
                crops.append(crop)
                labels.append(label)
                if label == 1:
                    n_pos += 1
                else:
                    n_neg += 1
        print(f'  {video_id}: cum n_pos={n_pos} n_neg={n_neg}')
    print(f'TOTAL training samples: pos={n_pos} neg={n_neg} total={n_pos+n_neg}')
    return crops, labels


def extract_features_batched(crops, extractor, batch_size=64):
    out = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        feats = extractor.features(batch).cpu().numpy()
        out.append(feats)
        if (i // batch_size) % 20 == 0:
            print(f'  features {i+len(batch)}/{len(crops)}')
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1024))


# ---------------------- MLP filter ----------------------

class TextFilterMLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X, y, val_frac=0.15, epochs=80, lr=1e-3, device='cuda'):
    """Returns (model, val_metrics). Class-weighted BCE loss."""
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = X.shape[0]
    perm = torch.randperm(n)
    nv = max(1, int(n * val_frac))
    idx_v, idx_t = perm[:nv], perm[nv:]
    Xt, yt = X[idx_t], y[idx_t]
    Xv, yv = X[idx_v], y[idx_v]
    pos_w = max((yt == 0).sum().item() / max((yt == 1).sum().item(), 1), 1.0)
    model = TextFilterMLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    best_val_auc = 0
    best_state = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv))
                pred = (pv > 0.5).float()
                acc = (pred == yv).float().mean().item()
                tp = ((pred == 1) & (yv == 1)).sum().item()
                fp = ((pred == 1) & (yv == 0)).sum().item()
                fn = ((pred == 0) & (yv == 1)).sum().item()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f'  ep={ep}  loss={loss.item():.4f}  val acc={acc:.4f} '
                  f'prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}')
            if f1 > best_val_auc:
                best_val_auc = f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_val_f1=best_val_auc, n_train=len(yt), n_val=len(yv),
                       pos_weight=pos_w)


# ---------------------- Apply filter ----------------------

def apply_filter(pred_jsons_dir, frames_root, mlp_path, out_dir,
                 score_thresh=0.5, min_track_keep=3, device='cuda'):
    pred_dir = Path(pred_jsons_dir)
    frames_root = Path(frames_root)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f'loading MLP from {mlp_path}')
    state = torch.load(mlp_path, map_location=device)
    model = TextFilterMLP(in_dim=state['in_dim']).to(device).eval()
    model.load_state_dict(state['mlp'])
    extractor = ClipFeatureExtractor(device=device)
    jsons = sorted(pred_dir.glob('*.json'))
    print(f'filtering {len(jsons)} videos at score_thresh={score_thresh}')
    grand_kept = grand_pre = 0
    for jp in jsons:
        video_id = jp.stem
        with open(jp) as f:
            preds = json.load(f)
        new_preds = {}
        n_pre = sum(len(v) for v in preds.values())
        # First pass: score every box, group by track
        per_track_scores = defaultdict(list)
        per_track_dets = defaultdict(list)
        for fid_str, dets in preds.items():
            try:
                fid = int(fid_str)
            except ValueError:
                new_preds[fid_str] = dets; continue
            # locate frame
            frame_path = None
            for ext in ('.jpg', '.png', '.jpeg'):
                p = frames_root / video_id / f'{fid}{ext}'
                if p.exists():
                    frame_path = p; break
                p = frames_root / video_id / f'{str(fid).zfill(6)}{ext}'
                if p.exists():
                    frame_path = p; break
            if frame_path is None:
                new_preds[fid_str] = dets; continue
            img = Image.open(frame_path).convert('RGB')
            crops_local = []; idx_local = []
            for i, det in enumerate(dets):
                pts = det.get('points', [])
                if len(pts) < 6:
                    continue
                c = crop_polygon(img, pts)
                if c is None:
                    continue
                crops_local.append(c); idx_local.append(i)
            if not crops_local:
                new_preds[fid_str] = []; continue
            with torch.no_grad():
                feats = extractor.features(crops_local)
                scores = torch.sigmoid(model(feats)).cpu().tolist()
            for li, det_idx in enumerate(idx_local):
                det = dets[det_idx]
                tid = det.get('ID', det.get('id', -1))
                per_track_scores[(video_id, tid)].append(scores[li])
                per_track_dets[(video_id, tid)].append((fid_str, det_idx, det, scores[li]))
        # Decide which tracks to keep: mean score above threshold AND min count
        kept_tracks = set()
        for tk, scores in per_track_scores.items():
            mean_s = sum(scores) / len(scores)
            if mean_s >= score_thresh and len(scores) >= min_track_keep:
                kept_tracks.add(tk)
            elif mean_s >= score_thresh + 0.15:  # very confident even on short track
                kept_tracks.add(tk)
        # Re-emit only detections whose track was kept (AND keep per-det score >= 0.25 floor)
        for fid_str, dets in preds.items():
            kept_here = []
            for det in dets:
                tid = det.get('ID', det.get('id', -1))
                if (video_id, tid) in kept_tracks:
                    kept_here.append(det)
            new_preds[fid_str] = kept_here
        n_post = sum(len(v) for v in new_preds.values())
        with open(out / jp.name, 'w') as f:
            json.dump(new_preds, f)
        print(f'  {video_id}: pre={n_pre} post={n_post} kept={n_post/max(n_pre,1)*100:.1f}% '
              f'tracks_pre={len(per_track_scores)} tracks_kept={len(kept_tracks)}')
        grand_pre += n_pre; grand_kept += n_post
    print(f'\nDONE. grand_pre={grand_pre} grand_kept={grand_kept} '
          f'kept_ratio={grand_kept/max(grand_pre,1)*100:.2f}%')


# ---------------------- Score dump (expensive, run once) ----------------------

def dump_scores(pred_jsons_dir, frames_root, mlp_path, out_scores, device='cuda'):
    """Run CLIP+MLP on every detection box, save per-box scores to a JSON cache.
    Layout: {video_id: {frame_id_str: [score_float, ...] aligned with dets}}.
    This is the 2-hour CLIP step. Once cached, refilter is instant."""
    pred_dir = Path(pred_jsons_dir)
    frames_root = Path(frames_root)
    state = torch.load(mlp_path, map_location=device)
    model = TextFilterMLP(in_dim=state['in_dim']).to(device).eval()
    model.load_state_dict(state['mlp'])
    extractor = ClipFeatureExtractor(device=device)
    jsons = sorted(pred_dir.glob('*.json'))
    print(f'scoring {len(jsons)} videos -> {out_scores}')
    all_scores = {}
    for jp in jsons:
        video_id = jp.stem
        with open(jp) as f:
            preds = json.load(f)
        vid_scores = {}
        for fid_str, dets in preds.items():
            try:
                fid = int(fid_str)
            except ValueError:
                vid_scores[fid_str] = [1.0] * len(dets); continue
            frame_path = None
            for ext in ('.jpg', '.png', '.jpeg'):
                p = frames_root / video_id / f'{fid}{ext}'
                if p.exists():
                    frame_path = p; break
                p = frames_root / video_id / f'{str(fid).zfill(6)}{ext}'
                if p.exists():
                    frame_path = p; break
            if frame_path is None:
                vid_scores[fid_str] = [0.0] * len(dets); continue
            img = Image.open(frame_path).convert('RGB')
            scores = [0.0] * len(dets)
            crops, idxs = [], []
            for i, det in enumerate(dets):
                pts = det.get('points', [])
                if len(pts) < 6:
                    continue
                c = crop_polygon(img, pts)
                if c is None:
                    continue
                crops.append(c); idxs.append(i)
            if crops:
                with torch.no_grad():
                    feats = extractor.features(crops)
                    sc = torch.sigmoid(model(feats)).cpu().tolist()
                for li, di in enumerate(idxs):
                    scores[di] = sc[li]
            vid_scores[fid_str] = scores
        all_scores[video_id] = vid_scores
        print(f'  {video_id}: {sum(len(v) for v in vid_scores.values())} boxes scored')
    Path(out_scores).parent.mkdir(parents=True, exist_ok=True)
    with open(out_scores, 'w') as f:
        json.dump(all_scores, f)
    print(f'saved scores to {out_scores}')


def refilter_from_scores(pred_jsons_dir, scores_json, out_dir,
                         score_thresh=0.5, min_track_keep=3,
                         conf_short_track=0.65):
    """Instant re-filter using cached scores. No GPU."""
    pred_dir = Path(pred_jsons_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(scores_json) as f:
        all_scores = json.load(f)
    jsons = sorted(pred_dir.glob('*.json'))
    grand_pre = grand_kept = 0
    for jp in jsons:
        video_id = jp.stem
        with open(jp) as f:
            preds = json.load(f)
        vid_scores = all_scores.get(video_id, {})
        per_track_scores = defaultdict(list)
        for fid_str, dets in preds.items():
            sc = vid_scores.get(fid_str, [0.5] * len(dets))
            for i, det in enumerate(dets):
                tid = det.get('ID', det.get('id', -1))
                s = sc[i] if i < len(sc) else 0.5
                per_track_scores[tid].append(s)
        kept_tracks = set()
        for tid, scs in per_track_scores.items():
            mean_s = sum(scs) / len(scs)
            if mean_s >= score_thresh and len(scs) >= min_track_keep:
                kept_tracks.add(tid)
            elif mean_s >= conf_short_track:
                kept_tracks.add(tid)
        new_preds = {}
        n_pre = sum(len(v) for v in preds.values())
        for fid_str, dets in preds.items():
            new_preds[fid_str] = [d for d in dets
                                  if d.get('ID', d.get('id', -1)) in kept_tracks]
        n_post = sum(len(v) for v in new_preds.values())
        grand_pre += n_pre; grand_kept += n_post
        with open(out / jp.name, 'w') as f:
            json.dump(new_preds, f)
        print(f'  {video_id}: pre={n_pre} post={n_post} '
              f'tracks {len(per_track_scores)}->{len(kept_tracks)}')
    print(f'DONE kept_ratio={grand_kept/max(grand_pre,1)*100:.1f}% '
          f'(thresh={score_thresh} min_track={min_track_keep})')


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    s_sc = sub.add_parser('score')
    s_sc.add_argument('--pred-jsons', required=True)
    s_sc.add_argument('--frames-root', required=True)
    s_sc.add_argument('--mlp', required=True)
    s_sc.add_argument('--out-scores', required=True)

    s_rf = sub.add_parser('refilter')
    s_rf.add_argument('--pred-jsons', required=True)
    s_rf.add_argument('--scores', required=True)
    s_rf.add_argument('--out-dir', required=True)
    s_rf.add_argument('--score-thresh', type=float, default=0.5)
    s_rf.add_argument('--min-track-keep', type=int, default=3)
    s_rf.add_argument('--conf-short-track', type=float, default=0.65)

    s_tr = sub.add_parser('train')
    s_tr.add_argument('--gt-json', required=True, help='COCO-format train.json')
    s_tr.add_argument('--pred-jsons', required=True, help='Per-video detection JSON dir')
    s_tr.add_argument('--frames-root', required=True)
    s_tr.add_argument('--out-mlp', required=True)
    s_tr.add_argument('--limit-videos', type=int, default=0)

    s_ap = sub.add_parser('apply')
    s_ap.add_argument('--pred-jsons', required=True)
    s_ap.add_argument('--frames-root', required=True)
    s_ap.add_argument('--mlp', required=True)
    s_ap.add_argument('--out-dir', required=True)
    s_ap.add_argument('--score-thresh', type=float, default=0.5)
    s_ap.add_argument('--min-track-keep', type=int, default=3)

    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.cmd == 'train':
        gt = load_gt_per_frame(args.gt_json)
        crops, labels = build_training_data(args.pred_jsons, gt, args.frames_root,
                                             limit_videos=args.limit_videos)
        if len(crops) < 50:
            print(f'ERROR: only {len(crops)} samples, too few')
            return
        print(f'extracting CLIP features for {len(crops)} samples...')
        extractor = ClipFeatureExtractor(device=device)
        X = extract_features_batched(crops, extractor, batch_size=64)
        print(f'X shape: {X.shape}')
        del crops; del extractor
        torch.cuda.empty_cache()
        y = np.asarray(labels, dtype=np.float32)
        model, metrics = train_mlp(X, y, device=device)
        print(f'TRAIN_METRICS: {metrics}')
        Path(args.out_mlp).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'mlp': model.state_dict(), 'in_dim': X.shape[1],
                    'metrics': metrics}, args.out_mlp)
        print(f'saved MLP to {args.out_mlp}')

    elif args.cmd == 'apply':
        apply_filter(args.pred_jsons, args.frames_root, args.mlp,
                     args.out_dir, score_thresh=args.score_thresh,
                     min_track_keep=args.min_track_keep, device=device)

    elif args.cmd == 'score':
        dump_scores(args.pred_jsons, args.frames_root, args.mlp,
                    args.out_scores, device=device)

    elif args.cmd == 'refilter':
        refilter_from_scores(args.pred_jsons, args.scores, args.out_dir,
                             score_thresh=args.score_thresh,
                             min_track_keep=args.min_track_keep,
                             conf_short_track=args.conf_short_track)


if __name__ == '__main__':
    main()
