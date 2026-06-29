"""Per-process lazy cache loader for v7 DenseTrack.

The GoMatching dataloader spawns N worker processes. Each worker opens its
own HDF5 / NPZ file handles on first access — file handles don't pickle.

Used by the patched GoMDatasetMapper to attach per-frame v7 data to each
dataset_dict it returns. The meta-arch (gom_lstmatcher.forward) then
aggregates per-clip via _v7_set_roi_ctx.

Two cache formats are supported (per-instance preferred; spatial is fallback):

  A) PER-INSTANCE (preferred; built by tools/build_per_instance_cache.py)
     <encoder_root_per_inst>/<video_id>.npz
        frame_ids       : (F,) int32
        box_offsets     : (F+1,) int32
        gt_boxes_xyxy   : (sum_M, 4) float32
        gt_track_ids    : (sum_M,) int32
        image_sizes     : (F, 2) int32
        <encoder_name>  : (sum_M, C_e) float16

  B) SPATIAL MAPS (legacy; built by tools/cache_encoder_features.py)
     <encoder_root>/<encoder_name>/<video_id>.h5
        features:       (N, C, 32, 32) fp16
        frame_indices:  (N,) int32     1-indexed GT frame_id

  SAM proposals (independent of encoder cache):
     <sam_root>/<video_id>.npz
        frame_indices    : (F,) int32
        n_masks_per_frame: (F,) int32
        polygons_flat    : object[]   flat polygon lists per mask
        boxes            : (sum_n, 4) float32 xyxy
        scores           : (sum_n,) float32   SAM stability
        clip_features    : (sum_n, 1024) float16  CLIP pooled inside mask
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def _video_id_from_path(file_name: str) -> str:
    """Extract 'Video_10_1_1' from '/.../Video_10_1_1/000000.jpg'."""
    return Path(file_name).parent.name


def _frame_id_from_path(file_name: str) -> int:
    """Filename '000000.jpg' (0-indexed on disk) corresponds to GT frame_id=1."""
    stem = Path(file_name).stem
    return int(stem) + 1


class V7Cache:
    """Lazy per-worker cache. Open on first access, keep handles for reuse."""

    def __init__(self,
                 encoder_cache_root: str,
                 sam_cache_root: str,
                 encoder_names: List[str],
                 enable_encoder_cache: bool = True,
                 enable_sam_cache: bool = True,
                 per_instance_cache_root: Optional[str] = None):
        """
        Args:
            per_instance_cache_root: if set, prefer the per-instance NPZ cache
                at this path. The legacy spatial-map cache is then used only as
                a fallback when a per-instance file is missing.
        """
        self.encoder_cache_root = encoder_cache_root
        self.sam_cache_root = sam_cache_root
        self.encoder_names = encoder_names
        self.enable_encoder_cache = enable_encoder_cache
        self.enable_sam_cache = enable_sam_cache
        self.per_instance_cache_root = per_instance_cache_root

        self._h5: Dict[tuple, "h5py.File"] = {}
        self._sam_data: Dict[str, dict] = {}  # video_id -> preloaded arrays
        self._per_inst: Dict[str, dict] = {}  # video_id -> preloaded npz arrays

        # cache miss reporting (per worker, just-in-time)
        self._reported_miss = set()

    # ----- per-instance cache (NEW, preferred) -----
    def _per_inst_for(self, video_id: str) -> Optional[dict]:
        if not self.per_instance_cache_root:
            return None
        if video_id in self._per_inst:
            return self._per_inst[video_id]
        path = os.path.join(self.per_instance_cache_root, f"{video_id}.npz")
        if not os.path.exists(path):
            key = ("per_inst", video_id)
            if key not in self._reported_miss:
                print(f"[v7-cache MISS per-instance] {path}")
                self._reported_miss.add(key)
            self._per_inst[video_id] = None
            return None
        npz = np.load(path, allow_pickle=False)
        loaded = {k: npz[k] for k in npz.files}
        self._per_inst[video_id] = loaded
        return loaded

    def get_per_instance_features(self, video_id: str, frame_id: int) -> Optional[Dict]:
        """Return per-instance pooled features for one frame, or None if absent.

        Output dict:
            {"gt_boxes":    (M, 4) float32 tensor (image coords),
             "gt_track_ids":(M,)    int32   tensor,
             "image_size":  (H, W),
             "features":    {enc_name: (M, C_e) float32 tensor}}
        """
        data = self._per_inst_for(video_id)
        if data is None:
            return None
        idxs = np.where(data["frame_ids"] == frame_id)[0]
        if len(idxs) == 0:
            return None
        i = int(idxs[0])
        start = int(data["box_offsets"][i])
        end = int(data["box_offsets"][i + 1])
        if end == start:
            return {"gt_boxes": torch.zeros(0, 4),
                    "gt_track_ids": torch.zeros(0, dtype=torch.int32),
                    "image_size": tuple(int(x) for x in data["image_sizes"][i]),
                    "features": {}}
        feats = {}
        for enc in self.encoder_names:
            if enc not in data:
                continue
            arr = data[enc][start:end]
            feats[enc] = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return {
            "gt_boxes": torch.from_numpy(
                np.asarray(data["gt_boxes_xyxy"][start:end], dtype=np.float32)),
            "gt_track_ids": torch.from_numpy(
                np.asarray(data["gt_track_ids"][start:end], dtype=np.int32)),
            "image_size": tuple(int(x) for x in data["image_sizes"][i]),
            "features": feats,
        }

    # ----- encoder features -----
    def _h5_for(self, enc: str, video_id: str):
        if not self.enable_encoder_cache:
            return None
        key = (enc, video_id)
        if key in self._h5:
            return self._h5[key]
        path = os.path.join(self.encoder_cache_root, enc, f"{video_id}.h5")
        if not os.path.exists(path):
            if key not in self._reported_miss:
                print(f"[v7-cache MISS] {path}")
                self._reported_miss.add(key)
            self._h5[key] = None
            return None
        import h5py
        self._h5[key] = h5py.File(path, "r")
        return self._h5[key]

    def get_encoder_features(self, video_id: str, frame_id: int) -> Dict[str, torch.Tensor]:
        """Return {encoder_name: (C, 32, 32) float32 tensor} for one frame.
        Missing encoders or frames yield zero tensors (downstream-safe)."""
        if not self.enable_encoder_cache:
            return {}
        out = {}
        for enc in self.encoder_names:
            f = self._h5_for(enc, video_id)
            if f is None:
                continue
            # find index for this frame_id
            fids = f["frame_indices"][:]
            idx = np.where(fids == frame_id)[0]
            if len(idx) == 0:
                continue
            feats = f["features"][idx[0]]  # (C, 32, 32) fp16
            out[enc] = torch.from_numpy(np.asarray(feats, dtype=np.float32))
        return out

    # ----- SAM proposals -----
    def _sam_for(self, video_id: str) -> Optional[dict]:
        if not self.enable_sam_cache:
            return None
        if video_id in self._sam_data:
            return self._sam_data[video_id]
        path = os.path.join(self.sam_cache_root, f"{video_id}.npz")
        if not os.path.exists(path):
            key = ("sam", video_id)
            if key not in self._reported_miss:
                print(f"[v7-cache MISS] {path}")
                self._reported_miss.add(key)
            self._sam_data[video_id] = None
            return None
        npz = np.load(path, allow_pickle=True)
        # Pre-extract small index arrays
        self._sam_data[video_id] = {
            "frame_indices": npz["frame_indices"][:],
            "n_masks_per_frame": npz["n_masks_per_frame"][:],
            "polygons_flat": npz["polygons_flat"],  # keep object array as-is
            "boxes": npz["boxes"],
            "scores": npz["scores"],
            "clip_features": npz["clip_features"],
        }
        return self._sam_data[video_id]

    def get_sam_proposals(self, video_id: str, frame_id: int) -> Dict[str, object]:
        """Return per-frame SAM data.

        Output dict has keys (canonical names used by gom_lstmatcher):
            sam_polygons        : list[list[float]]  (flat polygons)
            sam_clip_features   : (M, 1024) float32 tensor
            sam_stability_scores: (M,) float32 tensor
        Empty lists/tensors if no entry for this frame.
        """
        if not self.enable_sam_cache:
            return {"sam_polygons": [], "sam_clip_features": torch.zeros(0, 1024),
                    "sam_stability_scores": torch.zeros(0)}
        data = self._sam_for(video_id)
        if data is None:
            return {"sam_polygons": [], "sam_clip_features": torch.zeros(0, 1024),
                    "sam_stability_scores": torch.zeros(0)}
        idx = np.where(data["frame_indices"] == frame_id)[0]
        if len(idx) == 0:
            return {"sam_polygons": [], "sam_clip_features": torch.zeros(0, 1024),
                    "sam_stability_scores": torch.zeros(0)}
        i = int(idx[0])
        n_before = int(data["n_masks_per_frame"][:i].sum())
        n_here = int(data["n_masks_per_frame"][i])
        end = n_before + n_here
        polys = data["polygons_flat"][n_before:end].tolist()
        clip_feats = torch.from_numpy(
            np.asarray(data["clip_features"][n_before:end], dtype=np.float32))
        stab = torch.from_numpy(
            np.asarray(data["scores"][n_before:end], dtype=np.float32))
        return {"sam_polygons": polys,
                "sam_clip_features": clip_feats,
                "sam_stability_scores": stab}

    # ----- attach to a dataset_dict (called by mapper) -----
    def attach(self, dataset_dict: dict):
        """Mutate dataset_dict to add v7_* fields based on file_name.

        Per-instance cache (preferred) is attached as v7_per_instance.
        Spatial-map cache is attached as v7_encoder_features (legacy fallback).
        """
        fn = dataset_dict.get("file_name", "")
        if not fn:
            return
        video_id = _video_id_from_path(fn)
        frame_id = _frame_id_from_path(fn)
        per_inst = self.get_per_instance_features(video_id, frame_id)
        if per_inst is not None:
            dataset_dict["v7_per_instance"] = per_inst
        else:
            enc_feats = self.get_encoder_features(video_id, frame_id)
            if enc_feats:
                dataset_dict["v7_encoder_features"] = enc_feats
        sam = self.get_sam_proposals(video_id, frame_id)
        dataset_dict["v7_sam_polygons"] = sam["sam_polygons"]
        dataset_dict["v7_sam_clip_features"] = sam["sam_clip_features"]
        dataset_dict["v7_sam_stability_scores"] = sam["sam_stability_scores"]


def build_v7_cache_from_cfg(cfg) -> Optional[V7Cache]:
    """Construct a V7Cache from cfg, or None if v7 not enabled."""
    dt = getattr(cfg.MODEL, "DENSETRACK", None)
    if dt is None or not dt.ENABLED:
        return None
    enable_enc = dt.COMP2_ENABLED and bool(dt.COMP2_CACHE_ROOT)
    enable_sam = dt.COMP1_ENABLED and bool(dt.COMP1_CACHE_ROOT)
    per_inst_root = getattr(dt, "COMP2_PER_INSTANCE_CACHE_ROOT", "") or None
    if not (enable_enc or enable_sam or per_inst_root):
        return None
    return V7Cache(
        encoder_cache_root=dt.COMP2_CACHE_ROOT,
        sam_cache_root=dt.COMP1_CACHE_ROOT,
        encoder_names=list(dt.COMP2_ENCODERS),
        enable_encoder_cache=enable_enc,
        enable_sam_cache=enable_sam,
        per_instance_cache_root=per_inst_root,
    )
