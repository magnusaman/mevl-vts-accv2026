"""Frozen foundation-model encoder loaders for DenseTrack components 2 and 3.

All encoders are loaded with `requires_grad=False` and `.eval()`. They are not
fine-tuned. Each loader returns:
    - the model (frozen)
    - the preprocess function (image -> tensor)
    - the output channel dim (for sizing downstream projections)

Encoders chosen for ORTHOGONALITY of pretraining signal:
    clip-l-336      : web image-text contrastive  (1024 d)
    dinov2-l        : self-sup, semantic+geometric (1024 d)
    sam-h           : segmentation prompt-tuned    (1280 d)
    convnext-l      : ImageNet supervised, strong stride-8 (1536 d)

For Component 3 (VLM content matching), we use the user's Qwen-VL LoRA
(trained separately) — handled in content_matcher.py, not here.

Caching: encoder feature maps are 32x32x{Ci} float16 per frame per encoder.
At 4 encoders × 32 × 32 × 1024 (avg) × 2 bytes = 8 MB / frame.
For IC15-V (~14k frames): ~110 GB total. So we cache per-VIDEO (not per-clip)
and load on-demand during training.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


SUPPORTED_ENCODERS = {
    # name -> (output_dim, expected_input_size)
    "clip-l-336":  (1024, 336),
    "dinov2-l":    (1024, 518),
    "sam-h":       (1280, 1024),
    "convnext-l":  (1536, 384),
}


def load_frozen_encoder(name: str, device: str = "cuda") -> Tuple[nn.Module, callable, int]:
    """Return (model, preprocess_fn, out_dim). All frozen, eval mode.

    preprocess_fn takes a (B, 3, H, W) tensor in [0, 255] uint8 or [0, 1] float
    and returns a (B, 3, H', W') tensor ready for the encoder.
    """
    if name not in SUPPORTED_ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Supported: {list(SUPPORTED_ENCODERS)}")
    out_dim, in_size = SUPPORTED_ENCODERS[name]

    if name == "clip-l-336":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-336", pretrained="openai")
        model = model.visual.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def pp(x):
            # x: (B, 3, H, W) in [0, 255] uint8
            x = x.float() / 255.0
            x = nn.functional.interpolate(x, size=(in_size, in_size),
                                          mode="bilinear", align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                               device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std
        return model, pp, out_dim

    elif name == "dinov2-l":
        # Loaded via torch.hub at first call; caches automatically.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def pp(x):
            x = x.float() / 255.0
            x = nn.functional.interpolate(x, size=(in_size, in_size),
                                          mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std
        return model, pp, out_dim

    elif name == "sam-h":
        # Use HF transformers SAM
        from transformers import SamModel
        sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        # We only need the vision encoder for feature extraction
        vision_encoder = sam.vision_encoder

        def pp(x):
            x = x.float() / 255.0
            x = nn.functional.interpolate(x, size=(in_size, in_size),
                                          mode="bilinear", align_corners=False)
            mean = torch.tensor([123.675, 116.28, 103.53],
                                device=x.device).view(1, 3, 1, 1) / 255.0
            std = torch.tensor([58.395, 57.12, 57.375],
                               device=x.device).view(1, 3, 1, 1) / 255.0
            return (x - mean) / std

        # Wrap the vision encoder so .forward returns (B, C, H', W') feature map
        class SAMVisionWrapper(nn.Module):
            def __init__(self, ve):
                super().__init__()
                self.ve = ve
            def forward(self, x):
                # SAM vision encoder returns (B, H'*W', C); reshape to (B, C, H', W')
                out = self.ve(x)
                feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                B, N, C = feats.shape
                H = W = int(N ** 0.5)
                return feats.transpose(1, 2).reshape(B, C, H, W)

        return SAMVisionWrapper(vision_encoder).to(device).eval(), pp, out_dim

    elif name == "convnext-l":
        import timm
        model = timm.create_model("convnext_large.fb_in22k_ft_in1k_384",
                                  pretrained=True, num_classes=0,
                                  global_pool="").to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def pp(x):
            x = x.float() / 255.0
            x = nn.functional.interpolate(x, size=(in_size, in_size),
                                          mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std

        # timm model.forward_features returns (B, C, H', W')
        class ConvNextWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m.forward_features(x)

        return ConvNextWrapper(model).to(device).eval(), pp, out_dim

    else:
        raise NotImplementedError(name)


class EncoderFeatureCache:
    """Loads pre-computed encoder feature maps from HDF5 files.

    Cache layout:
        <root>/<encoder_name>/<video_id>.h5
        each h5 has dataset 'features' of shape (num_frames, C, H, W) float16
        and 'frame_ids' of shape (num_frames,) int32

    Use case at training: dataloader gives a clip of (video_id, [frame_ids]).
    Cache returns a tensor of shape (num_encoders, T, C_max, H, W) by stacking
    each encoder's features and zero-padding to the largest C dim.
    """
    def __init__(self, root: str, encoders: list, device: str = "cuda"):
        self.root = root
        self.encoders = encoders  # list of encoder names in canonical order
        self.device = device
        self._handles: Dict[str, Dict[str, "h5py.File"]] = {}

    def get_clip_features(self, video_id: str, frame_ids: list) -> Dict[str, torch.Tensor]:
        """Return dict of {encoder_name: (T, C, H, W) tensor on device}.

        Note: frames are indexed by raw frame_id (1-indexed for IC15-V).
        Missing frames are zero-filled.
        """
        import h5py
        import os
        out = {}
        for enc in self.encoders:
            h5_path = os.path.join(self.root, enc, f"{video_id}.h5")
            if not os.path.exists(h5_path):
                # Missing cache for this video — return zeros (caller should warn)
                C = SUPPORTED_ENCODERS[enc][0]
                out[enc] = torch.zeros(len(frame_ids), C, 32, 32, device=self.device)
                continue
            with h5py.File(h5_path, "r") as f:
                feats = f["features"]
                fids = f["frame_ids"][:]
                C = feats.shape[1]; H = feats.shape[2]; W = feats.shape[3]
                clip_feats = torch.zeros(len(frame_ids), C, H, W,
                                          dtype=torch.float16, device="cpu")
                fid_to_idx = {int(fid): i for i, fid in enumerate(fids)}
                for k, fid in enumerate(frame_ids):
                    if int(fid) in fid_to_idx:
                        clip_feats[k] = torch.from_numpy(feats[fid_to_idx[int(fid)]])
                out[enc] = clip_feats.to(self.device, dtype=torch.float32, non_blocking=True)
        return out
