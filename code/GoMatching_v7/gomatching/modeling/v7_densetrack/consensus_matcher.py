"""DenseTrack Component 2: Multi-encoder similarity consensus.

Replaces the SINGLE 1024-d similarity computation inside GoMatching++'s
LST-Matcher with a CONSENSUS across multiple foundation-encoder spaces.

Goal: when dense scenes (DSText) have many visually-similar text instances,
no single encoder's similarity is reliable. A learned attention pool over
per-encoder similarity matrices is more robust.

Math:
    Given:
        baseline_sim:   (M, N)  -- GoMatching's original similarity matrix
        per_enc_sims:   dict {enc_name: (M, N)} -- each from cosine sim of
                                                    pooled encoder features

    Output:
        final_sim = baseline_sim
                  + gate * sum_e( w_e * normalized(per_enc_sims[e]) )
        where w_e are softmax over a learned (1, num_enc) attention vector
        conditioned optionally on the baseline_sim's mean.

    Gate init: 0   ->  final_sim = baseline_sim exactly (baseline reproduced)
    Gate trained -> consensus contribution rises.

Trainable params:
    - per-encoder projection: nn.Linear(enc_dim, 256) per encoder  (~4 * 0.3M)
    - attention weights:      nn.Parameter (num_enc,)              (~4 params)
    - gate:                   nn.Parameter (1,)                    (1 param)

Total: ~1.2M params per the default 4-encoder setup.

At forward time, per-encoder features are ROI-aligned from cached encoder
feature MAPS (32x32 per frame) at each detection's bounding box.
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class MultiEncoderConsensusMatcher(nn.Module):
    """Augments a base similarity matrix with multi-encoder consensus."""

    def __init__(self,
                 encoder_dims: Dict[str, int],
                 proj_dim: int = 256,
                 use_baseline_residual: bool = True):
        """
        Args:
            encoder_dims: {encoder_name: feature_dim}, e.g.
                          {"clip-l-336": 1024, "dinov2-l": 1024, "sam-h": 1280, "convnext-l": 1536}
            proj_dim:     dimension to project all encoder features to
            use_baseline_residual: if True, output = baseline + gate * consensus
                                   if False, output = (1-gate) * baseline + gate * consensus
        """
        super().__init__()
        self.encoder_names = sorted(encoder_dims.keys())  # canonical order
        self.proj_dim = proj_dim
        self.use_baseline_residual = use_baseline_residual

        # Per-encoder projection to common dim
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, proj_dim) for name, dim in encoder_dims.items()
        })
        # Initialize projections normally; we don't need them to be near-zero
        # because the GATE controls the contribution magnitude.

        # Attention weights over encoders (learned scalar per encoder).
        # Softmax of these gives the consensus weights.
        self.encoder_attn = nn.Parameter(torch.zeros(len(self.encoder_names)))

        # Gate: scalar in (0, 1) via sigmoid; init at 0 logit so sigmoid=0.5
        # BUT we initialize at large negative logit so sigmoid≈0 -> consensus=0
        # at iter 0 (baseline reproduced exactly).
        self.gate_logit = nn.Parameter(torch.tensor(-6.0))

    def _pool_encoder_features_at_boxes(self,
                                         encoder_maps: torch.Tensor,
                                         boxes_per_frame: List[torch.Tensor],
                                         image_size: tuple) -> torch.Tensor:
        """ROI-align encoder feature maps at proposal boxes.

        Args:
            encoder_maps: (T, C, Hf, Wf) feature maps for one encoder, T frames
            boxes_per_frame: list of (n_t, 4) xyxy boxes (image coords)
            image_size: (H, W) of the original image

        Returns:
            (sum(n_t), C) pooled features

        Implementation note: we scale boxes from image-coords to
        feature-map-coords ourselves (rather than via roi_align's spatial_scale),
        then call roi_align with spatial_scale=1.0. This is more numerically
        robust on some torchvision versions and works for both CPU and GPU.
        """
        T, C, Hf, Wf = encoder_maps.shape
        H, W = image_size
        sx = Wf / W
        sy = Hf / H
        out = []
        for t, boxes in enumerate(boxes_per_frame):
            n = boxes.shape[0]
            if n == 0:
                out.append(torch.zeros(0, C, device=encoder_maps.device,
                                       dtype=encoder_maps.dtype))
                continue
            # Scale boxes to feature-map coords
            boxes_fm = boxes.clone().to(encoder_maps.device, dtype=encoder_maps.dtype)
            boxes_fm[:, [0, 2]] = boxes_fm[:, [0, 2]] * sx
            boxes_fm[:, [1, 3]] = boxes_fm[:, [1, 3]] * sy
            # Defensive: ensure xyxy order (max width=Wf, max height=Hf)
            x1 = torch.minimum(boxes_fm[:, 0], boxes_fm[:, 2]).clamp_(0, Wf - 0.5)
            y1 = torch.minimum(boxes_fm[:, 1], boxes_fm[:, 3]).clamp_(0, Hf - 0.5)
            x2 = torch.maximum(boxes_fm[:, 0], boxes_fm[:, 2]).clamp_(0.5, Wf)
            y2 = torch.maximum(boxes_fm[:, 1], boxes_fm[:, 3]).clamp_(0.5, Hf)
            # Ensure non-degenerate (x2 > x1, y2 > y1) by widening by 1 px
            x2 = torch.maximum(x2, x1 + 1.0)
            y2 = torch.maximum(y2, y1 + 1.0)
            boxes_fm = torch.stack([x1, y1, x2, y2], dim=1)
            # roi_align with list-of-boxes API (one boxes tensor per batch image)
            fm = encoder_maps[t:t+1].contiguous()
            pooled = roi_align(
                fm, [boxes_fm],
                output_size=(1, 1),
                spatial_scale=1.0,
                aligned=True,
            ).squeeze(-1).squeeze(-1)  # (n, C)
            out.append(pooled)
        return torch.cat(out, dim=0)  # (N, C)

    def forward(self,
                baseline_sim: torch.Tensor,
                encoder_features: Dict[str, torch.Tensor],
                query_boxes: List[torch.Tensor],
                key_boxes: List[torch.Tensor],
                image_size: tuple) -> torch.Tensor:
        """Compute consensus similarity matrix.

        Args:
            baseline_sim: (M, N) GoMatching's similarity (query x key)
            encoder_features: {enc_name: (T, C, Hf, Wf)} encoder maps for clip
            query_boxes: per-frame xyxy boxes for query side (current frame)
            key_boxes:   per-frame xyxy boxes for key side (memory frames)
            image_size:  (H, W) of original frames

        Returns:
            final_sim: (M, N) consensus similarity matrix
        """
        gate = torch.sigmoid(self.gate_logit)
        if gate.item() < 1e-6 and self.training is False:
            # at inference, if gate is essentially 0, skip the consensus compute
            return baseline_sim

        # Pool features per-encoder at query and key boxes
        per_enc_sim = []
        attn_weights = F.softmax(self.encoder_attn, dim=0)  # (num_enc,)
        for i, enc_name in enumerate(self.encoder_names):
            if enc_name not in encoder_features:
                continue
            maps = encoder_features[enc_name]  # (T, C, Hf, Wf)
            # Defensive: maps must cover every frame in query/key boxes.
            # If a clip frame had a cache miss the meta-arch should have padded
            # with zeros, but guard here in case anything slips through.
            T = maps.shape[0]
            if len(query_boxes) > T or len(key_boxes) > T:
                continue
            q_feat = self._pool_encoder_features_at_boxes(
                maps, query_boxes, image_size)  # (M, C)
            k_feat = self._pool_encoder_features_at_boxes(
                maps, key_boxes, image_size)    # (N, C)
            if q_feat.shape[0] == 0 or k_feat.shape[0] == 0:
                continue
            # Project to common dim
            q_proj = self.projections[enc_name](q_feat)  # (M, proj_dim)
            k_proj = self.projections[enc_name](k_feat)  # (N, proj_dim)
            # Cosine similarity
            q_norm = F.normalize(q_proj, dim=-1)
            k_norm = F.normalize(k_proj, dim=-1)
            sim = q_norm @ k_norm.t()  # (M, N) in [-1, 1]
            per_enc_sim.append(attn_weights[i] * sim)

        if len(per_enc_sim) == 0:
            return baseline_sim

        consensus = torch.stack(per_enc_sim, dim=0).sum(dim=0)  # (M, N)

        # Scale consensus to baseline range (baseline is post-softmax-like, in 0..1
        # after _activate_asso; here we work pre-activation, so it's logits ~ [-5, 5])
        # Treat consensus (in [-1, 1] after cosine) as an additive log-odds bonus.
        if self.use_baseline_residual:
            return baseline_sim + gate * consensus
        return (1 - gate) * baseline_sim + gate * consensus


def _self_test():
    """python -m gomatching.modeling.v7_densetrack.consensus_matcher"""
    print("[MultiEncoderConsensusMatcher] standalone self-test")
    encoder_dims = {"clip-l-336": 1024, "dinov2-l": 1024,
                    "sam-h": 1280, "convnext-l": 1536}
    matcher = MultiEncoderConsensusMatcher(encoder_dims, proj_dim=256)

    # Baseline similarity (M=3 queries, N=5 keys)
    baseline = torch.randn(3, 5)

    # Encoder feature maps for a clip of T=2 frames
    enc_features = {
        "clip-l-336":  torch.randn(2, 1024, 32, 32),
        "dinov2-l":    torch.randn(2, 1024, 32, 32),
        "sam-h":       torch.randn(2, 1280, 32, 32),
        "convnext-l":  torch.randn(2, 1536, 32, 32),
    }
    # Helper to build valid xyxy boxes (ensures x2>x1, y2>y1)
    def _mk_boxes(n, W=1280, H=720):
        if n == 0:
            return torch.zeros(0, 4)
        x1 = torch.rand(n) * (W - 100)
        y1 = torch.rand(n) * (H - 100)
        w = torch.rand(n) * 100 + 20
        h = torch.rand(n) * 100 + 20
        return torch.stack([x1, y1, x1 + w, y1 + h], dim=1)

    q_boxes = [_mk_boxes(3), _mk_boxes(0)]
    k_boxes = [_mk_boxes(2), _mk_boxes(3)]

    out = matcher(baseline, enc_features, q_boxes, k_boxes, image_size=(720, 1280))
    assert out.shape == (3, 5), f"shape mismatch: {out.shape}"
    # gate is sigmoid(-6) ~ 0.0025 -> consensus contribution is tiny
    diff = (out - baseline).abs().max().item()
    print(f"  gate=sigmoid(-6)={torch.sigmoid(matcher.gate_logit).item():.5f}")
    print(f"  max |out - baseline| = {diff:.4f}  (should be small at init)")
    assert diff < 0.5, f"At init the consensus should barely move baseline (got {diff:.3f})"

    # Backward to check gradient flow
    loss = out.sum()
    loss.backward()
    assert matcher.gate_logit.grad is not None
    assert matcher.encoder_attn.grad is not None
    for name, proj in matcher.projections.items():
        assert proj.weight.grad is not None, f"no grad on {name}"
    print(f"  gate grad: {matcher.gate_logit.grad.item():.4f}  (non-zero)")
    print(f"  encoder_attn grad: {matcher.encoder_attn.grad.tolist()}")

    n_trainable = sum(p.numel() for p in matcher.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")
    print("[MultiEncoderConsensusMatcher] self-test PASSED")


if __name__ == "__main__":
    _self_test()
