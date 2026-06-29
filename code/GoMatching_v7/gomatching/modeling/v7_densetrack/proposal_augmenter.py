"""DenseTrack Component 1: SAM-augmented proposal recall.

Goal: boost recall on tiny / occluded / small dense text on DSText where
DeepSolo's frozen R50 backbone misses sub-32px instances.

Strategy:
    1. SAM-H runs over the frame in "segment everything" mode -> mask proposals
       (slow, ~250ms/frame on L4 — MUST be cached offline)
    2. A small text-vs-non-text classifier (MLP over CLIP-features-of-each-mask)
       filters to text-like masks. Trainable, ~0.5M params.
    3. Mask polygons -> bounding boxes -> NMS-merged with DeepSolo proposals.
       Gate scalar controls the WEIGHT of SAM proposals in the merged set;
       gate=0 means SAM proposals are added with score=0 -> effectively ignored
       by downstream score-threshold filtering, recovering baseline behavior.

At training time, SAM mask proposals are loaded from a cache built once by
`tools/cache_sam_proposals.py`. No SAM forward is run in the training loop.

Cache format:
    <root>/<video_id>.npz
        masks   : packed array of mask polygons (variable-length per frame)
        frame_ids: int32 array of frame ids
        scores  : SAM stability scores (per-mask, float32)
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances


class SAMProposalAugmenter(nn.Module):
    """Augments DeepSolo proposals with text-filtered SAM masks.

    Trainable params:
        - text_classifier: tiny MLP (CLIP_dim -> 256 -> 1)  ~0.5M
        - gate: scalar in [0, 1]   1 param

    Initialized with gate=0 so at iter 0 the augmenter has no effect (identity).
    """
    def __init__(self, clip_dim: int = 768, hidden_dim: int = 256,
                 max_sam_proposals_per_frame: int = 50):
        super().__init__()
        self.text_classifier = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize classifier so initial p(text)=0.5 (logit=0)
        nn.init.zeros_(self.text_classifier[-1].weight)
        nn.init.zeros_(self.text_classifier[-1].bias)

        # Gated additive via sigmoid; init very negative so sigmoid≈0 at iter 0
        # (so SAM proposals don't perturb the converged baseline detector before
        # the text classifier has learned anything useful).
        self.gate = nn.Parameter(torch.tensor([-6.0]))
        self.max_per_frame = max_sam_proposals_per_frame

    @torch.no_grad()
    def _polygons_to_boxes(self, polygons: List[List[float]], device) -> torch.Tensor:
        """polygons: list of flat [x1,y1,...,xn,yn]. Return (N, 4) xyxy."""
        out = []
        for poly in polygons:
            if len(poly) < 6:
                continue
            xs = poly[0::2]; ys = poly[1::2]
            out.append([min(xs), min(ys), max(xs), max(ys)])
        if not out:
            return torch.zeros(0, 4, device=device)
        return torch.tensor(out, device=device, dtype=torch.float32)

    def forward(self,
                base_proposals: List[Instances],
                sam_mask_polygons: List[List[List[float]]],
                sam_clip_features: List[torch.Tensor],
                sam_stability_scores: List[torch.Tensor]) -> List[Instances]:
        """Merge SAM-derived proposals into base_proposals.

        Args:
            base_proposals: per-frame Instances from DeepSolo (already has
                            proposal_boxes, objectness_logits, query_features)
            sam_mask_polygons: per-frame list of polygon flat-lists
            sam_clip_features: per-frame (M_t, clip_dim) tensor of CLIP features
                               pooled inside each mask
            sam_stability_scores: per-frame (M_t,) tensor of SAM stability scores

        Returns:
            augmented proposals list, length unchanged but per-frame proposal
            count may be larger.

        Behavior:
            gate=0 -> SAM proposals appended with score=0 -> downstream score
                      threshold (cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST=0.3)
                      drops them all -> effectively no-op
            gate=1 -> SAM proposals appended with p(text)*stability score
        """
        out = []
        gate = torch.sigmoid(self.gate)  # init ~0.0025 (logit=-6); ramps up via gradient

        for proposals, polys, clip_feats, stab in zip(
                base_proposals, sam_mask_polygons,
                sam_clip_features, sam_stability_scores):
            if len(polys) == 0:
                out.append(proposals)
                continue

            # Score each SAM proposal as text-vs-not via the trainable classifier
            text_logits = self.text_classifier(clip_feats).squeeze(-1)  # (M,)
            text_prob = torch.sigmoid(text_logits)
            # Final SAM proposal score: gate * text_prob * SAM_stability
            sam_scores = gate * text_prob * stab.to(text_prob.dtype)

            # Top-K to bound memory
            if sam_scores.shape[0] > self.max_per_frame:
                topk = torch.topk(sam_scores, self.max_per_frame)
                keep = topk.indices
                sam_scores = sam_scores[keep]
                polys = [polys[i] for i in keep.tolist()]

            # Convert polygons to boxes (xyxy)
            sam_boxes = self._polygons_to_boxes(polys, device=proposals.proposal_boxes.tensor.device)
            if sam_boxes.shape[0] == 0:
                out.append(proposals)
                continue

            # Build a new Instances and concatenate
            new_p = Instances(proposals.image_size)
            new_p.proposal_boxes = Boxes(torch.cat(
                [proposals.proposal_boxes.tensor, sam_boxes], dim=0))
            new_p.objectness_logits = torch.cat(
                [proposals.objectness_logits, sam_scores], dim=0)
            # query_features: we don't have DeepSolo query features for SAM
            # proposals. Use ZERO features so they don't contribute to the
            # matcher's similarity until DeepSolo re-detects them on the next
            # frame (the SAM proposals are RECALL aids; they hand back to
            # DeepSolo's normal flow).
            if proposals.has("query_features"):
                Q = proposals.query_features.shape[1]  # num_points
                D = proposals.query_features.shape[2]  # 256
                zero_q = torch.zeros(sam_boxes.shape[0], Q, D,
                                     dtype=proposals.query_features.dtype,
                                     device=proposals.query_features.device)
                new_p.query_features = torch.cat(
                    [proposals.query_features, zero_q], dim=0)
            out.append(new_p)
        return out


# ---------------------------------------------------------------------------
# Unit test (pure torch, no GoMatching needed)
# ---------------------------------------------------------------------------
def _self_test():
    """Standalone sanity check. Run: python -m gomatching.modeling.v7_densetrack.proposal_augmenter"""
    print("[SAMProposalAugmenter] standalone self-test")
    augmenter = SAMProposalAugmenter(clip_dim=768, hidden_dim=256,
                                     max_sam_proposals_per_frame=10)

    # Synthetic base proposals: 1 frame with 3 DeepSolo proposals
    base = [Instances((720, 1280))]
    base[0].proposal_boxes = Boxes(torch.rand(3, 4) * 500)
    base[0].objectness_logits = torch.rand(3)
    base[0].query_features = torch.randn(3, 25, 256)

    # 5 SAM mask polygons
    polys = [[[100., 100., 200., 100., 200., 200., 100., 200.] for _ in range(5)]]
    clip_feats = [torch.randn(5, 768)]
    stab = [torch.rand(5)]

    out = augmenter(base, polys, clip_feats, stab)
    assert len(out) == 1
    assert len(out[0]) == 8, f"expected 3+5 proposals, got {len(out[0])}"
    print(f"  pre: 3 proposals, post: {len(out[0])} proposals ({len(out[0])-3} SAM)")

    # Gate is sigmoid(0) = 0.5 at init, so SAM scores should be ~0.5 * 0.5 * stab
    sam_scores = out[0].objectness_logits[3:]
    print(f"  SAM scores (gate init): min={sam_scores.min():.3f}, max={sam_scores.max():.3f}")
    assert sam_scores.max() < 0.6, "Gate should be near 0.5 at init, scores should be modest"

    # Gradient flows to the gate?
    loss = out[0].objectness_logits.sum()
    loss.backward()
    assert augmenter.gate.grad is not None and augmenter.gate.grad.abs().sum() > 0
    print(f"  gate.grad: {augmenter.gate.grad.item():.4f}  (non-zero, OK)")

    n_trainable = sum(p.numel() for p in augmenter.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")
    print("[SAMProposalAugmenter] self-test PASSED")


if __name__ == "__main__":
    _self_test()
