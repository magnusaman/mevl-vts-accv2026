"""DenseTrack Component 3: VLM content matching.

Adds recognized-text content similarity as ANOTHER term in the matcher.

Goal: when visually similar text instances exist (dense scenes), or when
visually different appearances correspond to the same text (motion blur,
rotation), the matcher needs content-awareness. We use a frozen Qwen-VL LoRA
to read the text per detection, maintain a per-track CONSENSUS text via
multi-frame majority voting, and add an edit-distance-based similarity term
to the matcher.

Distinction from LSTrack-SV (Hu et al. 2024):
    - LSTrack-SV uses per-frame CTC recognition (single shot) + raw edit
      distance + ByteTrack-style associator. Single-shot OCR per frame.
    - We use multi-frame VLM consensus (track-level text) + learned fuzzy
      similarity + inside a frozen DETR-query matcher.

Caching strategy:
    Qwen-VL is expensive (~200ms/crop). We cache transcriptions per
    (video, frame, proposal_id) offline once via `tools/cache_vlm_text.py`.
    At training time we just look up strings.

Trainable params:
    - content_weight: scalar in (0, 1) via sigmoid    (1 param)
    - sim_temperature: scalar                          (1 param)
    - Optional: small MLP that maps (edit_dist, len_diff, consensus_conf)
      to a [-1, 1] similarity score. ~5K params.
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def _ratio(a: str, b: str) -> float:
    """Normalized edit-distance similarity in [0, 1] (1.0 = identical)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        # rapidfuzz.fuzz.ratio returns a percentage in [0, 100]
        return _rf_fuzz.ratio(a, b) / 100.0
    # Pure-python Levenshtein fallback
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = cur
    return 1.0 - prev[n] / max(m, n)


class VLMContentMatcher(nn.Module):
    """Adds a content-similarity term to a base similarity matrix."""

    def __init__(self, use_mlp_fusion: bool = True):
        super().__init__()
        # Content weight in (0, 1) via sigmoid. Init at -6 -> 0.0025 (negligible).
        self.content_logit = nn.Parameter(torch.tensor(-6.0))
        self.sim_temperature = nn.Parameter(torch.tensor(2.0))

        if use_mlp_fusion:
            # Maps (edit_ratio, len_diff_norm, consensus_conf) -> sim in [-1, 1]
            self.fuse = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Tanh(),
            )
        else:
            self.fuse = None

    @staticmethod
    def consensus_text(votes: List[str]) -> tuple:
        """Multi-frame majority vote for track text. Returns (text, confidence)."""
        if not votes:
            return ("", 0.0)
        # Drop empty/None
        votes = [v for v in votes if v]
        if not votes:
            return ("", 0.0)
        from collections import Counter
        c = Counter(votes)
        most_common, count = c.most_common(1)[0]
        conf = count / len(votes)
        return most_common, conf

    def forward(self,
                base_sim: torch.Tensor,
                query_texts: List[str],
                key_track_consensus: List[tuple]) -> torch.Tensor:
        """Add content-similarity term to base_sim.

        Args:
            base_sim: (M, N) baseline similarity
            query_texts: list of M strings (current frame's transcriptions)
            key_track_consensus: list of N (consensus_text, confidence) tuples
                                  (one per track in the memory bank)

        Returns:
            (M, N) augmented similarity
        """
        content_w = torch.sigmoid(self.content_logit)
        # Skip computation if weight is essentially zero (inference shortcut)
        if not self.training and content_w.item() < 1e-6:
            return base_sim

        M, N = base_sim.shape
        device = base_sim.device
        if len(query_texts) != M or len(key_track_consensus) != N:
            # Mismatch — skip, return base
            return base_sim

        # Build (M, N) feature tensor of [edit_ratio, len_diff_norm, conf]
        feats = torch.zeros(M, N, 3, device=device)
        for i, q in enumerate(query_texts):
            for j, (k_text, k_conf) in enumerate(key_track_consensus):
                er = _ratio(q or "", k_text or "")
                ld = abs(len(q or "") - len(k_text or "")) / max(
                    len(q or "") + len(k_text or ""), 1)
                feats[i, j, 0] = er
                feats[i, j, 1] = ld
                feats[i, j, 2] = float(k_conf)

        if self.fuse is not None:
            content_sim = self.fuse(feats).squeeze(-1)  # (M, N) in [-1, 1]
        else:
            content_sim = feats[..., 0] * 2 - 1  # raw ratio mapped to [-1, 1]

        # Scale by content_weight and temperature
        return base_sim + content_w * self.sim_temperature * content_sim


def _self_test():
    """python -m gomatching.modeling.v7_densetrack.content_matcher"""
    print("[VLMContentMatcher] standalone self-test")
    cm = VLMContentMatcher(use_mlp_fusion=True)

    base = torch.randn(3, 4)
    query_texts = ["HELLO", "WORLD", "FOO"]
    consensus = [("HELLO", 0.9), ("WORLD", 0.7), ("BAR", 0.5), ("HELLO", 0.6)]

    out = cm(base, query_texts, consensus)
    assert out.shape == (3, 4)
    diff = (out - base).abs().max().item()
    print(f"  content_w=sigmoid(-6)={torch.sigmoid(cm.content_logit).item():.5f}")
    print(f"  max |out - base| = {diff:.4f}  (should be small at init)")
    assert diff < 0.1, f"Content term should barely move base at init (got {diff:.3f})"

    # Backward
    loss = out.sum()
    loss.backward()
    assert cm.content_logit.grad is not None
    print(f"  content_logit grad: {cm.content_logit.grad.item():.4f}")

    # Consensus voting test
    consensus_text, conf = cm.consensus_text(["abc", "abc", "abd", "abc"])
    assert consensus_text == "abc" and conf == 0.75, f"got {consensus_text!r}, {conf}"
    print(f"  consensus(['abc','abc','abd','abc']) = ({consensus_text!r}, {conf})")

    n_trainable = sum(p.numel() for p in cm.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")
    print("[VLMContentMatcher] self-test PASSED")


if __name__ == "__main__":
    _self_test()
