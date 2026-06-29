"""TIM-VTS missed-detection recovery verifier.

This module is intentionally small: it scores a candidate recovered text
instance from temporal memory features. It can be trained standalone from the
CSV produced by `scripts/tim_recovery_dataset.py` +
`scripts/tim_add_doctr_ocr_features.py`, then later plugged into GoMatching's
association/recovery path.
"""

from __future__ import annotations

import torch
from torch import nn


TIM_RECOVERY_FEATURES = [
    "gap_len",
    "step",
    "track_len",
    "text_len",
    "text_stability",
    "median_w",
    "median_h",
    "median_area",
    "candidate_area",
    "area_ratio",
    "width_ratio",
    "dist_per_frame",
    "scale_norm_dist",
    "feat_sem_cos",
    "feat_spa_cos",
    "feat_tex_cos",
    "crop_valid",
    "crop_sharpness",
    "crop_contrast",
    "crop_brightness",
    "crop_edge_density",
    "crop_area_px",
    "ocr_conf",
    "ocr_text_sim",
    "ocr_exact",
    "ocr_len",
    "frame_pred_count",
]


class TIMRecoveryVerifier(nn.Module):
    """MLP verifier for memory-proposed recovered detections."""

    def __init__(self, in_dim: int = len(TIM_RECOVERY_FEATURES), hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for candidate validity.

        Args:
            features: tensor of shape (N, F)

        Returns:
            logits of shape (N,)
        """
        return self.net(features).squeeze(-1)
