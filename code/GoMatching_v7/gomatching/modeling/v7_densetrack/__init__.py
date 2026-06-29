"""DenseTrack v7 — three orthogonal components on top of frozen GoMatching++.

Component 1: SAM-augmented proposal recall  (proposal_augmenter.py)
Component 2: Multi-encoder similarity consensus matcher  (consensus_matcher.py)
Component 3: VLM content matching  (content_matcher.py)

Each component has a learnable scalar gate initialized to 0, so the model
behaves IDENTICALLY to GoMatching++ baseline at iter 0. Gates ramp up after
warmup. This gives a clean rollback path if a component hurts performance.

All three components are gated by config flags:
    MODEL.DENSETRACK.COMP1_ENABLED  (SAM proposals)
    MODEL.DENSETRACK.COMP2_ENABLED  (consensus matcher)
    MODEL.DENSETRACK.COMP3_ENABLED  (content matcher)
"""
from .proposal_augmenter import SAMProposalAugmenter
from .consensus_matcher import MultiEncoderConsensusMatcher
from .content_matcher import VLMContentMatcher
from .encoder_loaders import (
    load_frozen_encoder,
    EncoderFeatureCache,
    SUPPORTED_ENCODERS,
)

__all__ = [
    "SAMProposalAugmenter",
    "MultiEncoderConsensusMatcher",
    "VLMContentMatcher",
    "load_frozen_encoder",
    "EncoderFeatureCache",
    "SUPPORTED_ENCODERS",
]
