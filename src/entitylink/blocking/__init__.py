"""Blocking and candidate generation module."""

from entitylink.blocking.candidate_generator import (
    generate_candidates,
    MultiStrategyBlockingEngine,
    CandidatePair,
)

__all__ = [
    "generate_candidates",
    "MultiStrategyBlockingEngine",
    "CandidatePair",
]
