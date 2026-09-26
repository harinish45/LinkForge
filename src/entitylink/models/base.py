"""Abstract base class for EntityLink matching models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
from entitylink.data.schema import EntityRecord


class BaseMatcher(ABC):
    """Abstract interface for all pair matching models."""

    @abstractmethod
    def fit(
        self,
        pairs: List[Tuple[EntityRecord, EntityRecord]],
        labels: List[int],
    ) -> "BaseMatcher":
        """Train or fit the matching model on labeled pairs."""
        pass

    @abstractmethod
    def score_pair(
        self,
        record1: EntityRecord,
        record2: EntityRecord,
    ) -> float:
        """Output a continuous match score / probability in [0.0, 1.0]."""
        pass

    def score_candidates(
        self,
        s1_record: EntityRecord,
        candidates: List[EntityRecord],
    ) -> List[Tuple[str, float]]:
        """Score a list of candidate records against a Source 1 entity.
        
        Returns:
            List of (candidate_entity_id, score) tuples.
        """
        scores: List[Tuple[str, float]] = []
        for cand in candidates:
            score = self.score_pair(s1_record, cand)
            scores.append((cand.entity_id, score))
        return scores
