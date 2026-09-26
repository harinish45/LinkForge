"""Threshold calibration and multi-match decision engine."""

from typing import Dict, List, Optional, Set, Tuple


class DecisionCalibrator:
    """Decides final matches from scored candidate pairs.
    
    Supports:
    - Primary threshold cutoff.
    - Confidence gap filtering (e.g. discard candidates far below top candidate).
    - Conservative singleton handling (return empty set when confidence is below threshold).
    - Hard candidate limit per S1 entity.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        confidence_gap: float = 0.15,
        max_matches: int = 10,
    ):
        self.threshold = threshold
        self.confidence_gap = confidence_gap
        self.max_matches = max_matches

    def select_matches(
        self,
        candidate_scores: List[Tuple[str, float]],
    ) -> List[str]:
        """Select passing candidate IDs from a list of (candidate_id, score) tuples.
        
        Args:
            candidate_scores: List of tuples (cand_id, score).
            
        Returns:
            List of accepted match IDs, sorted by score descending.
        """
        if not candidate_scores:
            return []

        # Sort descending by score
        sorted_cands = sorted(candidate_scores, key=lambda x: x[1], reverse=True)

        # Check top score against primary threshold
        top_id, top_score = sorted_cands[0]
        if top_score < self.threshold:
            # Singleton / no-match entity
            return []

        matches: List[str] = [top_id]
        score_floor = max(self.threshold, top_score - self.confidence_gap)

        # Consider remaining candidates within confidence gap
        for cand_id, score in sorted_cands[1:]:
            if len(matches) >= self.max_matches:
                break
            if score >= score_floor:
                matches.append(cand_id)
            else:
                break

        return matches
