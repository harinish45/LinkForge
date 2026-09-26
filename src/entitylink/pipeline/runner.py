"""End-to-end pipeline execution engine."""

import os
import time
from typing import Dict, List, Optional, Set, Tuple
from entitylink.data.loader import load_records_dict, stream_tsv_records
from entitylink.data.schema import EntityRecord
from entitylink.blocking.candidate_generator import generate_candidates
from entitylink.models.base import BaseMatcher
from entitylink.models.rule_matcher import RuleBasedMatcher
from entitylink.models.calibrator import DecisionCalibrator
from entitylink.pipeline.writer import write_matching_results, write_candidate_pairs


class PipelineRunner:
    """Orchestrates end-to-end entity resolution inference from source files to output TSVs."""

    def __init__(
        self,
        matcher: Optional[BaseMatcher] = None,
        calibrator: Optional[DecisionCalibrator] = None,
        blocking_config: Optional[Dict] = None,
    ):
        self.matcher = matcher or RuleBasedMatcher()
        self.calibrator = calibrator or DecisionCalibrator(threshold=0.62)
        self.blocking_config = blocking_config or {}

    def run_inference(
        self,
        test_dir: str,
        output_dir: str,
        max_rows: Optional[int] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
        """Execute complete inference on test set and write submission files."""
        start_time = time.time()
        print(f"[*] Starting EntityLink AI inference on {test_dir}...")

        s1_path = os.path.join(test_dir, "test_source1.tsv")
        s2_path = os.path.join(test_dir, "test_source2.tsv")
        s3_path = os.path.join(test_dir, "test_source3.tsv")

        # 1. Load data
        print("  [1/5] Loading source records...")
        s1_records = load_records_dict(s1_path, max_rows=max_rows)
        s2_records = load_records_dict(s2_path, max_rows=max_rows)
        s3_records = load_records_dict(s3_path, max_rows=max_rows)
        print(f"        Loaded S1: {len(s1_records)}, S2: {len(s2_records)}, S3: {len(s3_records)}")

        # Keep original ordered list of S1 entity IDs
        ordered_s1_ids = list(s1_records.keys())

        # 2. Candidate generation (Blocking)
        print("  [2/5] Generating candidate pairs...")
        candidates = generate_candidates(
            source1=s1_records,
            source2=s2_records,
            source3=s3_records,
            config=self.blocking_config,
        )
        total_candidates = sum(len(c) for c in candidates.values())
        print(f"        Generated {total_candidates} total candidate pairs.")

        # Combined target records lookup
        target_records: Dict[str, EntityRecord] = {}
        target_records.update(s2_records)
        target_records.update(s3_records)

        # 3. Scoring & Match Decision
        print("  [3/5] Scoring pairs & calibrating decisions...")
        final_matches: Dict[str, List[str]] = {}

        for s1_id, cand_ids in candidates.items():
            s1_rec = s1_records[s1_id]
            valid_cand_records = [target_records[cid] for cid in cand_ids if cid in target_records]

            # Score each candidate
            cand_scores: List[Tuple[str, float]] = []
            for cand_rec in valid_cand_records:
                score = self.matcher.score_pair(s1_rec, cand_rec)
                cand_scores.append((cand_rec.entity_id, score))

            # Calibrate and select final matches
            selected = self.calibrator.select_matches(cand_scores)

            # Strict contract: final matches MUST be a subset of candidates
            validated_selected = [m for m in selected if m in cand_ids]
            final_matches[s1_id] = validated_selected

        # 4. Writing Output Files
        print("  [4/5] Writing TSV output files...")
        matching_path = os.path.join(output_dir, "matching_results.tsv")
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

        write_matching_results(matching_path, final_matches, required_s1_ids=ordered_s1_ids)
        write_candidate_pairs(candidate_path, candidates, required_s1_ids=ordered_s1_ids)
        print(f"        Saved matching results: {matching_path}")
        print(f"        Saved candidate pairs: {candidate_path}")

        elapsed = time.time() - start_time
        print(f"  [5/5] Inference complete in {elapsed:.2f} seconds.")
        return final_matches, candidates
