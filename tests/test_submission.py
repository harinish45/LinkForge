"""Unit tests for submission validator."""

import os
import tempfile
import pytest
from student_resource.utils.validate_submission import validate


def test_submission_validator_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test_source1.tsv
        s1_path = os.path.join(tmpdir, "test_source1.tsv")
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
            f.write("S1-002\tBeta Ltd\t45 High St\tUK\n")

        # Create matching_results.tsv
        match_path = os.path.join(tmpdir, "matching_results.tsv")
        with open(match_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001,S3-001\n")
            f.write("S1-002\t\n")

        # Create candidate_pairs.tsv
        cand_path = os.path.join(tmpdir, "candidate_pairs.tsv")
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-001,S3-001\n")
            f.write("S1-002\tS2-002\n")

        errors, warnings = validate(
            matching_path=match_path,
            candidate_path=cand_path,
            test_dir=tmpdir,
            check_ids=False,
        )

        assert len(errors) == 0


def test_submission_validator_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test_source1.tsv
        s1_path = os.path.join(tmpdir, "test_source1.tsv")
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
            f.write("S1-002\tBeta Ltd\t45 High St\tUK\n")

        # Create matching_results.tsv with duplicate inside list and self match
        match_path = os.path.join(tmpdir, "matching_results.tsv")
        with open(match_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001,S2-001\n")  # Intra-list duplicate
            f.write("S1-002\tS1-002\n")        # Self-match (S1- inside match list)

        errors, warnings = validate(
            matching_path=match_path,
            candidate_path=None,
            test_dir=tmpdir,
            check_ids=False,
        )

        assert len(errors) > 0
        error_text = " ".join(errors)
        assert "repeated ID inside" in error_text or "duplicate" in error_text
        assert "self-matches" in error_text or "Source-1 IDs" in error_text
