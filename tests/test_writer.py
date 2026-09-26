"""Unit tests for submission TSV writer formatting."""

import os
import tempfile
import pytest
from entitylink.pipeline.writer import write_matching_results, write_candidate_pairs


def test_write_matching_results(tmp_path):
    output_file = tmp_path / "matching_results.tsv"
    matches = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": [],  # singleton
        "S1-3": ["S2-30", "S2-30"],  # duplicate should be deduplicated
    }
    required_s1 = ["S1-1", "S1-2", "S1-3"]

    write_matching_results(str(output_file), matches, required_s1_ids=required_s1)

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-10,S3-20"
    assert lines[2] == "S1-2\t"
    assert lines[3] == "S1-3\tS2-30"  # Deduplicated and cleaned
