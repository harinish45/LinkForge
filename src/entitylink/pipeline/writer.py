"""Robust output writer for challenge submission TSV files."""

import os
from typing import Dict, Iterable, List, Optional, Set

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
DELIM = "\t"


def write_id_list_file(
    output_path: str,
    id_mapping: Dict[str, Iterable[str]],
    required_s1_ids: Optional[List[str]] = None,
    header: Optional[List[str]] = None,
) -> None:
    """Write tab-separated file with exactly two columns: source1_id and comma-separated IDs.
    
    Ensures:
    - Every required S1 entity ID is included (empty string if no matches/candidates).
    - No duplicate IDs inside the comma-separated list.
    - Stable sorting of IDs for complete determinism.
    - Exact tab separation and UTF-8 encoding.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    header = header or MATCHING_HEADER

    # Determine ordered list of S1 IDs
    if required_s1_ids is not None:
        s1_order = required_s1_ids
    else:
        s1_order = sorted(id_mapping.keys())

    with open(output_path, mode="w", encoding="utf-8", newline="\n") as f:
        # Write header
        f.write(f"{header[0]}{DELIM}{header[1]}\n")

        for s1_id in s1_order:
            clean_s1 = s1_id.strip()
            ids = id_mapping.get(clean_s1, [])
            
            # Deterministic, unique, clean IDs
            cleaned_unique_ids = sorted({i.strip() for i in ids if i.strip()})
            joined_ids = ",".join(cleaned_unique_ids)

            f.write(f"{clean_s1}{DELIM}{joined_ids}\n")


def write_matching_results(
    output_path: str,
    matches: Dict[str, Iterable[str]],
    required_s1_ids: Optional[List[str]] = None,
) -> None:
    """Write matching_results.tsv formatted to pass competition validator."""
    write_id_list_file(
        output_path=output_path,
        id_mapping=matches,
        required_s1_ids=required_s1_ids,
        header=MATCHING_HEADER,
    )


def write_candidate_pairs(
    output_path: str,
    candidates: Dict[str, Iterable[str]],
    required_s1_ids: Optional[List[str]] = None,
) -> None:
    """Write candidate_pairs.tsv formatted to pass competition validator."""
    write_id_list_file(
        output_path=output_path,
        id_mapping=candidates,
        required_s1_ids=required_s1_ids,
        header=CANDIDATE_HEADER,
    )
