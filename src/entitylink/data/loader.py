"""Fast, memory-efficient data loader for TSV sources and ground truth."""

import csv
import os
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple
from entitylink.data.schema import EntityRecord, REQUIRED_SOURCE_COLUMNS, REQUIRED_GROUND_TRUTH_COLUMNS


def stream_tsv_records(
    file_path: str,
    max_rows: Optional[int] = None,
    encoding: str = "utf-8"
) -> Iterator[EntityRecord]:
    """Stream EntityRecord objects one by one without reading the entire file into memory.
    
    Args:
        file_path: Absolute or relative path to TSV file.
        max_rows: Optional row count limit for profiling or quick debugging.
        encoding: File character encoding (default: utf-8).
    
    Yields:
        EntityRecord instances.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    with open(file_path, mode="r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return

        header_cols = [c.strip().lower() for c in header]
        # Verify required columns exist
        col_idx = {col: i for i, col in enumerate(header_cols)}
        for req in REQUIRED_SOURCE_COLUMNS:
            if req not in col_idx:
                raise ValueError(
                    f"File '{file_path}' is missing required column '{req}'. Found: {header_cols}"
                )

        id_idx = col_idx["entity_id"]
        name_idx = col_idx["business_name"]
        addr_idx = col_idx["business_address"]
        country_idx = col_idx["country"]

        count = 0
        for row in reader:
            if not row or not any(row):
                continue
            
            # Extract fields defensively
            eid = row[id_idx].strip() if id_idx < len(row) else ""
            bname = row[name_idx].strip() if name_idx < len(row) else ""
            baddr = row[addr_idx].strip() if addr_idx < len(row) else ""
            bcountry = row[country_idx].strip() if country_idx < len(row) else ""

            yield EntityRecord(
                entity_id=eid,
                business_name=bname,
                business_address=baddr,
                country=bcountry
            )
            count += 1
            if max_rows is not None and count >= max_rows:
                break


def load_records_dict(
    file_path: str,
    max_rows: Optional[int] = None
) -> Dict[str, EntityRecord]:
    """Load records into an entity_id -> EntityRecord lookup dictionary."""
    records: Dict[str, EntityRecord] = {}
    for record in stream_tsv_records(file_path, max_rows=max_rows):
        records[record.entity_id] = record
    return records


def load_ground_truth(
    file_path: str,
    max_rows: Optional[int] = None,
    encoding: str = "utf-8"
) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set(matched_entity_ids)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    ground_truth: Dict[str, Set[str]] = {}
    with open(file_path, mode="r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return ground_truth

        header_cols = [c.strip().lower() for c in header]
        col_idx = {col: i for i, col in enumerate(header_cols)}
        for req in REQUIRED_GROUND_TRUTH_COLUMNS:
            if req not in col_idx:
                raise ValueError(
                    f"Ground truth file is missing column '{req}'. Found: {header_cols}"
                )

        s1_idx = col_idx["source1_entity_id"]
        matched_idx = col_idx["matched_entity_ids"]

        count = 0
        for row in reader:
            if not row or not any(row):
                continue
            
            s1_id = row[s1_idx].strip() if s1_idx < len(row) else ""
            raw_matches = row[matched_idx].strip() if matched_idx < len(row) else ""
            
            if raw_matches:
                matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
            else:
                matches = set()
            
            ground_truth[s1_id] = matches
            count += 1
            if max_rows is not None and count >= max_rows:
                break

    return ground_truth
