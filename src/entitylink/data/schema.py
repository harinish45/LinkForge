"""Data schema definition and integrity validation for EntityLink AI."""

from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any
import re


REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
VALID_ID_PREFIXES = ("S1-", "S2-", "S3-")


@dataclass(frozen=True)
class EntityRecord:
    """Represents a single business entity record."""
    entity_id: str
    business_name: str
    business_address: str
    country: str

    @property
    def source(self) -> str:
        """Derive the source identifier from entity_id prefix."""
        if self.entity_id.startswith("S1-"):
            return "S1"
        if self.entity_id.startswith("S2-"):
            return "S2"
        if self.entity_id.startswith("S3-"):
            return "S3"
        return "UNKNOWN"


@dataclass
class SchemaValidationReport:
    """Encapsulates validation results and diagnostics."""
    is_valid: bool
    total_records: int
    missing_columns: List[str]
    malformed_ids: List[str]
    duplicate_ids: List[str]
    empty_field_counts: Dict[str, int]
    warnings: List[str]


def validate_entity_id(entity_id: str, expected_prefix: Optional[str] = None) -> bool:
    """Validate whether an entity_id conforms to the required pattern.
    
    Format: S1-12345, S2-12345, S3-12345.
    """
    if not isinstance(entity_id, str) or not entity_id.strip():
        return False
    
    if expected_prefix:
        if not entity_id.startswith(expected_prefix):
            return False
    elif not entity_id.startswith(VALID_ID_PREFIXES):
        return False
    
    # Check that after prefix there is an identifier
    parts = entity_id.split("-", 1)
    return len(parts) == 2 and len(parts[1].strip()) > 0


def validate_records(
    records: List[EntityRecord],
    expected_prefix: Optional[str] = None
) -> SchemaValidationReport:
    """Perform rigorous integrity checks on a collection of EntityRecords."""
    seen_ids: Set[str] = set()
    duplicate_ids: List[str] = []
    malformed_ids: List[str] = []
    empty_fields: Dict[str, int] = {"name": 0, "address": 0, "country": 0}
    warnings: List[str] = []

    for r in records:
        # Check duplicate IDs
        if r.entity_id in seen_ids:
            if len(duplicate_ids) < 10:
                duplicate_ids.append(r.entity_id)
        seen_ids.add(r.entity_id)

        # Check malformed IDs
        if not validate_entity_id(r.entity_id, expected_prefix):
            if len(malformed_ids) < 10:
                malformed_ids.append(r.entity_id)

        # Check empty fields
        if not r.business_name or not r.business_name.strip():
            empty_fields["name"] += 1
        if not r.business_address or not r.business_address.strip():
            empty_fields["address"] += 1
        if not r.country or not r.country.strip():
            empty_fields["country"] += 1

    is_valid = len(duplicate_ids) == 0 and len(malformed_ids) == 0
    return SchemaValidationReport(
        is_valid=is_valid,
        total_records=len(records),
        missing_columns=[],
        malformed_ids=malformed_ids,
        duplicate_ids=duplicate_ids,
        empty_field_counts=empty_fields,
        warnings=warnings,
    )
