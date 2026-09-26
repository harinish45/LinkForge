"""Data loading and schema module for EntityLink AI."""

from entitylink.data.schema import EntityRecord, SchemaValidationReport, validate_records, validate_entity_id
from entitylink.data.loader import stream_tsv_records, load_records_dict, load_ground_truth

__all__ = [
    "EntityRecord",
    "SchemaValidationReport",
    "validate_records",
    "validate_entity_id",
    "stream_tsv_records",
    "load_records_dict",
    "load_ground_truth",
]
