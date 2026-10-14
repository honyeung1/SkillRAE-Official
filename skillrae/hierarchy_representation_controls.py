"""Default-off, one-factor controls for the frozen A2 experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ARMS = ("a2_without_l2", "a2_without_l0", "a2_source_spans")


def retrieve_without_channel(
    retriever: Any, instruction: str, arm: str, *, k: int
) -> tuple[Any, Any]:
    """Use the production formula without retuning the surviving weights."""
    if arm not in (*ARMS, "a2_reference"):
        raise ValueError(f"Unknown hierarchy/representation arm: {arm}")
    field = {"a2_without_l2": "l2_embeddings", "a2_without_l0": "subunit_embeddings"}.get(arm)
    if field is None:
        return retriever.retrieve_with_metadata(instruction, k=k)
    original = getattr(retriever, field)
    try:
        setattr(retriever, field, None)
        return retriever.retrieve_with_metadata(instruction, k=k)
    finally:
        setattr(retriever, field, original)


def source_span_view(binding: dict[str, Any]) -> dict[str, Any]:
    """Remove preassigned capability semantics, not public evidence or IDs.

    The evidence envelope and its upstream nomination stay fixed. This tests
    structured representation at B's input, not removal of all V2 processing.
    """
    result = deepcopy(binding)
    unit = result["capability_unit"]
    for field in (
        "canonical_label", "heading_path", "external_prerequisite",
        "roles_present", "slot_evidence",
    ):
        unit.pop(field, None)
    spans = unit.get("support_spans", [])
    for span in spans:
        span.pop("role", None)
    spans.sort(key=lambda row: (
        str(row["source_file"]), int(row["line_start"]),
        int(row["line_end"]), str(row["evidence_id"]),
    ))
    if public_evidence_signature(result) != public_evidence_signature(binding):
        raise ValueError("Source-span control changed public evidence")
    return result


def public_evidence_signature(binding: dict[str, Any]) -> dict[str, Any]:
    unit = binding["capability_unit"]
    return {
        "requirement": binding["requirement"],
        "skill_id": unit["skill_id"],
        "capability_unit_id": unit["capability_unit_id"],
        "resources": unit.get("resources", []),
        "spans": sorted(
            [{key: value for key, value in span.items() if key != "role"}
             for span in unit.get("support_spans", [])],
            key=lambda row: (row["source_file"], row["line_start"], row["line_end"], row["evidence_id"]),
        ),
    }
