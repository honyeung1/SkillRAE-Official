from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal


PROVENANCE_SOURCE_TYPES = {
    "instruction",
    "public_task_asset",
    "public_skill_catalog",
    "selected_skill",
    "generic_scaffold",
}
FORBIDDEN_SOURCE_PARTS = {"tests", "solution"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def stable_id(prefix: str, value: Any, length: int = 20) -> str:
    return f"{prefix}_{content_sha256(value)[:length]}"


def provenance_id(ref: "ProvenanceRef") -> str:
    return stable_id("prov", asdict(ref))


def _path_parts(path: str) -> tuple[str, ...]:
    normalized = path.replace("\\", "/")
    return tuple(part.lower() for part in PurePosixPath(normalized).parts)


@dataclass(frozen=True)
class ProvenanceRef:
    source_file: str
    source_type: str
    line_start: int
    line_end: int
    exact_text: str
    source_sha256: str
    extraction_rule: str

    def validate(self, repo_root: Path | None = None) -> list[str]:
        errors: list[str] = []
        if self.source_type not in PROVENANCE_SOURCE_TYPES:
            errors.append(f"unknown_source_type:{self.source_type}")
        if FORBIDDEN_SOURCE_PARTS.intersection(_path_parts(self.source_file)):
            errors.append(f"forbidden_source_path:{self.source_file}")
        if not self.source_file:
            errors.append("missing_source_file")
        if self.line_start < 1 or self.line_end < self.line_start:
            errors.append("invalid_line_range")
        if not self.exact_text:
            errors.append("empty_exact_text")
        if not self.source_sha256:
            errors.append("missing_source_sha256")
        if not self.extraction_rule:
            errors.append("missing_extraction_rule")

        if repo_root is not None:
            source = (repo_root / self.source_file).resolve()
            try:
                source.relative_to(repo_root.resolve())
            except ValueError:
                errors.append(f"source_outside_repo:{self.source_file}")
                return errors
            if not source.is_file():
                errors.append(f"source_missing:{self.source_file}")
                return errors
            raw = source.read_bytes()
            if sha256(raw).hexdigest() != self.source_sha256:
                errors.append(f"source_hash_mismatch:{self.source_file}")
            lines = raw.decode("utf-8").splitlines(keepends=True)
            if self.line_end > len(lines):
                errors.append(f"line_range_out_of_bounds:{self.source_file}")
            else:
                observed = "".join(lines[self.line_start - 1 : self.line_end])
                if observed != self.exact_text:
                    errors.append(f"quote_mismatch:{self.source_file}:{self.line_start}")
        return errors


SubunitRole = Literal[
    "STEP",
    "TOOL",
    "ARTIFACT",
    "CONSTRAINT",
    "VALIDATION",
    "REFERENCE",
]


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    source_dir: str
    skill_md: str
    skill_md_sha256: str
    tree_sha256: str


@dataclass(frozen=True)
class SubunitInstance:
    subunit_id: str
    skill_id: str
    role: SubunitRole
    heading_path: tuple[str, ...]
    provenance: ProvenanceRef
    executable: bool


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: Literal["TOOL", "ARTIFACT", "VALIDATION", "CONCEPT"]
    canonical_label: str
    evidence: tuple[ProvenanceRef, ...]


RelationStatus = Literal["CERTIFIED", "CANDIDATE", "REJECTED", "UNRESOLVED"]


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    relation_type: str
    source_id: str
    target_id: str
    status: RelationStatus
    evidence: tuple[ProvenanceRef, ...]
    construction_rule: str


@dataclass(frozen=True)
class CommunityRecord:
    community_id: str
    member_skill_ids: tuple[str, ...]
    member_subunit_ids: tuple[str, ...] = ()
    status: RelationStatus = "UNRESOLVED"
    construction_rule: str = ""
    evidence_relation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementRecord:
    requirement_id: str
    requirement_type: str
    criticality: Literal["HARD", "CORE", "SUPPORT"]
    provenance: ProvenanceRef


@dataclass(frozen=True)
class CandidateSkill:
    skill_id: str
    c0_rank: int
    community_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportPath:
    path_id: str
    requirement_ids: tuple[str, ...]
    parent_skill_ids: tuple[str, ...]
    subunit_ids: tuple[str, ...]
    community_ids: tuple[str, ...] = ()
    prerequisite_path_ids: tuple[str, ...] = ()
    unresolved_dependencies: tuple[str, ...] = ()
    validation_present: bool = False
    contradicted: bool = False
    conflict_ids: tuple[str, ...] = ()
    status: RelationStatus = "UNRESOLVED"
    provenance_ids: tuple[str, ...] = ()

    @property
    def is_closed_candidate(self) -> bool:
        return (
            self.status == "CERTIFIED"
            and not self.unresolved_dependencies
            and not self.contradicted
            and not self.conflict_ids
        )


@dataclass(frozen=True)
class CompilerInput:
    task_id: str
    requirements: tuple[RequirementRecord, ...]
    candidates: tuple[CandidateSkill, ...]
    communities: tuple[CommunityRecord, ...]
    support_paths: tuple[SupportPath, ...]
    subunits: tuple[SubunitInstance, ...]
    top_k: int = 5

    def candidate_by_id(self) -> dict[str, CandidateSkill]:
        return {candidate.skill_id: candidate for candidate in self.candidates}


@dataclass(frozen=True)
class GuideLine:
    line_number: int
    content: str
    semantic_role: str
    source_ids: tuple[str, ...]
    destination: str = "PROCEDURAL_SLICE_GUIDE.md"


@dataclass(frozen=True)
class CompilerOutput:
    task_id: str
    policy: str
    selected_skill_ids: tuple[str, ...]
    selected_path_ids: tuple[str, ...]
    closed_requirement_ids: tuple[str, ...]
    guide_text: str
    guide_lines: tuple[GuideLine, ...]
    decision_log: tuple[dict[str, Any], ...] = ()

    @property
    def output_sha256(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class GraphPackage:
    schema_version: str
    source_root: str
    skills: tuple[SkillRecord, ...]
    subunits: tuple[SubunitInstance, ...]
    entities: tuple[EntityRecord, ...]
    relations: tuple[RelationRecord, ...]
    communities: tuple[CommunityRecord, ...] = ()
    construction_config: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def package_sha256(self) -> str:
        return content_sha256(self.payload())


def validate_provenance_refs(
    refs: Iterable[ProvenanceRef], repo_root: Path | None = None
) -> list[str]:
    errors: list[str] = []
    for ref in refs:
        errors.extend(ref.validate(repo_root=repo_root))
    return errors
