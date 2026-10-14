from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

from markdown_it import MarkdownIt

from .schema import (
    EntityRecord,
    GraphPackage,
    ProvenanceRef,
    RelationRecord,
    SkillRecord,
    SubunitInstance,
    canonical_json_bytes,
    stable_id,
)


SCHEMA_VERSION = "hcsc-graph-v2.0"
LEGACY_RETRIEVAL_FILENAMES = {
    "skill_nodes.json",
    "subunit_nodes.json",
    "edges.json",
    "canonical_skill_representations.json",
    "skill_l2_mapping.json",
    "subunit_ids.json",
    "subunit_embeddings.npy",
}

_ACTION_MARKERS = re.compile(
    r"\b(?:use|run|create|write|save|open|load|read|extract|compute|calculate|"
    r"recalculate|verify|validate|check|inspect|convert|render|export|import|"
    r"install|call|execute|preserve|avoid|never|must|should|required)\b",
    re.IGNORECASE,
)
_VALIDATION_MARKERS = re.compile(
    r"\b(?:verify|validate|check|test|reopen|re-open|recalculate|confirm|assert|"
    r"error|consistency)\b",
    re.IGNORECASE,
)
_CONSTRAINT_MARKERS = re.compile(
    r"\b(?:must|must not|never|do not|don't|avoid|required|only if|preserve|"
    r"cannot|should not)\b",
    re.IGNORECASE,
)
_ARTIFACT_MARKERS = re.compile(
    r"\b(?:save|write|create|export|output|artifact|file|workbook|document|"
    r"presentation|image|video|report)\b",
    re.IGNORECASE,
)
_FILE_TOKEN = re.compile(
    r"(?<![\w.-])([\w./-]+\.(?:xlsx|xlsm|csv|tsv|json|yaml|yml|xml|pdf|"
    r"pptx|docx|png|jpg|jpeg|svg|mp4|wav|txt|md|py|sh|js|ts|html|css))(?![\w.-])",
    re.IGNORECASE,
)
_INLINE_CODE = re.compile(r"`([^`\n]{1,160})`")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return sha256(canonical_json_bytes(entries)).hexdigest()


def _headings_by_line(tokens: list[object]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if getattr(token, "type", "") != "heading_open" or not token.map:
            continue
        level = int(str(token.tag)[1:])
        title = ""
        if index + 1 < len(tokens) and getattr(tokens[index + 1], "type", "") == "inline":
            title = str(tokens[index + 1].content).strip()
        headings.append((int(token.map[0]), level, title))
    return headings


def _heading_path_at(line_index: int, headings: list[tuple[int, int, str]]) -> tuple[str, ...]:
    stack: dict[int, str] = {}
    for heading_line, level, title in headings:
        if heading_line > line_index:
            break
        stack = {key: value for key, value in stack.items() if key < level}
        stack[level] = title
    return tuple(stack[level] for level in sorted(stack) if stack[level])


def _classify_role(text: str, token_type: str) -> tuple[str, bool]:
    if token_type == "fence":
        return "TOOL", True
    if _VALIDATION_MARKERS.search(text):
        return "VALIDATION", True
    if _CONSTRAINT_MARKERS.search(text):
        return "CONSTRAINT", True
    if _ARTIFACT_MARKERS.search(text):
        return "ARTIFACT", True
    if _ACTION_MARKERS.search(text):
        return "STEP", True
    return "REFERENCE", False


def _candidate_spans(tokens: list[object]) -> list[tuple[int, int, str, str]]:
    list_spans: list[tuple[int, int]] = []
    candidates: list[tuple[int, int, str, str]] = []
    for token in tokens:
        token_type = getattr(token, "type", "")
        mapping = getattr(token, "map", None)
        if not mapping:
            continue
        start, end = int(mapping[0]), int(mapping[1])
        if token_type == "list_item_open":
            list_spans.append((start, end))
            candidates.append((start, end, token_type, "commonmark_list_item"))
        elif token_type == "fence":
            candidates.append((start, end, token_type, "commonmark_fenced_code"))

    for token in tokens:
        if getattr(token, "type", "") != "paragraph_open" or not token.map:
            continue
        start, end = int(token.map[0]), int(token.map[1])
        if any(list_start <= start and end <= list_end for list_start, list_end in list_spans):
            continue
        candidates.append((start, end, "paragraph_open", "actionable_paragraph"))

    return sorted(set(candidates), key=lambda row: (row[0], row[1], row[2], row[3]))


def parse_skill_markdown(skill_dir: Path, repo_root: Path) -> tuple[SkillRecord, tuple[SubunitInstance, ...]]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"Missing SKILL.md under {skill_dir}")

    raw = skill_md.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    source_sha = sha256(raw).hexdigest()
    tree_sha = hash_tree(skill_dir)
    relative_md = skill_md.relative_to(repo_root).as_posix()
    relative_dir = skill_dir.relative_to(repo_root).as_posix()
    skill_id = skill_dir.name
    skill = SkillRecord(
        skill_id=skill_id,
        source_dir=relative_dir,
        skill_md=relative_md,
        skill_md_sha256=source_sha,
        tree_sha256=tree_sha,
    )

    parser = MarkdownIt("commonmark")
    tokens = parser.parse(text)
    headings = _headings_by_line(tokens)
    subunits: list[SubunitInstance] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for start, end, token_type, extraction_rule in _candidate_spans(tokens):
        if start < 0 or end <= start or end > len(lines):
            continue
        exact_text = "".join(lines[start:end])
        if token_type == "paragraph_open" and not _ACTION_MARKERS.search(exact_text):
            continue
        role, executable = _classify_role(exact_text, token_type)
        span_key = (start, end, exact_text)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        provenance = ProvenanceRef(
            source_file=relative_md,
            source_type="public_skill_catalog",
            line_start=start + 1,
            line_end=end,
            exact_text=exact_text,
            source_sha256=source_sha,
            extraction_rule=extraction_rule,
        )
        subunit_id = stable_id(
            "sui",
            {
                "tree_sha256": tree_sha,
                "source_file": relative_md,
                "line_start": start + 1,
                "line_end": end,
                "exact_text": exact_text,
            },
        )
        subunits.append(
            SubunitInstance(
                subunit_id=subunit_id,
                skill_id=skill_id,
                role=role,  # type: ignore[arg-type]
                heading_path=_heading_path_at(start, headings),
                provenance=provenance,
                executable=executable,
            )
        )
    return skill, tuple(subunits)


def _entity_mentions(subunit: SubunitInstance) -> Iterable[tuple[str, str]]:
    text = subunit.provenance.exact_text
    seen: set[tuple[str, str]] = set()
    for match in _FILE_TOKEN.finditer(text):
        item = ("ARTIFACT", match.group(1))
        normalized = (item[0], " ".join(item[1].lower().split()))
        if normalized not in seen:
            seen.add(normalized)
            yield item
    for match in _INLINE_CODE.finditer(text):
        label = match.group(1).strip()
        if not label or _FILE_TOKEN.fullmatch(label):
            continue
        item = ("TOOL", label)
        normalized = (item[0], " ".join(item[1].lower().split()))
        if normalized not in seen:
            seen.add(normalized)
            yield item
    if subunit.role == "VALIDATION":
        item = ("VALIDATION", "public-validation-operation")
        if item not in seen:
            yield item


def build_graph_package(pool_dir: Path, repo_root: Path) -> GraphPackage:
    pool_dir = pool_dir.resolve()
    repo_root = repo_root.resolve()
    try:
        pool_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Skill pool must be under repository root: {pool_dir}") from exc

    skills: list[SkillRecord] = []
    subunits: list[SubunitInstance] = []
    relations: list[RelationRecord] = []
    entities_by_key: dict[tuple[str, str], EntityRecord] = {}

    skill_dirs = sorted(
        path for path in pool_dir.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
    )
    for skill_dir in skill_dirs:
        skill, skill_subunits = parse_skill_markdown(skill_dir, repo_root)
        skills.append(skill)
        subunits.extend(skill_subunits)
        for subunit in skill_subunits:
            evidence = (subunit.provenance,)
            relations.append(
                RelationRecord(
                    relation_id=stable_id(
                        "rel",
                        ["SKILL_HAS_INSTANCE", skill.skill_id, subunit.subunit_id],
                    ),
                    relation_type="SKILL_HAS_INSTANCE",
                    source_id=skill.skill_id,
                    target_id=subunit.subunit_id,
                    status="CERTIFIED",
                    evidence=evidence,
                    construction_rule="deterministic_parent_membership",
                )
            )
            for entity_type, label in _entity_mentions(subunit):
                normalized_label = " ".join(label.lower().split())
                key = (entity_type, normalized_label)
                entity = entities_by_key.get(key)
                if entity is None:
                    entity = EntityRecord(
                        entity_id=stable_id("ent", [entity_type, normalized_label]),
                        entity_type=entity_type,  # type: ignore[arg-type]
                        canonical_label=normalized_label,
                        evidence=evidence,
                    )
                    entities_by_key[key] = entity
                relation_type = {
                    "TOOL": "USES_TOOL",
                    "ARTIFACT": "MENTIONS_ARTIFACT",
                    "VALIDATION": "HAS_VALIDATION_OPERATION",
                }[entity_type]
                relations.append(
                    RelationRecord(
                        relation_id=stable_id(
                            "rel",
                            [relation_type, subunit.subunit_id, entity.entity_id],
                        ),
                        relation_type=relation_type,
                        source_id=subunit.subunit_id,
                        target_id=entity.entity_id,
                        status="CERTIFIED",
                        evidence=evidence,
                        construction_rule="deterministic_exact_mention",
                    )
                )

        by_heading: dict[tuple[str, ...], list[SubunitInstance]] = {}
        for subunit in skill_subunits:
            by_heading.setdefault(subunit.heading_path, []).append(subunit)
        for heading_subunits in by_heading.values():
            ordered = sorted(heading_subunits, key=lambda item: item.provenance.line_start)
            for left, right in zip(ordered, ordered[1:]):
                relations.append(
                    RelationRecord(
                        relation_id=stable_id(
                            "rel",
                            ["PRECEDES", left.subunit_id, right.subunit_id],
                        ),
                        relation_type="PRECEDES",
                        source_id=left.subunit_id,
                        target_id=right.subunit_id,
                        status="CERTIFIED",
                        evidence=(left.provenance, right.provenance),
                        construction_rule="deterministic_source_order_same_heading",
                    )
                )

    return GraphPackage(
        schema_version=SCHEMA_VERSION,
        source_root=pool_dir.relative_to(repo_root).as_posix(),
        skills=tuple(skills),
        subunits=tuple(sorted(subunits, key=lambda item: item.subunit_id)),
        entities=tuple(sorted(entities_by_key.values(), key=lambda item: item.entity_id)),
        relations=tuple(sorted(relations, key=lambda item: item.relation_id)),
        communities=tuple(),
        construction_config={
            "markdown_parser": "markdown-it-py/commonmark",
            "community_status": "NOT_BUILT",
            "semantic_relation_status": "NOT_BUILT",
            "legacy_artifacts_modified": False,
        },
    )


def write_graph_package(package: GraphPackage, output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.name in LEGACY_RETRIEVAL_FILENAMES:
        raise ValueError("Graph V2 output root must not be a legacy retrieval artifact")
    package_dir = output_root / package.package_sha256
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = package_dir / "graph_v2_manifest.json"
    payload = package.payload()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Immutable package collision at {manifest_path}")
    manifest_path.write_text(serialized, encoding="utf-8")
    manifest_file_sha256 = sha256(serialized.encode("utf-8")).hexdigest()
    (package_dir / "graph_v2_manifest.sha256").write_text(
        f"{manifest_file_sha256}  graph_v2_manifest.json\n",
        encoding="utf-8",
    )
    return package_dir
