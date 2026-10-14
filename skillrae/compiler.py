from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable

from markdown_it import MarkdownIt

from .schema import (
    CompilerInput,
    CompilerOutput,
    GuideLine,
    ProvenanceRef,
    RequirementRecord,
    SupportPath,
    provenance_id,
    stable_id,
)


_HARD_MARKERS = re.compile(
    r"\b(?:must|required|do not|don't|never|preserve|deliver|submit|output|"
    r"save as|write to|cannot|only)\b",
    re.IGNORECASE,
)
_CORE_MARKERS = re.compile(
    r"\b(?:create|build|implement|calculate|compute|analyze|extract|convert|"
    r"generate|modify|update|validate|verify|check|produce|fill|answer)\b",
    re.IGNORECASE,
)
_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OUTPUT_PATH", re.compile(r"\b(?:path|filename|file name|save|write|output)\b", re.I)),
    ("FORMAT", re.compile(r"\b(?:format|json|csv|xlsx|pdf|pptx|docx|schema)\b", re.I)),
    ("PRESERVATION", re.compile(r"\b(?:preserve|unchanged|do not modify|keep)\b", re.I)),
    ("VALIDATION", re.compile(r"\b(?:validate|verify|check|recalculate|test)\b", re.I)),
    ("CONSISTENCY", re.compile(r"\b(?:consistent|consistency|match|reconcile)\b", re.I)),
    ("COMPLETION", re.compile(r"\b(?:complete|finish|deliver|submit)\b", re.I)),
    ("ARTIFACT", re.compile(r"\b(?:artifact|file|workbook|document|presentation|report)\b", re.I)),
)


def _requirement_type(text: str) -> str:
    for requirement_type, pattern in _TYPE_RULES:
        if pattern.search(text):
            return requirement_type
    return "DOMAIN_ACTION"


def _criticality(text: str) -> str:
    if _HARD_MARKERS.search(text):
        return "HARD"
    if _CORE_MARKERS.search(text):
        return "CORE"
    return "SUPPORT"


def extract_public_requirements(instruction_path: Path, repo_root: Path) -> tuple[RequirementRecord, ...]:
    instruction_path = instruction_path.resolve()
    repo_root = repo_root.resolve()
    raw = instruction_path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    source_sha = sha256(raw).hexdigest()
    relative_path = instruction_path.relative_to(repo_root).as_posix()
    tokens = MarkdownIt("commonmark").parse(text)

    list_spans: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []
    for token in tokens:
        if not token.map:
            continue
        start, end = int(token.map[0]), int(token.map[1])
        if token.type == "list_item_open":
            list_spans.append((start, end))
            spans.append((start, end, "commonmark_list_item"))
    for token in tokens:
        if token.type != "paragraph_open" or not token.map:
            continue
        start, end = int(token.map[0]), int(token.map[1])
        if any(left <= start and end <= right for left, right in list_spans):
            continue
        exact = "".join(lines[start:end])
        if _HARD_MARKERS.search(exact) or _CORE_MARKERS.search(exact):
            spans.append((start, end, "actionable_paragraph"))

    if not spans and text:
        spans = [(0, len(lines), "whole_instruction_fallback")]

    requirements: list[RequirementRecord] = []
    seen: set[tuple[int, int, str]] = set()
    for start, end, rule in sorted(spans):
        exact_text = "".join(lines[start:end])
        key = (start, end, exact_text)
        if not exact_text.strip() or key in seen:
            continue
        seen.add(key)
        provenance = ProvenanceRef(
            source_file=relative_path,
            source_type="instruction",
            line_start=start + 1,
            line_end=end,
            exact_text=exact_text,
            source_sha256=source_sha,
            extraction_rule=rule,
        )
        requirements.append(
            RequirementRecord(
                requirement_id=stable_id(
                    "req",
                    [relative_path, start + 1, end, exact_text],
                ),
                requirement_type=_requirement_type(exact_text),
                criticality=_criticality(exact_text),  # type: ignore[arg-type]
                provenance=provenance,
            )
        )
    return tuple(requirements)


def _valid_paths(data: CompilerInput) -> tuple[SupportPath, ...]:
    requirement_ids = {item.requirement_id for item in data.requirements}
    candidate_ids = {item.skill_id for item in data.candidates}
    subunit_ids = {item.subunit_id for item in data.subunits}
    return tuple(
        path
        for path in data.support_paths
        if path.is_closed_candidate
        and set(path.requirement_ids).issubset(requirement_ids)
        and set(path.parent_skill_ids).issubset(candidate_ids)
        and set(path.subunit_ids).issubset(subunit_ids)
    )


def _path_bundle(
    path: SupportPath,
    paths_by_id: dict[str, SupportPath],
) -> tuple[SupportPath, ...] | None:
    bundle: dict[str, SupportPath] = {}
    visiting: set[str] = set()

    def visit(current: SupportPath) -> bool:
        if current.path_id in bundle:
            return True
        if current.path_id in visiting or not current.is_closed_candidate:
            return False
        visiting.add(current.path_id)
        for dependency_id in current.prerequisite_path_ids:
            dependency = paths_by_id.get(dependency_id)
            if dependency is None or not visit(dependency):
                return False
        visiting.remove(current.path_id)
        bundle[current.path_id] = current
        return True

    if not visit(path):
        return None
    bundle_ids = set(bundle)
    for item in bundle.values():
        if bundle_ids.intersection(item.conflict_ids):
            return None
    return tuple(sorted(bundle.values(), key=lambda item: item.path_id))


def closed_paths_for_selection(data: CompilerInput, selected_skill_ids: Iterable[str]) -> tuple[SupportPath, ...]:
    selected = set(selected_skill_ids)
    valid = _valid_paths(data)
    valid_ids = {path.path_id for path in valid}
    result: list[SupportPath] = []
    for path in valid:
        if not set(path.parent_skill_ids).issubset(selected):
            continue
        if not set(path.prerequisite_path_ids).issubset(valid_ids):
            continue
        result.append(path)
    result_ids = {path.path_id for path in result}
    return tuple(
        path
        for path in result
        if set(path.prerequisite_path_ids).issubset(result_ids)
        and not set(path.conflict_ids).intersection(result_ids)
    )


def closed_requirements(data: CompilerInput, selected_skill_ids: Iterable[str]) -> tuple[str, ...]:
    covered = {
        requirement_id
        for path in closed_paths_for_selection(data, selected_skill_ids)
        for requirement_id in path.requirement_ids
    }
    return tuple(sorted(covered))


def _c0_order(data: CompilerInput) -> list[str]:
    return [
        item.skill_id
        for item in sorted(data.candidates, key=lambda candidate: (candidate.c0_rank, candidate.skill_id))
    ]


def _fill_from_c0(data: CompilerInput, selected: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for skill_id in selected:
        if skill_id not in result:
            result.append(skill_id)
    for skill_id in _c0_order(data):
        if skill_id not in result:
            result.append(skill_id)
        if len(result) == data.top_k:
            break
    if len(result) != data.top_k:
        raise ValueError(f"Cannot produce exactly top_k={data.top_k} skills")
    return tuple(result)


def select_c0(data: CompilerInput) -> tuple[str, ...]:
    return _fill_from_c0(data, ())


def select_rf(data: CompilerInput, k_rrf: int = 60) -> tuple[str, ...]:
    c0 = _c0_order(data)
    valid_paths = _valid_paths(data)
    relation_score: dict[str, tuple[int, int]] = {}
    closure_score: dict[str, tuple[int, int]] = {}
    for candidate in data.candidates:
        candidate_paths = [path for path in valid_paths if candidate.skill_id in path.parent_skill_ids]
        communities = {community for path in candidate_paths for community in path.community_ids}
        requirements = {requirement for path in candidate_paths for requirement in path.requirement_ids}
        validation_count = sum(path.validation_present for path in candidate_paths)
        relation_score[candidate.skill_id] = (len(communities), len(candidate_paths))
        closure_score[candidate.skill_id] = (len(requirements), validation_count)

    def rank_from_scores(scores: dict[str, tuple[int, int]]) -> list[str]:
        by_id = data.candidate_by_id()
        return sorted(
            scores,
            key=lambda skill_id: (
                -scores[skill_id][0],
                -scores[skill_id][1],
                by_id[skill_id].c0_rank,
                skill_id,
            ),
        )

    rankings = [c0, rank_from_scores(relation_score), rank_from_scores(closure_score)]
    fused: dict[str, float] = {skill_id: 0.0 for skill_id in c0}
    for ranking in rankings:
        for index, skill_id in enumerate(ranking, start=1):
            fused[skill_id] += 1.0 / (k_rrf + index)
    selected = sorted(c0, key=lambda skill_id: (-fused[skill_id], c0.index(skill_id), skill_id))
    return tuple(selected[: data.top_k])


def select_hq(data: CompilerInput) -> tuple[str, ...]:
    valid_community_ids = {
        community.community_id
        for community in data.communities
        if community.status == "CERTIFIED"
    }
    valid_paths = _valid_paths(data)
    candidate_by_id = data.candidate_by_id()
    selected: list[str] = []
    covered: set[str] = set()
    remaining_communities = set(valid_community_ids)

    while remaining_communities and len(selected) < data.top_k:
        best: tuple[tuple[int, int, str], str, str, set[str]] | None = None
        for community_id in sorted(remaining_communities):
            for candidate in data.candidates:
                if candidate.skill_id in selected or community_id not in candidate.community_ids:
                    continue
                candidate_requirements = {
                    requirement_id
                    for path in valid_paths
                    if candidate.skill_id in path.parent_skill_ids
                    and community_id in path.community_ids
                    for requirement_id in path.requirement_ids
                }
                newly_covered = candidate_requirements - covered
                if not newly_covered:
                    continue
                key = (
                    len(newly_covered),
                    -candidate_by_id[candidate.skill_id].c0_rank,
                    candidate.skill_id,
                )
                record = (key, community_id, candidate.skill_id, newly_covered)
                if best is None or record[0] > best[0]:
                    best = record
        if best is None:
            break
        _, community_id, skill_id, newly_covered = best
        selected.append(skill_id)
        covered.update(newly_covered)
        remaining_communities.remove(community_id)
    return _fill_from_c0(data, selected)


def select_jg(data: CompilerInput) -> tuple[tuple[str, ...], tuple[str, ...]]:
    valid_paths = _valid_paths(data)
    paths_by_id = {path.path_id: path for path in valid_paths}
    requirement_by_id = {item.requirement_id: item for item in data.requirements}
    candidate_by_id = data.candidate_by_id()
    selected_skills: set[str] = set()
    selected_paths: set[str] = set()
    covered: set[str] = set()

    while True:
        best: tuple[tuple[int, int, int, int, int, str], tuple[SupportPath, ...]] | None = None
        for path in valid_paths:
            bundle = _path_bundle(path, paths_by_id)
            if bundle is None:
                continue
            bundle_path_ids = {item.path_id for item in bundle}
            if bundle_path_ids.issubset(selected_paths):
                continue
            if any(selected_paths.intersection(item.conflict_ids) for item in bundle):
                continue
            bundle_skills = {skill for item in bundle for skill in item.parent_skill_ids}
            new_skills = bundle_skills - selected_skills
            if len(selected_skills | bundle_skills) > data.top_k:
                continue
            bundle_requirements = {req for item in bundle for req in item.requirement_ids}
            newly_covered = bundle_requirements - covered
            new_critical = sum(
                requirement_by_id[req].criticality in {"HARD", "CORE"}
                for req in newly_covered
                if req in requirement_by_id
            )
            if not newly_covered and not new_skills:
                continue
            validation_gain = sum(item.validation_present for item in bundle if item.path_id not in selected_paths)
            c0_rank_sum = sum(candidate_by_id[skill].c0_rank for skill in new_skills)
            key = (
                new_critical,
                len(newly_covered),
                validation_gain,
                -len(new_skills),
                -c0_rank_sum,
                path.path_id,
            )
            record = (key, bundle)
            if best is None or record[0] > best[0]:
                best = record
        if best is None or best[0][0] <= 0:
            break
        bundle = best[1]
        selected_paths.update(item.path_id for item in bundle)
        selected_skills.update(skill for item in bundle for skill in item.parent_skill_ids)
        covered.update(req for item in bundle for req in item.requirement_ids)

    ordered_selected = sorted(
        selected_skills,
        key=lambda skill_id: (candidate_by_id[skill_id].c0_rank, skill_id),
    )
    final_skills = _fill_from_c0(data, ordered_selected)
    final_paths = tuple(
        path.path_id
        for path in closed_paths_for_selection(data, final_skills)
        if path.path_id in selected_paths
    )
    return final_skills, final_paths


def authorize_pc(data: CompilerInput, proposed_skill_ids: tuple[str, ...]) -> tuple[str, ...]:
    baseline = select_c0(data)
    baseline_closed = set(closed_requirements(data, baseline))
    proposed_closed = set(closed_requirements(data, proposed_skill_ids))
    if baseline_closed < proposed_closed:
        return proposed_skill_ids
    return baseline


def _ordered_paths(paths: Iterable[SupportPath]) -> list[SupportPath]:
    paths_by_id = {path.path_id: path for path in paths}
    emitted: set[str] = set()
    result: list[SupportPath] = []

    def emit(path: SupportPath) -> None:
        if path.path_id in emitted:
            return
        for dependency_id in path.prerequisite_path_ids:
            dependency = paths_by_id.get(dependency_id)
            if dependency is not None:
                emit(dependency)
        emitted.add(path.path_id)
        result.append(path)

    for path in sorted(paths_by_id.values(), key=lambda item: item.path_id):
        emit(path)
    return result


def render_common_guide(
    data: CompilerInput,
    selected_skill_ids: tuple[str, ...],
    preferred_path_ids: Iterable[str] = (),
) -> tuple[str, tuple[GuideLine, ...], tuple[str, ...]]:
    requirements = {item.requirement_id: item for item in data.requirements}
    subunits = {item.subunit_id: item for item in data.subunits}
    paths = closed_paths_for_selection(data, selected_skill_ids)
    preferred = set(preferred_path_ids)
    if preferred:
        paths = tuple(path for path in paths if path.path_id in preferred)
    ordered = _ordered_paths(paths)
    raw_lines: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "# Procedural Slice Guide",
            "generic_heading",
            ("generic_scaffold:hcsc_procedural_slice_guide_v1",),
        ),
        (
            "Use only the source-grounded routes below when they help the public task.",
            "generic_instruction",
            ("generic_scaffold:hcsc_procedural_slice_guide_v1",),
        ),
    ]
    for path in ordered:
        req_refs = tuple(
            provenance_id(requirements[req].provenance)
            for req in path.requirement_ids
            if req in requirements
        )
        raw_lines.append(
            (
                f"Requirement: {', '.join(path.requirement_ids)}",
                "requirement_route",
                req_refs,
            )
        )
        raw_lines.append(
            (
                f"Capability: {', '.join(path.community_ids) if path.community_ids else 'unclassified'}",
                "capability_route",
                tuple(path.provenance_ids) or req_refs,
            )
        )
        raw_lines.append(
            (
                f"Skills: {', '.join(path.parent_skill_ids)}",
                "selected_skill_route",
                tuple(path.provenance_ids) or req_refs,
            )
        )
        for subunit_id in path.subunit_ids:
            subunit = subunits[subunit_id]
            location = (
                f"{subunit.provenance.source_file}:"
                f"L{subunit.provenance.line_start}-L{subunit.provenance.line_end}"
            )
            raw_lines.append(
                (
                    f"Operation ({subunit.role}): {location}",
                    "subunit_route",
                    (provenance_id(subunit.provenance),),
                )
            )

    guide_lines = tuple(
        GuideLine(
            line_number=index,
            content=content,
            semantic_role=role,
            source_ids=source_ids,
        )
        for index, (content, role, source_ids) in enumerate(raw_lines, start=1)
    )
    guide_text = "\n".join(line.content for line in guide_lines) + "\n"
    return guide_text, guide_lines, tuple(path.path_id for path in ordered)


def compile_policy(data: CompilerInput, policy: str) -> CompilerOutput:
    decision_log: list[dict[str, object]] = []
    preferred_paths: tuple[str, ...] = ()
    if policy in {"c0", "c0_guide"}:
        selected = select_c0(data)
    elif policy == "rf":
        selected = select_rf(data)
    elif policy == "hq":
        selected = select_hq(data)
    elif policy in {"jg", "jg_pc"}:
        proposed, preferred_paths = select_jg(data)
        if policy == "jg_pc":
            selected = authorize_pc(data, proposed)
            if selected != proposed:
                preferred_paths = ()
                decision_log.append({"decision": "ABSTAIN_TO_C0", "reason": "NO_STRICT_PARETO_GAIN"})
            else:
                decision_log.append({"decision": "AUTHORIZE", "reason": "STRICT_CLOSURE_SUPERSET"})
        else:
            selected = proposed
    else:
        raise ValueError(f"Unknown HCSC policy: {policy}")

    if policy == "c0":
        guide_text = ""
        guide_lines: tuple[GuideLine, ...] = ()
        selected_path_ids: tuple[str, ...] = ()
    else:
        guide_text, guide_lines, selected_path_ids = render_common_guide(
            data,
            selected,
            preferred_path_ids=preferred_paths,
        )
    return CompilerOutput(
        task_id=data.task_id,
        policy=policy,
        selected_skill_ids=selected,
        selected_path_ids=selected_path_ids,
        closed_requirement_ids=closed_requirements(data, selected),
        guide_text=guide_text,
        guide_lines=guide_lines,
        decision_log=tuple(decision_log),
    )
