from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ._tokens import count_tokens

from .compiler import extract_public_requirements


GUIDE_VERSION = "hcsc-bounded-complete-capability-guide-v1"
DEFAULT_GUIDE_BUDGET = 256
DEFAULT_MAX_ROUTES = 2
CRITICALITY_ORDER = {"HARD": 0, "CORE": 1, "SUPPORT": 2}
REQUIREMENT_TYPE_ORDER = {
    "OUTPUT_PATH": 0,
    "ARTIFACT": 1,
    "FORMAT": 2,
    "VALIDATION": 3,
    "COMPLETION": 4,
}


def _line_ranges(spans: Iterable[dict[str, Any]]) -> str:
    ranges = sorted(
        {
            (int(span["line_start"]), int(span["line_end"]))
            for span in spans
        }
    )
    return ",".join(
        f"L{start}" if start == end else f"L{start}-L{end}"
        for start, end in ranges
    )


def _roles(spans: Iterable[dict[str, Any]]) -> str:
    return ",".join(sorted({str(span["role"]) for span in spans}))


def _operational_spans(unit: dict[str, Any]) -> list[dict[str, Any]]:
    ids = set(map(str, unit["operational_provenance_ids"]))
    spans = [
        span
        for span in unit["support_spans"]
        if str(span["evidence_id"]) in ids
    ]
    if {str(span["evidence_id"]) for span in spans} != ids:
        raise ValueError(f"Capability unit lost operational evidence: {unit['candidate_id']}")
    return sorted(
        spans,
        key=lambda row: (
            str(row["source_file"]),
            int(row["line_start"]),
            int(row["line_end"]),
            str(row["evidence_id"]),
        ),
    )


def _completion_spans(unit: dict[str, Any]) -> list[dict[str, Any]]:
    ids = set(map(str, unit["slot_evidence"]["completion_evidence_ids"]))
    return [
        span
        for span in unit["support_spans"]
        if str(span["evidence_id"]) in ids
    ]


def _route_sort_key(
    route: dict[str, Any],
    requirement_by_id: dict[str, Any],
    selected_rank: dict[str, int],
) -> tuple[Any, ...]:
    requirement_id = str(route["requirement_ids"][0])
    requirement = requirement_by_id[requirement_id]
    skill_id = str(route["parent_skill_ids"][0])
    return (
        CRITICALITY_ORDER.get(requirement.criticality, 9),
        REQUIREMENT_TYPE_ORDER.get(requirement.requirement_type, 9),
        selected_rank.get(skill_id, 99),
        requirement_id,
        str(route["path_id"]),
    )


def _diverse_route_order(
    routes: list[dict[str, Any]],
    requirement_by_id: dict[str, Any],
    selected_rank: dict[str, int],
) -> list[dict[str, Any]]:
    ordered = sorted(
        routes,
        key=lambda route: _route_sort_key(route, requirement_by_id, selected_rank),
    )
    result: list[dict[str, Any]] = []
    seen_requirements: set[str] = set()
    seen_skills: set[str] = set()
    for route in ordered:
        requirement_id = str(route["requirement_ids"][0])
        skill_id = str(route["parent_skill_ids"][0])
        if requirement_id in seen_requirements or skill_id in seen_skills:
            continue
        result.append(route)
        seen_requirements.add(requirement_id)
        seen_skills.add(skill_id)
    result.extend(route for route in ordered if route not in result)
    return result


def _route_lines(
    route_index: int,
    route: dict[str, Any],
    unit: dict[str, Any],
    requirement: Any,
    community_ids: tuple[str, ...],
) -> list[tuple[str, str, list[dict[str, str]]]]:
    skill_id = str(unit["skill_id"])
    operational = _operational_spans(unit)
    completion = _completion_spans(unit)
    mounted = f"/root/.agents/skills/{skill_id}/SKILL.md"
    requirement_source = (
        f"{requirement.provenance.source_file}:"
        f"L{requirement.provenance.line_start}-L{requirement.provenance.line_end}"
    )
    source_rows = [
        {
            "source_file": requirement.provenance.source_file,
            "source_type": requirement.provenance.source_type,
            "source_id": requirement.requirement_id,
        },
        *[
            {
                "source_file": str(span["source_file"]),
                "source_type": "selected_skill",
                "source_id": str(span["evidence_id"]),
            }
            for span in operational
        ],
    ]
    community = community_ids[0] if community_ids else "none"
    first = (
        f"{route_index}. [{requirement.criticality}/{requirement.requirement_type}] "
        f"`{requirement_source}` -> `{skill_id}` / `{unit['candidate_id']}` "
        f"(group `{community}`)."
    )
    second = (
        f"   Open `{mounted}:{_line_ranges(operational)}` "
        f"[{_roles(operational)}]; completion evidence `{_line_ranges(completion)}`."
    )
    resources = [
        f"/root/.agents/skills/{skill_id}/{row['relative_skill_path']}"
        for row in unit.get("resources", [])
    ]
    lines = [
        (first, "capability_route", source_rows),
        (second, "operation_route", source_rows),
    ]
    if resources:
        lines.append(
            (
                f"   Public package resource: `{resources[0]}`.",
                "resource_route",
                [
                    {
                        "source_file": str(unit["resources"][0]["source_file"]),
                        "source_type": "selected_skill",
                        "source_id": str(unit["resources"][0]["resource_id"]),
                    }
                ],
            )
        )
    return lines


def build_bounded_runtime_guide(
    *,
    repo_root: Path,
    task_id: str,
    selected_skill_ids: list[str],
    selected_path_ids: list[str],
    task_support_path: Path,
    capability_units_path: Path,
    skill_communities: dict[str, tuple[str, ...]],
    budget_tokens: int = DEFAULT_GUIDE_BUDGET,
    max_routes: int = DEFAULT_MAX_ROUTES,
) -> dict[str, Any]:
    if not selected_path_ids:
        return {
            "schema_version": GUIDE_VERSION,
            "task_id": task_id,
            "text": "",
            "tokens": 0,
            "route_count": 0,
            "selected_path_ids": [],
            "line_provenance": [],
            "budget_tokens": budget_tokens,
            "max_routes": max_routes,
        }
    support = json.loads(task_support_path.read_text(encoding="utf-8"))
    units_payload = json.loads(capability_units_path.read_text(encoding="utf-8"))
    path_by_id = {str(row["path_id"]): row for row in support["support_paths"]}
    unit_by_id = {
        str(row["candidate_id"]): row
        for row in units_payload["units"]
        if row.get("status") == "CERTIFIED"
    }
    requirements = extract_public_requirements(
        repo_root / "tasks" / task_id / "instruction.md", repo_root
    )
    requirement_by_id = {row.requirement_id: row for row in requirements}
    routes: list[dict[str, Any]] = []
    for path_id in selected_path_ids:
        route = path_by_id.get(path_id)
        if route is None:
            raise ValueError(f"Unknown selected HCSC path: {task_id}/{path_id}")
        unit = unit_by_id.get(str(route["capability_unit_id"]))
        if unit is None:
            raise ValueError(f"Uncertified selected HCSC unit: {task_id}/{path_id}")
        if str(unit["skill_id"]) not in selected_skill_ids:
            raise ValueError(f"HCSC route points outside selected skills: {task_id}/{path_id}")
        routes.append(route)
    selected_rank = {
        skill_id: rank for rank, skill_id in enumerate(selected_skill_ids, start=1)
    }
    ordered = _diverse_route_order(routes, requirement_by_id, selected_rank)
    raw_lines: list[tuple[str, str, list[dict[str, str]]]] = [
        (
            "## Procedural Slice Guide",
            "generic_heading",
            [
                {
                    "source_file": "generator:hcsc_runtime_guide_v1",
                    "source_type": "generic_scaffold",
                    "source_id": "generic_scaffold:hcsc_runtime_guide_v1",
                }
            ],
        ),
        (
            "Open the cited mounted lines before using a route; the original task instruction remains authoritative.",
            "generic_instruction",
            [
                {
                    "source_file": "generator:hcsc_runtime_guide_v1",
                    "source_type": "generic_scaffold",
                    "source_id": "generic_scaffold:hcsc_runtime_guide_v1",
                }
            ],
        ),
    ]
    accepted_paths: list[str] = []
    for route in ordered[:max_routes]:
        requirement_id = str(route["requirement_ids"][0])
        unit = unit_by_id[str(route["capability_unit_id"])]
        skill_id = str(unit["skill_id"])
        candidate = raw_lines + _route_lines(
            len(accepted_paths) + 1,
            route,
            unit,
            requirement_by_id[requirement_id],
            skill_communities.get(skill_id, ()),
        )
        text = "\n".join(line[0] for line in candidate) + "\n"
        if count_tokens(text)[0] > budget_tokens:
            continue
        raw_lines = candidate
        accepted_paths.append(str(route["path_id"]))
    if not accepted_paths:
        raise ValueError(f"No complete HCSC route fits the frozen guide budget: {task_id}")
    text = "\n".join(line[0] for line in raw_lines) + "\n"
    tokens = count_tokens(text)[0]
    provenance = [
        {
            "line": index,
            "content": content,
            "semantic_role": role,
            "sources": sources,
            "task_id": task_id,
            "destination_artifact": "PROCEDURAL_SLICE_GUIDE.md",
        }
        for index, (content, role, sources) in enumerate(raw_lines, start=1)
        if content.strip()
    ]
    return {
        "schema_version": GUIDE_VERSION,
        "task_id": task_id,
        "text": text,
        "tokens": tokens,
        "route_count": len(accepted_paths),
        "selected_path_ids": accepted_paths,
        "candidate_path_count": len(routes),
        "line_provenance": provenance,
        "budget_tokens": budget_tokens,
        "max_routes": max_routes,
    }
