from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from .compiler import closed_paths_for_selection, closed_requirements, select_c0
from .schema import CompilerInput, SupportPath


CRITICALITIES = {"HARD", "CORE"}


@dataclass(frozen=True)
class MechanismDecision:
    selected_skill_ids: tuple[str, ...]
    preferred_path_ids: tuple[str, ...]
    decision: str
    audit: dict[str, Any]


def _requirements(data: CompilerInput) -> dict[str, Any]:
    return {row.requirement_id: row for row in data.requirements}


def _critical_ids(data: CompilerInput, values: Iterable[str]) -> set[str]:
    by_id = _requirements(data)
    return {
        value
        for value in values
        if value in by_id and by_id[value].criticality in CRITICALITIES
    }


def _validation_requirement_ids(
    data: CompilerInput, selected_skill_ids: Iterable[str]
) -> set[str]:
    return {
        requirement_id
        for path in closed_paths_for_selection(data, selected_skill_ids)
        if path.validation_present
        for requirement_id in path.requirement_ids
    }


def _new_critical_paths(
    data: CompilerInput,
    selected_skill_ids: tuple[str, ...],
    baseline_closed: set[str],
    incoming_skill_ids: set[str],
) -> tuple[str, ...]:
    by_id = _requirements(data)
    candidates: list[SupportPath] = []
    for path in closed_paths_for_selection(data, selected_skill_ids):
        if not incoming_skill_ids.intersection(path.parent_skill_ids):
            continue
        if not any(
            requirement_id not in baseline_closed
            and by_id.get(requirement_id) is not None
            and by_id[requirement_id].criticality in CRITICALITIES
            for requirement_id in path.requirement_ids
        ):
            continue
        candidates.append(path)
    candidates.sort(
        key=lambda path: (
            -int(path.validation_present),
            len(path.parent_skill_ids),
            path.path_id,
        )
    )
    result: list[str] = []
    seen_requirements: set[str] = set()
    for path in candidates:
        new_requirements = set(path.requirement_ids) - baseline_closed
        if not new_requirements - seen_requirements:
            continue
        result.append(path.path_id)
        seen_requirements.update(new_requirements)
    return tuple(result)


def select_hq_residual(
    data: CompilerInput,
    *,
    max_swaps: int = 1,
) -> MechanismDecision:
    """Allocate one residual capability-region slot without losing critical closure."""

    if max_swaps != 1:
        raise ValueError("HQ residual v2 is frozen to exactly one possible swap")
    baseline = select_c0(data)
    baseline_closed = set(closed_requirements(data, baseline))
    baseline_critical = _critical_ids(data, baseline_closed)
    candidate_by_id = data.candidate_by_id()
    baseline_set = set(baseline)
    tail = [
        row.skill_id
        for row in sorted(data.candidates, key=lambda row: (row.c0_rank, row.skill_id))
        if row.skill_id not in baseline_set
    ]
    proposals: list[tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]] = []
    for incoming in tail:
        incoming_communities = set(candidate_by_id[incoming].community_ids)
        if not incoming_communities:
            continue
        for dropped in baseline:
            proposed = tuple(
                incoming if skill_id == dropped else skill_id for skill_id in baseline
            )
            proposed_closed = set(closed_requirements(data, proposed))
            proposed_critical = _critical_ids(data, proposed_closed)
            if not baseline_critical.issubset(proposed_critical):
                continue
            added_critical = proposed_critical - baseline_critical
            if not added_critical:
                continue
            remaining_communities = {
                community_id
                for skill_id in proposed
                if skill_id != incoming
                for community_id in candidate_by_id[skill_id].community_ids
            }
            community_gain = incoming_communities - remaining_communities
            if not community_gain:
                continue
            lost_noncritical = baseline_closed - proposed_closed
            added_all = proposed_closed - baseline_closed
            key = (
                len(added_critical),
                len(added_all),
                len(community_gain),
                -len(lost_noncritical),
                -candidate_by_id[incoming].c0_rank,
                candidate_by_id[dropped].c0_rank,
                incoming,
                dropped,
            )
            proposals.append(
                (
                    key,
                    proposed,
                    {
                        "incoming_skill_id": incoming,
                        "dropped_skill_id": dropped,
                        "added_critical_requirement_ids": sorted(added_critical),
                        "added_requirement_ids": sorted(added_all),
                        "lost_noncritical_requirement_ids": sorted(lost_noncritical),
                        "new_community_ids": sorted(community_gain),
                    },
                )
            )
    if not proposals:
        return MechanismDecision(
            selected_skill_ids=baseline,
            preferred_path_ids=(),
            decision="ABSTAIN_TO_C0_BINDING",
            audit={
                "baseline_closed_requirement_ids": sorted(baseline_closed),
                "baseline_critical_requirement_ids": sorted(baseline_critical),
                "qualified_swap_count": 0,
            },
        )
    _, selected, audit = max(proposals, key=lambda row: row[0])
    incoming = set(selected) - baseline_set
    preferred = _new_critical_paths(
        data, selected, baseline_closed, incoming
    )
    if not preferred:
        raise ValueError("HQ selected a strict critical gain without a supporting path")
    return MechanismDecision(
        selected_skill_ids=selected,
        preferred_path_ids=preferred,
        decision="AUTHORIZE_ONE_COMMUNITY_RESIDUAL_SWAP",
        audit={
            **audit,
            "baseline_closed_requirement_ids": sorted(baseline_closed),
            "baseline_critical_requirement_ids": sorted(baseline_critical),
            "qualified_swap_count": len(proposals),
        },
    )


def _community_matching(
    data: CompilerInput,
    selected_skill_ids: tuple[str, ...],
    allowed_requirement_ids: set[str] | None = None,
) -> tuple[int, dict[str, str]]:
    edges: dict[str, set[str]] = {}
    for path in closed_paths_for_selection(data, selected_skill_ids):
        for requirement_id in path.requirement_ids:
            if allowed_requirement_ids is not None and requirement_id not in allowed_requirement_ids:
                continue
            edges.setdefault(requirement_id, set()).update(path.community_ids)
    requirement_by_community: dict[str, str] = {}

    def assign(requirement_id: str, visited: set[str]) -> bool:
        for community_id in sorted(edges.get(requirement_id, ())):
            if community_id in visited:
                continue
            visited.add(community_id)
            previous = requirement_by_community.get(community_id)
            if previous is None or assign(previous, visited):
                requirement_by_community[community_id] = requirement_id
                return True
        return False

    for requirement_id in sorted(edges):
        assign(requirement_id, set())
    matching = {
        requirement_id: community_id
        for community_id, requirement_id in requirement_by_community.items()
    }
    return len(matching), matching


def select_hq_community_exact(data: CompilerInput) -> MechanismDecision:
    """Exact set-level community quota allocation under critical-closure safety."""

    baseline = select_c0(data)
    baseline_set = set(baseline)
    baseline_closed = set(closed_requirements(data, baseline))
    baseline_critical = _critical_ids(data, baseline_closed)
    candidate_by_id = data.candidate_by_id()
    ordered_candidates = tuple(
        row.skill_id
        for row in sorted(data.candidates, key=lambda row: (row.c0_rank, row.skill_id))
    )
    qualified: list[tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]] = []
    for unordered in combinations(ordered_candidates, data.top_k):
        proposed = tuple(sorted(unordered, key=lambda value: candidate_by_id[value].c0_rank))
        proposed_closed = set(closed_requirements(data, proposed))
        proposed_critical = _critical_ids(data, proposed_closed)
        if not baseline_critical.issubset(proposed_critical):
            continue
        added_critical = proposed_critical - baseline_critical
        if not added_critical:
            continue
        critical_quota, critical_assignment = _community_matching(
            data, proposed, proposed_critical
        )
        all_quota, all_assignment = _community_matching(data, proposed)
        if critical_quota == 0:
            continue
        incoming = set(proposed) - baseline_set
        changes = len(incoming)
        validation = _validation_requirement_ids(data, proposed)
        rank_sum = sum(candidate_by_id[value].c0_rank for value in proposed)
        key = (
            len(proposed_critical),
            critical_quota,
            all_quota,
            len(proposed_closed),
            len(validation),
            -changes,
            -rank_sum,
            tuple(-candidate_by_id[value].c0_rank for value in proposed),
        )
        qualified.append(
            (
                key,
                proposed,
                {
                    "incoming_skill_ids": sorted(incoming),
                    "dropped_skill_ids": sorted(baseline_set - set(proposed)),
                    "added_critical_requirement_ids": sorted(added_critical),
                    "added_requirement_ids": sorted(proposed_closed - baseline_closed),
                    "lost_noncritical_requirement_ids": sorted(
                        baseline_closed - proposed_closed
                    ),
                    "critical_community_quota": critical_quota,
                    "all_requirement_community_quota": all_quota,
                    "critical_requirement_community_assignment": critical_assignment,
                    "all_requirement_community_assignment": all_assignment,
                    "changed_slot_count": changes,
                },
            )
        )
    if not qualified:
        return MechanismDecision(
            selected_skill_ids=baseline,
            preferred_path_ids=(),
            decision="ABSTAIN_TO_C0_BINDING",
            audit={
                "baseline_closed_requirement_ids": sorted(baseline_closed),
                "baseline_critical_requirement_ids": sorted(baseline_critical),
                "qualified_set_count": 0,
            },
        )
    _, selected, audit = max(qualified, key=lambda row: row[0])
    incoming = set(selected) - baseline_set
    preferred = _new_critical_paths(data, selected, baseline_closed, incoming)
    if not preferred:
        raise ValueError("HQ exact selected a strict critical gain without a supporting path")
    return MechanismDecision(
        selected_skill_ids=selected,
        preferred_path_ids=preferred,
        decision="AUTHORIZE_EXACT_COMMUNITY_QUOTA_SET",
        audit={
            **audit,
            "baseline_closed_requirement_ids": sorted(baseline_closed),
            "baseline_critical_requirement_ids": sorted(baseline_critical),
            "qualified_set_count": len(qualified),
        },
    )


def select_jg_pc_exact(data: CompilerInput) -> MechanismDecision:
    """Exact bounded set search under the frozen Pareto-closure authorization."""

    baseline = select_c0(data)
    baseline_set = set(baseline)
    baseline_closed = set(closed_requirements(data, baseline))
    baseline_critical = _critical_ids(data, baseline_closed)
    candidate_by_id = data.candidate_by_id()
    ordered_candidates = tuple(
        row.skill_id
        for row in sorted(data.candidates, key=lambda row: (row.c0_rank, row.skill_id))
    )
    if len(ordered_candidates) < data.top_k:
        raise ValueError("JG-PC exact search has fewer candidates than top_k")
    qualified: list[tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]] = []
    for unordered in combinations(ordered_candidates, data.top_k):
        proposed = tuple(sorted(unordered, key=lambda value: candidate_by_id[value].c0_rank))
        proposed_closed = set(closed_requirements(data, proposed))
        if not baseline_closed.issubset(proposed_closed):
            continue
        proposed_critical = _critical_ids(data, proposed_closed)
        added_critical = proposed_critical - baseline_critical
        if not added_critical:
            continue
        incoming = set(proposed) - baseline_set
        changes = len(incoming)
        validation = _validation_requirement_ids(data, proposed)
        rank_sum = sum(candidate_by_id[value].c0_rank for value in proposed)
        key = (
            len(proposed_critical),
            len(proposed_closed),
            len(validation),
            -changes,
            -rank_sum,
            tuple(-candidate_by_id[value].c0_rank for value in proposed),
        )
        qualified.append(
            (
                key,
                proposed,
                {
                    "incoming_skill_ids": sorted(incoming),
                    "dropped_skill_ids": sorted(baseline_set - set(proposed)),
                    "added_critical_requirement_ids": sorted(added_critical),
                    "added_requirement_ids": sorted(proposed_closed - baseline_closed),
                    "validation_requirement_ids": sorted(validation),
                    "changed_slot_count": changes,
                },
            )
        )
    if not qualified:
        return MechanismDecision(
            selected_skill_ids=baseline,
            preferred_path_ids=(),
            decision="ABSTAIN_TO_C0_BINDING",
            audit={
                "baseline_closed_requirement_ids": sorted(baseline_closed),
                "baseline_critical_requirement_ids": sorted(baseline_critical),
                "qualified_set_count": 0,
            },
        )
    _, selected, audit = max(qualified, key=lambda row: row[0])
    incoming = set(selected) - baseline_set
    preferred = _new_critical_paths(
        data, selected, baseline_closed, incoming
    )
    if not preferred:
        raise ValueError("JG-PC selected a strict critical gain without a supporting path")
    return MechanismDecision(
        selected_skill_ids=selected,
        preferred_path_ids=preferred,
        decision="AUTHORIZE_EXACT_PARETO_CLOSURE_SET",
        audit={
            **audit,
            "baseline_closed_requirement_ids": sorted(baseline_closed),
            "baseline_critical_requirement_ids": sorted(baseline_critical),
            "qualified_set_count": len(qualified),
        },
    )
