from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from .compiler import (
    _fill_from_c0,
    _valid_paths,
    closed_requirements,
    render_common_guide,
    select_c0,
    select_jg,
)
from .schema import (
    CandidateSkill,
    CommunityRecord,
    CompilerInput,
    CompilerOutput,
    SupportPath,
)


DUAL_POLICIES = ("c0", "c0_guide", "rf", "hq", "jg", "jg_pc")


def load_legacy_nomination_communities(
    repo_root: Path,
) -> tuple[tuple[CommunityRecord, ...], dict[str, tuple[str, ...]]]:
    """Load legacy L2 groups as nomination regions, never as semantic evidence."""
    nodes = json.loads((repo_root / "skill_nodes.json").read_text(encoding="utf-8"))
    mapping = json.loads((repo_root / "skill_l2_mapping.json").read_text(encoding="utf-8"))
    name_by_node = {str(row["id"]): str(row["name"]) for row in nodes}
    communities: list[CommunityRecord] = []
    memberships: dict[str, list[str]] = {}
    for community_id, row in sorted(mapping.items()):
        unknown = sorted(set(map(str, row.get("skills", []))) - set(name_by_node))
        if unknown:
            raise ValueError(f"Legacy community {community_id} has unknown skill nodes: {unknown}")
        members = tuple(name_by_node[str(value)] for value in row.get("skills", []))
        communities.append(
            CommunityRecord(
                community_id=str(community_id),
                member_skill_ids=members,
                status="CANDIDATE",
                construction_rule="legacy_l2_nomination_only_no_evidence_authority",
            )
        )
        for skill_id in members:
            memberships.setdefault(skill_id, []).append(str(community_id))
    return tuple(communities), {
        skill_id: tuple(sorted(community_ids))
        for skill_id, community_ids in memberships.items()
    }


def certified_foundation_subunits(
    profiles_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for profile in profiles_payload.get("profiles", []):
        if profile.get("status") != "CERTIFIED":
            continue
        support = list(map(str, profile.get("support_subunit_ids", [])))
        if len(support) != 1:
            continue
        result[support[0]] = profile
    return result


def certify_task_support_paths(
    paths: Iterable[SupportPath],
    profiles_payload: dict[str, Any],
) -> tuple[tuple[SupportPath, ...], tuple[dict[str, Any], ...]]:
    """Require both task entailment and independent executable-operation certification."""
    certified = certified_foundation_subunits(profiles_payload)
    accepted: list[SupportPath] = []
    audit: list[dict[str, Any]] = []
    for path in paths:
        reasons: list[str] = []
        if path.status != "CERTIFIED":
            reasons.append(f"task_path_status:{path.status}")
        if path.unresolved_dependencies:
            reasons.append("unresolved_dependencies")
        if path.contradicted:
            reasons.append("contradicted")
        if path.conflict_ids:
            reasons.append("conflict_ids_present")
        uncertified = sorted(set(path.subunit_ids) - set(certified))
        if uncertified:
            reasons.extend(f"foundation_operation_not_certified:{value}" for value in uncertified)
        if not reasons:
            accepted.append(path)
        audit.append(
            {
                "path_id": path.path_id,
                "requirement_ids": list(path.requirement_ids),
                "parent_skill_ids": list(path.parent_skill_ids),
                "subunit_ids": list(path.subunit_ids),
                "accepted": not reasons,
                "reasons": reasons,
            }
        )
    return tuple(accepted), tuple(audit)


def attach_nomination_communities(
    paths: Iterable[SupportPath],
    memberships: dict[str, tuple[str, ...]],
) -> tuple[SupportPath, ...]:
    result: list[SupportPath] = []
    for path in paths:
        community_ids = tuple(
            sorted(
                {
                    community_id
                    for skill_id in path.parent_skill_ids
                    for community_id in memberships.get(skill_id, ())
                }
            )
        )
        result.append(
            SupportPath(
                path_id=path.path_id,
                requirement_ids=path.requirement_ids,
                parent_skill_ids=path.parent_skill_ids,
                subunit_ids=path.subunit_ids,
                community_ids=community_ids,
                prerequisite_path_ids=path.prerequisite_path_ids,
                unresolved_dependencies=path.unresolved_dependencies,
                validation_present=path.validation_present,
                contradicted=path.contradicted,
                conflict_ids=path.conflict_ids,
                status=path.status,
                provenance_ids=path.provenance_ids,
            )
        )
    return tuple(result)


def select_rf_dual(data: CompilerInput, k_rrf: int = 60) -> tuple[str, ...]:
    """Fuse C0 with task support and executable-closure residuals only."""
    c0 = [
        item.skill_id
        for item in sorted(data.candidates, key=lambda item: (item.c0_rank, item.skill_id))
    ]
    valid_paths = _valid_paths(data)
    candidate_by_id = data.candidate_by_id()

    support_score: dict[str, tuple[int, int]] = {}
    closure_score: dict[str, tuple[int, int]] = {}
    for candidate in data.candidates:
        candidate_paths = [
            path for path in valid_paths if candidate.skill_id in path.parent_skill_ids
        ]
        support_score[candidate.skill_id] = (
            len({req for path in candidate_paths for req in path.requirement_ids}),
            len({subunit for path in candidate_paths for subunit in path.subunit_ids}),
        )
        closure_score[candidate.skill_id] = (
            sum(path.validation_present for path in candidate_paths),
            sum(not path.unresolved_dependencies for path in candidate_paths),
        )

    def ranked(scores: dict[str, tuple[int, int]]) -> list[str]:
        return sorted(
            scores,
            key=lambda skill_id: (
                -scores[skill_id][0],
                -scores[skill_id][1],
                candidate_by_id[skill_id].c0_rank,
                skill_id,
            ),
        )

    fused = {skill_id: 0.0 for skill_id in c0}
    for ranking in (c0, ranked(support_score), ranked(closure_score)):
        for rank, skill_id in enumerate(ranking, start=1):
            fused[skill_id] += 1.0 / (k_rrf + rank)
    selected = sorted(c0, key=lambda skill_id: (-fused[skill_id], c0.index(skill_id), skill_id))
    return tuple(selected[: data.top_k])


def select_hq_dual(data: CompilerInput) -> tuple[str, ...]:
    """Use legacy communities to nominate slots, with V2 paths as sole authorization."""
    valid_paths = _valid_paths(data)
    candidate_by_id = data.candidate_by_id()
    selected: list[str] = []
    covered: set[str] = set()
    available_communities = {community.community_id for community in data.communities}

    while available_communities and len(selected) < data.top_k:
        best: tuple[tuple[int, int, str], str, str, set[str]] | None = None
        for community_id in sorted(available_communities):
            for candidate in data.candidates:
                if candidate.skill_id in selected or community_id not in candidate.community_ids:
                    continue
                requirements = {
                    requirement_id
                    for path in valid_paths
                    if candidate.skill_id in path.parent_skill_ids
                    and community_id in path.community_ids
                    for requirement_id in path.requirement_ids
                }
                marginal = requirements - covered
                if not marginal:
                    continue
                key = (len(marginal), -candidate_by_id[candidate.skill_id].c0_rank, candidate.skill_id)
                record = (key, community_id, candidate.skill_id, marginal)
                if best is None or record[0] > best[0]:
                    best = record
        if best is None:
            break
        _, community_id, skill_id, marginal = best
        selected.append(skill_id)
        covered.update(marginal)
        available_communities.remove(community_id)
    return _fill_from_c0(data, selected)


def authorize_pc_dual(
    data: CompilerInput,
    proposed_skill_ids: tuple[str, ...],
    hard_veto_skill_ids: Iterable[str] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    baseline = select_c0(data)
    requirements = {item.requirement_id: item for item in data.requirements}
    baseline_closed = set(closed_requirements(data, baseline))
    proposed_closed = set(closed_requirements(data, proposed_skill_ids))
    baseline_critical = {
        value
        for value in baseline_closed
        if requirements[value].criticality in {"HARD", "CORE"}
    }
    proposed_critical = {
        value
        for value in proposed_closed
        if requirements[value].criticality in {"HARD", "CORE"}
    }
    incoming = set(proposed_skill_ids) - set(baseline)
    veto_hits = sorted(incoming.intersection(set(hard_veto_skill_ids)))
    checks = {
        "preserves_all_baseline_closure": baseline_closed.issubset(proposed_closed),
        "adds_critical_closure": bool(proposed_critical - baseline_critical),
        "hard_veto_skill_ids": veto_hits,
    }
    if (
        checks["preserves_all_baseline_closure"]
        and checks["adds_critical_closure"]
        and not veto_hits
    ):
        return proposed_skill_ids, {"decision": "AUTHORIZE", "checks": checks}
    return baseline, {"decision": "ABSTAIN_TO_C0", "checks": checks}


def compile_dual_policy(
    data: CompilerInput,
    policy: str,
    hard_veto_skill_ids: Iterable[str] = (),
) -> CompilerOutput:
    preferred_paths: tuple[str, ...] = ()
    decision_log: list[dict[str, Any]] = []
    if policy in {"c0", "c0_guide"}:
        selected = select_c0(data)
    elif policy == "rf":
        selected = select_rf_dual(data)
    elif policy == "hq":
        selected = select_hq_dual(data)
    elif policy in {"jg", "jg_pc"}:
        proposed, preferred_paths = select_jg(data)
        if policy == "jg_pc":
            selected, decision = authorize_pc_dual(data, proposed, hard_veto_skill_ids)
            decision_log.append(decision)
            if selected != proposed:
                preferred_paths = ()
        else:
            selected = proposed
    else:
        raise ValueError(f"Unknown dual-graph HCSC policy: {policy}")

    if policy == "c0":
        guide_text = ""
        guide_lines = ()
        selected_path_ids = ()
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


def output_payload(output: CompilerOutput) -> dict[str, Any]:
    return asdict(output)
