from __future__ import annotations
from typing import Any
from .mechanism_policies_v2 import MechanismDecision

def _preferred_guide_paths(
    decision: MechanismDecision,
    baseline_path_ids: list[str],
    binding_by_path: dict[str, dict[str, Any]],
) -> list[str]:
    if decision.decision == "ABSTAIN_TO_C0_BINDING":
        return list(baseline_path_ids)
    selected = set(decision.selected_skill_ids)
    result: list[str] = []
    seen_requirements: set[str] = set()
    for path_id in [*decision.preferred_path_ids, *baseline_path_ids]:
        binding = binding_by_path.get(path_id)
        if binding is None:
            continue
        skill_id = str(binding["capability_unit"]["skill_id"])
        requirement_id = str(binding["requirement"]["requirement_id"])
        if skill_id not in selected or requirement_id in seen_requirements:
            continue
        result.append(path_id)
        seen_requirements.add(requirement_id)
        if len(result) == 2:
            break
    if not result:
        raise ValueError("Authorized mechanism change has no renderable guide route")
    return result
