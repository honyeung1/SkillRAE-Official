from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_PER_SKILL_PER_REQUIREMENT = 2
MAX_PER_REQUIREMENT = 10

def _unit_text(unit: dict[str, Any]) -> str:
    values = [
        str(unit.get("canonical_label", "")),
        *(str(span["exact_text"]) for span in unit.get("support_spans", [])),
        *(str(resource["relative_skill_path"]) for resource in unit.get("resources", [])),
    ]
    return "\n".join(value for value in values if value.strip())

def _nominate(
    requirements: tuple[Any, ...],
    units: list[dict[str, Any]],
) -> list[tuple[Any, dict[str, Any], float]]:
    if not requirements or not units:
        return []
    texts = [row.provenance.exact_text for row in requirements] + [_unit_text(row) for row in units]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
        scores = (matrix[: len(requirements)] @ matrix[len(requirements) :].T).toarray()
    except ValueError:
        scores = np.zeros((len(requirements), len(units)))
    nominations: list[tuple[Any, dict[str, Any], float]] = []
    for req_index, requirement in enumerate(requirements):
        skill_counts: dict[str, int] = {}
        selected = 0
        ranked = sorted(
            range(len(units)),
            key=lambda index: (
                -float(scores[req_index, index]),
                int(units[index]["c0_rank"]),
                str(units[index]["skill_id"]),
                str(units[index]["candidate_id"]),
            ),
        )
        for unit_index in ranked:
            unit = units[unit_index]
            score = float(scores[req_index, unit_index])
            if score <= 0.0:
                continue
            skill_id = str(unit["skill_id"])
            if skill_counts.get(skill_id, 0) >= MAX_PER_SKILL_PER_REQUIREMENT:
                continue
            nominations.append((requirement, unit, score))
            skill_counts[skill_id] = skill_counts.get(skill_id, 0) + 1
            selected += 1
            if selected >= MAX_PER_REQUIREMENT:
                break
    return nominations
