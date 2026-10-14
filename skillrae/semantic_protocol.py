from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from .compiler import extract_public_requirements
from .schema import GraphPackage, SubunitInstance, content_sha256, provenance_id


from .model_config import model_config

EXACT_MODEL = model_config().construction
PROTOCOL_VERSION = "hcsc-semantic-construction-v1"
FOUNDATION_BATCH_SIZE = 8
FOUNDATION_SUBUNITS_PER_SKILL = 14
TASK_SUBUNITS_PER_SKILL = 4
MAX_PARALLEL_CALLS = 4
MAX_FOUNDATION_GENERATION_TOKENS = 7000
MAX_FOUNDATION_VERIFICATION_TOKENS = 5000
MAX_PAIR_VERIFICATION_TOKENS = 5000
MAX_TASK_GENERATION_TOKENS = 6500
MAX_TASK_VERIFICATION_TOKENS = 5000


FOUNDATION_GENERATOR_SYSTEM = """
You construct a public-source procedural skill graph. Work only from the supplied
exact skill subunit spans. For each skill, identify one to six distinct executable
operation concepts that are genuinely present in those spans. Do not invent a
procedure, requirement, tool, artifact, dependency, or applicability condition.

Return one JSON object with this schema:
{
  "skills": [
    {
      "skill_id": "exact supplied skill_id",
      "concepts": [
        {
          "local_id": "c1",
          "canonical_label": "2-8 word operation label",
          "support_subunit_ids": ["exact supplied subunit id"],
          "action_verb": "concrete verb",
          "action_object": "concrete object",
          "tool_terms": ["terms copied exactly from support spans"],
          "input_terms": ["terms copied exactly from support spans"],
          "output_terms": ["terms copied exactly from support spans"],
          "validation_terms": ["terms copied exactly from support spans"],
          "dependencies": [
            {
              "requires_local_id": "another local concept id",
              "evidence_subunit_ids": ["exact supplied subunit id"]
            }
          ]
        }
      ]
    }
  ]
}

Rules:
- A concept must contain a concrete executable operation, not topical relevance.
- Use only supplied IDs. Every concept needs one to four support subunits.
- A dependency means the supported text makes the required operation a real
  prerequisite; document order, semantic similarity, and convenience are not enough.
- Empty arrays are valid. Prefer omission over unsupported semantic claims.
- Never mention tests, solutions, verifiers, rewards, trajectories, or hidden data.
""".strip()


FOUNDATION_VERIFIER_SYSTEM = """
You independently verify proposed public-source procedural graph concepts. You
receive the exact source spans and a proposal. Judge operational support, not topic
similarity. A proposal is CERTIFIED only when the cited spans contain the concrete
operation claimed. A dependency is CERTIFIED only when the cited text establishes a
real prerequisite; order or plausibility is insufficient.

Return one JSON object:
{
  "decisions": [
    {
      "skill_id": "...",
      "local_id": "...",
      "concept_status": "CERTIFIED|REJECTED|UNRESOLVED",
      "reason": "short evidence-grounded reason",
      "dependency_decisions": [
        {
          "requires_local_id": "...",
          "status": "CERTIFIED|REJECTED|UNRESOLVED",
          "reason": "..."
        }
      ]
    }
  ]
}

Do not repair or rewrite the proposal. Unknown evidence is UNRESOLVED, not negative.
""".strip()


PAIR_VERIFIER_SYSTEM = """
You verify candidate cross-skill operation equivalences. For each supplied pair,
return CERTIFIED only if both independently grounded concepts operationalize the
same executable operation with materially equivalent action semantics. Shared nouns,
topic similarity, or use of the same broad tool is insufficient.

Return exactly:
{
  "decisions": [
    {
      "pair_id": "exact supplied pair id",
      "status": "CERTIFIED|REJECTED|UNRESOLVED",
      "reason": "short evidence-grounded reason"
    }
  ]
}
""".strip()


TASK_GENERATOR_SYSTEM = """
You construct task-conditioned procedural support paths using only a public task
instruction and exact public selected-skill subunit spans. For each HARD or CORE
requirement, propose only source-grounded executable paths. A path may use multiple
skills when their operations form a real prerequisite chain. Do not infer private
evaluation requirements and do not generate missing domain knowledge.

Return one JSON object:
{
  "paths": [
    {
      "local_path_id": "p1",
      "requirement_ids": ["exact supplied requirement id"],
      "subunit_ids": ["exact supplied subunit id"],
      "parent_skill_ids": ["exact supplied skill id"],
      "prerequisite_local_path_ids": ["another proposed local path id"],
      "validation_present": true,
      "unresolved_dependencies": [],
      "contradicted": false,
      "conflict_local_path_ids": []
    }
  ]
}

Rules:
- Every path must provide a concrete operation for its cited public requirement.
- Validation is true only when a cited span gives a concrete public check.
- A path is contradicted only when the public requirement and cited skill text
  directly conflict. Unknown applicability is not contradiction.
- Prerequisites must be operational, not merely ordered prose.
- Empty output is valid and preferred to an unsupported path.
- Never use tests, solutions, verifier behavior, rewards, trajectories, or history.
""".strip()


TASK_VERIFIER_SYSTEM = """
You independently verify proposed task-conditioned procedural support paths. Work
only from the supplied public instruction and exact public skill spans. CERTIFIED
means the cited operations concretely support the cited requirement, all claimed
dependencies are evidenced, no unresolved prerequisite remains, and no public hard
constraint is violated. Topic relevance or a plausible plan is insufficient.

Return exactly:
{
  "decisions": [
    {
      "local_path_id": "exact proposed id",
      "status": "CERTIFIED|REJECTED|UNRESOLVED",
      "validation_supported": true,
      "dependency_status": "CERTIFIED|REJECTED|UNRESOLVED",
      "contradiction_status": "NONE|CERTIFIED|UNRESOLVED",
      "reason": "short evidence-grounded reason"
    }
  ]
}

Do not repair the proposal. Unknown evidence is UNRESOLVED, never a certified path.
""".strip()


_ROLE_PRIORITY = {
    "STEP": 0,
    "TOOL": 1,
    "VALIDATION": 2,
    "ARTIFACT": 3,
    "CONSTRAINT": 4,
    "REFERENCE": 5,
}


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_immutable_json(path: Path, value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Immutable artifact collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return sha256(serialized.encode("utf-8")).hexdigest()


def _source_row(subunit: SubunitInstance) -> dict[str, Any]:
    return {
        "subunit_id": subunit.subunit_id,
        "skill_id": subunit.skill_id,
        "role": subunit.role,
        "heading_path": list(subunit.heading_path),
        "source_file": subunit.provenance.source_file,
        "line_start": subunit.provenance.line_start,
        "line_end": subunit.provenance.line_end,
        "exact_text": subunit.provenance.exact_text,
        "source_sha256": subunit.provenance.source_sha256,
        "provenance_id": provenance_id(subunit.provenance),
    }


def select_foundation_subunits(
    subunits: Iterable[SubunitInstance],
    limit: int = FOUNDATION_SUBUNITS_PER_SKILL,
) -> tuple[SubunitInstance, ...]:
    eligible = [
        item
        for item in subunits
        if item.provenance.exact_text.strip() and len(item.provenance.exact_text) <= 1200
    ]
    ordered = sorted(
        eligible,
        key=lambda item: (
            _ROLE_PRIORITY[item.role],
            len(item.provenance.exact_text),
            item.provenance.line_start,
            item.subunit_id,
        ),
    )
    selected: list[SubunitInstance] = []
    seen_headings: set[tuple[str, ...]] = set()
    for item in ordered:
        if item.heading_path in seen_headings:
            continue
        selected.append(item)
        seen_headings.add(item.heading_path)
        if len(selected) == limit:
            return tuple(selected)
    for item in ordered:
        if item not in selected:
            selected.append(item)
        if len(selected) == limit:
            break
    return tuple(selected)


def select_task_subunits(
    instruction_text: str,
    by_skill: dict[str, tuple[SubunitInstance, ...]],
    skill_ids: tuple[str, ...],
    per_skill: int = TASK_SUBUNITS_PER_SKILL,
) -> tuple[SubunitInstance, ...]:
    selected: list[SubunitInstance] = []
    for skill_id in skill_ids:
        items = [
            item
            for item in by_skill.get(skill_id, ())
            if item.provenance.exact_text.strip() and len(item.provenance.exact_text) <= 1600
        ]
        if not items:
            continue
        corpus = [instruction_text] + [item.provenance.exact_text for item in items]
        try:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
            scores = (matrix[1:] @ matrix[0].T).toarray().reshape(-1)
        except ValueError:
            scores = [0.0] * len(items)
        ranked = sorted(
            zip(items, scores),
            key=lambda row: (
                -float(row[1]),
                _ROLE_PRIORITY[row[0].role],
                row[0].provenance.line_start,
                row[0].subunit_id,
            ),
        )
        chosen = [item for item, _ in ranked[:per_skill]]
        if not any(item.role == "VALIDATION" for item in chosen):
            validator = next((item for item, _ in ranked if item.role == "VALIDATION"), None)
            if validator is not None:
                chosen[-1:] = [validator]
        selected.extend(chosen)
    return tuple(selected)


def _foundation_batch_payload(
    graph: GraphPackage,
    skill_ids: tuple[str, ...],
    subunits_by_skill: dict[str, tuple[SubunitInstance, ...]],
) -> dict[str, Any]:
    skills = {item.skill_id: item for item in graph.skills}
    return {
        "schema_version": "hcsc-foundation-input-v1",
        "graph_package_sha256": graph.package_sha256,
        "public_sources_only": True,
        "skills": [
            {
                "skill_id": skill_id,
                "skill_md": skills[skill_id].skill_md,
                "skill_md_sha256": skills[skill_id].skill_md_sha256,
                "tree_sha256": skills[skill_id].tree_sha256,
                "subunits": [
                    _source_row(item)
                    for item in select_foundation_subunits(subunits_by_skill[skill_id])
                ],
            }
            for skill_id in skill_ids
        ],
    }


def _task_payload(
    repo_root: Path,
    graph: GraphPackage,
    ranking_row: dict[str, Any],
    subunits_by_skill: dict[str, tuple[SubunitInstance, ...]],
) -> dict[str, Any]:
    task_id = str(ranking_row["task_id"])
    instruction_path = repo_root / "tasks" / task_id / "instruction.md"
    instruction_text = instruction_path.read_text(encoding="utf-8")
    if _sha256_file(instruction_path) != ranking_row["instruction_sha256"]:
        raise ValueError(f"Instruction drift for {task_id}")
    skill_ids = tuple(str(value) for value in ranking_row["c0_ranking"][:20])
    requirements = extract_public_requirements(instruction_path, repo_root)
    selected_subunits = select_task_subunits(
        instruction_text,
        subunits_by_skill,
        skill_ids,
    )
    return {
        "schema_version": "hcsc-task-support-input-v1",
        "task_id": task_id,
        "graph_package_sha256": graph.package_sha256,
        "public_sources_only": True,
        "instruction": {
            "source_file": instruction_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(instruction_path),
            "exact_text": instruction_text,
        },
        "requirements": [
            {
                "requirement_id": item.requirement_id,
                "requirement_type": item.requirement_type,
                "criticality": item.criticality,
                "provenance_id": provenance_id(item.provenance),
                "source_file": item.provenance.source_file,
                "line_start": item.provenance.line_start,
                "line_end": item.provenance.line_end,
                "exact_text": item.provenance.exact_text,
            }
            for item in requirements
        ],
        "c0_top20": list(skill_ids),
        "subunits": [_source_row(item) for item in selected_subunits],
    }


def freeze_semantic_protocol(
    repo_root: Path,
    graph: GraphPackage,
    rankings_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    rankings_path = rankings_path.resolve()
    output_root = output_root.resolve()
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    if rankings.get("labels_rewards_or_trajectories_loaded") is not False:
        raise ValueError("Rankings are not certified outcome-blind")

    by_skill_lists: dict[str, list[SubunitInstance]] = {}
    for subunit in graph.subunits:
        by_skill_lists.setdefault(subunit.skill_id, []).append(subunit)
    by_skill = {
        skill_id: tuple(sorted(items, key=lambda item: item.subunit_id))
        for skill_id, items in by_skill_lists.items()
    }

    skill_ids = tuple(sorted(item.skill_id for item in graph.skills))
    foundation_rows: list[dict[str, Any]] = []
    for start in range(0, len(skill_ids), FOUNDATION_BATCH_SIZE):
        batch_skill_ids = skill_ids[start : start + FOUNDATION_BATCH_SIZE]
        payload = _foundation_batch_payload(graph, batch_skill_ids, by_skill)
        batch_id = f"foundation_{start // FOUNDATION_BATCH_SIZE + 1:03d}"
        path = output_root / "frozen_inputs" / "foundation" / f"{batch_id}.json"
        file_sha = _write_immutable_json(path, payload)
        foundation_rows.append(
            {
                "batch_id": batch_id,
                "skill_ids": list(batch_skill_ids),
                "input_path": str(path),
                "input_sha256": content_sha256(payload),
                "file_sha256": file_sha,
            }
        )

    task_rows: list[dict[str, Any]] = []
    for row in rankings["task_rows"]:
        payload = _task_payload(repo_root, graph, row, by_skill)
        task_id = payload["task_id"]
        path = output_root / "frozen_inputs" / "tasks" / f"{task_id}.json"
        file_sha = _write_immutable_json(path, payload)
        task_rows.append(
            {
                "task_id": task_id,
                "input_path": str(path),
                "input_sha256": content_sha256(payload),
                "file_sha256": file_sha,
                "candidate_skill_count": len(payload["c0_top20"]),
                "candidate_subunit_count": len(payload["subunits"]),
                "requirement_count": len(payload["requirements"]),
            }
        )

    prompts = {
        "foundation_generator": FOUNDATION_GENERATOR_SYSTEM,
        "foundation_verifier": FOUNDATION_VERIFIER_SYSTEM,
        "pair_verifier": PAIR_VERIFIER_SYSTEM,
        "task_generator": TASK_GENERATOR_SYSTEM,
        "task_verifier": TASK_VERIFIER_SYSTEM,
    }
    call_budget = {
        "provider_canary": 1,
        "foundation_generation": len(foundation_rows),
        "foundation_verification": len(foundation_rows),
        "cross_skill_pair_verification_max": 24,
        "task_generation": len(task_rows),
        "task_verification": len(task_rows),
    }
    call_budget["total_max"] = sum(call_budget.values())
    protocol = {
        "schema_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "model": EXACT_MODEL,
        "fallback_allowed": False,
        "public_sources_only": True,
        "reward_or_trajectory_loaded": False,
        "graph_package_sha256": graph.package_sha256,
        "rankings_path": str(rankings_path),
        "rankings_sha256": _sha256_file(rankings_path),
        "foundation_batch_size": FOUNDATION_BATCH_SIZE,
        "foundation_subunits_per_skill": FOUNDATION_SUBUNITS_PER_SKILL,
        "task_subunits_per_skill": TASK_SUBUNITS_PER_SKILL,
        "parallelism": MAX_PARALLEL_CALLS,
        "transport_retry_policy": {
            "max_attempts": 2,
            "retryable_status_codes": [408, 429, 500, 502, 503, 504],
            "same_frozen_request": True,
            "semantic_or_schema_retry": False,
        },
        "max_output_tokens": {
            "foundation_generation": MAX_FOUNDATION_GENERATION_TOKENS,
            "foundation_verification": MAX_FOUNDATION_VERIFICATION_TOKENS,
            "pair_verification": MAX_PAIR_VERIFICATION_TOKENS,
            "task_generation": MAX_TASK_GENERATION_TOKENS,
            "task_verification": MAX_TASK_VERIFICATION_TOKENS,
        },
        "prompts": {
            name: {"text": text, "sha256": sha256(text.encode("utf-8")).hexdigest()}
            for name, text in prompts.items()
        },
        "call_budget": call_budget,
        "foundation_batches": foundation_rows,
        "task_inputs": task_rows,
        "forbidden_sources": [
            "tests/",
            "solution/",
            "verifier",
            "reward",
            "trajectory",
            "curated membership",
        ],
    }
    protocol["protocol_sha256"] = content_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    path = output_root / "semantic_protocol.json"
    _write_immutable_json(path, protocol)
    protocol["protocol_path"] = str(path)
    return protocol


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_VERSION or payload.get("status") != "FROZEN":
        raise ValueError("Semantic construction protocol is not frozen")
    if payload.get("model") != EXACT_MODEL or payload.get("fallback_allowed") is not False:
        raise ValueError("Semantic construction model/fallback drift")
    expected = content_sha256(
        {key: value for key, value in payload.items() if key != "protocol_sha256"}
    )
    if payload.get("protocol_sha256") != expected:
        raise ValueError("Semantic construction protocol hash mismatch")
    return payload
