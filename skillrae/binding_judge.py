from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .compiler import extract_public_requirements
from .schema import content_sha256, stable_id
from .semantic_runner import _call_json as _semantic_call_json
from .model_config import model_config


EXACT_MODEL = model_config().binding
CONSTRUCTION_MODEL = model_config().construction


def _call_json(*args, **kwargs):
    return _semantic_call_json(*args, model=EXACT_MODEL, **kwargs)

MAX_PARALLEL_CALLS = 4
MAX_JUDGE_TOKENS = 14_000
MAX_VERIFIER_TOKENS = 8_000
MAX_BINDINGS_PER_CALL = 6

DIMENSIONS = (
    "action",
    "input_scope",
    "temporal_basis",
    "units",
    "filters",
    "output_contract",
    "validation",
)
DIMENSION_STATUSES = {"SUPPORTED", "UNRESOLVED", "CONTRADICTED", "NOT_APPLICABLE"}
JUDGE_VERDICTS = {"AUTHORIZED", "PARTIAL", "AMBIGUOUS", "CONTRADICTED"}
JUDGE_REASON_CODES = {
    "DIRECT_COMPLETE",
    "MISSING_SCOPE",
    "MISSING_TIME_BASIS",
    "MISSING_UNITS",
    "MISSING_FILTER",
    "MISSING_OUTPUT",
    "MISSING_VALIDATION",
    "INAPPLICABLE_ACTION",
    "EXPLICIT_CONTRADICTION",
    "OTHER_PUBLIC_GAP",
}
VERIFIER_REASON_CODES = {
    "COMPLETE_PUBLIC_BINDING",
    "ACTION_NOT_EXECUTABLE",
    "MATERIAL_SCOPE_GAP",
    "MATERIAL_TIME_GAP",
    "MATERIAL_UNIT_GAP",
    "MATERIAL_FILTER_GAP",
    "MATERIAL_OUTPUT_GAP",
    "MATERIAL_VALIDATION_GAP",
    "EVIDENCE_MISMATCH",
    "OTHER_PUBLIC_GAP",
}


JUDGE_SYSTEM = """You are the SkillRAE Requirement-OCU Semantic Binding Judge.
Return one JSON object only. Never judge topical similarity or reward detail.

For every supplied binding candidate, decide whether the exact public capability-unit
spans provide an executable method that is semantically applicable to the complete
meaning of the cited public task requirement. Check material scope, time basis, units,
filters, output semantics, and validation whenever they matter. Do not fill missing
conditions with world knowledge or assumptions. A useful but incomplete procedure is
PARTIAL, not AUTHORIZED. Missing public evidence is AMBIGUOUS. Use CONTRADICTED only
for an explicit incompatibility.

AUTHORIZED requires:
- a concrete executable action in the supplied OCU spans;
- no unresolved or contradicted material requirement dimension;
- an evidence-grounded completion or validation path when the requirement needs one;
- all cited evidence IDs to come from the supplied binding record.

For each binding return exactly:
{
  "binding_id": "...",
  "verdict": "AUTHORIZED|PARTIAL|AMBIGUOUS|CONTRADICTED",
  "reason_code": "DIRECT_COMPLETE|MISSING_SCOPE|MISSING_TIME_BASIS|MISSING_UNITS|MISSING_FILTER|MISSING_OUTPUT|MISSING_VALIDATION|INAPPLICABLE_ACTION|EXPLICIT_CONTRADICTION|OTHER_PUBLIC_GAP",
  "dimensions": {
    "action": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "input_scope": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "temporal_basis": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "units": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "filters": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "output_contract": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE",
    "validation": "SUPPORTED|UNRESOLVED|CONTRADICTED|NOT_APPLICABLE"
  },
  "operation_evidence_ids": ["..."],
  "completion_evidence_ids": ["..."],
  "applicability_evidence_ids": ["..."]
}

Return {"binding_decisions":[...]} with exactly one decision for every input
binding_id and no additional binding IDs. IDs are evidence pointers, not prose.
"""


VERIFIER_SYSTEM = """You are an independent verifier of proposed SkillRAE
Requirement-OCU bindings. Return one JSON object only. Evaluate operational and
semantic sufficiency, not topical relevance. The task instruction and OCU evidence
are the only authorities. Do not infer hidden benchmark expectations.

For every proposed AUTHORIZED binding, return exactly:
{
  "binding_id": "...",
  "verdict": "PASS|FAIL",
  "reason_code": "COMPLETE_PUBLIC_BINDING|ACTION_NOT_EXECUTABLE|MATERIAL_SCOPE_GAP|MATERIAL_TIME_GAP|MATERIAL_UNIT_GAP|MATERIAL_FILTER_GAP|MATERIAL_OUTPUT_GAP|MATERIAL_VALIDATION_GAP|EVIDENCE_MISMATCH|OTHER_PUBLIC_GAP",
  "failed_dimensions": ["action|input_scope|temporal_basis|units|filters|output_contract|validation"],
  "evidence_ids": ["..."]
}

PASS only when the proposed binding is directly executable and no material part of
the complete cited requirement is unsupported or contradicted. More detail is not
automatically sufficient. Return {"binding_verifications":[...]} with exactly one
row for every proposed binding and no extras.
"""


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_immutable_json(path: Path, value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Immutable binding artifact collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return sha256(serialized.encode("utf-8")).hexdigest()


def _forbidden_path(path: str) -> bool:
    parts = {part.lower() for part in Path(path.replace("\\", "/")).parts}
    return bool(parts.intersection({"tests", "solution"}))


def _public_asset_manifest(repo_root: Path, task_id: str) -> list[dict[str, Any]]:
    environment = repo_root / "tasks" / task_id / "environment"
    rows: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in environment.rglob("*") if candidate.is_file()):
        relative_environment = path.relative_to(environment)
        if relative_environment.parts and relative_environment.parts[0] == "skills":
            continue
        relative_repo = path.relative_to(repo_root).as_posix()
        if _forbidden_path(relative_repo):
            raise ValueError(f"Forbidden public-asset candidate: {relative_repo}")
        rows.append(
            {
                "source_file": relative_repo,
                "source_type": "public_task_asset",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _requirement_payload(requirement: Any) -> dict[str, Any]:
    return {
        "requirement_id": requirement.requirement_id,
        "requirement_type": requirement.requirement_type,
        "criticality": requirement.criticality,
        "provenance": asdict(requirement.provenance),
    }


def _unit_payload(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "capability_unit_id": str(unit["candidate_id"]),
        "skill_id": str(unit["skill_id"]),
        "canonical_label": str(unit["canonical_label"]),
        "heading_path": list(map(str, unit.get("heading_path", []))),
        "external_prerequisite": str(unit.get("external_prerequisite", "NONE")),
        "roles_present": list(map(str, unit.get("roles_present", []))),
        "slot_evidence": unit.get("slot_evidence", {}),
        "support_spans": unit.get("support_spans", []),
        "resources": unit.get("resources", []),
    }


def _validate_unit_sources(repo_root: Path, unit: dict[str, Any]) -> None:
    evidence_ids: set[str] = set()
    for span in unit.get("support_spans", []):
        source_file = str(span.get("source_file", ""))
        if _forbidden_path(source_file):
            raise ValueError(f"Forbidden OCU source: {source_file}")
        source = (repo_root / source_file).resolve()
        source.relative_to(repo_root.resolve())
        if not source.is_file() or sha256_file(source) != span.get("source_sha256"):
            raise ValueError(f"OCU source hash drift: {source_file}")
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        start = int(span["line_start"])
        end = int(span["line_end"])
        if "".join(lines[start - 1 : end]) != str(span["exact_text"]):
            raise ValueError(f"OCU exact span drift: {source_file}:{start}")
        evidence_id = str(span["evidence_id"])
        if not evidence_id or evidence_id in evidence_ids:
            raise ValueError(f"Duplicate/missing OCU evidence ID: {evidence_id}")
        evidence_ids.add(evidence_id)


def build_protocol(
    *,
    repo_root: Path,
    source_gate_path: Path,
    source_protocol_path: Path,
    capability_units_path: Path,
    output_root: Path,
) -> Path:
    repo_root = repo_root.resolve()
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    units_payload = json.loads(capability_units_path.read_text(encoding="utf-8"))
    if source_gate.get("status") != "PASS" or source_protocol.get("status") != "FROZEN":
        raise ValueError("Binding source HCSC artifacts are not frozen PASS")
    if units_payload.get("status") != "PASS" or units_payload.get("model") != CONSTRUCTION_MODEL:
        raise ValueError("Capability units must have PASS status and match the configured construction model")
    unit_by_id = {
        str(row["candidate_id"]): row
        for row in units_payload.get("units", [])
        if row.get("status") == "CERTIFIED"
    }
    for unit in unit_by_id.values():
        _validate_unit_sources(repo_root, unit)

    task_inputs: list[dict[str, Any]] = []
    total_bindings = 0
    for gate_row in source_gate["task_rows"]:
        task_id = str(gate_row["task_id"])
        instruction_path = repo_root / "tasks" / task_id / "instruction.md"
        requirements = extract_public_requirements(instruction_path, repo_root)
        requirement_by_id = {row.requirement_id: row for row in requirements}
        support_path = source_protocol_path.parent / "task_support" / f"{task_id}.json"
        support = json.loads(support_path.read_text(encoding="utf-8"))
        accepted_path_ids = {
            str(row["path_id"])
            for row in gate_row.get("path_certification_audit", [])
            if row.get("accepted") is True
        }
        bindings: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for path in support.get("support_paths", []):
            path_id = str(path["path_id"])
            if path_id not in accepted_path_ids:
                continue
            requirement_ids = list(map(str, path.get("requirement_ids", [])))
            if len(requirement_ids) != 1:
                raise ValueError(f"{task_id}/{path_id}: binding path is not unary")
            requirement_id = requirement_ids[0]
            unit_id = str(path["capability_unit_id"])
            if requirement_id not in requirement_by_id or unit_id not in unit_by_id:
                raise ValueError(f"{task_id}/{path_id}: unknown requirement or OCU")
            pair = (requirement_id, unit_id)
            if pair in seen_pairs:
                raise ValueError(f"{task_id}: duplicate requirement-OCU pair {pair}")
            seen_pairs.add(pair)
            unit = unit_by_id[unit_id]
            if tuple(map(str, path["parent_skill_ids"])) != (str(unit["skill_id"]),):
                raise ValueError(f"{task_id}/{path_id}: path/OCU parent mismatch")
            binding_id = stable_id("bind", [task_id, path_id, requirement_id, unit_id])
            bindings.append(
                {
                    "binding_id": binding_id,
                    "binding_alias": f"B{len(bindings) + 1:03d}",
                    "path_id": path_id,
                    "requirement": _requirement_payload(requirement_by_id[requirement_id]),
                    "capability_unit": _unit_payload(unit),
                }
            )
        input_payload = {
            "schema_version": "skillrae-binding-judge-input-v1",
            "task_id": task_id,
            "instruction": {
                "source_file": instruction_path.relative_to(repo_root).as_posix(),
                "source_type": "instruction",
                "sha256": sha256_file(instruction_path),
                "exact_text": instruction_path.read_text(encoding="utf-8"),
            },
            "current5_skill_ids": list(map(str, gate_row["outputs"]["c0"]["selected_skill_ids"])),
            "c0_top20_skill_ids": list(map(str, gate_row["c0_top20"])),
            "public_asset_manifest": _public_asset_manifest(repo_root, task_id),
            "binding_candidates": bindings,
            "forbidden_sources": [
                "tests/",
                "solution/",
                "verifier/reward/trajectory/oracle/ground-truth artifacts",
            ],
            "reward_or_trajectory_loaded": False,
        }
        input_path = output_root / "inputs" / f"{task_id}.json"
        _write_immutable_json(input_path, input_payload)
        task_inputs.append(
            {
                "task_id": task_id,
                "input_path": str(input_path),
                "input_file_sha256": sha256_file(input_path),
                "input_payload_sha256": content_sha256(input_payload),
                "binding_count": len(bindings),
                "batches": [
                    {
                        "batch_id": f"{task_id}__b{index // MAX_BINDINGS_PER_CALL + 1:02d}",
                        "binding_aliases": [
                            row["binding_alias"]
                            for row in bindings[index : index + MAX_BINDINGS_PER_CALL]
                        ],
                    }
                    for index in range(0, len(bindings), MAX_BINDINGS_PER_CALL)
                ],
            }
        )
        total_bindings += len(bindings)

    protocol = {
        "schema_version": "skillrae-requirement-ocu-binding-protocol-v2-batched-envelope",
        "status": "FROZEN",
        "model": EXACT_MODEL,
        "public_sources_only": True,
        "reward_or_trajectory_loaded": False,
        "source_gate_path": str(source_gate_path),
        "source_gate_sha256": sha256_file(source_gate_path),
        "source_protocol_path": str(source_protocol_path),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "capability_units_path": str(capability_units_path),
        "capability_units_sha256": sha256_file(capability_units_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "judge_prompt_sha256": sha256(JUDGE_SYSTEM.encode("utf-8")).hexdigest(),
        "verifier_prompt_sha256": sha256(VERIFIER_SYSTEM.encode("utf-8")).hexdigest(),
        "authorization_rule": (
            "judge AUTHORIZED with all material dimensions supported/not-applicable, "
            "then independent verifier PASS; all other states fail closed"
        ),
        "request_envelope": (
            "same semantic prompt and candidate records; deterministic batches of at most "
            f"{MAX_BINDINGS_PER_CALL} bindings use short aliases to prevent JSON omission/ID-copy errors"
        ),
        "task_count": len(task_inputs),
        "binding_count": total_bindings,
        "task_inputs": task_inputs,
    }
    protocol["protocol_sha256"] = content_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    output = output_root / "binding_protocol.json"
    _write_immutable_json(output, protocol)
    return output


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = content_sha256(
        {key: value for key, value in payload.items() if key != "protocol_sha256"}
    )
    if payload.get("status") != "FROZEN" or payload.get("model") != EXACT_MODEL:
        raise ValueError("Binding protocol is not frozen exact-model")
    if payload.get("protocol_sha256") != expected:
        raise ValueError("Binding protocol hash drift")
    if payload.get("reward_or_trajectory_loaded") is not False:
        raise ValueError("Binding protocol is not outcome blind")
    implementation = Path(str(payload.get("implementation_path", "")))
    if not implementation.is_file() or sha256_file(implementation) != payload.get(
        "implementation_sha256"
    ):
        raise ValueError("Binding implementation hash drift")
    return payload


def provider_canary(repo_root: Path, protocol_path: Path, output_root: Path) -> Path:
    protocol = _load_protocol(protocol_path)
    output = output_root / "provider_canary.json"
    artifact = _call_json(
        repo_root,
        "binding:provider_canary",
        "Return JSON only. Follow the user instruction exactly.",
        {
            "schema_version": "skillrae-binding-provider-canary-v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "instruction": "Return exactly the JSON object {\"status\":\"OK\"}.",
        },
        32,
        output,
    )
    if artifact["output"] != {"status": "OK"}:
        raise ValueError("Binding model canary returned unexpected content")
    return output


def _binding_index(input_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["binding_id"]): row for row in input_payload.get("binding_candidates", [])
    }


def _allowed_evidence(binding: dict[str, Any]) -> set[str]:
    return {
        str(binding["requirement"]["requirement_id"]),
        *(
            str(span["evidence_id"])
            for span in binding["capability_unit"].get("support_spans", [])
        ),
    }


def _validate_judge_output(
    input_payload: dict[str, Any], output: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = _binding_index(input_payload)
    rows = output.get("binding_decisions")
    if not isinstance(rows, list):
        raise ValueError("Judge output lacks binding_decisions list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Judge decision is not an object")
        binding_id = str(row.get("binding_id", ""))
        if binding_id not in expected or binding_id in indexed:
            raise ValueError(f"Judge emitted unknown/duplicate binding: {binding_id}")
        verdict = str(row.get("verdict", ""))
        dimensions = row.get("dimensions")
        if (
            verdict not in JUDGE_VERDICTS
            or row.get("reason_code") not in JUDGE_REASON_CODES
            or not isinstance(dimensions, dict)
        ):
            raise ValueError(f"Malformed judge verdict: {binding_id}")
        if set(dimensions) != set(DIMENSIONS) or any(
            str(dimensions[key]) not in DIMENSION_STATUSES for key in DIMENSIONS
        ):
            raise ValueError(f"Malformed semantic dimensions: {binding_id}")
        allowed = _allowed_evidence(expected[binding_id])
        for field in (
            "operation_evidence_ids",
            "completion_evidence_ids",
            "applicability_evidence_ids",
        ):
            values = row.get(field)
            if not isinstance(values, list) or any(str(value) not in allowed for value in values):
                raise ValueError(f"Invalid judge evidence pointers: {binding_id}/{field}")
        if verdict == "AUTHORIZED":
            if dimensions["action"] != "SUPPORTED":
                raise ValueError(f"Authorized binding lacks supported action: {binding_id}")
            if any(value in {"UNRESOLVED", "CONTRADICTED"} for value in dimensions.values()):
                raise ValueError(f"Authorized binding has open semantic dimension: {binding_id}")
            if not row["operation_evidence_ids"]:
                raise ValueError(f"Authorized binding lacks operation evidence: {binding_id}")
        indexed[binding_id] = row
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        raise ValueError(f"Judge omitted binding decisions: {missing[:5]}")
    return [indexed[binding_id] for binding_id in expected]


def _run_judge_task(repo_root: Path, task: dict[str, Any], output_root: Path) -> dict[str, Any]:
    input_path = Path(str(task["input_path"]))
    if sha256_file(input_path) != task["input_file_sha256"]:
        raise ValueError(f"Binding input file drift: {task['task_id']}")
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    if content_sha256(input_payload) != task["input_payload_sha256"]:
        raise ValueError(f"Binding input payload drift: {task['task_id']}")
    all_bindings = list(input_payload["binding_candidates"])
    if not all_bindings:
        return {
            "task_id": task["task_id"],
            "judge_artifact_path": None,
            "judge_artifact_sha256": None,
            "binding_decisions": [],
        }
    by_alias = {str(row["binding_alias"]): row for row in all_bindings}
    artifact_paths: list[str] = []
    artifact_hashes: list[str] = []
    decisions: list[dict[str, Any]] = []
    for batch in task["batches"]:
        aliases = list(map(str, batch["binding_aliases"]))
        batch_bindings = []
        alias_to_id: dict[str, str] = {}
        for alias in aliases:
            source = dict(by_alias[alias])
            actual_id = str(source.pop("binding_id"))
            source.pop("binding_alias", None)
            source["binding_id"] = alias
            alias_to_id[alias] = actual_id
            batch_bindings.append(source)
        batch_payload = {
            **{key: value for key, value in input_payload.items() if key != "binding_candidates"},
            "binding_candidates": batch_bindings,
            "batch_id": batch["batch_id"],
        }
        artifact_path = output_root / "calls" / "judge" / f"{batch['batch_id']}.json"
        artifact = _call_json(
            repo_root,
            f"binding:{batch['batch_id']}:judge",
            JUDGE_SYSTEM,
            batch_payload,
            MAX_JUDGE_TOKENS,
            artifact_path,
        )
        batch_decisions = _validate_judge_output(batch_payload, artifact["output"])
        for decision in batch_decisions:
            mapped = dict(decision)
            mapped["binding_alias"] = str(decision["binding_id"])
            mapped["binding_id"] = alias_to_id[str(decision["binding_id"])]
            decisions.append(mapped)
        artifact_paths.append(str(artifact_path))
        artifact_hashes.append(sha256_file(artifact_path))
    expected_ids = [str(row["binding_id"]) for row in all_bindings]
    indexed = {str(row["binding_id"]): row for row in decisions}
    if set(indexed) != set(expected_ids):
        raise ValueError(f"Batched judge coverage mismatch: {task['task_id']}")
    return {
        "task_id": task["task_id"],
        "judge_artifact_paths": artifact_paths,
        "judge_artifact_sha256s": artifact_hashes,
        "binding_decisions": [indexed[binding_id] for binding_id in expected_ids],
    }


def run_judges(repo_root: Path, protocol_path: Path, output_root: Path) -> Path:
    protocol = _load_protocol(protocol_path)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = {
            pool.submit(_run_judge_task, repo_root, task, output_root): task
            for task in protocol["task_inputs"]
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "task_id": str(task["task_id"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    payload = {
        "schema_version": "skillrae-binding-judge-summary-v1",
        "status": "PASS" if len(rows) == protocol["task_count"] and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "protocol_sha256": protocol["protocol_sha256"],
        "task_count": len(rows),
        "binding_count": sum(len(row["binding_decisions"]) for row in rows),
        "authorized_proposals": sum(
            decision["verdict"] == "AUTHORIZED"
            for row in rows
            for decision in row["binding_decisions"]
        ),
        "rows": sorted(rows, key=lambda row: row["task_id"]),
        "errors": sorted(errors, key=lambda row: row["task_id"]),
        "reward_or_trajectory_loaded": False,
    }
    payload["summary_sha256"] = content_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )
    output = output_root / "binding_judge_summary.json"
    _write_immutable_json(output, payload)
    return output


def _validate_verifier_output(
    proposed: dict[str, dict[str, Any]], output: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = output.get("binding_verifications")
    if not isinstance(rows, list):
        raise ValueError("Verifier output lacks binding_verifications list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Verifier row is not an object")
        binding_id = str(row.get("binding_id", ""))
        if binding_id not in proposed or binding_id in indexed:
            raise ValueError(f"Verifier emitted unknown/duplicate binding: {binding_id}")
        if row.get("verdict") not in {"PASS", "FAIL"}:
            raise ValueError(f"Verifier emitted invalid verdict: {binding_id}")
        if row.get("reason_code") not in VERIFIER_REASON_CODES:
            raise ValueError(f"Verifier emitted invalid reason code: {binding_id}")
        failed = row.get("failed_dimensions")
        if not isinstance(failed, list) or any(str(value) not in DIMENSIONS for value in failed):
            raise ValueError(f"Verifier emitted invalid dimensions: {binding_id}")
        evidence = row.get("evidence_ids")
        if not isinstance(evidence, list) or any(
            str(value) not in proposed[binding_id]["allowed_evidence_ids"] for value in evidence
        ):
            raise ValueError(f"Verifier emitted invalid evidence: {binding_id}")
        indexed[binding_id] = row
    if set(indexed) != set(proposed):
        raise ValueError("Verifier did not cover every proposed binding")
    return [indexed[binding_id] for binding_id in proposed]


def _run_verifier_task(
    repo_root: Path,
    task: dict[str, Any],
    judge_row: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    input_payload = json.loads(Path(str(task["input_path"])).read_text(encoding="utf-8"))
    bindings = _binding_index(input_payload)
    proposed_rows = [
        row for row in judge_row["binding_decisions"] if row["verdict"] == "AUTHORIZED"
    ]
    if not proposed_rows:
        return {
            "task_id": task["task_id"],
            "verifier_artifact_path": None,
            "verifier_artifact_sha256": None,
            "binding_verifications": [],
        }
    artifact_paths: list[str] = []
    artifact_hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    for start in range(0, len(proposed_rows), MAX_BINDINGS_PER_CALL):
        batch_rows = proposed_rows[start : start + MAX_BINDINGS_PER_CALL]
        proposed: dict[str, dict[str, Any]] = {}
        alias_to_id: dict[str, str] = {}
        verifier_bindings: list[dict[str, Any]] = []
        for offset, decision in enumerate(batch_rows, start=1):
            binding_id = str(decision["binding_id"])
            alias = f"V{offset:03d}"
            source = dict(bindings[binding_id])
            source.pop("binding_alias", None)
            source["binding_id"] = alias
            allowed = sorted(_allowed_evidence(bindings[binding_id]))
            proposed[alias] = {"allowed_evidence_ids": set(allowed)}
            alias_to_id[alias] = binding_id
            mapped_judge = dict(decision)
            mapped_judge["binding_id"] = alias
            verifier_bindings.append(
                {
                    **source,
                    "judge_decision": mapped_judge,
                    "allowed_evidence_ids": allowed,
                }
            )
        batch_id = f"{task['task_id']}__b{start // MAX_BINDINGS_PER_CALL + 1:02d}"
        verifier_input = {
            "schema_version": "skillrae-binding-verifier-input-v2-batched-envelope",
            "task_id": task["task_id"],
            "batch_id": batch_id,
            "instruction": input_payload["instruction"],
            "public_asset_manifest": input_payload["public_asset_manifest"],
            "proposed_bindings": verifier_bindings,
            "reward_or_trajectory_loaded": False,
        }
        artifact_path = output_root / "calls" / "verifier" / f"{batch_id}.json"
        artifact = _call_json(
            repo_root,
            f"binding:{batch_id}:verify",
            VERIFIER_SYSTEM,
            verifier_input,
            MAX_VERIFIER_TOKENS,
            artifact_path,
        )
        batch_verified = _validate_verifier_output(proposed, artifact["output"])
        for verified in batch_verified:
            mapped = dict(verified)
            mapped["binding_alias"] = str(verified["binding_id"])
            mapped["binding_id"] = alias_to_id[str(verified["binding_id"])]
            rows.append(mapped)
        artifact_paths.append(str(artifact_path))
        artifact_hashes.append(sha256_file(artifact_path))
    expected_ids = [str(row["binding_id"]) for row in proposed_rows]
    indexed = {str(row["binding_id"]): row for row in rows}
    if set(indexed) != set(expected_ids):
        raise ValueError(f"Batched verifier coverage mismatch: {task['task_id']}")
    return {
        "task_id": task["task_id"],
        "verifier_artifact_paths": artifact_paths,
        "verifier_artifact_sha256s": artifact_hashes,
        "binding_verifications": [indexed[binding_id] for binding_id in expected_ids],
    }


def run_verifiers(
    repo_root: Path,
    protocol_path: Path,
    judge_summary_path: Path,
    output_root: Path,
) -> Path:
    protocol = _load_protocol(protocol_path)
    judge = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    if judge.get("status") != "PASS" or judge.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Binding judge summary is not sealed PASS")
    judge_by_task = {str(row["task_id"]): row for row in judge["rows"]}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = {
            pool.submit(
                _run_verifier_task,
                repo_root,
                task,
                judge_by_task[str(task["task_id"])],
                output_root,
            ): task
            for task in protocol["task_inputs"]
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "task_id": str(task["task_id"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    payload = {
        "schema_version": "skillrae-binding-verifier-summary-v1",
        "status": "PASS" if len(rows) == protocol["task_count"] and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "protocol_sha256": protocol["protocol_sha256"],
        "judge_summary_sha256": sha256_file(judge_summary_path),
        "task_count": len(rows),
        "verified_pass_count": sum(
            item["verdict"] == "PASS"
            for row in rows
            for item in row["binding_verifications"]
        ),
        "rows": sorted(rows, key=lambda row: row["task_id"]),
        "errors": sorted(errors, key=lambda row: row["task_id"]),
        "reward_or_trajectory_loaded": False,
    }
    payload["summary_sha256"] = content_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )
    output = output_root / "binding_verifier_summary.json"
    _write_immutable_json(output, payload)
    return output


def normalize_bindings(
    protocol_path: Path,
    judge_summary_path: Path,
    verifier_summary_path: Path,
    output_root: Path,
) -> Path:
    protocol = _load_protocol(protocol_path)
    judge = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    verifier = json.loads(verifier_summary_path.read_text(encoding="utf-8"))
    if judge.get("status") != "PASS" or verifier.get("status") != "PASS":
        raise ValueError("Binding semantic stages are not PASS")
    judge_by_task = {str(row["task_id"]): row for row in judge["rows"]}
    verifier_by_task = {str(row["task_id"]): row for row in verifier["rows"]}
    task_rows: list[dict[str, Any]] = []
    for task in protocol["task_inputs"]:
        task_id = str(task["task_id"])
        input_payload = json.loads(Path(str(task["input_path"])).read_text(encoding="utf-8"))
        binding_by_id = _binding_index(input_payload)
        judge_rows = {
            str(row["binding_id"]): row for row in judge_by_task[task_id]["binding_decisions"]
        }
        verifier_rows = {
            str(row["binding_id"]): row
            for row in verifier_by_task[task_id]["binding_verifications"]
        }
        decisions: list[dict[str, Any]] = []
        authorized_path_ids: list[str] = []
        for binding_id, binding in binding_by_id.items():
            judged = judge_rows[binding_id]
            verified = verifier_rows.get(binding_id)
            authorized = judged["verdict"] == "AUTHORIZED" and verified is not None and verified["verdict"] == "PASS"
            if authorized:
                final_status = "AUTHORIZED"
                authorized_path_ids.append(str(binding["path_id"]))
            elif judged["verdict"] == "AUTHORIZED":
                final_status = "VERIFIER_REJECTED"
            else:
                final_status = str(judged["verdict"])
            decisions.append(
                {
                    "binding_id": binding_id,
                    "path_id": str(binding["path_id"]),
                    "requirement_id": str(binding["requirement"]["requirement_id"]),
                    "capability_unit_id": str(binding["capability_unit"]["capability_unit_id"]),
                    "skill_id": str(binding["capability_unit"]["skill_id"]),
                    "final_status": final_status,
                    "judge_decision": judged,
                    "verifier_decision": verified,
                    "provenance_complete": True,
                }
            )
        task_payload = {
            "schema_version": "skillrae-requirement-ocu-bindings-v1",
            "status": "PASS",
            "task_id": task_id,
            "input_payload_sha256": task["input_payload_sha256"],
            "public_sources_only": True,
            "reward_or_trajectory_loaded": False,
            "authorized_path_ids": sorted(authorized_path_ids),
            "decisions": decisions,
        }
        task_payload["normalized_sha256"] = content_sha256(
            {key: value for key, value in task_payload.items() if key != "normalized_sha256"}
        )
        task_path = output_root / "bindings" / f"{task_id}.json"
        _write_immutable_json(task_path, task_payload)
        task_rows.append(
            {
                "task_id": task_id,
                "binding_path": str(task_path),
                "binding_file_sha256": sha256_file(task_path),
                "candidate_count": len(decisions),
                "authorized_count": len(authorized_path_ids),
                "authorized_path_ids": sorted(authorized_path_ids),
            }
        )
    summary = {
        "schema_version": "skillrae-binding-normalization-summary-v1",
        "status": "PASS",
        "model": EXACT_MODEL,
        "protocol_sha256": protocol["protocol_sha256"],
        "judge_summary_sha256": sha256_file(judge_summary_path),
        "verifier_summary_sha256": sha256_file(verifier_summary_path),
        "task_count": len(task_rows),
        "candidate_count": sum(row["candidate_count"] for row in task_rows),
        "authorized_count": sum(row["authorized_count"] for row in task_rows),
        "tasks_with_authorized_bindings": sum(bool(row["authorized_count"]) for row in task_rows),
        "task_rows": task_rows,
        "public_sources_only": True,
        "reward_or_trajectory_loaded": False,
    }
    summary["summary_sha256"] = content_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    output = output_root / "binding_normalization_summary.json"
    _write_immutable_json(output, summary)
    return output


def authorized_paths(binding_summary_path: Path) -> dict[str, set[str]]:
    payload = json.loads(binding_summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("reward_or_trajectory_loaded") is not False:
        raise ValueError("Binding normalization is not clean PASS")
    return {
        str(row["task_id"]): set(map(str, row["authorized_path_ids"]))
        for row in payload["task_rows"]
    }
