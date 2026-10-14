from __future__ import annotations

import concurrent.futures
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from .schema import content_sha256
from .semantic_protocol import (
    EXACT_MODEL,
    FOUNDATION_GENERATOR_SYSTEM,
    FOUNDATION_VERIFIER_SYSTEM,
    MAX_FOUNDATION_GENERATION_TOKENS,
    MAX_FOUNDATION_VERIFICATION_TOKENS,
    MAX_PARALLEL_CALLS,
    MAX_PAIR_VERIFICATION_TOKENS,
    MAX_TASK_GENERATION_TOKENS,
    MAX_TASK_VERIFICATION_TOKENS,
    TASK_GENERATOR_SYSTEM,
    TASK_VERIFIER_SYSTEM,
    PAIR_VERIFIER_SYSTEM,
    load_protocol,
)


DEFAULT_BASE_URL = ""
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_TRANSPORT_ATTEMPTS = 2


class SemanticCallError(RuntimeError):
    pass


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _write_immutable_json(path: Path, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Immutable semantic artifact collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _load_api_env(repo_root: Path) -> None:
    # Release adapter: credentials are explicit, never read from a private checkout.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SemanticCallError("Set OPENAI_API_KEY in the environment")
    if not os.environ.get("OPENAI_BASE_URL"):
        raise SemanticCallError("Set OPENAI_BASE_URL to a Responses-compatible endpoint")


def api_client(repo_root: Path) -> tuple[OpenAI, str]:
    _load_api_env(repo_root)
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=300),
    )
    return client, urlsplit(base_url).netloc


def response_text(response: object) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)
    values: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                values.append(str(value))
    return "".join(values)


def _retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


def _call_json(
    repo_root: Path,
    call_id: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    max_output_tokens: int,
    artifact_path: Path,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    model = model or EXACT_MODEL
    input_sha = content_sha256(user_payload)
    prompt_sha = _sha256_text(system_prompt)
    if artifact_path.is_file():
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (existing.get("call_id") != call_id or existing.get("status") != "FROZEN"
                or existing.get("requested_model") != model
                or existing.get("input_sha256") != input_sha
                or existing.get("prompt_sha256") != prompt_sha):
            raise SemanticCallError(f"Invalid existing checkpoint: {artifact_path}")
        return existing

    user_text = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    input_sha = content_sha256(user_payload)
    prompt_sha = _sha256_text(system_prompt)
    last_error: Exception | None = None
    for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
        client, provider_host = api_client(repo_root)
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                text={"format": {"type": "json_object"}},
                max_output_tokens=max_output_tokens,
                timeout=300,
            )
            raw = response_text(response).strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise SemanticCallError(f"{call_id}: response is not a JSON object")
            reported_model = str(getattr(response, "model", ""))
            if reported_model and reported_model != model:
                raise SemanticCallError(
                    f"{call_id}: provider model mismatch {reported_model!r} != {model!r}"
                )
            artifact = {
                "schema_version": "hcsc-semantic-call-v1",
                "status": "FROZEN",
                "call_id": call_id,
                "requested_model": model,
                "reported_model": reported_model or model,
                "fallback_used": False,
                "provider_host": provider_host,
                "public_sources_only": True,
                "reward_or_trajectory_loaded": False,
                "input_sha256": input_sha,
                "prompt_sha256": prompt_sha,
                "raw_response_sha256": _sha256_text(raw),
                "response_id": str(getattr(response, "id", "")),
                "transport_attempt": attempt,
                "output": parsed,
            }
            artifact["artifact_sha256"] = content_sha256(
                {key: value for key, value in artifact.items() if key != "artifact_sha256"}
            )
            _write_immutable_json(artifact_path, artifact)
            return artifact
        except (json.JSONDecodeError, SemanticCallError):
            raise
        except Exception as exc:  # provider SDK has multiple transport subclasses
            last_error = exc
            if attempt >= MAX_TRANSPORT_ATTEMPTS or not _retryable(exc):
                break
            time.sleep(5 * attempt)
    raise SemanticCallError(f"{call_id}: provider transport failed: {type(last_error).__name__}")


def provider_canary(repo_root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    payload = {
        "schema_version": "hcsc-provider-canary-v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "instruction": "Return exactly the JSON object {\"status\":\"OK\"}.",
    }
    artifact = _call_json(
        repo_root,
        f"provider_canary:{EXACT_MODEL}",
        "Follow the user instruction exactly. Return JSON only.",
        payload,
        32,
        output_path,
    )
    if artifact["output"] != {"status": "OK"}:
        raise SemanticCallError("Construction model canary returned unexpected content")
    return artifact


def _support_texts(input_payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["subunit_id"]): str(item["exact_text"])
        for skill in input_payload.get("skills", [])
        for item in skill.get("subunits", [])
    }


def _validate_foundation_proposal(
    input_payload: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[tuple[str, str]]]:
    expected_skills = {str(item["skill_id"]): item for item in input_payload["skills"]}
    rows = proposal.get("skills")
    if not isinstance(rows, list):
        raise SemanticCallError("Foundation proposal missing skills list")
    observed_skills: set[str] = set()
    concepts: dict[tuple[str, str], dict[str, Any]] = {}
    deterministically_rejected: set[tuple[str, str]] = set()
    texts = _support_texts(input_payload)
    allowed_subunits = set(texts)
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticCallError("Invalid foundation skill row")
        skill_id = str(row.get("skill_id", ""))
        if skill_id not in expected_skills or skill_id in observed_skills:
            raise SemanticCallError(f"Unexpected/duplicate foundation skill: {skill_id}")
        observed_skills.add(skill_id)
        local_ids: set[str] = set()
        skill_concepts = row.get("concepts")
        if not isinstance(skill_concepts, list) or len(skill_concepts) > 6:
            raise SemanticCallError(f"{skill_id}: invalid concept list")
        for concept in skill_concepts:
            if not isinstance(concept, dict):
                raise SemanticCallError(f"{skill_id}: invalid concept")
            local_id = str(concept.get("local_id", ""))
            if not local_id or local_id in local_ids:
                raise SemanticCallError(f"{skill_id}: duplicate/missing concept local_id")
            local_ids.add(local_id)
            supports = concept.get("support_subunit_ids")
            if not isinstance(supports, list) or not 1 <= len(supports) <= 4:
                raise SemanticCallError(f"{skill_id}/{local_id}: invalid supports")
            if any(str(value) not in allowed_subunits for value in supports):
                raise SemanticCallError(f"{skill_id}/{local_id}: unknown support subunit")
            support_text = "\n".join(texts[str(value)] for value in supports)
            for field in ("tool_terms", "input_terms", "output_terms", "validation_terms"):
                terms = concept.get(field, [])
                if not isinstance(terms, list):
                    raise SemanticCallError(f"{skill_id}/{local_id}: invalid {field}")
                if any(str(term) not in support_text for term in terms):
                    deterministically_rejected.add((skill_id, local_id))
            concepts[(skill_id, local_id)] = concept
        for concept in skill_concepts:
            local_id = str(concept["local_id"])
            dependencies = concept.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise SemanticCallError(f"{skill_id}/{local_id}: invalid dependencies")
            for dependency in dependencies:
                required = str(dependency.get("requires_local_id", ""))
                evidence = dependency.get("evidence_subunit_ids")
                if required not in local_ids or required == local_id:
                    raise SemanticCallError(f"{skill_id}/{local_id}: invalid dependency target")
                if not isinstance(evidence, list) or not evidence:
                    raise SemanticCallError(f"{skill_id}/{local_id}: dependency lacks evidence")
                if any(str(value) not in allowed_subunits for value in evidence):
                    raise SemanticCallError(f"{skill_id}/{local_id}: unknown dependency evidence")
    if observed_skills != set(expected_skills):
        raise SemanticCallError("Foundation proposal omitted a supplied skill")
    return concepts, deterministically_rejected


def _validate_foundation_verification(
    concepts: dict[tuple[str, str], dict[str, Any]], verification: dict[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[tuple[str, str]]]:
    rows = verification.get("decisions")
    if not isinstance(rows, list):
        raise SemanticCallError("Foundation verification missing decisions")
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    incomplete: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticCallError("Invalid foundation verification row")
        key = (str(row.get("skill_id", "")), str(row.get("local_id", "")))
        if key not in concepts or key in decisions:
            raise SemanticCallError(f"Unexpected/duplicate verification decision: {key}")
        if row.get("concept_status") not in {"CERTIFIED", "REJECTED", "UNRESOLVED"}:
            raise SemanticCallError(f"Invalid concept status: {key}")
        proposed_dependencies = {
            str(item["requires_local_id"])
            for item in concepts[key].get("dependencies", [])
        }
        dependency_rows = row.get("dependency_decisions", [])
        if not isinstance(dependency_rows, list):
            raise SemanticCallError(f"Invalid dependency decisions: {key}")
        observed_dependencies: set[str] = set()
        for dependency in dependency_rows:
            required = str(dependency.get("requires_local_id", ""))
            if required not in proposed_dependencies or required in observed_dependencies:
                raise SemanticCallError(f"Unexpected dependency verification: {key}/{required}")
            if dependency.get("status") not in {"CERTIFIED", "REJECTED", "UNRESOLVED"}:
                raise SemanticCallError(f"Invalid dependency status: {key}/{required}")
            observed_dependencies.add(required)
        if observed_dependencies != proposed_dependencies:
            incomplete.add(key)
        decisions[key] = row
    incomplete.update(set(concepts) - set(decisions))
    return decisions, incomplete


def _run_foundation_batch(
    repo_root: Path,
    protocol: dict[str, Any],
    batch: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    input_payload = json.loads(Path(batch["input_path"]).read_text(encoding="utf-8"))
    if content_sha256(input_payload) != batch["input_sha256"]:
        raise SemanticCallError(f"Frozen foundation input drift: {batch['batch_id']}")
    generation_path = output_root / "calls" / "foundation" / f"{batch['batch_id']}.generate.json"
    generation = _call_json(
        repo_root,
        f"{batch['batch_id']}:generate",
        FOUNDATION_GENERATOR_SYSTEM,
        input_payload,
        MAX_FOUNDATION_GENERATION_TOKENS,
        generation_path,
    )
    concepts, deterministically_rejected = _validate_foundation_proposal(
        input_payload, generation["output"]
    )
    verifier_payload = {
        "schema_version": "hcsc-foundation-verification-input-v1",
        "source": input_payload,
        "proposal": generation["output"],
        "proposal_sha256": content_sha256(generation["output"]),
    }
    verification_path = output_root / "calls" / "foundation" / f"{batch['batch_id']}.verify.json"
    verification = _call_json(
        repo_root,
        f"{batch['batch_id']}:verify",
        FOUNDATION_VERIFIER_SYSTEM,
        verifier_payload,
        MAX_FOUNDATION_VERIFICATION_TOKENS,
        verification_path,
    )
    decisions, incomplete_verification = _validate_foundation_verification(
        concepts, verification["output"]
    )
    return {
        "batch_id": batch["batch_id"],
        "input_sha256": batch["input_sha256"],
        "generation_artifact_sha256": generation["artifact_sha256"],
        "verification_artifact_sha256": verification["artifact_sha256"],
        "proposed_concepts": len(concepts),
        "certified_concepts": sum(
            decision["concept_status"] == "CERTIFIED"
            and key not in deterministically_rejected
            and key not in incomplete_verification
            for key, decision in decisions.items()
        ),
        "deterministically_rejected_nonexact_concepts": len(deterministically_rejected),
        "incomplete_verification_concepts": len(incomplete_verification),
        "generation_path": str(generation_path),
        "verification_path": str(verification_path),
    }


def run_foundation(
    repo_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    batches = list(protocol["foundation_batches"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def run(batch: dict[str, Any]) -> dict[str, Any]:
        return _run_foundation_batch(repo_root, protocol, batch, output_root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = {pool.submit(run, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "batch_id": str(batch["batch_id"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    summary = {
        "schema_version": "hcsc-foundation-run-summary-v1",
        "status": "PASS" if len(rows) == len(batches) and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "fallback_used": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "expected_batches": len(batches),
        "completed_batches": len(rows),
        "rows": sorted(rows, key=lambda row: row["batch_id"]),
        "errors": sorted(errors, key=lambda row: row["batch_id"]),
    }
    summary["summary_sha256"] = content_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _write_immutable_json(output_root / "foundation_run_summary.json", summary)
    return summary


def _validate_task_proposal(input_payload: dict[str, Any], proposal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allowed_requirements = {str(item["requirement_id"]) for item in input_payload["requirements"]}
    subunit_to_skill = {
        str(item["subunit_id"]): str(item["skill_id"]) for item in input_payload["subunits"]
    }
    allowed_skills = set(input_payload["c0_top20"])
    rows = proposal.get("paths")
    if not isinstance(rows, list):
        raise SemanticCallError("Task proposal missing paths")
    paths: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticCallError("Invalid task path")
        local_id = str(row.get("local_path_id", ""))
        if not local_id or local_id in paths:
            raise SemanticCallError("Duplicate/missing local path id")
        requirement_ids = row.get("requirement_ids")
        subunit_ids = row.get("subunit_ids")
        parent_skill_ids = row.get("parent_skill_ids")
        if not isinstance(requirement_ids, list) or not requirement_ids:
            raise SemanticCallError(f"{local_id}: missing requirements")
        if not set(map(str, requirement_ids)).issubset(allowed_requirements):
            raise SemanticCallError(f"{local_id}: unknown requirement")
        if not isinstance(subunit_ids, list) or not subunit_ids:
            raise SemanticCallError(f"{local_id}: missing subunits")
        if not set(map(str, subunit_ids)).issubset(subunit_to_skill):
            raise SemanticCallError(f"{local_id}: unknown subunit")
        expected_parents = {subunit_to_skill[str(value)] for value in subunit_ids}
        if not isinstance(parent_skill_ids, list) or set(map(str, parent_skill_ids)) != expected_parents:
            raise SemanticCallError(f"{local_id}: parent skills do not match subunits")
        if not expected_parents.issubset(allowed_skills):
            raise SemanticCallError(f"{local_id}: parent outside C0 top20")
        paths[local_id] = row
    local_ids = set(paths)
    for local_id, row in paths.items():
        for field in ("prerequisite_local_path_ids", "conflict_local_path_ids"):
            values = row.get(field, [])
            if not isinstance(values, list) or not set(map(str, values)).issubset(local_ids - {local_id}):
                raise SemanticCallError(f"{local_id}: invalid {field}")
        unresolved = row.get("unresolved_dependencies", [])
        if not isinstance(unresolved, list):
            raise SemanticCallError(f"{local_id}: invalid unresolved_dependencies")
    return paths


def _validate_task_verification(
    paths: dict[str, dict[str, Any]], verification: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    rows = verification.get("decisions")
    if not isinstance(rows, list):
        raise SemanticCallError("Task verification missing decisions")
    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticCallError("Invalid task verification row")
        local_id = str(row.get("local_path_id", ""))
        if local_id not in paths or local_id in decisions:
            raise SemanticCallError(f"Unexpected/duplicate task decision: {local_id}")
        if row.get("status") not in {"CERTIFIED", "REJECTED", "UNRESOLVED"}:
            raise SemanticCallError(f"Invalid task path status: {local_id}")
        if row.get("dependency_status") not in {"CERTIFIED", "REJECTED", "UNRESOLVED"}:
            raise SemanticCallError(f"Invalid dependency status: {local_id}")
        if row.get("contradiction_status") not in {"NONE", "CERTIFIED", "UNRESOLVED"}:
            raise SemanticCallError(f"Invalid contradiction status: {local_id}")
        if not isinstance(row.get("validation_supported"), bool):
            raise SemanticCallError(f"Invalid validation status: {local_id}")
        decisions[local_id] = row
    if set(decisions) != set(paths):
        raise SemanticCallError("Task verifier omitted proposed paths")
    return decisions


def _run_task(
    repo_root: Path,
    protocol: dict[str, Any],
    row: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    input_payload = json.loads(Path(row["input_path"]).read_text(encoding="utf-8"))
    if content_sha256(input_payload) != row["input_sha256"]:
        raise SemanticCallError(f"Frozen task input drift: {task_id}")
    generation_path = output_root / "calls" / "tasks" / f"{task_id}.generate.json"
    generation = _call_json(
        repo_root,
        f"task:{task_id}:generate",
        TASK_GENERATOR_SYSTEM,
        input_payload,
        MAX_TASK_GENERATION_TOKENS,
        generation_path,
    )
    paths = _validate_task_proposal(input_payload, generation["output"])
    verifier_payload = {
        "schema_version": "hcsc-task-support-verification-input-v1",
        "source": input_payload,
        "proposal": generation["output"],
        "proposal_sha256": content_sha256(generation["output"]),
    }
    verification_path = output_root / "calls" / "tasks" / f"{task_id}.verify.json"
    verification = _call_json(
        repo_root,
        f"task:{task_id}:verify",
        TASK_VERIFIER_SYSTEM,
        verifier_payload,
        MAX_TASK_VERIFICATION_TOKENS,
        verification_path,
    )
    decisions = _validate_task_verification(paths, verification["output"])
    return {
        "task_id": task_id,
        "input_sha256": row["input_sha256"],
        "generation_artifact_sha256": generation["artifact_sha256"],
        "verification_artifact_sha256": verification["artifact_sha256"],
        "proposed_paths": len(paths),
        "certified_paths": sum(
            decision["status"] == "CERTIFIED"
            and decision["dependency_status"] == "CERTIFIED"
            and decision["contradiction_status"] == "NONE"
            for decision in decisions.values()
        ),
        "generation_path": str(generation_path),
        "verification_path": str(verification_path),
    }


def run_tasks(repo_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    task_inputs = list(protocol["task_inputs"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def run(row: dict[str, Any]) -> dict[str, Any]:
        return _run_task(repo_root, protocol, row, output_root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = {pool.submit(run, row): row for row in task_inputs}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "task_id": str(row["task_id"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    summary = {
        "schema_version": "hcsc-task-support-run-summary-v1",
        "status": "PASS" if len(rows) == len(task_inputs) and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "fallback_used": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "expected_tasks": len(task_inputs),
        "completed_tasks": len(rows),
        "rows": sorted(rows, key=lambda row: row["task_id"]),
        "errors": sorted(errors, key=lambda row: row["task_id"]),
    }
    summary["summary_sha256"] = content_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _write_immutable_json(output_root / "task_support_run_summary.json", summary)
    return summary


def _validate_pair_verification(
    input_payload: dict[str, Any], output: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    expected = {str(row["pair_id"]) for row in input_payload["pairs"]}
    rows = output.get("decisions")
    if not isinstance(rows, list):
        raise SemanticCallError("Pair verification missing decisions")
    decisions: dict[str, dict[str, Any]] = {}
    invalid_or_duplicate = 0
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticCallError("Invalid pair verification row")
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in expected or pair_id in decisions:
            invalid_or_duplicate += 1
            continue
        if row.get("status") not in {"CERTIFIED", "REJECTED", "UNRESOLVED"}:
            raise SemanticCallError(f"Invalid pair status: {pair_id}")
        decisions[pair_id] = row
    return decisions, expected - set(decisions), invalid_or_duplicate


def _run_pair_batch(
    repo_root: Path,
    pair_protocol: dict[str, Any],
    batch: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    input_payload = json.loads(Path(batch["input_path"]).read_text(encoding="utf-8"))
    if content_sha256(input_payload) != batch["input_sha256"]:
        raise SemanticCallError(f"Frozen pair input drift: {batch['batch_id']}")
    artifact_path = output_root / "calls" / "pairs" / f"{batch['batch_id']}.verify.json"
    artifact = _call_json(
        repo_root,
        f"{batch['batch_id']}:verify",
        PAIR_VERIFIER_SYSTEM,
        input_payload,
        MAX_PAIR_VERIFICATION_TOKENS,
        artifact_path,
    )
    decisions, incomplete, invalid_or_duplicate = _validate_pair_verification(
        input_payload, artifact["output"]
    )
    return {
        "batch_id": batch["batch_id"],
        "input_sha256": batch["input_sha256"],
        "verification_artifact_sha256": artifact["artifact_sha256"],
        "pair_count": len(input_payload["pairs"]),
        "certified_pairs": sum(row["status"] == "CERTIFIED" for row in decisions.values()),
        "incomplete_pairs": len(incomplete),
        "invalid_or_duplicate_output_rows": invalid_or_duplicate,
        "verification_path": str(artifact_path),
    }


def run_pairs(repo_root: Path, pair_protocol_path: Path, output_root: Path) -> dict[str, Any]:
    pair_protocol = json.loads(pair_protocol_path.read_text(encoding="utf-8"))
    if pair_protocol.get("schema_version") not in {
        "hcsc-cross-skill-pair-protocol-v1",
        "hcsc-cross-skill-pair-protocol-v1-transport-repair",
    }:
        raise SemanticCallError("Unexpected pair protocol schema")
    expected_hash = content_sha256(
        {key: value for key, value in pair_protocol.items() if key != "pair_protocol_sha256"}
    )
    if pair_protocol.get("pair_protocol_sha256") != expected_hash:
        raise SemanticCallError("Pair protocol hash mismatch")
    batches = list(pair_protocol["batches"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def run(batch: dict[str, Any]) -> dict[str, Any]:
        return _run_pair_batch(repo_root, pair_protocol, batch, output_root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = {pool.submit(run, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "batch_id": str(batch["batch_id"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    summary = {
        "schema_version": "hcsc-pair-run-summary-v1",
        "status": "PASS" if len(rows) == len(batches) and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "fallback_used": False,
        "pair_protocol_sha256": pair_protocol["pair_protocol_sha256"],
        "expected_batches": len(batches),
        "completed_batches": len(rows),
        "rows": sorted(rows, key=lambda row: row["batch_id"]),
        "errors": sorted(errors, key=lambda row: row["batch_id"]),
    }
    summary["summary_sha256"] = content_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    summary_name = f"pair_run_summary_{pair_protocol['pair_protocol_sha256'][:12]}.json"
    _write_immutable_json(output_root / summary_name, summary)
    summary["summary_path"] = str(output_root / summary_name)
    return summary
