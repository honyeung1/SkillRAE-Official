from __future__ import annotations

import concurrent.futures
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable

from .graph_v2 import build_graph_package, sha256_file
from .schema import GraphPackage, SubunitInstance, content_sha256, provenance_id, stable_id
from .semantic_protocol import EXACT_MODEL
from .semantic_runner import SemanticCallError, _call_json


PROTOCOL_VERSION = "hcsc-capability-unit-construction-v3"
MAX_PARALLEL_CALLS = 4
MAX_CANDIDATES_PER_BATCH = 16
MAX_CANDIDATES_PER_SKILL = 12
MAX_SUPPORT_SPANS = 8
MAX_SUPPORT_CHARS = 5200
MAX_CLASSIFICATION_TOKENS = 7500
MAX_VERIFICATION_TOKENS = 6500

_RESOURCE_TOKEN = re.compile(
    r"(?<![\w.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|sh|js|mjs|ts|json|xml|xsd|"
    r"md|txt|csv|yaml|yml|html|css|bib|skill))(?![\w.-])",
    re.IGNORECASE,
)


CAPABILITY_CLASSIFIER_SYSTEM = """
You classify deterministic candidate Operational Capability Units built only from
exact public skill spans and public files. Do not add an operation, procedure,
resource, prerequisite, output, or validation rule.

Return JSON only with this schema:
{
  "decisions": [
    {
      "candidate_alias": "exact supplied alias",
      "status": "COMPLETE|INCOMPLETE|UNRESOLVED",
      "canonical_label": "2-8 word operation label",
      "action_evidence_ids": ["exact supplied evidence id"],
      "method_evidence_ids": ["exact supplied evidence id"],
      "input_evidence_ids": ["exact supplied evidence id"],
      "output_evidence_ids": ["exact supplied evidence id"],
      "completion_evidence_ids": ["exact supplied evidence id"],
      "resource_ids": ["exact supplied resource id"],
      "external_prerequisite": "NONE|REQUIRED|UNRESOLVED",
      "reason": "short source-grounded reason"
    }
  ]
}

COMPLETE means the cited evidence forms one coherent executable capability with:
(1) a concrete action, (2) an executable method/tool/algorithm, (3) an observable
output or effect, and (4) a public validation or completion predicate. One exact
span may support several slots when it truly contains them. Inputs may be empty
when no special input is required. A listed public resource may support the method,
but its existence alone is not a capability. Any required external prerequisite
outside the candidate makes the unit INCOMPLETE. Use supplied IDs only. Omission,
topic similarity, generic advice, or inferred missing steps are not COMPLETE.
Never mention tests, solutions, verifiers, rewards, trajectories, or hidden data.
""".strip()


CAPABILITY_VERIFIER_SYSTEM = """
You independently verify proposed Operational Capability Units using only the
supplied exact public skill spans, public resource inventory, and the proposal.
Do not repair, complete, or rewrite a proposal.

Return JSON only:
{
  "decisions": [
    {
      "candidate_alias": "exact supplied alias",
      "status": "CERTIFIED|REJECTED|UNRESOLVED",
      "coherent_single_capability": true,
      "action_supported": true,
      "method_supported": true,
      "output_supported": true,
      "completion_supported": true,
      "self_contained": true,
      "reason": "short evidence-grounded reason"
    }
  ]
}

CERTIFIED requires every boolean to be true and the proposal to identify a single
executable action-to-observable-completion chain. Source order, related vocabulary,
or a collection of individually useful fragments is insufficient. Unknown evidence
is UNRESOLVED. Use aliases exactly and never use private evaluation information.
""".strip()


def _write_immutable_json(path: Path, value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Immutable capability-unit collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return sha256(serialized.encode("utf-8")).hexdigest()


def _common_heading_depth(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    depth = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        depth += 1
    return depth


def _resource_inventory(skill_dir: Path, repo_root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in skill_dir.rglob("*") if item.is_file()):
        if path.name == "SKILL.md":
            continue
        relative_skill = path.relative_to(skill_dir).as_posix()
        relative_repo = path.relative_to(repo_root).as_posix()
        rows.append(
            {
                "resource_id": stable_id(
                    "res",
                    [skill_dir.name, relative_skill, sha256_file(path)],
                ),
                "skill_id": skill_dir.name,
                "source_file": relative_repo,
                "relative_skill_path": relative_skill,
                "source_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            }
        )
    return tuple(rows)


def _mentioned_resources(
    spans: Iterable[SubunitInstance],
    inventory: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    by_path = {str(row["relative_skill_path"]).lower(): row for row in inventory}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in inventory:
        by_name.setdefault(Path(str(row["relative_skill_path"])).name.lower(), []).append(row)
    matched: dict[str, dict[str, Any]] = {}
    mentions: dict[str, set[str]] = {}
    for span in spans:
        evidence_id = provenance_id(span.provenance)
        for token in _RESOURCE_TOKEN.findall(span.provenance.exact_text):
            normalized = token.lower().lstrip("./")
            resource = by_path.get(normalized)
            if resource is None:
                named = by_name.get(Path(normalized).name, [])
                if len(named) == 1:
                    resource = named[0]
            if resource is None:
                continue
            resource_id = str(resource["resource_id"])
            matched[resource_id] = resource
            mentions.setdefault(resource_id, set()).add(evidence_id)
    return tuple(
        {
            **matched[resource_id],
            "mention_provenance_ids": sorted(mentions[resource_id]),
        }
        for resource_id in sorted(matched)
    )


def _candidate_spans_for_anchor(
    anchor: SubunitInstance,
    skill_subunits: tuple[SubunitInstance, ...],
) -> tuple[SubunitInstance, ...]:
    same_heading = [
        item
        for item in skill_subunits
        if item.heading_path == anchor.heading_path
        and abs(item.provenance.line_start - anchor.provenance.line_start) <= 120
    ]
    selected: list[SubunitInstance] = sorted(
        same_heading,
        key=lambda item: (item.provenance.line_start, item.subunit_id),
    )
    if anchor not in selected:
        selected.append(anchor)

    required_roles = {"VALIDATION", "CONSTRAINT", "ARTIFACT"}
    present_roles = {item.role for item in selected}
    for role in sorted(required_roles - present_roles):
        candidates = [
            item
            for item in skill_subunits
            if item.role == role
            and _common_heading_depth(item.heading_path, anchor.heading_path) >= max(
                1, len(anchor.heading_path) - 1
            )
            and abs(item.provenance.line_start - anchor.provenance.line_start) <= 180
        ]
        if candidates:
            selected.append(
                min(
                    candidates,
                    key=lambda item: (
                        abs(item.provenance.line_start - anchor.provenance.line_start),
                        item.provenance.line_start,
                        item.subunit_id,
                    ),
                )
            )

    ordered = sorted(
        {item.subunit_id: item for item in selected}.values(),
        key=lambda item: (
            0 if item.subunit_id == anchor.subunit_id else 1,
            abs(item.provenance.line_start - anchor.provenance.line_start),
            item.provenance.line_start,
            item.subunit_id,
        ),
    )
    accepted: list[SubunitInstance] = []
    chars = 0
    for item in ordered:
        text_length = len(item.provenance.exact_text)
        if accepted and (len(accepted) >= MAX_SUPPORT_SPANS or chars + text_length > MAX_SUPPORT_CHARS):
            continue
        accepted.append(item)
        chars += text_length
    return tuple(sorted(accepted, key=lambda item: (item.provenance.line_start, item.subunit_id)))


def build_capability_candidates(
    graph: GraphPackage,
    repo_root: Path,
    foundation_profiles: dict[str, Any],
    allowed_skill_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    repo_root = repo_root.resolve()
    allowed = set(map(str, allowed_skill_ids))
    graph_sha256 = graph.package_sha256
    subunit_by_id = {item.subunit_id: item for item in graph.subunits}
    by_skill: dict[str, list[SubunitInstance]] = {}
    for subunit in graph.subunits:
        if subunit.skill_id in allowed:
            by_skill.setdefault(subunit.skill_id, []).append(subunit)
    skill_by_id = {item.skill_id: item for item in graph.skills}
    anchors: dict[str, list[dict[str, Any]]] = {}
    for profile in foundation_profiles.get("profiles", []):
        if profile.get("status") != "CERTIFIED" or profile.get("skill_id") not in allowed:
            continue
        support_ids = list(map(str, profile.get("support_subunit_ids", [])))
        if len(support_ids) != 1 or support_ids[0] not in subunit_by_id:
            continue
        anchors.setdefault(str(profile["skill_id"]), []).append(profile)

    rows: list[dict[str, Any]] = []
    for skill_id in sorted(allowed):
        skill = skill_by_id.get(skill_id)
        if skill is None:
            raise ValueError(f"Graph misses allowed skill: {skill_id}")
        skill_dir = repo_root / skill.source_dir
        inventory = _resource_inventory(skill_dir, repo_root)
        candidates: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
        for profile in anchors.get(skill_id, []):
            anchor_id = str(profile["support_subunit_ids"][0])
            anchor = subunit_by_id[anchor_id]
            spans = _candidate_spans_for_anchor(anchor, tuple(by_skill.get(skill_id, [])))
            resources = _mentioned_resources(spans, inventory)
            support_ids = tuple(item.subunit_id for item in spans)
            resource_ids = tuple(str(item["resource_id"]) for item in resources)
            key = (support_ids, resource_ids)
            roles = sorted({item.role for item in spans})
            row = {
                "candidate_id": stable_id(
                    "ocu",
                    [graph_sha256, skill_id, anchor_id, support_ids, resource_ids],
                ),
                "skill_id": skill_id,
                "anchor_concept_id": str(profile["concept_id"]),
                "anchor_subunit_id": anchor_id,
                "anchor_label": str(profile.get("canonical_label", "")),
                "heading_path": list(anchor.heading_path),
                "support_subunit_ids": list(support_ids),
                "support_provenance_ids": [provenance_id(item.provenance) for item in spans],
                "roles_present": roles,
                "support_spans": [
                    {
                        "evidence_id": provenance_id(item.provenance),
                        "subunit_id": item.subunit_id,
                        "role": item.role,
                        "source_file": item.provenance.source_file,
                        "line_start": item.provenance.line_start,
                        "line_end": item.provenance.line_end,
                        "exact_text": item.provenance.exact_text,
                        "source_sha256": item.provenance.source_sha256,
                    }
                    for item in spans
                ],
                "resources": list(resources),
                "candidate_rule": "certified_anchor_plus_same_section_and_nearest_completion_evidence",
            }
            existing = candidates.get(key)
            if existing is None or row["candidate_id"] < existing["candidate_id"]:
                candidates[key] = row

        ranked = sorted(
            candidates.values(),
            key=lambda row: (
                -("VALIDATION" in row["roles_present"]),
                -bool(row["resources"]),
                -len(row["roles_present"]),
                -len(row["support_subunit_ids"]),
                row["candidate_id"],
            ),
        )[:MAX_CANDIDATES_PER_SKILL]
        rows.extend(ranked)
    return tuple(sorted(rows, key=lambda row: (row["skill_id"], row["candidate_id"])))


def _validate_candidate_sources(
    graph: GraphPackage,
    repo_root: Path,
    candidates: Iterable[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    subunits = {item.subunit_id: item for item in graph.subunits}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        for subunit_id in candidate["support_subunit_ids"]:
            subunit = subunits.get(str(subunit_id))
            if subunit is None:
                errors.append(f"unknown_subunit:{candidate_id}:{subunit_id}")
                continue
            errors.extend(subunit.provenance.validate(repo_root=repo_root))
        for resource in candidate["resources"]:
            source = (repo_root / str(resource["source_file"])).resolve()
            try:
                source.relative_to(repo_root)
            except ValueError:
                errors.append(f"resource_outside_repo:{candidate_id}:{source}")
                continue
            parts = {part.lower() for part in source.parts}
            if {"tests", "solution"}.intersection(parts):
                errors.append(f"forbidden_resource:{candidate_id}:{source}")
            elif not source.is_file():
                errors.append(f"missing_resource:{candidate_id}:{source}")
            elif sha256_file(source) != resource["source_sha256"]:
                errors.append(f"resource_hash_mismatch:{candidate_id}:{source}")
    return sorted(set(errors))


def freeze_capability_unit_protocol(
    repo_root: Path,
    rankings_path: Path,
    foundation_profiles_path: Path,
    output_root: Path,
    *,
    pool_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    rankings_path = rankings_path.resolve()
    foundation_profiles_path = foundation_profiles_path.resolve()
    output_root = output_root.resolve()
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    if rankings.get("labels_rewards_or_trajectories_loaded") is not False:
        raise ValueError("Rankings are not outcome-blind")
    task_rows = list(rankings.get("task_rows", []))
    task_ids = [str(row["task_id"]) for row in task_rows]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Rankings must contain nonempty, unique task IDs")
    allowed_skill_ids = sorted(
        {
            str(skill_id)
            for row in task_rows
            for skill_id in row.get("c0_ranking", [])[:20]
        }
    )
    graph = build_graph_package(pool_dir or repo_root / "global_skill_pool", repo_root)
    profiles = json.loads(foundation_profiles_path.read_text(encoding="utf-8"))
    if profiles.get("status") != "PASS" or profiles.get("model") != EXACT_MODEL:
        raise ValueError("Foundation profiles must have PASS status and match the configured construction model")
    if profiles.get("graph_package_sha256") != graph.package_sha256:
        raise ValueError("Foundation profile graph drift")
    candidates = build_capability_candidates(graph, repo_root, profiles, allowed_skill_ids)
    source_errors = _validate_candidate_sources(graph, repo_root, candidates)
    if source_errors:
        raise ValueError(f"Capability candidate provenance failed: {source_errors[:5]}")
    candidate_payload = {
        "schema_version": "hcsc-capability-candidates-v3",
        "status": "FROZEN",
        "graph_package_sha256": graph.package_sha256,
        "foundation_profiles_sha256": profiles["normalized_output_sha256"],
        "allowed_skill_ids": allowed_skill_ids,
        "candidates": list(candidates),
    }
    candidate_path = output_root / "capability_candidates.json"
    candidate_file_sha = _write_immutable_json(candidate_path, candidate_payload)

    batches: list[dict[str, Any]] = []
    for start in range(0, len(candidates), MAX_CANDIDATES_PER_BATCH):
        source_rows = list(candidates[start : start + MAX_CANDIDATES_PER_BATCH])
        batch_id = f"capability_{start // MAX_CANDIDATES_PER_BATCH + 1:03d}"
        aliased = [
            {"candidate_alias": f"u{index:02d}", **row}
            for index, row in enumerate(source_rows, start=1)
        ]
        payload = {
            "schema_version": "hcsc-capability-classification-input-v3",
            "batch_id": batch_id,
            "graph_package_sha256": graph.package_sha256,
            "public_sources_only": True,
            "candidates": aliased,
        }
        path = output_root / "frozen_inputs" / f"{batch_id}.json"
        file_sha = _write_immutable_json(path, payload)
        batches.append(
            {
                "batch_id": batch_id,
                "input_path": str(path),
                "input_sha256": content_sha256(payload),
                "file_sha256": file_sha,
                "candidate_ids": [row["candidate_id"] for row in source_rows],
                "candidate_count": len(source_rows),
            }
        )
    result = {
        "schema_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "model": EXACT_MODEL,
        "fallback_allowed": False,
        "public_sources_only": True,
        "reward_or_trajectory_loaded": False,
        "graph_package_sha256": graph.package_sha256,
        "rankings_path": str(rankings_path),
        "rankings_sha256": sha256_file(rankings_path),
        "foundation_profiles_path": str(foundation_profiles_path),
        "foundation_profiles_sha256": profiles["normalized_output_sha256"],
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": candidate_file_sha,
        "allowed_skill_ids": allowed_skill_ids,
        "candidate_count": len(candidates),
        "batches": batches,
        "parallelism": MAX_PARALLEL_CALLS,
        "semantic_or_schema_retry": False,
        "prompts": {
            "classifier": {
                "text": CAPABILITY_CLASSIFIER_SYSTEM,
                "sha256": sha256(CAPABILITY_CLASSIFIER_SYSTEM.encode("utf-8")).hexdigest(),
            },
            "verifier": {
                "text": CAPABILITY_VERIFIER_SYSTEM,
                "sha256": sha256(CAPABILITY_VERIFIER_SYSTEM.encode("utf-8")).hexdigest(),
            },
        },
        "call_budget": {
            "classification": len(batches),
            "independent_verification": len(batches),
            "total": 2 * len(batches),
        },
        "forbidden_sources": [
            "tests/",
            "solution/",
            "verifier",
            "oracle",
            "reward",
            "trajectory",
            "historical outcome",
        ],
    }
    result["protocol_sha256"] = content_sha256(
        {key: value for key, value in result.items() if key != "protocol_sha256"}
    )
    protocol_path = output_root / "capability_unit_protocol_v3.json"
    _write_immutable_json(protocol_path, result)
    result["protocol_path"] = str(protocol_path)
    return result


def load_capability_unit_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_VERSION or payload.get("status") != "FROZEN":
        raise ValueError("Capability-unit V3 protocol is not frozen")
    if payload.get("model") != EXACT_MODEL or payload.get("fallback_allowed") is not False:
        raise ValueError("Capability-unit V3 model/fallback drift")
    expected = content_sha256(
        {key: value for key, value in payload.items() if key != "protocol_sha256"}
    )
    if payload.get("protocol_sha256") != expected:
        raise ValueError("Capability-unit V3 protocol hash mismatch")
    return payload


def _index_decisions(rows: Any, aliases: set[str]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            alias = str(row.get("candidate_alias", ""))
            if alias in aliases:
                grouped.setdefault(alias, []).append(row)
    accepted = {alias: values[0] for alias, values in grouped.items() if len(values) == 1}
    unresolved = {alias for alias in aliases if alias not in accepted}
    return accepted, unresolved


def _proposal_errors(candidate: dict[str, Any], proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence_ids = {str(row["evidence_id"]) for row in candidate["support_spans"]}
    resource_ids = {str(row["resource_id"]) for row in candidate["resources"]}
    evidence_fields = (
        "action_evidence_ids",
        "method_evidence_ids",
        "input_evidence_ids",
        "output_evidence_ids",
        "completion_evidence_ids",
    )
    for field in evidence_fields:
        values = proposal.get(field)
        if not isinstance(values, list) or any(str(value) not in evidence_ids for value in values):
            errors.append(f"invalid_{field}")
    if not isinstance(proposal.get("resource_ids"), list) or any(
        str(value) not in resource_ids for value in proposal.get("resource_ids", [])
    ):
        errors.append("invalid_resource_ids")
    if proposal.get("status") not in {"COMPLETE", "INCOMPLETE", "UNRESOLVED"}:
        errors.append("invalid_status")
    if proposal.get("external_prerequisite") not in {"NONE", "REQUIRED", "UNRESOLVED"}:
        errors.append("invalid_external_prerequisite")
    if proposal.get("status") == "COMPLETE":
        for field in (
            "action_evidence_ids",
            "method_evidence_ids",
            "output_evidence_ids",
            "completion_evidence_ids",
        ):
            if not proposal.get(field):
                errors.append(f"complete_missing_{field}")
        if proposal.get("external_prerequisite") != "NONE":
            errors.append("complete_not_self_contained")
        if not str(proposal.get("canonical_label", "")).strip():
            errors.append("complete_missing_label")
    return errors


def _run_capability_batch(
    repo_root: Path,
    batch: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    source = json.loads(Path(batch["input_path"]).read_text(encoding="utf-8"))
    if content_sha256(source) != batch["input_sha256"]:
        raise SemanticCallError(f"Frozen capability input drift: {batch['batch_id']}")
    classification_path = output_root / "calls" / f"{batch['batch_id']}.classify.json"
    classification = _call_json(
        repo_root,
        f"v3:{batch['batch_id']}:classify",
        CAPABILITY_CLASSIFIER_SYSTEM,
        source,
        MAX_CLASSIFICATION_TOKENS,
        classification_path,
    )
    aliases = {str(row["candidate_alias"]) for row in source["candidates"]}
    proposals, missing_proposals = _index_decisions(
        classification["output"].get("decisions"), aliases
    )
    verifier_payload = {
        "schema_version": "hcsc-capability-verification-input-v3",
        "batch_id": batch["batch_id"],
        "public_sources_only": True,
        "candidates": source["candidates"],
        "proposal": classification["output"],
        "proposal_sha256": content_sha256(classification["output"]),
    }
    verification_path = output_root / "calls" / f"{batch['batch_id']}.verify.json"
    verification = _call_json(
        repo_root,
        f"v3:{batch['batch_id']}:verify",
        CAPABILITY_VERIFIER_SYSTEM,
        verifier_payload,
        MAX_VERIFICATION_TOKENS,
        verification_path,
    )
    verifications, missing_verifications = _index_decisions(
        verification["output"].get("decisions"), aliases
    )
    candidate_by_alias = {str(row["candidate_alias"]): row for row in source["candidates"]}
    deterministic_rejections = {
        alias: _proposal_errors(candidate_by_alias[alias], proposals[alias])
        for alias in aliases.intersection(proposals)
    }
    certified = 0
    for alias in aliases:
        proposal = proposals.get(alias, {})
        decision = verifications.get(alias, {})
        booleans = (
            "coherent_single_capability",
            "action_supported",
            "method_supported",
            "output_supported",
            "completion_supported",
            "self_contained",
        )
        if (
            proposal.get("status") == "COMPLETE"
            and not deterministic_rejections.get(alias)
            and decision.get("status") == "CERTIFIED"
            and all(decision.get(field) is True for field in booleans)
        ):
            certified += 1
    return {
        "batch_id": batch["batch_id"],
        "candidate_count": len(aliases),
        "certified_count": certified,
        "missing_proposals": len(missing_proposals),
        "missing_verifications": len(missing_verifications),
        "deterministic_rejection_count": sum(bool(value) for value in deterministic_rejections.values()),
        "classification_artifact_sha256": classification["artifact_sha256"],
        "verification_artifact_sha256": verification["artifact_sha256"],
    }


def run_capability_unit_protocol(
    repo_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = load_capability_unit_protocol(protocol_path)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=protocol["parallelism"]) as pool:
        futures = {
            pool.submit(_run_capability_batch, repo_root, batch, output_root): batch
            for batch in protocol["batches"]
        }
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
    result = {
        "schema_version": "hcsc-capability-unit-run-summary-v3",
        "status": "PASS" if len(rows) == len(protocol["batches"]) and not errors else "BLOCKED",
        "model": EXACT_MODEL,
        "fallback_used": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "expected_batches": len(protocol["batches"]),
        "completed_batches": len(rows),
        "rows": sorted(rows, key=lambda row: row["batch_id"]),
        "errors": sorted(errors, key=lambda row: row["batch_id"]),
    }
    result["summary_sha256"] = content_sha256(
        {key: value for key, value in result.items() if key != "summary_sha256"}
    )
    _write_immutable_json(output_root / "capability_unit_run_summary_v3.json", result)
    return result


def normalize_capability_units(
    repo_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = load_capability_unit_protocol(protocol_path)
    graph = build_graph_package(repo_root / "global_skill_pool", repo_root)
    if graph.package_sha256 != protocol["graph_package_sha256"]:
        raise ValueError("Capability-unit graph drift")
    normalized: list[dict[str, Any]] = []
    missing_batches: list[str] = []
    for batch in protocol["batches"]:
        source = json.loads(Path(batch["input_path"]).read_text(encoding="utf-8"))
        classification_path = output_root / "calls" / f"{batch['batch_id']}.classify.json"
        verification_path = output_root / "calls" / f"{batch['batch_id']}.verify.json"
        if not classification_path.is_file() or not verification_path.is_file():
            missing_batches.append(str(batch["batch_id"]))
            continue
        classification = json.loads(classification_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        aliases = {str(row["candidate_alias"]) for row in source["candidates"]}
        proposals, missing_proposals = _index_decisions(
            classification["output"].get("decisions"), aliases
        )
        verifications, missing_verifications = _index_decisions(
            verification["output"].get("decisions"), aliases
        )
        for candidate in source["candidates"]:
            alias = str(candidate["candidate_alias"])
            proposal = proposals.get(alias, {})
            decision = verifications.get(alias, {})
            rejection_reasons = _proposal_errors(candidate, proposal) if proposal else ["missing_proposal"]
            if alias in missing_proposals:
                rejection_reasons.append("missing_or_duplicate_proposal")
            if alias in missing_verifications:
                rejection_reasons.append("missing_or_duplicate_verification")
            booleans = (
                "coherent_single_capability",
                "action_supported",
                "method_supported",
                "output_supported",
                "completion_supported",
                "self_contained",
            )
            status = "CERTIFIED" if (
                proposal.get("status") == "COMPLETE"
                and not rejection_reasons
                and decision.get("status") == "CERTIFIED"
                and all(decision.get(field) is True for field in booleans)
            ) else "REJECTED" if (
                proposal.get("status") == "INCOMPLETE" or decision.get("status") == "REJECTED"
            ) else "UNRESOLVED"
            normalized.append(
                {
                    **{key: value for key, value in candidate.items() if key != "candidate_alias"},
                    "status": status,
                    "canonical_label": str(proposal.get("canonical_label", "")),
                    "slot_evidence": {
                        field: list(map(str, proposal.get(field, [])))
                        for field in (
                            "action_evidence_ids",
                            "method_evidence_ids",
                            "input_evidence_ids",
                            "output_evidence_ids",
                            "completion_evidence_ids",
                        )
                    },
                    "operational_provenance_ids": sorted(
                        {
                            str(evidence_id)
                            for field in (
                                "action_evidence_ids",
                                "method_evidence_ids",
                                "input_evidence_ids",
                                "output_evidence_ids",
                                "completion_evidence_ids",
                            )
                            for evidence_id in proposal.get(field, [])
                        }
                    ),
                    "operational_subunit_ids": sorted(
                        {
                            str(span["subunit_id"])
                            for span in candidate["support_spans"]
                            if str(span["evidence_id"])
                            in {
                                str(evidence_id)
                                for field in (
                                    "action_evidence_ids",
                                    "method_evidence_ids",
                                    "input_evidence_ids",
                                    "output_evidence_ids",
                                    "completion_evidence_ids",
                                )
                                for evidence_id in proposal.get(field, [])
                            }
                        }
                    ),
                    "selected_resource_ids": list(map(str, proposal.get("resource_ids", []))),
                    "external_prerequisite": str(proposal.get("external_prerequisite", "UNRESOLVED")),
                    "classifier_reason": str(proposal.get("reason", "")),
                    "verifier_reason": str(decision.get("reason", "")),
                    "rejection_reasons": sorted(set(rejection_reasons)),
                    "classification_artifact_sha256": classification["artifact_sha256"],
                    "verification_artifact_sha256": verification["artifact_sha256"],
                }
            )
    source_errors = _validate_candidate_sources(graph, repo_root, normalized)
    if source_errors:
        raise ValueError(f"Normalized capability provenance failed: {source_errors[:5]}")
    certified = [row for row in normalized if row["status"] == "CERTIFIED"]
    result = {
        "schema_version": "hcsc-capability-units-v3.1-operational-spans",
        "status": "PASS" if not missing_batches else "BLOCKED",
        "model": EXACT_MODEL,
        "fallback_used": False,
        "public_sources_only": True,
        "reward_or_trajectory_loaded": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "graph_package_sha256": graph.package_sha256,
        "foundation_profiles_sha256": protocol["foundation_profiles_sha256"],
        "missing_batches": missing_batches,
        "candidate_count": len(normalized),
        "certified_unit_count": len(certified),
        "certified_skill_count": len({row["skill_id"] for row in certified}),
        "units": sorted(normalized, key=lambda row: row["candidate_id"]),
    }
    result["normalized_output_sha256"] = content_sha256(
        {key: value for key, value in result.items() if key != "normalized_output_sha256"}
    )
    path = output_root / "capability_units_v3_operational.json"
    _write_immutable_json(path, result)
    result["path"] = str(path)
    return result
