"""Offline source-scoped L0 nomination graph; never a capability certificate."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from markdown_it import MarkdownIt


SCHEMA = "skillrae-source-scoped-l0-v1"
BODY_TOKENS = 320
CONTEXT_TOKENS = 96
ARMS = (
    "legacy", "without_l0", "legacy_max", "source_sum",
    "source_max", "source_parent_budget",
)


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_sections(path: Path, repo: Path, skill_id: str) -> list[dict[str, Any]]:
    """Partition Markdown at real headings, keeping code fences and exact source."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = MarkdownIt("commonmark").parse(text)
    boundaries: list[tuple[int, tuple[dict[str, Any], ...]]] = [(0, ())]
    stack: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        if token.type != "heading_open" or not token.map:
            continue
        start, end = token.map
        level = int(token.tag[1:])
        stack = [item for item in stack if item["level"] < level]
        stack.append({"level": level, "text": tokens[i + 1].content,
                      "line_start": start + 1, "line_end": end,
                      "exact_text": "".join(lines[start:end])})
        if boundaries[-1][0] == start:
            boundaries[-1] = (start, tuple(stack))
        else:
            boundaries.append((start, tuple(stack)))
    result = []
    for i, (start, headings) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        content = "".join(lines[start:end])
        if not content.strip():
            continue
        row = {
            "skill_id": skill_id, "source_file": path.relative_to(repo).as_posix(),
            "source_sha256": sha256(raw).hexdigest(), "source_type": "public_skill_catalog",
            "line_start": start + 1, "line_end": end, "char_start": offsets[start],
            "char_end": offsets[end], "exact_text": content,
            "heading_path": list(headings), "rule": "commonmark_section_partition",
            "authority": "NOMINATE_ONLY",
        }
        row["section_id"] = "l0s_" + digest(row)[:24]
        result.append(row)
    return result


def index_windows(section: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Bound embedding input explicitly; windows reference a complete source section."""
    text = section["exact_text"]
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                         truncation=False, verbose=False)
    offsets = encoding["offset_mapping"]
    if not offsets:
        return []
    context = section["skill_id"] + "\n" + " > ".join(h["text"] for h in section["heading_path"])
    context_offsets = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True,
                                truncation=False, verbose=False)["offset_mapping"]
    if len(context_offsets) > CONTEXT_TOKENS:
        context = context[:context_offsets[CONTEXT_TOKENS - 1][1]]
    result = []
    for index in range(0, len(offsets), BODY_TOKENS):
        start = 0 if index == 0 else offsets[index][0]
        stop = index + BODY_TOKENS
        end = offsets[stop][0] if stop < len(offsets) else len(text)
        body = text[start:end]
        rendered = context + "\n\n" + body
        token_count = len(tokenizer(rendered, truncation=False, verbose=False)["input_ids"])
        if token_count > tokenizer.model_max_length:
            raise ValueError("Embedding window would be silently truncated")
        row = {
            "skill_id": section["skill_id"], "section_id": section["section_id"],
            "source_file": section["source_file"], "source_sha256": section["source_sha256"],
            "char_start": section["char_start"] + start,
            "char_end": section["char_start"] + end,
            "exact_text": body, "embedding_text": rendered, "embedding_tokens": token_count,
            "context_text": context, "authority": "NOMINATE_ONLY",
            "rule": "nonoverlapping_token_bounded_exact_source_window",
        }
        row["window_id"] = "l0w_" + digest(row)[:24]
        result.append(row)
    return result


def validate_graph(graph: dict[str, Any], repo: Path) -> dict[str, int]:
    sources = {}
    sections = {item["section_id"]: item for item in graph["sections"]}
    if len(sections) != len(graph["sections"]):
        raise ValueError("Duplicate section identity")
    window_ids = set()
    source_segments = defaultdict(list)
    section_windows = defaultdict(list)
    for row in [*graph["sections"], *graph["windows"]]:
        relative = Path(row["source_file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("global_skill_pool",):
            raise ValueError("Forbidden source")
        if any(part in {"tests", "solution"} for part in relative.parts):
            raise ValueError("Forbidden source")
        path = repo / relative
        if path.resolve().relative_to(repo.resolve()).parts[:1] != ("global_skill_pool",):
            raise ValueError("Escaped public pool")
        if row["source_file"] not in sources:
            sources[row["source_file"]] = path.read_bytes()
        raw = sources[row["source_file"]]
        if sha256(raw).hexdigest() != row["source_sha256"]:
            raise ValueError("Source changed")
        text = raw.decode("utf-8")
        if text[row["char_start"]:row["char_end"]] != row["exact_text"]:
            raise ValueError("Invalid exact source span")
        if row["authority"] != "NOMINATE_ONLY":
            raise ValueError("Retrieval evidence cannot authorize execution")
        if "window_id" in row:
            if row["window_id"] in window_ids:
                raise ValueError("Duplicate window identity")
            window_ids.add(row["window_id"])
            parent = sections[row["section_id"]]
            if parent["skill_id"] != row["skill_id"] or parent["source_file"] != row["source_file"]:
                raise ValueError("Wrong parent")
            if not parent["char_start"] <= row["char_start"] < row["char_end"] <= parent["char_end"]:
                raise ValueError("Window outside section")
            section_windows[row["section_id"]].append(row)
        else:
            lines = text.splitlines(keepends=True)
            if "".join(lines[row["line_start"] - 1:row["line_end"]]) != row["exact_text"]:
                raise ValueError("Invalid line range")
            for heading in row["heading_path"]:
                if "".join(lines[heading["line_start"] - 1:heading["line_end"]]) != heading["exact_text"]:
                    raise ValueError("Invalid heading provenance")
            source_segments[row["source_file"]].append(row)
    for path, rows in source_segments.items():
        restored = "".join(r["exact_text"] for r in sorted(rows, key=lambda r: r["char_start"]))
        if restored != sources[path].decode("utf-8"):
            raise ValueError("Incomplete section partition")
    for sid, section in sections.items():
        restored = "".join(r["exact_text"] for r in sorted(section_windows[sid], key=lambda r: r["char_start"]))
        if restored != section["exact_text"]:
            raise ValueError("Incomplete embedding-window coverage")
    return {"sources": len(sources), "sections": len(sections), "windows": len(window_ids)}


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {key: 0.5 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def legacy_projection(similarities: np.ndarray, ids: list[str], parents: dict[str, list[str]],
                      *, maximum: bool = False, limit: int = 30) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index in np.argsort(similarities)[-limit:][::-1]:
        linked = parents.get(ids[index], [])
        if not linked or len(linked) > 15:
            continue
        contribution = float(similarities[index]) / len(linked)
        for skill in linked:
            scores[skill] = (max(scores.get(skill, float("-inf")), contribution) if maximum
                             else scores.get(skill, 0.0) + contribution)
    return normalize(scores)


def source_projection(similarities: np.ndarray, windows: list[dict[str, Any]],
                      *, policy: str, limit: int = 30) -> dict[str, float]:
    if policy not in {"sum", "max", "parent_budget"}:
        raise ValueError("Unknown source aggregation")
    # Exact duplicate content in the same source scope has only one retrieval vote.
    candidates: dict[tuple[str, str, str], float] = {}
    for value, row in zip(similarities, windows, strict=True):
        key = (row["skill_id"], row["context_text"], row["exact_text"])
        candidates[key] = max(candidates.get(key, float("-inf")), float(value))
    if policy == "parent_budget":
        parent_best: dict[str, float] = {}
        for (skill, _, _), score in candidates.items():
            parent_best[skill] = max(parent_best.get(skill, float("-inf")), score)
        ranked = sorted(parent_best.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return normalize(dict(ranked))
    scores: dict[str, float] = {}
    ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:limit]
    for (skill, _, _), value in ranked:
        scores[skill] = (scores.get(skill, 0.0) + value if policy == "sum"
                         else max(scores.get(skill, float("-inf")), value))
    return normalize(scores)


def fuse(skill_ids: list[str], l1: dict[str, float], prior: dict[str, float],
         boosted: set[str], l0: dict[str, float]) -> list[tuple[str, float]]:
    values = []
    for sid in skill_ids:
        score = 0.6 * l1.get(sid, 0.0) + 0.4 * l0.get(sid, 0.0) + 0.15 * prior.get(sid, 0.0)
        values.append((sid, score * (1.10 if sid in boosted else 1.0)))
    return sorted(values, key=lambda item: item[1], reverse=True)
