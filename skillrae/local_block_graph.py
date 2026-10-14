"""Source-local procedural blocks for nomination; no capability authorization."""

from collections import Counter
import math
import re

import numpy as np
from markdown_it import MarkdownIt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import source_scoped_l0 as source


ACTION = re.compile(
    r'\b(read|write|create|load|save|open|parse|extract|convert|run|execute|install|'
    r'validate|verify|check|compute|calculate|recalculate|train|build|test|query|'
    r'filter|export|import|update|delete|remove|add|set|use|apply|return|select|'
    r'compare|render|compile|fit|solve|download|inspect)\b', re.I)


def procedural_units(section):
    """Keep whole outer lists/code blocks, with neighboring explanatory paragraphs.

    Syntax bounds the source unit, not its semantic completeness. Long embedding
    windows retain a pointer to this complete source span and its parent section.
    """
    lines = section['exact_text'].splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks = [t for t in MarkdownIt('commonmark').parse(section['exact_text'])
              if t.level == 0 and t.map and t.type in
              {'paragraph_open', 'bullet_list_open', 'ordered_list_open', 'fence', 'code_block'}]
    spans = []
    for i, token in enumerate(blocks):
        a, b = token.map
        text = ''.join(lines[a:b])
        executable = token.type in {'fence', 'code_block'} or ACTION.search(text)
        if not executable:
            continue
        # Do not cross a heading or an unrelated block when restoring context.
        if i and blocks[i - 1].type == 'paragraph_open':
            pa, pb = blocks[i - 1].map
            if not ''.join(lines[pb:a]).strip():
                a = pa
        if i + 1 < len(blocks) and blocks[i + 1].type == 'paragraph_open':
            na, nb = blocks[i + 1].map
            if not ''.join(lines[b:na]).strip():
                b = nb
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(b, spans[-1][1]))
        else:
            spans.append((a, b))
    result = []
    for a, b in spans:
        row = dict(section)
        row.update(parent_section_id=section['section_id'],
                   line_start=section['line_start'] + a,
                   line_end=section['line_start'] + b - 1,
                   char_start=section['char_start'] + offsets[a],
                   char_end=section['char_start'] + offsets[b],
                   exact_text=''.join(lines[a:b]),
                   rule='source_local_outer_procedure_block_with_adjacent_explanation',
                   semantic_completeness='NOT_CERTIFIED')
        row['section_id'] = 'l0u_' + source.digest(row)[:24]
        result.append(row)
    return result


def project(similarities, windows, name_to_id, limit=30):
    """Keep old top-30/sum projection, with one vote per full source unit.

    Multiple windows are encoding views of one unit, not additional graph nodes.
    Exact duplicate text within one skill/heading scope also has only one vote.
    """
    unique = {}
    for score, row in zip(similarities, windows, strict=True):
        key = (row['skill_id'], row['context_text'], row['unit_exact_sha256'])
        unique[key] = max(unique.get(key, -math.inf), float(score))
    scores = {}
    for (name, _, _), score in sorted(unique.items(), key=lambda x: (-x[1], x[0]))[:limit]:
        if name in name_to_id:
            sid = name_to_id[name]
            scores[sid] = scores.get(sid, 0.0) + score
    return source.normalize(scores)


def communities(embeddings, skill_ids, old_labels):
    """Rebuild recall taxonomy in the current L1 space; never typed relations."""
    k = int(math.sqrt(len(skill_ids)))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
    centers = np.stack([embeddings[labels == i].mean(axis=0) for i in range(k)])
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Empty L2 centroid')
    centers /= norms
    rows = []
    for i in range(k):
        members = [j for j, label in enumerate(labels) if label == i]
        members.sort(key=lambda j: (-float(embeddings[j] @ centers[i]), skill_ids[j]))
        rows.append({'id': f'recall_l2_{i:03d}', 'skills': [skill_ids[j] for j in members],
                     'authority': 'NOMINATE_ONLY', 'label_type': 'centroid_member_ids_not_generated_claim'})
    quality = {'clusters': k, 'active_skills': len(skill_ids),
               'sizes': sorted(Counter(labels.tolist()).values()),
               'cosine_silhouette': float(silhouette_score(embeddings, labels, metric='cosine')),
               'old_cosine_silhouette_same_space': float(silhouette_score(embeddings, old_labels, metric='cosine')),
               'shared_capability_equivalence_claimed': False}
    quality['pass'] = (quality['cosine_silhouette'] > quality['old_cosine_silhouette_same_space']
                       and max(quality['sizes']) < len(skill_ids) / 2)
    return rows, centers, quality
