"""Bounded L1 support map; no synthetic OCU certification or execution order."""
import json
from pathlib import Path

from ._tokens import count_tokens
from .compiler import extract_public_requirements
from .runtime_guide import DEFAULT_GUIDE_BUDGET, DEFAULT_MAX_ROUTES, _diverse_route_order


def build_whole_skill_guide(*, repo_root, task_id, selected_skill_ids, selected_path_ids,
                           task_support_path, capability_units_path, skill_communities,
                           budget_tokens=DEFAULT_GUIDE_BUDGET, max_routes=DEFAULT_MAX_ROUTES):
    empty = {'schema_version': 'whole-l1-support-guide-v1', 'task_id': task_id, 'text': '',
             'tokens': 0, 'route_count': 0, 'selected_path_ids': [], 'line_provenance': [],
             'budget_tokens': budget_tokens, 'max_routes': max_routes}
    if not selected_path_ids:
        return empty
    support = json.loads(Path(task_support_path).read_text())
    documents = json.loads(Path(capability_units_path).read_text())
    if documents.get('entity_level') != 'L1_WHOLE_SKILL':
        raise ValueError('Expected explicit whole-skill documents, not OCU units')
    document_by_id = {x['document_id']: x for x in documents['documents']}
    path_by_id = {x['path_id']: x for x in support['support_paths']}
    requirements = {x.requirement_id: x for x in extract_public_requirements(
        Path(repo_root) / 'tasks' / task_id / 'instruction.md', Path(repo_root))}
    routes = [path_by_id[key] for key in selected_path_ids]
    ranks = {skill: i for i, skill in enumerate(selected_skill_ids, 1)}
    ordered = _diverse_route_order(routes, requirements, ranks)
    scaffold = [{'source_file': 'generator:whole_l1_support_guide_v1', 'source_type': 'generic_scaffold',
                 'source_id': 'generic_scaffold:whole_l1_support_guide_v1'}]
    lines = [('## Procedural Slice Guide', 'generic_heading', scaffold),
             ('Open the cited mounted lines before using a route; the original task instruction remains authoritative.',
              'generic_instruction', scaffold)]
    accepted = []
    for route in ordered[:max_routes]:
        document = document_by_id[route['document_id']]
        skill = document['skill_id']
        if skill not in selected_skill_ids or route['parent_skill_ids'] != [skill]:
            raise ValueError('Whole-skill support outside selected skills')
        req = requirements[route['requirement_ids'][0]]
        evidence = {x['evidence_id']: x for x in document['support_spans']}
        ids = list(dict.fromkeys(route['operation_evidence_ids'] + route['completion_evidence_ids']))
        if not route['operation_evidence_ids'] or set(ids) - evidence.keys():
            raise ValueError('Whole-skill route lacks valid public action evidence')
        refs = [{'source_file': req.provenance.source_file, 'source_type': req.provenance.source_type,
                 'source_id': req.requirement_id}]
        refs.extend({'source_file': evidence[key]['source_file'], 'source_type': 'selected_skill',
                     'source_id': key} for key in ids)
        group = skill_communities.get(skill, ('none',))[0] if skill_communities.get(skill) else 'none'
        added = [(f'{len(accepted) + 1}. [{req.criticality}/{req.requirement_type}] '
                  f'`{req.provenance.source_file}:L{req.provenance.line_start}-L{req.provenance.line_end}` '
                  f'-> `{skill}` (group `{group}`).', 'whole_skill_support', refs),
                 (f'   Open `/root/.agents/skills/{skill}/SKILL.md`; support is assessed at whole-skill level.',
                  'whole_skill_document', refs)]
        text = '\n'.join(x[0] for x in lines + added) + '\n'
        if count_tokens(text)[0] > budget_tokens:
            continue
        lines += added
        accepted.append(route['path_id'])
    if not accepted:
        raise ValueError('No whole-skill support fits the unchanged guide budget')
    text = '\n'.join(x[0] for x in lines) + '\n'
    provenance = [{'line': i, 'content': text_, 'semantic_role': role, 'sources': refs,
                   'task_id': task_id, 'destination_artifact': 'PROCEDURAL_SLICE_GUIDE.md'}
                  for i, (text_, role, refs) in enumerate(lines)]
    return {**empty, 'text': text, 'tokens': count_tokens(text)[0], 'route_count': len(accepted),
            'selected_path_ids': accepted, 'candidate_path_count': len(routes), 'line_provenance': provenance}
