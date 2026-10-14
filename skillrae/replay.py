"""Replay selection and context compilation from recorded public-source inputs."""
from __future__ import annotations
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
from .cli import load_input
from .compiler import extract_public_requirements, closed_requirements
from .guide_selection import _preferred_guide_paths
from .mechanism_policies_v2 import select_hq_community_exact
from .schema import SupportPath
from .runtime_guide import build_bounded_runtime_guide

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def json_read(path):
    return json.loads(path.read_text())

def within(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes snapshot root')
    return path

def validate_snapshot(root):
    for name, expected in json_read(root/'SHA256SUMS.json').items():
        path=within(root,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f'Snapshot file differs: {name}')

def replay_task(root: Path, task: str, *, output: Path | None = None):
    taskdir=within(root,'tasks/'+task)
    raw=json_read(taskdir/'input.json');expected=json_read(taskdir/'expected.json')
    graph=json_read(root/'graph.json');source_root=root/'sources'
    instruction=within(source_root,raw['instruction_source'])
    extracted=extract_public_requirements(instruction,source_root)
    if len(extracted)!=len(raw['requirements']):
        raise ValueError(f'Requirement extraction differs: {task}')
    requirements=[]
    for record, stored in zip(extracted, raw['requirements']):
        # Identity is stable across distribution path relocation; all semantic
        # attributes and exact source spans must still match fresh extraction.
        observed=asdict(record)
        observed.pop('requirement_id')
        reference={k:v for k,v in stored.items() if k!='requirement_id'}
        if json.loads(json.dumps(observed))!=reference:
            raise ValueError(f'Requirement extraction differs: {task}')
        requirements.append(replace(record,requirement_id=stored['requirement_id']))
    requirements=tuple(requirements)
    ids={r['skill_id'] for r in raw['candidates']}
    keys={f.name for f in fields(SupportPath)}
    data=load_input({**raw,'communities':graph['communities'],
        'subunits':[s for s in graph['subunits'] if s['skill_id'] in ids],
        'support_paths':[{k:v for k,v in p.items() if k in keys} for p in raw['support_paths']]})
    decision=select_hq_community_exact(data)
    binding={b['path_id']:b for b in raw['accepted_bindings']}
    paths=_preferred_guide_paths(decision,raw['baseline_path_ids'],binding)
    # Reuse the checked records with their original identifiers in the guide.
    with patch('skillrae.runtime_guide.extract_public_requirements',return_value=requirements):
        guide=build_bounded_runtime_guide(repo_root=source_root,task_id=task,
            selected_skill_ids=list(decision.selected_skill_ids),selected_path_ids=paths,
            task_support_path=taskdir/'input.json',capability_units_path=within(root,raw['units_file']),
            skill_communities={k:tuple(v) for k,v in graph['memberships'].items()},
            budget_tokens=raw['guide_compilation_budget'],max_routes=2)
    if raw['relative_executor_mount']:
        guide['text']=guide['text'].replace('/root/.agents/skills/','../codex_home/skills/')
        for line in guide['line_provenance']:
            line['content']=line['content'].replace('/root/.agents/skills/','../codex_home/skills/')
        from ._tokens import count_tokens
        guide['tokens']=count_tokens(guide['text'])[0];guide['budget_tokens']=256
    observed={'selected_skill_ids':list(decision.selected_skill_ids),'selected_path_ids':paths,
        'closed_requirement_ids':list(closed_requirements(data,decision.selected_skill_ids)),
        'guide_sha256':sha256(guide['text'].encode()).hexdigest(),'guide_tokens':guide['tokens']}
    for k,v in expected.items():
        if observed[k]!=v:
            raise ValueError(f'{task}: {k} differs from recorded output: {observed[k]!r} != {v!r}')
    if guide['text']!=(taskdir/'expected_guide.md').read_text():
        raise ValueError(f'{task}: guide text differs')
    config=json_read(root/'manifest.json')
    packages=[{'skill_id':s,'directory':str(within(source_root,config['pool_source_dir']+'/'+s)),
               'included':within(source_root,config['pool_source_dir']+'/'+s).is_dir()}
              for s in decision.selected_skill_ids]
    result={'task_id':task,'status':'PASS',**observed,'packages':packages,
            'guide_route_count':guide['route_count'],'changed_set':list(decision.selected_skill_ids)!=[x['skill_id'] for x in raw['candidates'][:data.top_k]]}
    if output:
        dest=output/task;dest.mkdir(parents=True,exist_ok=True)
        (dest/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
        (dest/'PROCEDURAL_SLICE_GUIDE.md').write_text(guide['text'])
        (dest/'guide_provenance.json').write_text(json.dumps(guide['line_provenance'],indent=2)+'\n')
    return result

def run(root: Path, task: str | None, output: Path | None):
    root=root.resolve()
    if output:
        output=output.resolve()
        if output.is_relative_to(root):
            raise ValueError('Write replay outputs outside the immutable snapshot')
    validate_snapshot(root)
    config=json_read(root/'manifest.json')
    tasks=config['tasks'] if task=='all' else [task or config['default_task']]
    rows=[]
    for tid in tasks:
        if tid not in config['tasks']:
            raise ValueError(f'Unknown snapshot task: {tid}')
        result=replay_task(root,tid,output=output);rows.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    summary={'benchmark':config['benchmark'],'verified_tasks':len(rows),'changed_sets':sum(r['changed_set'] for r in rows),
             'nonempty_guides':sum(r['guide_tokens']>0 for r in rows),'model_calls':0,'agent_executions':0}
    print(json.dumps(summary),flush=True)
    return summary


def check_pool(snapshot: Path, pool: Path):
    """Check a supplied full skill pool against the evaluated file identities."""
    expected=json_read(snapshot/'pool_files.json')
    differences=[]
    for name,value in expected.items():
        path=within(pool,name)
        if not path.is_file() or digest(path)!=value:
            differences.append(name)
    actual={str(p.relative_to(pool)) for p in pool.rglob('*') if p.is_file() and len(p.relative_to(pool).parts)>1}
    extra=sorted(actual-set(expected))
    if differences or extra:
        raise ValueError(f'Pool mismatch: {len(differences)} missing/changed, {len(extra)} extra files; first differences: {(differences+extra)[:10]}')
    print(json.dumps({'pool_check':'PASS','matched_files':len(expected)}))
