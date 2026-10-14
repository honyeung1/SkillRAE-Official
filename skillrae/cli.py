"""Portable entry points around the evaluated core; no implicit model calls."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from .schema import (CompilerInput,RequirementRecord,ProvenanceRef,CandidateSkill,
                     CommunityRecord,SupportPath,SubunitInstance)

def load_input(raw):
    requirements=tuple(RequirementRecord(**{**r,'provenance':ProvenanceRef(**r['provenance'])}) for r in raw['requirements'])
    subunits=tuple(SubunitInstance(**{**r,'provenance':ProvenanceRef(**r['provenance'])}) for r in raw.get('subunits',[]))
    return CompilerInput(task_id=raw['task_id'],requirements=requirements,
        candidates=tuple(CandidateSkill(**r) for r in raw['candidates']),
        communities=tuple(CommunityRecord(**r) for r in raw.get('communities',[])),
        support_paths=tuple(SupportPath(**r) for r in raw.get('support_paths',[])),
        subunits=subunits,top_k=int(raw.get('top_k',5)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from .model_config import PRESETS
    parser.add_argument('--preset', choices=tuple(PRESETS), help='Construction/binding model configuration')
    subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('models',help='Show the active construction and binding model identifiers')
    select=subs.add_parser('select',help='Select from an accepted-support CompilerInput JSON')
    select.add_argument('input',type=Path)
    req=subs.add_parser('requirements',help='Extract requirements from a public instruction')
    req.add_argument('instruction',type=Path);req.add_argument('--root',type=Path,required=True)
    graph=subs.add_parser('source-graph',help='Parse public skill sources; semantic certification is separate')
    graph.add_argument('pool',type=Path);graph.add_argument('--root',type=Path,required=True)
    ocu=subs.add_parser('ocu-candidates',help='Build source-linked OCU candidates from assessed foundation profiles')
    ocu.add_argument('pool',type=Path);ocu.add_argument('profiles',type=Path)
    ocu.add_argument('--root',type=Path,required=True)
    ocu.add_argument('--skills',nargs='+',help='Skill IDs to include; default: all parsed skills')
    replay=subs.add_parser('replay',help='Verify a frozen method snapshot against recorded selections and guides')
    replay.add_argument('snapshot',type=Path)
    replay.add_argument('--task',help='Task ID, or all; default is the documented example')
    replay.add_argument('--output',type=Path,help='Optional output directory outside the snapshot')
    poolcheck=subs.add_parser('check-pool',help='Verify a supplied skill library against the evaluation snapshot')
    poolcheck.add_argument('snapshot',type=Path);poolcheck.add_argument('pool',type=Path)
    args=parser.parse_args()
    if args.preset:
        os.environ['SKILLRAE_MODEL_PRESET']=args.preset
    if args.command=='models':
        from .model_config import model_config
        print(json.dumps(model_config().payload(),indent=2))
        return
    if args.command=='check-pool':
        from .replay import check_pool
        check_pool(args.snapshot,args.pool)
        return
    if args.command=='replay':
        from .replay import run
        run(args.snapshot,args.task,args.output)
        return
    if args.command=='select':
        from .mechanism_policies_v2 import select_hq_community_exact
        result=asdict(select_hq_community_exact(load_input(json.loads(args.input.read_text()))))
    elif args.command=='requirements':
        from .compiler import extract_public_requirements
        result=[asdict(r) for r in extract_public_requirements(args.instruction,args.root)]
    elif args.command=='ocu-candidates':
        from .graph_v2 import build_graph_package
        from .capability_units_v3 import build_capability_candidates, _validate_candidate_sources
        from .semantic_protocol import EXACT_MODEL
        graph=build_graph_package(args.pool.resolve(),args.root.resolve())
        profiles=json.loads(args.profiles.read_text())
        if profiles.get('status')!='PASS' or profiles.get('model')!=EXACT_MODEL:
            parser.error('Foundation profiles must have PASS status and the construction model')
        if profiles.get('graph_package_sha256')!=graph.package_sha256:
            parser.error('Foundation profile graph hash does not match the supplied skill sources')
        allowed=args.skills if args.skills is not None else [skill.skill_id for skill in graph.skills]
        candidates=build_capability_candidates(graph,args.root.resolve(),profiles,allowed)
        errors=_validate_candidate_sources(graph,args.root.resolve(),candidates)
        if errors:
            parser.error('Candidate source validation failed: '+str(errors[:5]))
        result={'graph_package_sha256':graph.package_sha256,'allowed_skill_ids':allowed,
                'candidates':list(candidates)}
    else:
        from .graph_v2 import build_graph_package
        result=build_graph_package(args.pool.resolve(),args.root.resolve()).payload()
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
