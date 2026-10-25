# Pipeline interfaces

The CLI exposes source parsing, requirement extraction, OCU candidate construction,
and joint selection. The Python interfaces below cover
the intervening retrieval and semantic steps.

## Offline representations

`graph_v2.build_graph_package(pool_dir, repo_root)` parses skill Markdown and
retains source identity. `source_scoped_l0.source_sections` and
`local_block_graph.procedural_units` retain source-local procedure blocks.
`local_block_graph.communities` builds communities from skill embeddings.

`capability_units_v3.build_capability_candidates(graph, repo_root,
foundation_profiles, allowed_skill_ids)` constructs OCU candidates from source
anchors assessed by `semantic_protocol` and `semantic_runner`.

```bash
skillrae --preset skillsbench ocu-candidates /data/skills /data/profiles.json --root /data
```

Foundation profiles contain `status: "PASS"`, the configured construction `model`,
matching `graph_package_sha256`, and `profiles`. Each usable profile has
`status: "CERTIFIED"`, `skill_id`, one `support_subunit_ids` anchor, `concept_id`,
and an optional `canonical_label`. The command checks source identity and accepts
all skills by default; `--skills skill-a skill-b` restricts the candidates.

The classification and independent-review prompts, runners, and validators are
in `capability_units_v3`. Its protocol builder accepts task rankings and an
optional `pool_dir`; its normalization step produces accepted OCU records.

## Models and requests

Set `SKILLRAE_MODEL_PRESET=skillsbench` or `agentskillos` before importing the
semantic modules. All presets use `gpt-5.5` for construction and binding.
`SKILLRAE_CONSTRUCTION_MODEL` and `SKILLRAE_BINDING_MODEL`
override each stage independently. The CLI also accepts `--preset` before the
subcommand. Use `skillrae models` to inspect the configuration.

Construction checks use the construction model; requirement judges and reviewers
use the binding model. Protocols record the model identity. Cached responses must
match the model, prompt, and input. Credentials come from `OPENAI_API_KEY` and
`OPENAI_BASE_URL`.

## Retrieval and binding

`retrieval.SkillRetriever(data_dir)` loads the index artifacts enumerated in
`retrieval.REQUIRED_RETRIEVAL_ARTIFACTS`. Source-local retrieval also uses procedure
windows, their embeddings, and community centroids. `local_block_graph.project`
aggregates block evidence; `source_scoped_l0.fuse` combines retrieval channels.
Prepare the index for the same library used to construct the OCU records.

`compiler.extract_public_requirements` extracts requirements from the instruction.
`nomination._nominate` uses these requirements and candidate OCUs with `c0_rank`
to nominate pairs. Similarity determines which pairs receive semantic assessment.

`binding_judge.build_protocol` assembles source-checked pair inputs;
`run_judges` and `run_verifiers` perform assessment and independent review.
`normalize_bindings` accepts a support relation after both assessments. These
Python interfaces consume protocol manifests and source hashes as specified by
their signatures. An accepted relation connects a requirement to an OCU, its
parent skill, and its community memberships.

## Joint selection

```bash
skillrae select /data/task_selection.json
```

The JSON follows `CompilerInput`; [the example](../examples/selection.json)
provides a complete small input:

- `task_id`, `top_k`: task identity and skill budget.
- `requirements`: identity, criticality, type, and source provenance.
- `candidates`, `communities`: skill ranking and community membership.
- `subunits`: available source units and parent skills.
- `support_paths`: accepted support, parent skills, memberships, and dependencies.

`mechanism_policies_v2.select_hq_community_exact` returns the selected skill IDs,
support paths, and decision metadata. A support path is active when its required
parents and dependencies are present.

## Execution context

`runtime_guide.build_bounded_runtime_guide` consumes the task source root,
selected skill/path IDs, accepted task support, OCU records, and community
memberships. It returns the bounded guide and its source references.

Provide the complete selected package trees to the executor. The default guide
uses `/root/.agents/skills` as the package mount location. The
[usage guide](REPRODUCING.md) describes setup and a synthetic selection example.
For a new task, use the returned guide and selected packages with your executor.
