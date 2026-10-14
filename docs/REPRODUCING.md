# Reproducing selection and execution guidance

The release includes candidate rankings, accepted support, OCU records, task
instructions, and reference outputs for **87 SkillsBench tasks and 30 AgentSkillOS
tasks**. Replay recomputes joint selection and source-linked guidance from these
recorded inputs. Agent execution and official scoring use the benchmark environments.

## Run a real task

From the repository root or extracted code package:

```bash
python -m pip install -e . -c requirements-replay.txt
skillrae replay reproduce/skillsbench --output ./replay-output
```

The default task is `adaptive-cruise-control`. The selector replaces
`simulation-metrics` with `yaml-config`, retains five packages, and compiles a
two-route, 210-token guide. All five selected packages are included.

Inspect `replay-output/adaptive-cruise-control/`:

- `selection.json`: selected skills, support paths, requirements, and package locations.
- `PROCEDURAL_SLICE_GUIDE.md`: execution guidance.
- `guide_provenance.json`: source references.

The command re-extracts requirements from the instruction and checks the
computed selections, support paths, coverage, and guide against reference outputs.

## Check both benchmarks

```bash
skillrae replay reproduce/skillsbench --task all
skillrae replay reproduce/agentskillos --task all
```

| Snapshot content | Purpose |
|---|---|
| `manifest.json` | Tasks, budgets, library identity, construction and binding models |
| `graph.json` | Source-unit metadata and community memberships |
| `capabilities/` | Capability records, keyed by content identity |
| `tasks/<id>/` | Method inputs and reference selection/guide |
| `sources/tasks/`, `sources/skills/` | Instructions and distributable packages |
| `pool_files.json`, `external_skills.json` | Library file identities and separately obtained resources |
| `SHA256SUMS.json` | Snapshot integrity checks |

Record identifiers and source-file hashes retain their recorded identities.
Source locations use package-relative paths; guide references follow those paths.

## Skill packages

Included packages retain their own license notices. Packages requiring separate
acquisition are listed in `external_skills.json`, alongside file hashes and
verified version-specific source links. Obtain these packages from their
authorized sources before supplying them to an executor.

For each skill, `verified_file_paths` maps package-relative filenames to upstream
files at the specified commit. A `benchmark_source` entry supplies task-specific
files from the pinned SkillsBench revision. Combine these files under the skill
ID; every listed file has a verified source location and SHA-256 identity.

Verify a full skill library against its recorded file identities:

```bash
skillrae check-pool reproduce/skillsbench /data/skillsbench-pool
skillrae check-pool reproduce/agentskillos /data/agentskillos-pool
```

## New tasks and benchmark execution

[Pipeline inputs](PIPELINE.md) describes new source graphs, OCU assessment,
retrieval, requirement binding, and selector inputs. Model presets in the
[README](../README.md#models) configure new construction and binding calls.
Recorded-input replay uses the original inputs and their model metadata, independently
of the current live presets.

Benchmark environments and scoring tools are maintained upstream:

- [SkillsBench](https://github.com/benchflow-ai/skillsbench)
- [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS), under `benchmark/`

Full score reproduction requires the corresponding benchmark version, complete
selected packages, executor configuration, and scoring environment. The bundled
inputs cover the Codex-side selection and guidance computations.
