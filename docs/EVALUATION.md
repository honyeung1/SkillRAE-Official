# Evaluation settings

## Current run configuration

Capability construction and requirement binding both use **`gpt-5.5`** in every
preset. These settings govern new semantic calls:

| Preset | Capability construction | Requirement binding |
|---|---|---|
| `default` | `gpt-5.5` | `gpt-5.5` |
| `skillsbench` | `gpt-5.5` | `gpt-5.5` |
| `agentskillos` | `gpt-5.5` | `gpt-5.5` |

Choose a preset before starting the process, as described in [Models](../README.md#models).
Environment overrides take precedence; `skillrae models` shows the active values.

## Benchmark settings and recorded inputs

The paper evaluates 87 SkillsBench tasks with a 207-skill library and 30
AgentSkillOS tasks with a 200-skill library. The evaluated configuration retrieves
20 candidates and selects five complete packages. Guidance is bounded to 256
tokens and two entries. Execution settings are Codex CLI/GPT-5.5 and
Grok Build CLI/Grok-4.6.

Recorded-input replay uses the original construction and binding identities in
the [SkillsBench manifest](../reproduce/skillsbench/manifest.json) and
[AgentSkillOS manifest](../reproduce/agentskillos/manifest.json). Current run
presets do not change those recorded inputs or their model provenance.

## Component comparisons

| Component comparison | Implementation correspondence |
|---|---|
| Full SkillRAE | Hierarchical retrieval, accepted OCU support, joint selection, bounded guide |
| Without hierarchical retrieval | Flat skill-level candidates with the second stage retained |
| Without Stage II | First-stage selection without binding, joint selection, or guidance |
| Without OCU-level evidence | Whole-skill support with the selection machinery retained |

`hierarchy_representation_controls.py` and `whole_skill_runtime_guide.py` provide
component-comparison utilities. Each comparison uses its corresponding input
representation. See [Reproducing](REPRODUCING.md) for the available method inputs
and the benchmark execution requirements.
