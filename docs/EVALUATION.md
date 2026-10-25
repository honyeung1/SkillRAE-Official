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

## Benchmark settings

The paper evaluates 87 SkillsBench tasks with a 207-skill library and 30
AgentSkillOS tasks with a 200-skill library. The evaluated configuration retrieves
20 candidates and selects five complete packages. Guidance is bounded to 256
tokens and two entries. Execution settings are Codex CLI/GPT-5.5 and
Grok Build CLI/Grok-4.6.

This release distributes the implementation, model configuration, and a synthetic
selection example. Frozen benchmark inputs and execution logs are not bundled.
Model presets configure new semantic calls.

## Component comparisons

| Component comparison | Implementation correspondence |
|---|---|
| Full SkillRAE | Hierarchical retrieval, accepted OCU support, joint selection, bounded guide |
| Without hierarchical retrieval | Flat skill-level candidates with the second stage retained |
| Without Stage II | First-stage selection without binding, joint selection, or guidance |
| Without OCU-level evidence | Whole-skill support with the selection machinery retained |

`hierarchy_representation_controls.py` and `whole_skill_runtime_guide.py` provide
component-comparison utilities. Each comparison uses its corresponding input
representation. See [Setup and usage](REPRODUCING.md) for input preparation and benchmark execution
requirements.
