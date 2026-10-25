# Running the implementation

This repository provides the SkillRAE implementation, model configuration,
and a synthetic selection example. Supply task instructions, a skill library,
and model access to use the method on new inputs. Frozen benchmark inputs and
execution logs are not bundled.

## Install

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . -c constraints.txt
```

For embedding-based retrieval, install the optional dependency:

```bash
python -m pip install -e '.[retrieval]' -c constraints.txt
```

## Inspect the selection mechanism

```bash
skillrae select examples/selection.json
```

This synthetic example selects `s1` and `s3`, preserving critical support and
adding support under a two-skill budget. `examples/fixture.md` contains the
source passages referenced by the example. This command requires no model API.

## Configure semantic calls

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` for a Responses-compatible provider:

```bash
export SKILLRAE_MODEL_PRESET=skillsbench
skillrae models
```

The current presets use `gpt-5.5` for construction and binding. Construction
and binding models can be overridden independently; see [evaluation settings](EVALUATION.md).

## Supply new inputs

The [pipeline guide](PIPELINE.md) documents source parsing, OCU construction,
retrieval, requirement binding, joint selection, and guide assembly. Its commands
and Python interfaces consume explicit inputs; the CLI does not launch a complete
benchmark evaluation automatically.

```bash
skillrae requirements /data/tasks/task-id/instruction.md --root /data
skillrae source-graph /data/skills --root /data
skillrae --preset skillsbench ocu-candidates /data/skills /data/foundation_profiles.json --root /data
skillrae select /data/task_selection.json
```

`foundation_profiles.json` contains assessed source profiles in the schema described
in the pipeline guide. The selection input contains accepted requirement support.
Provide the selected complete packages and generated guide to the chosen executor.
Benchmark execution and official scoring use the upstream environments:

- [SkillsBench](https://github.com/benchflow-ai/skillsbench)
- [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS), under `benchmark/`

## Local checks

```bash
python -m unittest discover -s tests -v
```

These checks exercise model routing, source-identity validation, and synthetic
selection cases. They do not rerun benchmark experiments.
