# PIE: Prune, Interpret, Evaluate

**A cross-layer transcoder framework for budgeted circuit discovery and feature interpretation.**

PIE helps you select a small set of CLT feature occurrences, describe the retained features, and evaluate those descriptions. The workflow follows three stages: **Prune → Interpret → Evaluate**.

This repository contains the tool and data-generation code. Research datasets, collected feature histories, API responses and experimental results are kept private and are not distributed.

## Key features

- **CLT-native selection:** use Feature Attribution Patching (FAP) or a final-position attribution readout.
- **Interaction-aware refinement:** apply Synergy reranking or bounded Swap search at a fixed occurrence budget.
- **Behavioral evaluation:** measure final-position and sequence-mean KL, with activation, CLT-RelP, FActP and random references available.
- **Budgeted interpretation:** generate activation-based descriptions for retained feature identities.
- **FADE-style evaluation:** assess clarity, purity and responsiveness on a separate text corpus.

The model profiles support Gemma-2-2B and Llama-3.2-1B with their corresponding CLTs. The task loaders accept IOI and DocString paired inputs.

## Installation

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install ./vendor/circuit-tracer --no-deps
python -m pip check
```

Pruning and explanation evaluation require a CUDA GPU and access to the model checkpoints. Interpretation and FADE evaluation use an OpenAI-compatible API: set `OPENAI_API_KEY` and set `PIE_LLM_MODEL` to the API model you want to use. `--base-url` selects a different compatible endpoint.

See [installation details](docs/INSTALL.md) for model setup and dependency versions.

## Usage

Use the installed `pie` command, `python -m pie`, or the stage scripts listed below. Run any command with `--help` for its full options.

### Generate local inputs

The IOI generator from the original framework creates paired prompts locally:

```bash
python scripts/ioi.py \
  --N 100 --seed 0 --device cpu \
  --out data/tasks/ioi_pairs.jsonl --format jsonl
```

The generator loads a tokenizer and may download its files on first use. Generated files stay local and are ignored by Git. For DocString, supply your own paired input file following the [task format](docs/TASKS.md).

### Stage I: Prune

Select a budget of occurrences and measure their behavioral fidelity:

```bash
pie prune \
  --model llama --task ioi --method fap \
  --data-file data/tasks/ioi_pairs.jsonl \
  --num-samples 10 --keep-topk 100 \
  --output-dir outputs/llama_ioi
```

Add `--allow-download` for the initial checkpoint download. Cached pinned checkpoints are used by default. Use `--task docstring --data-file <local-file>` for DocString inputs.

| Method | Selection |
| --- | --- |
| `fap`, `lastlse` | Sequence-summed or final-position attribution |
| `fap-swap`, `lastlse-swap` | Attribution followed by bounded Swap refinement |
| `synergy` | FAP with boundary interaction reranking |
| `activation`, `clt-relp`, `factp` | Alternative selection methods |
| `random` | Clean-active random reference with zero ablation |

Use `--keep-topk 50,100,200` to compare budgets. FActP scores the full active universe and can take substantially longer. `native-replay` is an advanced interface requiring a compatible local graph bundle; bundles are not distributed.

**Local outputs:** `per_prompt.jsonl` contains selections and fidelity for each prompt/budget; `summary.json` aggregates the KL endpoints; `run_manifest.json` records the configuration. Each run requires a new output directory. Keep outputs private.

K counts `(layer, token position, feature index)` occurrences. Stage II deduplicates retained occurrences into `(layer, feature index)` identities across prompts. A K100 selection per prompt can therefore yield more than 100 identities over a dataset.

### Stage II: Interpret

Generate descriptions for the retained identities:

```bash
pie interpret \
  --pruned-file outputs/llama_ioi/per_prompt.jsonl --keep-topk 100 \
  --scan "$PIE_FEATURE_SCAN" \
  --llm-model "$PIE_LLM_MODEL" \
  --out-file outputs/llama_ioi/descriptions.json
```

Set `PIE_FEATURE_SCAN` to a compatible Hugging Face feature-history repository you can access, optionally with `@revision`. It must contain `features/index.json.gz` and the indexed binary feature-example shards. Feature histories are not bundled.

When several pruning budgets are present, choose one with `--keep-topk`. Descriptions are saved locally as a JSON mapping from feature IDs to text. `--max-concurrent` controls API concurrency.

Activation descriptions summarize observed patterns. The separate [relation tools](docs/INTERPRETATION.md) support role explanations grounded in measured graph evidence.

### Stage III: Evaluate

Evaluate descriptions on a separate corpus:

```bash
pie evaluate \
  --model llama \
  --descriptions outputs/llama_ioi/descriptions.json \
  --corpus sentence-transformers/wikipedia-en-sentences --num-samples 200 \
  --llm-model "$PIE_LLM_MODEL" \
  --out-file outputs/llama_ioi/evaluation.json
```

The corpus is loaded at runtime and is not redistributed here. Use text separate from the feature histories used to write descriptions. The model profile supplies its matching CLT; full model IDs from `configs/models.json` are accepted too.

| Metric | What it measures |
| --- | --- |
| Clarity | Activation separation between generated concept examples and natural text |
| Purity | Average precision of activation for clearly expressed concepts |
| Responsiveness | Activation separation between concept-positive and other rated text |

The evaluator collects activations at the CLT encoder input hooks, applies the CLT's activation function, and excludes BOS and padding from the maximum. Exact activation ties are handled consistently. Undefined class comparisons are written as `null`; invalid API replies are reported as errors.

### Script entry points

| Command | Equivalent script |
| --- | --- |
| `pie prune` | `python scripts/run_pruning.py` |
| `pie interpret` | `python scripts/run_description.py` |
| `pie evaluate` | `python scripts/run_evaluation.py` |

`run_behavior.py` remains available for behavioral comparisons. The description/evaluation wrappers under `legacy/scripts/` forward to the maintained commands. Both hyphenated and historical underscore flags are accepted by the description and evaluation commands.

## Repository structure

| Path | Contents |
| --- | --- |
| `pie/behavior/`, `pie/pruning/`, `pie/exchange/` | Feature selection, behavioral fidelity and Swap refinement |
| `pie/interpretation/` | Activation-history loading, prompts and interpretation CLI |
| `pie/evaluation/`, `pie/models/` | FADE-style metrics and activation collection |
| `pie/relations/` | Role explanations and relation recovery |
| `scripts/` | Tool entry points and the IOI data generator |
| `configs/models.json` | Supported checkpoint profiles |
| `vendor/circuit-tracer/` | Bundled CLT loading and patching implementation |
| `legacy/` | Earlier pruning scripts and compatibility wrappers |
| `tests/`, `docs/` | Synthetic software tests and usage documentation |

Generated data, caches and outputs are ignored by Git. The repository contains no research datasets or result exports.

## License

First-party code is distributed under the [MIT License](LICENSE). Bundled and adapted dependencies retain their own [third-party notices](THIRD_PARTY_NOTICES.md). Model checkpoints and external corpora retain their original terms.
