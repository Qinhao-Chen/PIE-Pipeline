# Interpretation and relation tools

The main interpretation stage describes retained feature identities using activation histories:

```bash
pie interpret --help
```

Provide a local pruning output, select one budget with `--keep-topk`, and supply a compatible feature-history repository with `--scan`. Descriptions are written locally. No feature histories, collected descriptions or evaluation results are distributed.

## Role explanations

The [relation package](../pie/relations/) supplies tools for explanations grounded in measured graph evidence:

- `description_prompts.py` renders activation-condition and cross-layer-role prompts.
- `activation_length.py` provides bounded description variants.
- `rank_band_prompts.py` renders activation, role, metadata and shuffled-role conditions.
- `run_task_b.py` builds relation questions, collects descriptions or evaluator replies, and scores responses for user-supplied graphs.

Inspect the relation CLI before preparing your own inputs:

```bash
python -m pie.relations.run_task_b --help
```

Its subcommands include `split`, `describe`, `questions`, `evaluate`, `score` and `paired-bootstrap`. Supply your own graphs, split, descriptions and model/CLT revisions. Keep generated exports and API caches local.

Role prompts use measured discovery evidence. Activation examples alone do not establish a cross-layer role. Question construction separates public prompts from private labels and uses the supplied split to keep discovery, development and evaluation inputs separate.

## Synthetic checks

The tests and `scripts/run_relation_fixture.py` use synthetic graphs and scripted responses to exercise question construction, scoring and leakage controls. These are software fixtures, not collected research data or experimental results. See [testing](VALIDATION.md).
