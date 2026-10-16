# Software checks

Tests use synthetic tensors, graphs, text and scripted replies. They do not require private datasets, saved experimental results, model checkpoints or live API requests.

For CPU testing, install the package and test dependencies in a Python 3.10 or 3.11 environment:

```bash
python -m pip install -e '.[api]' numpy==1.24.4 tqdm==4.67.1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Run the relevant checks:

```bash
python -m unittest discover -s tests -p test_cli_evaluation.py -v
python -m unittest discover -s tests -p test_behavior.py -v
python -m unittest discover -s tests -p test_relations.py -v
```

The CLI/evaluation checks cover stage entry points, reading pruning budgets, CLT input hooks, encoder thresholds, BOS exclusion, tied scores and invalid replies. Behavioral checks cover task-prefix boundaries, occurrence/write alignment, KL endpoints, selection and Swap. Relation checks cover question construction, private labels, missing replies and paired scoring.

A real model run requires your own local inputs and the [GPU environment](INSTALL.md). Keep its data, logs, API responses and outputs private. No measured performance claims or archived run outputs are included in this repository.
