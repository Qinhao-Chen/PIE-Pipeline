# PIE: Prune, Interpret, Evaluate

**A Cross-Layer Transcoder-Native Framework for Efficient Circuit Discovery via Feature Attribution**

This repository contains the official implementation of the paper **"Prune, Interpret, Evaluate (PIE)"**.

PIE is an end-to-end framework designed to solve the interpretability efficiency bottleneck. Instead of interpreting millions of features in sparse dictionaries, PIE identifies a sparse "Causal Core" of features that are behaviorally necessary for a task using **Feature Attribution Patching (FAP)** and **Synergy-Aware Reranking**.

## Key Features

* **CLT-Native Pruning:** operates directly on Cross-Layer Transcoder feature-write edges.
* **FAP (Feature Attribution Patching):** A fast, gradient-based approximation for feature importance that scales to millions of units.
* **Synergy Reranking:** Detects and rescues "boundary features" that are individually weak but highly synergistic (e.g., non-additive interactions).
* **Budgeted Interpretation:** Reduces downstream LLM interpretation costs by ~40x by focusing only on the causal core.
* **FADE-style Evaluation:** Automated scoring of feature explanations for Clarity, Purity, and Responsiveness.

## Installation

### Prerequisites

* Python 3.10.14

### Setup

1. Clone this repository:

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The `pie` package is used in source form — there is no `setup.py` / `pyproject.toml`. Run all scripts from the repository root and add it to `PYTHONPATH`:

```bash
cd PIE-Pipeline
export PYTHONPATH="$PWD:$PYTHONPATH"
```


3. **External Dependency:** This codebase relies on `circuit-tracer` for CLT loading and patching utilities.
```bash
git clone https://github.com/safety-research/circuit-tracer.git
pip install -e circuit-tracer

```


4. **API Keys:** For Stages II and III (Interpretation & Evaluation), you need an OpenAI API key.
```bash
export OPENAI_API_KEY="sk-..."

```


## Usage

The PIE workflow consists of three sequential stages.


### Stage I: PRUNE (Feature Selection)

Filter the massive search space of CLT features down to a small budget (e.g., 100 features) using FAP and Synergy Reranking.

**Example: Pruning Llama-3.2-1B on IOI Task**

```bash
python scripts/run_pruning.py \
    --model "meta-llama/Llama-3.2-1B" \
    --transcoder_set "mntss/clt-llama-3.2-1b-524k" \
    --data_file "ioi_pairs.json" \
    --output_dir "./results/llama_ioi" \
    --keep_topk 100 \
    --feat_budget_type "global" \
    --synergy "boundary" \
    --lambda_syn 3.0 \
    --boundary_percent 25.0 \
    --corruption "corrupted_prompt" \
    --record_per_prompt

```

* **`--synergy boundary`**: Enables the interaction-aware reranking described in the paper.
* **`--lambda_syn 3.0`**: The optimal synergy weight found in the paper's sensitivity analysis.
* **Output:** Generates a `.jsonl` file containing the pruned feature indices (the "Causal Core").

### Stage II: INTERPRET (Auto-Explanation)

Generate natural language descriptions *only* for the features retained in Stage I. This saves massive compute compared to explaining the full dictionary.

```bash
python scripts/run_description.py \
    --scan "mntss/clt-llama-3.2-1b-524k" \
    --pruned_file (your file path for the pruning results) \
    --llm_model "gpt-5.2" \
    --out_file "./results/llama_ioi/descriptions.json"

```

* **Input:** The pruned JSONL from Stage I.
* **Output:** A JSON mapping Feature IDs to natural language explanations.

### Stage III: EVALUATE (Quality Assessment)

Assess the quality of the explanations and the semantic distinctness of the features using FADE metrics (Clarity, Purity, Responsiveness) on a held-out corpus (Wikipedia).

```bash
python scripts/run_evaluation.py \
    --model "meta-llama/Llama-3.2-1B" \
    --clt "mntss/clt-llama-3.2-1b-524k" \
    --descriptions "./results/llama_ioi/descriptions.json" \
    --corpus "sentence-transformers/wikipedia-en-sentences" \
    --out_file "./results/llama_ioi/fade_metrics.json" \
    --llm_model "gpt-5-mini"

```

* **Metrics Calculated:**
* **Clarity:** Can an LLM generate synthetic activations based on the description?
* **Purity:** How well does the description predict feature activity on real text?
* **Responsiveness:** Do relevant texts consistently trigger the feature?



## Reproducing Paper Results

To reproduce the specific results for **Gemma-2-2B** and **Llama-3.2-1B** on the IOI task:

1. **Download Data:** Ensure you have the Indirect Object Identification (IOI) dataset (formatted as expected by `load_samples`).
2. **Models:** The paper uses the following public checkpoints:
* `mntss/clt-gemma-2-2b-426k`
* `mntss/clt-llama-3.2-1b-524k`


3. **Hyperparameters:**
* Set `keep_topk` to 100.
* For FAP-Synergy: `lambda_syn=3`, `boundary_percent=25`, `boundary_cap=32`.



## Results Summary

As demonstrated in the paper, PIE achieves significant efficiency gains:

| Model | Method | Features Kept () | KL Divergence (nats) ↓ | Efficiency vs Random |
| --- | --- | --- | --- | --- |
| **Gemma-2-2B** | Random (Active Set) | 4,250 | 0.69 | 1x |
| **Gemma-2-2B** | **PIE (FAP-Synergy)** | **100** | **0.73** | **~40x** |
| **Llama-3.2-1B** | Random (Active Set) | 3,750 | 1.15 | 1x |
| **Llama-3.2-1B** | **PIE (FAP-Synergy)** | **100** | **1.12** | **~40x** |

*Random baselines require thousands of features to match the behavioral fidelity that PIE achieves with just 100 features.*

## More Baselines

### RelP (Relevance Patching)

`scripts/run_relp_pruning.py` is a drop-in alternative to Stage I that replaces the gradient-based FAP score with **LRP-style relevance coefficients** (Identity-rule on activations, Half-rule on the gated MLP, gradient stop on attention pattern and LayerNorm scales). It scores once per prompt and slices selections for many K budgets in one pass, also reporting Faithfulness and Completeness per K.

The command below was smoke-tested on a single A40 with `--num_samples 2` against IOI:

```bash
python scripts/run_relp_pruning.py \
    --model "meta-llama/Llama-3.2-1B" \
    --transcoder_set "mntss/clt-llama-3.2-1b-524k" \
    --data_file "/path/to/ioi_pairs.jsonl" \
    --output_dir "./results/llama_ioi_relp" \
    --keep_topk "50,100,200,400,800" \
    --feat_budget_type "global" \
    --corruption "corrupted_prompt" \
    --metric "last_logit_diff_json"
```

Notes on the data file:
* `--data_file` must be JSONL — one record per line. RelP reads `text_clean` (required), `text_corr` (required when `--corruption corrupted_prompt`), and `io_clean` / `s_clean` (used as `pos_target` / `neg_target` for `--metric last_logit_diff_json`). If `io_clean`/`s_clean` are missing, the metric silently falls back to `logsumexp`.

Outputs (per `--output_dir`):
* `selected_features_k{K}.txt` / `.json` — full ranked feature list (across all prompts), sorted by kept-count then mean |score|; one row per encoder-occurrence feature ID (Cantor-paired `(layer, feat_id)`).
* `final_summary.json` — per-K aggregates (`mean_kl`, `std_kl`, `prediction_change_rate`, `mean_faithfulness`, `mean_completeness`, `num_samples`).
* `per_prompt_rank{r}.jsonl` / `per_prompt_merged.jsonl` — per-prompt records with `kl_by_k`, `pred_changed_by_k`, `faithfulness_by_k`, `completeness_by_k`, and `kept_fids_max_k`.
* `run_manifest.json` — run id, args, world size, and SHA-256 of all artifacts.

## License

This project is licensed under the MIT License.
