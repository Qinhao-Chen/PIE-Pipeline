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

We evaluate PIE on **Llama-3.2-1B** and **Gemma-2-2B** across two tasks (**IOI**, **Doc-String**) and five budgets (K ∈ {50, 100, 200, 400, 800}). The summary below focuses on the strict-budget regime (K ≤ 200) where method choice most affects fidelity; extended results (K ∈ {400, 800}, full FADE metrics, sensitivity sweeps) are reported in the paper appendices.

### Behavioral fidelity on IOI (last-token KL ↓)

Mean ± std across prompts. FAP-Synergy dominates at strict budgets; gradient-based methods converge as K grows.

| Model | Method | K=50 | K=100 | K=200 |
| --- | --- | --- | --- | --- |
| **Llama-3.2-1B** | CLT-RelP | 1.32 ± 0.62 | 1.15 ± 0.54 | 0.87 ± 0.41 |
| | FAP | 1.33 ± 0.60 | 1.13 ± 0.53 | **0.85 ± 0.42** |
| | **FAP-Synergy** | **1.22 ± 0.56** | **1.12 ± 0.52** | **0.85 ± 0.42** |
| | Activation-Magnitude | 1.59 ± 0.69 | 1.52 ± 0.65 | 1.32 ± 0.59 |
| | FActP | 1.26 ± 0.63 | 1.24 ± 0.64 | 1.22 ± 0.63 |
| **Gemma-2-2B** | CLT-RelP | 0.86 ± 0.43 | 0.75 ± 0.37 | 0.53 ± 0.28 |
| | FAP | 0.91 ± 0.46 | 0.74 ± 0.37 | **0.52 ± 0.29** |
| | **FAP-Synergy** | **0.82 ± 0.42** | **0.73 ± 0.36** | **0.52 ± 0.29** |
| | Activation-Magnitude | 1.29 ± 0.59 | 1.25 ± 0.58 | 1.18 ± 0.56 |
| | FActP | 0.81 ± 0.37 | 0.77 ± 0.38 | 0.76 ± 0.37 |

### Effective Budget — ~33% downstream cost reduction

On IOI, **FAP-Synergy at K=50 functionally matches base FAP and CLT-RelP at K=75** (Llama: 1.22 vs. 1.22 / 1.23; Gemma: 0.82 vs. 0.81 / 0.80). Because interpretation and evaluation costs scale linearly per retained feature, synergy effectively grants the pipeline 25 "free" feature occurrences and cuts downstream API spend by **~33%**.

### Causal sparsity gap — ~40× compression vs. active-set random

PIE at K=100 reaches a fidelity level that random sampling from the prompt-active feature set only achieves with **≈4,000 features** (Figure 2 in the paper). This ≈40× gap shows that the vast majority of *active* CLT features are causally irrelevant — strongly motivating the *prune-first, interpret-later* paradigm.

### Interpretability quality (FADE metrics, K=50 Doc-String)

FAP-Synergy yields the highest overall Clarity / Purity / Responsiveness in the strictest budget regime (mean ± std):

| Model | Method | Clarity | Purity | Responsiveness |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B | CLT-RelP | 0.710 ± 0.043 | 0.501 ± 0.052 | 0.586 ± 0.050 |
| Llama-3.2-1B | FAP | 0.733 ± 0.042 | 0.540 ± 0.051 | 0.610 ± 0.048 |
| Llama-3.2-1B | **FAP-Synergy** | **0.765 ± 0.039** | **0.581 ± 0.049** | **0.652 ± 0.049** |
| Gemma-2-2B | CLT-RelP | 0.739 ± 0.040 | 0.631 ± 0.045 | 0.708 ± 0.041 |
| Gemma-2-2B | FAP | 0.770 ± 0.035 | 0.644 ± 0.046 | 0.722 ± 0.042 |
| Gemma-2-2B | **FAP-Synergy** | **0.774 ± 0.037** | **0.675 ± 0.044** | **0.735 ± 0.039** |

As budgets relax (K ≥ 400), CLT-RelP, base FAP, and FAP-Synergy converge — for relaxed-sparsity workflows, base FAP or CLT-RelP is the computationally optimal choice.

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
