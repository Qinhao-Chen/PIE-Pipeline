# Tool organization

PIE is organized around the Prune → Interpret → Evaluate workflow.

- The `pie` CLI exposes all three stages, with matching scripts under `scripts/`.
- `run_behavior.py` remains a compatibility entry point for pruning and behavioral comparisons.
- Earlier description and evaluation paths forward to the maintained stage commands.
- Interpretation reads current pruning JSONL at an explicitly selected budget, alongside historical input formats.
- Explanation evaluation uses CLT encoder input hooks and the CLT activation function, excludes BOS/padding, handles tied scores, and reports invalid replies.
- The original IOI generator is available in `scripts/ioi.py` for local input generation.

Saved-analysis commands, their helper package, bundled datasets, result exports and reviewer-release artifacts have been removed. Generated data, caches and outputs are ignored by Git. Selection algorithms remain available in their existing modules.
