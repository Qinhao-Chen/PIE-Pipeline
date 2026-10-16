# Local task inputs

Datasets are not distributed with PIE. Generate inputs locally or provide your own JSONL file, then pass its path through `--data-file`. Generated files under `data/` are ignored by Git.

## IOI generation

The original framework's generator is retained in `scripts/ioi.py`:

```bash
python scripts/ioi.py --N 100 --seed 0 --device cpu \
  --out data/tasks/ioi_pairs.jsonl --format jsonl
```

It generates clean prompts and aligned corrupted variants using configurable templates, a random seed and a corruption specification. Use `--help` for the available options. The generator loads a tokenizer and may download its files on first use. PIE validates the generated pairs again with the selected subject-model tokenizer.

## Paired file format

Use JSONL with one record per line and unique pair IDs.

| Task | Shared fields | Additional fields |
| --- | --- | --- |
| IOI | `id`, `text_clean`, `text_corr` | `io_clean`, `io_corr` |
| DocString | `id`, `text_clean`, `text_corr` | `target_clean`, `target_corr` |

For IOI, both source strings must end in the literal `io_clean` answer. The loader removes that suffix and its leading whitespace before inference. `io_corr` records the corrupted task's answer label.

For DocString, supply prefixes ending at `:param` and their corresponding target names. A DocString dataset or generator is not included.

The [prefix checker](../pie/behavior/protocol.py) validates answer-token boundaries, requires equal clean/corrupted prefix lengths, and accepts prefixes between 8 and 128 model tokens. It rejects invalid pairs rather than truncating the answer boundary.

If `--data-file` is omitted, the pruning CLI looks for your local `data/tasks/{task}_pairs.jsonl`. Those paths are not populated by installation.
