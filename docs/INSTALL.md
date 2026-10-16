# Installation

Use Python 3.10 or 3.11 and run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install ./vendor/circuit-tracer --no-deps
python -m pip check
```

The editable installation exposes the `pie` command and Python package without setting `PYTHONPATH`. `requirements.txt` installs the GPU and API extras. Install `requirements-gpu.txt` or `requirements-api.txt` separately when only one is needed.

## GPU and model setup

Pruning and fresh activation collection require a CUDA-capable PyTorch installation and enough GPU memory for the model and CLT. The bundled tracer includes the helper methods used by PIE; install it after the runtime dependencies.

| Dependency | Version |
| --- | --- |
| NumPy | 1.24.4 |
| PyTorch | 2.6.0 |
| Transformers | 4.51.0 |
| TransformerLens | 2.16.1 |
| huggingface-hub | 0.34.1 |
| datasets | 3.6.0 |
| tqdm | 4.67.1 |
| safetensors | 0.5.3 |
| circuit-tracer | Bundled source |

These are direct dependency pins. Pip resolves transitive dependencies. Use a CUDA driver and PyTorch wheel compatible with your machine.

[Model profiles](../configs/models.json) specify the supported Gemma and Llama model/CLT pairs. Obtain checkpoint access and configure Hugging Face authentication locally. Add `--allow-download` to pruning or evaluation for the initial download; subsequent offline use needs a complete local cache.

## Local inputs

No datasets or experimental results are distributed. [Generate IOI inputs or prepare a local task file](TASKS.md), then follow the [pruning workflow](../README.md#stage-i-prune). Generated data and outputs are ignored by Git.

## API setup

The interpretation and FADE stages read `OPENAI_API_KEY` from the environment. Set `PIE_LLM_MODEL` to the API model you want to use in the README examples. Pass `--base-url` for another OpenAI-compatible endpoint. Keep credentials outside the code tree.

Interpretation requires a compatible feature-history repository supplied through `--scan`. Its activation histories are separate from model/CLT weights and are not bundled.

See [software checks](VALIDATION.md) for tests using synthetic inputs.
