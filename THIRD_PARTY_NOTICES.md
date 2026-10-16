# Third-party notices

The root [MIT License](LICENSE) applies to first-party PIE code. It does not replace the licenses of bundled dependencies, adapted code, model checkpoints or external corpora. Third-party attribution is preserved in the source tree.

| Component | Use in PIE | License and attribution |
| --- | --- | --- |
| [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) | Replacement model, CLT loading, attribution and graph support | Bundled under its MIT license in [vendor/circuit-tracer/LICENSE](vendor/circuit-tracer/LICENSE) |
| [RelP](https://github.com/FarnoushRJ/RelP) | Relevance-propagation rules adapted for CLT occurrence selection | Apache-2.0; full text in [licenses/RelP-APACHE-2.0.txt](licenses/RelP-APACHE-2.0.txt) |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Model hooks and adapted gated-MLP forward logic | MIT, copyright 2022 TransformerLensOrg; full notice in [licenses/TransformerLens-MIT.txt](licenses/TransformerLens-MIT.txt) |

The bundled tracer is based on commit `23a2c1040d5f245beac702540255fe776a68f20b` and includes local helper additions. Use the bundled source for the implementation provided here.

The CLT-RelP implementation in `pie/pruning/relp.py` and its adapter in `pie/behavior/clt_relp.py` apply relevance-propagation rules in the replacement-model/CLT setting. These files are adaptations. Changes include CLT write scoring, occurrence-based selection and unconditional restoration of temporary model hooks; upstream notices remain applicable to adapted portions.

Python dependencies installed separately retain their distributed licenses. The package does not bundle the Gemma/Llama model weights, transcoder weights or Wikipedia corpus. Checkpoint identifiers and revisions are listed in [configs/models.json](configs/models.json); acquire those resources from their original hosts under their original terms.
