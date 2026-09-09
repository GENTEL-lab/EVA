# Comparison-model resources

Third-party models can be prepared locally and run in their own environments.
Use the supported [local-checkpoint interfaces](BENCHMARK_PROTOCOLS.md#comparison-model-execution-not-plotting),
or export predictions from an upstream scorer to the
[common metric evaluator](BENCHMARK_PROTOCOLS.md#using-your-own-model-outputs).
The benchmark step does not require downloading a model from its host.

| Method | Resources | Components to prepare locally |
|---|---|---|
| RNA-FM, RNABERT, RNA-MSM | [RNA-FM upstream](https://github.com/ml4bio/RNA-FM); [recorded model identities](../reproduction/release/recovered_rna_snapshots_20260909.json) | Model/config/tokenizer snapshot and a compatible Transformers/Multimolecule environment; choose the appropriate masking, length and sequence/MSA protocol. |
| GENA-LM | [Official code and models](https://github.com/AIRI-Institute/GENA_LM) | Checkpoint, DNA tokenizer and any required custom model code. |
| ERNIE-RNA | [Official repository](https://github.com/Bruce-ywj/ERNIE-RNA/blob/main/README.md) | ERNIE/fairseq environment and the pretrained checkpoint selected for the comparison. |
| Evo2 | [Official models and runtime](https://github.com/arcinstitute/evo2) | Selected model size/checkpoint and its Vortex/Transformer Engine environment. |
| CodonFM | [Official repository](https://github.com/NVIDIA-BioNeMo/CodonFM) | Checkpoint, configuration, CodonFM environment and preprocessing settings. |
| AIDO.RNA | [ModelGenerator](https://github.com/genbio-ai/ModelGenerator), [AIDO](https://github.com/genbio-ai/AIDO) | ModelGenerator environment, matching API and local checkpoint. |
| GenerRNA | [Official checkpoint and tokenizer](https://huggingface.co/pfnet/GenerRNA) | Model, complete tokenizer and the upstream likelihood-scoring implementation. |
| CodonGPT | [Preserved scoring worker](../reproduction/benchmark/upstream/rna/external_scripts/compute_codongpt_ll.py) | Compatible checkpoint/tokenizer, Transformers version and codon preprocessing. |
| ESM | [Official models and fair-esm](https://github.com/facebookresearch/esm) | Local weights and companion files, reference/DMS inputs and the selected scoring strategy or ensemble. |
| ProGen3 | [Official models and scoring CLI](https://github.com/Profluent-AI/progen3) | Local weights and the official runtime; record the scorer's forward/reverse convention. |

Use whichever local model version your benchmark specifies and record its
identity, environment and scoring protocol with the output. Reproducing a
particular published row additionally requires that row's checkpoint and
protocol. [Paper workflow coverage](PAPER_WORKFLOWS.md) records the established
historical bindings and the components still needed for that purpose.

## Recovered comparison-model files: September 9, 2026

The [RNA snapshot record](../reproduction/release/recovered_rna_snapshots_20260909.json)
contains revision identifiers and SHA256 values for 15 RNA-FM, RNABERT and
RNA-MSM weight/config/tokenizer files. Their saved configs record Transformers
4.50.0; the exact historical Multimolecule version has not been established.
Matching local files can use the supplied manifests in
`reproduction/release/model_manifests/`. Keep each manifest outside its model
folder. For a different local checkpoint, create a manifest with the
[existing snapshot command](BENCHMARK_PROTOCOLS.md#comparison-model-execution-not-plotting).

The [additional asset record](../reproduction/release/recovered_competitor_assets_20260909.json)
contains 13 file identities from CodonFM 600M/1B, ERNIE-RNA, AIDO.RNA and
GenerRNA caches, including an ERNIE environment-file hash. These records help
identify local assets; execution scope is documented in the workflow guide.
