# Official comparison-model resources

Links checked on September 9, 2026. These are resource and runtime entry points,
not new claims of paper checkpoint identity. Use the recorded original revision
when it is available; unverified current weights are not substituted. Exact
execution gaps are listed in [PAPER_WORKFLOWS.md](PAPER_WORKFLOWS.md).

| Method | Upstream resources | Environment boundary |
|---|---|---|
| RNA-FM, RNABERT, RNA-MSM | [RNA-FM upstream](https://github.com/ml4bio/RNA-FM); the historical Multimolecule ports use three revisions that still return HTTP 404 | Matching cached checkpoint/tokenizer hashes are now recorded below; public download and the original Multimolecule runtime remain unresolved. |
| GENA-LM | [Official code and models](https://github.com/AIRI-Institute/GENA_LM) | DNA tokenizer/custom-code runtime; recorded snapshot resolves but model-name binding is unresolved. |
| ERNIE-RNA | [Official repository and checkpoint instructions](https://github.com/Bruce-ywj/ERNIE-RNA/blob/main/README.md) | Use its fairseq environment and original pretrained checkpoint, not a structure fine-tuned model. |
| Evo2 | [Official models and runtime](https://github.com/arcinstitute/evo2) | Separate Vortex/Transformer Engine environment; exact paper model and runtime revisions remain required. |
| CodonFM | [Official repository and model downloads](https://github.com/NVIDIA-BioNeMo/CodonFM) | Cached 600M/1B weights and config hashes are recorded below; public revision/runtime linkage is still required. |
| AIDO.RNA | [ModelGenerator](https://github.com/genbio-ai/ModelGenerator), [AIDO model collection](https://github.com/genbio-ai/AIDO) | Match the paper backbone and ModelGenerator API version; current releases do not prove compatibility with the historical worker. |
| GenerRNA | [Official model checkpoint and tokenizer](https://huggingface.co/pfnet/GenerRNA) | Cached model_updated.pt and tokenizer-config hashes are recorded below; the complete tokenizer/runtime and public-version binding remain required. |
| CodonGPT | The paper-linked NanilTx/codonGPT_pub repository returned HTTP 404 in this check | Obtain the original repository/version and weights; a similarly named software project is not a substitute. |
| ESM | [Official model downloads and fair-esm environment](https://github.com/facebookresearch/esm) | Recover exact model size/ensemble and original ProteinGym reference/DMS inputs. |
| ProGen3 | [Official models, Docker runtime and scoring CLI](https://github.com/Profluent-AI/progen3) | Official scorer averages forward/reverse likelihood; verify this against the original EVA comparison protocol and fix the corresponding weight revision. |

`reproduction/release/competitor_resource_verification.json` records the
read-only Hugging Face revision checks and available LFS object hashes.
No third-party weight download or inference is claimed by this resource index.

## Recovered comparison-model files: September 9, 2026

The original RNA-FM, RNABERT and RNA-MSM cache snapshots were recovered and
all 15 weight/config/tokenizer files were hashed. Their model configurations
record Transformers 4.50.0; the exact Multimolecule runtime is still unbound.
The [recovered identity record](../reproduction/release/recovered_rna_snapshots_20260909.json)
contains each historical revision and every file SHA256. Ready-to-use manifests
for the existing `--model-manifest` interface are in
`reproduction/release/model_manifests/`. Keep a manifest outside its model folder.

The historical public revisions still return 404. Current public weights have
different file hashes; no tensor-level equivalence is claimed and no current
weight is substituted. The recovered files are not rehosted in this source
repository. Original environment/protocol correspondence and a working public
download remain required before these methods can be claimed fully reproducible.

The [additional recovered-asset record](../reproduction/release/recovered_competitor_assets_20260909.json)
adds 13 file identities from CodonFM 600M/1B, ERNIE-RNA, AIDO.RNA and GenerRNA
caches. These include configurations and the ERNIE environment-file hash; the
two ERNIE copies have identical hashes. A local cached weight is not itself a
public download or proof that its full environment/assay protocol is recovered.
No additional competitor inference was performed in this repository update.
