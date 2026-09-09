# Official comparison-model resources

Links checked on September 9, 2026. These are resource and runtime entry points,
not new claims of paper checkpoint identity. Use the recorded original revision
when it is available; unverified current weights are not substituted. Exact
execution gaps are listed in [PAPER_WORKFLOWS.md](PAPER_WORKFLOWS.md).

| Method | Upstream resources | Environment boundary |
|---|---|---|
| RNA-FM, RNABERT, RNA-MSM | Historical configuration uses the Multimolecule model ports; the three recorded snapshot revisions currently return HTTP 404 | Recover the matching checkpoint plus tokenizer and Multimolecule package version before inference. |
| GENA-LM | [Official code and models](https://github.com/AIRI-Institute/GENA_LM) | DNA tokenizer/custom-code runtime; recorded snapshot resolves but model-name binding is unresolved. |
| ERNIE-RNA | [Official repository and checkpoint instructions](https://github.com/Bruce-ywj/ERNIE-RNA/blob/main/README.md) | Use its fairseq environment and original pretrained checkpoint, not a structure fine-tuned model. |
| Evo2 | [Official models and runtime](https://github.com/arcinstitute/evo2) | Separate Vortex/Transformer Engine environment; exact paper model and runtime revisions remain required. |
| CodonFM | [Official repository and model downloads](https://github.com/NVIDIA-BioNeMo/CodonFM) | Its own Docker recipe and codon tokenizer; original and accelerated implementations must not be interchanged without a protocol check. |
| AIDO.RNA | [ModelGenerator](https://github.com/genbio-ai/ModelGenerator), [AIDO model collection](https://github.com/genbio-ai/AIDO) | Match the paper backbone and ModelGenerator API version; current releases do not prove compatibility with the historical worker. |
| GenerRNA | [Official model checkpoint and tokenizer](https://huggingface.co/pfnet/GenerRNA) | Pair model_updated.pt with its corresponding tokenizer and architecture. Original file hashes remain needed. |
| CodonGPT | The paper-linked NanilTx/codonGPT_pub repository returned HTTP 404 in this check | Obtain the original repository/version and weights; a similarly named software project is not a substitute. |
| ESM | [Official model downloads and fair-esm environment](https://github.com/facebookresearch/esm) | Recover exact model size/ensemble and original ProteinGym reference/DMS inputs. |
| ProGen3 | [Official models, Docker runtime and scoring CLI](https://github.com/Profluent-AI/progen3) | Official scorer averages forward/reverse likelihood; verify this against the original EVA comparison protocol and fix the corresponding weight revision. |

`reproduction/release/competitor_resource_verification.json` records the
read-only Hugging Face revision checks and available LFS object hashes.
No third-party weight download or inference is claimed by this resource index.
