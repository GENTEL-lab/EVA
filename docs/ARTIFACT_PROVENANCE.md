# Manuscript artifact correspondence

The author confirmed on September 9, 2026 that the released checkpoint and dataset files listed below were used in the manuscript. This document maps the named manuscript resources to immutable public snapshots and the accompanying file-level inventory. The confirmation establishes paper-use correspondence; checksum evidence establishes file identity.

## Models and training data

| Manuscript resource | Released artifact | Fixed revision |
|---|---|---|
| EVA 21M | [EVA_21M/](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_21M) (weights, configuration, tokenizer and training configuration) | `514db6705637c1ec963b728768fc9b34728699ee` |
| EVA 145M | [EVA_145M/](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_145M) (weights, configuration, tokenizer and training configuration) | `514db6705637c1ec963b728768fc9b34728699ee` |
| EVA 437M | [EVA_437M/](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_437M) (weights, configuration, tokenizer and training configuration) | `514db6705637c1ec963b728768fc9b34728699ee` |
| EVA 1.4B GLM | [EVA_1.4B_GLM/](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_1.4B_GLM) (weights, configuration, tokenizer and training configuration) | `514db6705637c1ec963b728768fc9b34728699ee` |
| EVA 1.4B CLM | [EVA_1.4B_CLM/](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_1.4B_CLM) (weights, configuration, tokenizer and training configuration) | `514db6705637c1ec963b728768fc9b34728699ee` |
| OpenRNA v1 training corpus | [OpenRNA_v1_114M.fa.gz](https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M/blob/103c79aab4d625828721a25ab98be51763c326bb/OpenRNA_v1_114M.fa.gz) | `103c79aab4d625828721a25ab98be51763c326bb` |
| Released validation set | [eva_training_data/EVA_validation_100k.fa](https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M/blob/103c79aab4d625828721a25ab98be51763c326bb/eva_training_data/EVA_validation_100k.fa) | `103c79aab4d625828721a25ab98be51763c326bb` |

The model card distinguishes the primary 1.4B mixed-objective checkpoint (`EVA_1.4B_GLM`) from the separate CLM-only checkpoint (`EVA_1.4B_CLM`). The representative Milena reproduction explicitly uses the latter. Parameter count alone does not identify a checkpoint.

## Benchmark inputs and results

| Manuscript component | Exact artifact binding |
|---|---|
| Representative Milena benchmark (Figure 2) | [Manifest](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/milena_14b/manifest.json): EVA_1.4B_CLM, 135-sequence FASTA, paired labels and archived prediction file; all are identified by hashes. |
| ncRNA benchmark summary (Figure 2) | [ncRNA_13datasets_spearman.csv](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark_release/reference/ncRNA_13datasets_spearman.csv) |
| mRNA benchmark summary (Figure 2) | [mRNA_5datasets_spearman.csv](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark_release/reference/mRNA_5datasets_spearman.csv) |
| Protein benchmark summary (Figure 2) | [protein_20datasets_spearman.csv](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark_release/reference/protein_20datasets_spearman.csv) |

## File identity and verification

[CSV inventory](../examples/reproduction/release/artifact_provenance/EVA_Artifact_Provenance.csv) and [JSON inventory](../examples/reproduction/release/artifact_provenance/EVA_Artifact_Provenance.json) contain 44 records: 35 model components, two OpenRNA files and seven benchmark input/result files. Each record includes repository, full revision, exact path, size, SHA256, intended role and the source of verification evidence.

Large-file checksums come from Hugging Face LFS metadata. Small model files and benchmark files were downloaded and hashed; benchmark hashes were matched against the released manifests. See [verification instructions](../examples/reproduction/release/artifact_provenance/README.md).

This inventory covers released EVA models, OpenRNA and the listed benchmark artifacts. The score-table identities do not specify every third-party model run or every experimental artifact in the manuscript. Other workflows are indexed in [Paper workflows](PAPER_WORKFLOWS.md).
