# Models and data

The model/download metadata below were read from the public APIs on
September 9, 2026. File sizes use decimal MB/GB. The machine-readable record is
`reproduction/release/public_resource_metadata_20260909.json`.

## EVA checkpoints

[Fixed Hugging Face revision](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee)
contains the following weight files. Each checkpoint also needs its matching
`config.json` and `tokenizer.json`.

| Model directory | Weight bytes | Approximate size | Download |
|---|---:|---:|---|
| `EVA_1.4B_CLM` | 3,057,231,170 | 3.057 GB | [Files](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_1.4B_CLM) |
| `EVA_1.4B_GLM` | 3,057,231,170 | 3.057 GB | [Files](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_1.4B_GLM) |
| `EVA_145M` | 361,225,818 | 0.361 GB | [Files](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_145M) |
| `EVA_21M` | 44,860,956 | 0.045 GB | [Files](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_21M) |
| `EVA_437M` | 1,013,942,634 | 1.014 GB | [Files](https://huggingface.co/GENTEL-Lab/EVA/tree/514db6705637c1ec963b728768fc9b34728699ee/EVA_437M) |

For the representative example, the [reproduction guide](REPRODUCTION.md)
downloads `EVA_1.4B_CLM` through the bundled downloader and verifies all three
files against its frozen manifest. The same downloader supports `EVA_21M` and
`EVA_1.4B_GLM`. Other directories are available through the fixed file listing;
public availability alone does not establish correspondence to every paper row.

## OpenRNA training data

[OpenRNA v1](https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M) provides
the released sequence collection. Use its file listing to choose the required
subset and record the downloaded revision and hashes. Dataset download size
varies by subset. Original training splits and every run-to-checkpoint binding
are not reconstructed by this repository's synthetic training example.

## Deposited analysis data

[Benchmark and experiment data](https://doi.org/10.5281/zenodo.19027142):

| Archive | Bytes | Contents |
|---|---:|---|
| `mrna_codon_optimization.tar.gz` | 25,628 | Archived case-study results |
| `generated_rna_sequences.tar.gz` | 162,986,756 | Generated/reference sequences and calculated features |
| `generrna_generation.tar.gz` | 13,487,400 | Generation-comparison data |
| `mrna_6species.tar.gz` | 39,780,876 | Six-species sequence and feature comparisons |

These data deposits are distinct from the code archive and do not have to be
downloaded for the bundled Milena example. Consult [paper workflows](PAPER_WORKFLOWS.md)
for which figures have a portable command and complete inputs.

## Runtime image

Prefer building the current [Docker recipe](INSTALLATION.md#gpu-runtime-docker)
for the repaired code. The legacy [environment deposit](https://doi.org/10.5281/zenodo.18786989)
currently resolves to a record with eight split archive parts, not a single
`eva_latest.tar.gz` download. The retrieved version DOI is
[10.5281/zenodo.22093406](https://doi.org/10.5281/zenodo.22093406).
Its files are not evidence that it contains the latest main-branch repairs.
The checked metadata and advertised checksums are retained in the JSON record.

## Comparison models and access checks

[Official comparison-model resources](OFFICIAL_MODEL_RESOURCES.md) links the
upstream code and records unresolved historical bindings. Run the independent
access report from the repository root with Python 3.10 or 3.11:

```bash
python scripts/check_resource_links.py --output results/resource-access.json
```

Exit 0 means the report was written. Inspect each `access` field: network
errors and unavailable revisions are reported without changing CPU regression
status. The tool retrieves metadata only; it neither downloads model bodies
nor replaces missing weights. CPU CI performs offline file/hash checks.

The [September 9 access report](../reproduction/release/resource_access_20260909.json)
is a dated observation. External services and their current default branches
can change. Each data/model resource retains its upstream license and terms.
