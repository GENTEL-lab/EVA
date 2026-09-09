# Software requirements and evidence

This page maps the six software requests in the September 2, 2026 review to
public entry points and validation evidence. It describes release scope,
not editorial acceptance.

| Requirement | Public entry and evidence | Status and remaining scope |
|---|---|---|
| Benchmark scores for EVA and comparison methods | [Milena workflow](REPRODUCTION.md#gpu-run-the-complete-milena-benchmark), [comparison protocols](BENCHMARK_PROTOCOLS.md), [workflow index](PAPER_WORKFLOWS.md) | EVA representative inference verified; competition-wide reproduction remains partial where original resources or producers are missing. |
| Executable training/evaluation components, or explicit omissions | [Training](../training/pretrain/README.md), [fine-tuning](../training/finetune/aptamer/script/README.md), [smoke workflow](REPRODUCTION.md#training-and-fine-tuning) | Small training and save/reload verified. Full pretraining, missing dense-model code and unresolved paper configurations are not claimed. |
| Explicit dependency failures | `tests/test_reproducibility.py`; [validation](RELEASE_STATUS.md) | Dependency failures, nonfinite scores and invalid training paths tested. Archived legacy sources are not the supported CLI. |
| Precise checkpoint and dataset identities | `examples/reproduction/milena_14b/manifest.json`, [resources](RESOURCES.md), [comparison resources](OFFICIAL_MODEL_RESOURCES.md) | Representative versions/hashes fixed. Other missing versions remain explicit. Code archive: [Figshare](https://doi.org/10.6084/m9.figshare.33490096.v3). |
| Public prompt, conditioning and input specifications | [CLI reference](USAGE.md), [Milena protocol](REPRODUCTION.md#gpu-run-the-complete-milena-benchmark), [benchmark protocols](BENCHMARK_PROTOCOLS.md) | Documentation covers supported entry points; omitted components identified separately. |
| One end-to-end benchmark | [Download → inference → metrics → plots](REPRODUCTION.md#gpu-run-the-complete-milena-benchmark), [expected output](../examples/reproduction/milena_14b/expected/README.md) | Completed for 135 Milena samples. Spearman is 0.84, rounded to two decimal places; predictions and execution records are provided. |

## Checking the evidence

1. Follow [Installation](INSTALLATION.md), then the selected path in
   [Reproduction](REPRODUCTION.md). CPU recalculation and GPU inference are distinct.
2. Inspect predictions/reports: IDs, counts, hashes, environment and comparison
   results. `--check-only` verifies resources without running inference.
3. Use [Paper workflows](PAPER_WORKFLOWS.md) for method-specific requirements.
   Historical source availability is not counted as executed inference.

The 128-test clean-archive record and the later 139-test/container and two-mode
GPU records are described in [validation status](RELEASE_STATUS.md), with their
respective scope and source identities. GitHub Actions checks current CPU behavior
separately. A passing CPU check or representative
example does not establish reproduction of all manuscript experiments.
The v1.2.0 code release and its [DOI](https://doi.org/10.6084/m9.figshare.33490096.v3) are public.
