# Reproducing EVA computations

Choose one of the three workflows below. Paths are relative to the complete
source checkout; the Python wheel does not contain benchmark inputs. Use a new
output directory for each run. See [Installation](INSTALLATION.md) for setup
and [validation status](RELEASE_STATUS.md) for dated execution evidence.

## CPU: recalculate stored predictions

**Requirements:** Python 3.10 or 3.11; standard library only, no model or GPU.
From the repository root:

```bash
python3 scripts/reproduce_historical_benchmark.py \
  --artifact-only --output results/milena_stored
```

This verifies bundled hashes and recomputes the metric from all 135 archived
Milena predictions. Expected outputs:

| File | Contents |
|---|---|
| `results/milena_stored/report.json` | `n: 135`, `arithmetic_match: true`, Spearman `0.8360456283218484` |
| `results/milena_stored/sequence_label_audit.csv` | 135 rows with sequence IDs, labels and stored scores |

The explicit `--artifact-only` mode returns 0 for successful recalculation.
It does not execute a model. Historical metadata columns are retained for
traceability and do not replace the selected labels.
For table-wide summaries, see [benchmark calculations](BENCHMARK_RELEASE.md).
That separate audit can return 2 for documented coverage/sample-count issues
while still writing its reports.

## GPU: run the complete Milena benchmark

**Requirements:** the [Docker GPU runtime](INSTALLATION.md#gpu-runtime-docker),
one NVIDIA GPU and about 3.06 GB of downloaded weights. The measured runtime is
A100/PyTorch 2.5.1/CUDA 12.4; other hardware has not been certified.
After entering the container with the checkout mounted at `/eva`, run:

```bash
python scripts/download_reproduction_checkpoint.py \
  --model EVA_1.4B_CLM --destination checkpoint
python scripts/reproduce_milena.py \
  --checkpoint checkpoint/EVA_1.4B_CLM --output results/milena_fresh
```

The download fixes Hugging Face revision
`514db6705637c1ec963b728768fc9b34728699ee`. Before inference, the runner checks
weights, config, tokenizer, FASTA and label hashes against
`examples/reproduction/milena_14b/manifest.json`.
The protocol uses all 135 samples, the bundled historical implementation,
BF16, batch size 1, no conditioning and summed causal log likelihood excluding
direction/EOS targets. The prompt is `<bos>5{RNA}3<eos>`. Input order and
sequence-to-label association are frozen by hashes. The public config changes
historical `moe_world_size` from 4 to 1; this workflow uses one GPU.

| Output inside `results/milena_fresh/` | Contents |
|---|---|
| `predictions.csv` | 135 keyed predictions with sequence and experimental value |
| `report.json` | Metric, comparison, execution status, timing, environment and hashes |
| `inference.log`, `inference/report.json` | Backend log and detailed execution evidence |
| `plot/` | PNG/SVG diagnostics and plot report |

### Expected result and exit codes

The clean-environment run produced **0.8394237924835843**, compared with the
archived **0.8360456283218484**: difference **0.0033781641617359748**. This
observation is retained and accepted for the representative example; it is
not an execution failure. The original table and predictions are unchanged.
[Expected outputs](../examples/reproduction/milena_14b/expected/README.md) give the evidence.

That run took 12.08 seconds for inference and 23.40 seconds for the runner,
with 3,157,299,200 bytes (2.94 GiB) peak PyTorch-allocated GPU memory on an
A100-SXM4-80GB. Downloads are excluded. Allocated memory is not total device
usage or a minimum GPU specification.

- Default: returns **0** after valid inference, metrics and plots, including
  when the reference comparison differs.
- `--strict-reference`: returns **2** after an otherwise valid run if comparison
  at the stored reference precision fails. This is diagnostic, not a scientific
  tolerance. Use a new output directory when rerunning in strict mode.
- `--check-only`: returns **0** for input/checkpoint verification only;
  `fresh_inference` remains false and no model is run.
- Missing dependencies, changed inputs/weights, invalid predictions and plotting
  failures remain errors. Failures after output initialization are recorded in
  `report.json`. Existing output directories are never overwritten.

The report separates `execution_status` (`COMPLETED`, `FAILED`,
`INPUTS_VERIFIED_ONLY`) from `comparison_status` (`MATCH`, `DIFFERENCE`, `NOT_RUN`).
Existing numerical comparison fields are preserved. The legacy
`paper_result_reproduced` field is not set automatically by a successful command;
this example does not establish reproduction of every paper result.
The older `scripts/reproduce_dms.py` retains its EVA-21M/mean-scoring default.
Changing its reference row does not select the historical 1.4B protocol.

## Training and fine-tuning

**Requirements:** complete checkout and the GPU runtime. This small synthetic
engineering example needs no downloaded paper checkpoint:

```bash
python scripts/smoke_workflows.py \
  --training-only --output results/training_smoke
```

It performs two optimizer steps each for pretraining, mid-training and
fine-tuning, verifies parameter updates, then saves and reloads weights.
Inspect the output report for each stage. It validates the exposed training
machinery; it does not rerun large-scale pretraining.

- [Training](../training/pretrain/README.md): stage entry points and configuration fields.
- [Fine-tuning](../training/finetune/aptamer/script/README.md): checkpoint, data and container paths.
- [CLI guide](USAGE.md): inputs, conditioning and task parameters.
- [Paper workflows](PAPER_WORKFLOWS.md): result/resource mapping and omitted components.
- [Reviewer requirements](REVIEWER_REQUIREMENTS.md): six requests and supporting evidence.

## Other analyses and historical sources

[Notebooks](../examples/notebooks/README.md) provide interactive analysis and plotting;
[reproduction resources](../examples/reproduction/README.md) hold fixed inputs, protocols,
reference outputs and historical implementations. They work together with the
command-line entry points in `scripts/`. For example, the benchmark notebook
calls `scripts/benchmark_release.py` with manifests and tables from
`examples/reproduction/benchmark_release/`.

Third-party models can run from local checkpoints or in their own upstream
environments. Export keyed predictions to use the common metric evaluator;
see [using your own model outputs](BENCHMARK_PROTOCOLS.md#using-your-own-model-outputs).

[Benchmark protocols](BENCHMARK_PROTOCOLS.md), [essentiality](ESSENTIALITY_REPRODUCTION.md)
and [SAE analyses](SAE_REPRODUCTION.md) distinguish stored-result calculations
from inference. Sources under `examples/reproduction/benchmark/upstream/` and
`examples/reproduction/sae/archive/` preserve historical paths and behavior. Follow the
documented portable entry points instead of executing these archives unchanged.
