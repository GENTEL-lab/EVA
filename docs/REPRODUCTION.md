# Reproducing EVA computations

This guide is the entry point for the current source candidate. It separates
fresh model inference, recalculation of stored predictions, and summary plotting.
Use the full source checkout: the Python wheel does not contain experiment inputs.
Current validation is reported in [RELEASE_STATUS.md](RELEASE_STATUS.md).

## Install the full runtime

The Docker recipe targets Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
Transformers 4.55.0 and MegaBlocks 0.7.0. A CUDA GPU is needed for the model.
From the repository root:

```bash
docker build -f docker/Dockerfile -t eva-repro:local .
mkdir -p checkpoint results
docker run --rm -it --gpus device=0 \
  -v "$PWD":/eva -w /eva eva-repro:local bash
python -m pytest -q tests
```

If the default package registry is unreachable, select an accessible index
with `docker build --build-arg PIP_INDEX_URL=INDEX_URL ...`; dependency versions
remain pinned. Record the selected index with the build log.

On an existing compatible CUDA runtime, use an isolated environment and
`python -m pip install -e '.[benchmark,design,dev,notebook]'`.
This does not build MegaBlocks/grouped-GEMM; use the Docker recipe for those
compiled dependencies. EVA attention uses PyTorch SDPA. The unused external
`flash-attn` package is not required or installed by this recipe. Missing dependencies fail explicitly.

## One complete benchmark from a public checkpoint

```bash
python scripts/download_reproduction_checkpoint.py \
  --model EVA_1.4B_CLM --destination checkpoint
python scripts/reproduce_milena.py \
  --checkpoint checkpoint/EVA_1.4B_CLM --output results/milena_14b
```

The downloader fixes Hugging Face revision
`514db6705637c1ec963b728768fc9b34728699ee`. The workflow verifies the weight,
config, tokenizer and input hashes before scoring all 135 Milena sequences.
It uses the bundled historical model implementation, BF16, batch size 1,
no conditioning, `<bos>5{RNA}3<eos>`, and summed causal log likelihood excluding
`5`, `3` and `<eos>` targets. No alternative objective is chosen to fit a score.
The public config differs from the historical config only in `moe_world_size`
(1 instead of 4); single-GPU execution sets this to 1 in both cases.

The current submission labels are retained as confirmed by the author. Their
positional association is frozen by the exact FASTA and label hashes. Header
metadata from old archives is not used to replace the experimental readouts.
The fixed protocol and resource identities are in
`reproduction/milena_14b/manifest.json`.

Outputs are `predictions.csv`, `report.json`, `inference.log`, the detailed
inference report, and `plot/` with PNG/SVG diagnostics. The report contains
Spearman, its full-precision difference from the current table, timing, GPU
memory, package versions and hashes. The current reference is
0.8360456283218484. The machine comparison uses the stored CSV precision
as a diagnostic, not as a scientific acceptance threshold. The manuscript
plot does not annotate an individual Milena coefficient. The observed delta
and archive-vector comparison are disclosed separately in the expected output.
Exit 2 means the run finished but the metric differs;
other failures raise an error and preserve a failure report. Exit 0 for
`--check-only` verifies inputs only. A numerical match alone does not establish
all scientific provenance; inspect the report and release status.

The older `scripts/reproduce_dms.py` retains its mean-scoring behavior and
EVA-21M default for compatibility. Changing its reference row does not select
the historical 1.4B protocol. Use `reproduce_milena.py` for this example.

## Existing predictions and summary tables

For a CPU-only summary audit (no EVA installation required):

```bash
python scripts/benchmark_release.py audit-reference \
  --manifest reproduction/benchmark_release/reference_manifest.json \
  --data-root reproduction/benchmark_release/reference \
  --output results/reference_audit --plot
```

This command currently returns 2 for documented coverage/sample-count issues;
its outputs are still written. It does not run a model. See
[BENCHMARK_RELEASE.md](BENCHMARK_RELEASE.md) for keyed prediction CSVs,
metric calculation and the notebook. Author confirmation of data correctness
does not by itself resolve an undocumented difference in sample inclusion.

## Training and other analyses

For a small synthetic pretraining/midtraining/fine-tuning check on one GPU:

```bash
python scripts/smoke_workflows.py --training-only --output results/training_smoke
```

This performs two optimizer steps per training stage and checks exact saved
weight reloads; it does not consume a paper checkpoint or reproduce pretraining.


- [Training](../training/pretrain/README.md) and
  [fine-tuning](../finetune/aptamer/script/README.md): complete-source entry
  points and the synthetic forward/backward/save/reload smoke workflow.
- [Benchmark protocols](BENCHMARK_PROTOCOLS.md) and
  [paper workflow coverage](PAPER_WORKFLOWS.md): competitor sources and
  method-specific limitations.
- [Essentiality](ESSENTIALITY_REPRODUCTION.md): recomputation of stored
  predictions with stable row identity and separate ablation fractions.
- [SAE analyses](SAE_REPRODUCTION.md): stored-result calculations and
  separately identified inference demonstrations.

Do not treat small synthetic training tests as a rerun of pretraining or a
reproduction of all reported experiments. Historical manifests retain their
original paths for traceability; they are not executable installation commands.
