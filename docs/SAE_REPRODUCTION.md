# Historical SAE results: portable CPU reproduction

This source-checkout workflow reconstructs archived analyses; it does **not**
load EVA/SAE checkpoints, rerun model generation, retrain an SAE, or refold RNA.
The original scientific selection rules and author-selected examples are retained.
It is separate from the small engineering smoke test in `tools/sae_steering.py`.
The source bundle is not included in the EVA wheel.

## Install and run

From the repository root, with Python 3.10 or 3.11:

```bash
python3.11 -m venv .venv-sae-reproduction
source .venv-sae-reproduction/bin/activate
python -m pip install -r examples/reproduction/sae/requirements.txt
python scripts/reproduce_sae.py --output results/sae-archive-run1 --plot
python -m unittest discover -s tests -p test_sae_reproduction.py -v
```

The output path must not already exist. Input hashes are checked before analysis.
The CPU dependency versions are pinned: NumPy 2.0.2, SciPy 1.14.1 and Matplotlib
3.10.5. The complete command was validated with Python 3.11.10 in an existing
Docker image, with the checkout mounted read-only, networking disabled, and no
GPU. This is not a claim that a new Docker image was built. No unpublished EVA
module, weight, training dataset, or absolute server path is needed by this entry
point. The three non-likelihood modes `examples`, `table`, and `coverage` use only
the Python standard library; use `--mode NAME` to run one mode.

## Analysis contracts

| Output | What is reproduced | What is not established |
| --- | --- | --- |
| Likelihood | Eight archived raw JSON files → original summaries → selected 401 case/direction/feature groups → original statistics | New model predictions or an unselected population success rate |
| Eight structure illustrations | Exact author-selected paired records, source cases, diagnostic-pair arithmetic and original SVG | Random sampling or an aggregate generation success rate |
| Table S30 | Archived CSV → all numerical LaTeX rows, checked byte-for-byte against the archived table | Complete raw-record recomputation of 213/225 |
| Generation coverage | Fixed saved cohort → source array indices → available same-run recorded-seed pairs | Exhaustive recovery of all historical runs, or replacement paper numbers |

### Likelihood: preserve the original selection

The original `create_likelihood_causal_figure.py` reads eight summary tables.
The portable entry point first reconstructs those summaries from 4,183 raw rows:

1. Group by `(case_index, record_id, direction, feature_id)`.
2. Choose the first maximum `delta_margin` over **all** archived conditions,
   including `no_steer` and `0x`; ties retain the first occurrence.
3. From the 487 resulting groups, retain only those whose chosen
   `best_condition` is neither `no_steer` nor `0x`.
4. Compute the greater-alternative Wilcoxon signed-rank statistic and
   `mean(delta) / std(delta, ddof=0)`, matching the original effect-size formula.

All fields and the row order of the eight reconstructed summaries match the
archived CSVs. The selected result is 401/401 positive groups, W = 80,601,
p = 9.370116963725616e-68, and d = 1.6985262810017903 (displayed as 1.70).
The displayed statistics match the archived SVG text exactly. Per-dataset
selected counts are 15, 16, 42, 42, 87, 85, 84 and 30, in the order stored in
`examples/reproduction/sae/protocol.json`.

These are **selected best-condition summaries**, not 401 independent generated
RNAs or evidence that every prespecified intervention improved an unselected
population. The command does not replace this historical analysis with a
different scale-selection rule. `--plot` redraws the same data/statistics in a
portable PNG/SVG layout; it does not claim pixel identity with the archived SVG.

### Eight author-selected structure illustrations

The examples are the eight cases explicitly selected by the authors for visual
illustration. Each selector in `protocol.json` fixes the source CSV, source
`case_index`, record ID, direction, feature, condition, sample index and nonempty
sample seed. Exactly one unsteered and one steered row must match. The runner
also matches span, anchor and unchanged prefix/suffix context, validates the
source case, and recomputes diagnostic-pair hits from the stored dot-bracket
structures. It does not call RNAfold again.

`0x` is feature ablation: the selected SAE latent is clamped to zero. It is not
`no_steer`. In the historical steering implementation, `1x` is based on the
selected feature activation at the diagnostic anchor in the full target-state
reference sequence; it is not necessarily the activation of the unsteered
generated sample. The original scoring and hook source is archived for inspection.

The illustration displays **WT-specific diagnostic-pair recovery**. The archived
Table S30 description instead uses direction-dependent
`net_target_score = target_hits - opposite_hits`. Six selected illustrations use
`mutant_state`: their positive WT recovery corresponds to a negative change in
that net target score. Both values are exported, without relabeling or replacing
the author's examples. The original eight-panel SVG is copied byte-for-byte.

### Table S30 and coverage

The archived table contains nine dataset totals summing to 213/225. Its numerical
LaTeX rows can be reconstructed exactly from `summary_by_dataset.csv`; its
original caption is retained as historical text, not newly validated by this
command. The upstream script that produced those success counts has not been
recovered. Consequently the runner does not label 213/225 as raw-record reproduced.

The saved 225-row cohort matches stable descending `bp_distance` sorting of the
archived source cases, retaining the original order for ties. Constructed
mutation cases are identified by the source cases file and original array index,
not by `record_id` alone: the saved cohort contains 152 distinct record IDs.

The separate coverage report keeps runs separate and never infers pairs from row
order or sample index alone. Blank/missing seeds are excluded rather than treated
as equal. Missing scores are errors, not zeros; absent cases remain unverified.
Ambiguous duplicate baselines are excluded. The observed positive-case count is
only a lower bound over available recorded-seed pairs. A case with available
pairs but no observed positive delta is **not** established as a historical
failure, because the full historical run/search budget is not recovered.

The current bundle covers 104/225 fixed-cohort cases with recorded-seed pairs;
98 have at least one positive net-target delta among the archived runs, six have
available pairs without one, and 121 remain unverified. These audit values are
not substituted for the paper's archived totals.

## Files and provenance

`examples/reproduction/sae/manifest.json` records the original host (`EVA_a100`), full
source paths, sizes and SHA-256 values for 110 archived files (about 24 MB).
The `archive/` tree preserves original relative paths and original source code,
including plotting, scoring, case finding and SAE-training metadata. Its scripts
retain historical absolute paths and are evidence, not portable entry points.
The new runner does not import or execute those scripts.

For 30 generation CSVs, `generation_runs` also records original JSON argument
dictionaries and the original JSON checksum when found. The legacy status string
`archived_original_json` means that the JSON was read and its arguments/checksum
were captured in the manifest; it does **not** mean every complete generation
JSON is duplicated in the bundle. Original generation JSON metadata is still
missing for `andreasson2020_glms_generation_v2`, `milena2021_cata_generation_v2`,
and `chen2024_myo_generation_v2`. Their CSV seeds/contexts can be checked, but
complete original run-argument provenance cannot be claimed.

The source checkout HEAD was `4a081f2db94238be0ce0ac3f9cad035d85a94b34`.
Some historical scripts were untracked, so this commit alone does not identify
their contents; the individual file checksums are the relevant evidence.
The historical layer-13 SAE has hidden width 1,024, feature width 8,192, and a
step-200,000 checkpoint. The original checkpoint and base-model paths/hashes are
recorded in `protocol.json`, along with the public-checkpoint correspondence
limitation. Model weights and the approximately 61-GB training FASTA are not
included. Matching model tensor contents to the public release and publishing
these historical resources remain separate tasks.

## Outputs and validation

The command writes:

- `likelihood_selected_case_features.csv`, dataset summaries, statistics JSON,
  and optional reconstructed PNG/SVG;
- `author_selected_examples.csv`, exact sequences/structures/reference cases,
  the original vector figure, and a selector/provenance report;
- archived Table S30 CSV/TeX, CSV-reconstructed TeX, and an explicitly scoped report;
- `generation_coverage.csv` and per-run case-index/exclusion details in JSON;
- `run_report.json` with Python/platform, source and runner hashes, verified
  input count and `model_inference_performed: false`;
- `output_checksums.json` for all generated outputs except the checksum file itself.

Twenty CPU regression tests cover empty and missing seed handling, inconsistent
seeds/contexts/case identities, ablation, missing and non-finite scores, duplicate
records, the original likelihood tie/selection rule, all eight real selectors,
archive hashes, S30 reconstruction, and refusal to overwrite existing output.
The new pairing helper fixes the old summarizer's empty-string-equals-empty-string
pairing bug. The archived historical source is intentionally unchanged.

No complete historical SAE cohort-level model-inference reproduction is claimed
by the CPU command. A separate successful 21M/layer-1 engineering smoke test does
not establish equivalence to the historical 1.4B/layer-13 paper experiment.

## Historical 1.4B/layer-13 single-case inference

The additional `examples/reproduction/sae/run_historical_smoke.py` was run against the
actual historical checkpoint and SAE on EVA_a100, after verifying both SHA-256
values. It uses the archived generation/scoring functions and an isolated copy
of the original `eva/` and `tools/utils/` model code, with no source changes and
no checkpoint copy into the repository.

Observed result: Pepper case 0, WT-state direction, feature 2008, seed 60,
7-nt span, `no_steer` versus `0x` ablation. Both generated spans/full sequences,
RNAfold structures, target/opposite hits, net target scores, anchor and span
coordinates exactly match the archived author-selected sample 18. The diagnostic
anchor activation also matches exactly. The generated spans are `CUGCUAG` and
`CUGCUUG`; WT hits change from 0 to 11 and net target score from -5 to 11.
The sample index is 0 in the new one-sample run, explicitly mapped to archived
sample index 18 by using its original seed 60 directly.

This check took 25.5 seconds and used one shared GPU, with observed process peak
3,612 MiB; PyTorch peak allocation was 3,237,024,256 bytes. The helper imposes a
4-GiB allocator cap, a 5-GiB process-memory monitoring threshold and free-memory
guards. The process exited and released its GPU allocation. The original
checkout's tokenizer and loader hashes were unchanged before/after the test.

The original MoE class can initialize CUDA experts in FP32 even when the loader
requests CPU. To avoid that transient allocation, the helper sets `config.bf16`
during original-class construction, retains the original MegaBlocks backend and
the original loader's final BF16 precision, uses strict checkpoint loading, and
checks all 238 final state tensors for exact equality with the source tensor
cast to the same dtype. This memory-only loading adjustment is recorded in the
report; architecture, steering settings, span and sampling rule are unchanged.

Validated existing environment:

- Python 3.11.14 at
  `/data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/envs/70_RNAVerse/bin/python`;
- PyTorch 2.7.0+cu126, Transformers 4.55.0, tokenizers 0.21.4, existing MegaBlocks;
- RNAfold 2.4.7 at
  `/data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/bin/RNAfold`.

On EVA_a100, from a source checkout containing this bundle, the corresponding
isolated setup and command are:

```bash
SAE_ORIGINAL_ROOT=/data/yanjie_huang/enzyme1_server/eva/EVA1
SAE_RUN_ROOT=$(mktemp -d /data/yanjie_huang/enzyme1_server/eva/sae_validation_XXXXXX)
mkdir -p "$SAE_RUN_ROOT/code/tools"
rsync -a --include='*/' --include='*.py' --include='*.json' --exclude='*' "$SAE_ORIGINAL_ROOT/eva" "$SAE_RUN_ROOT/code/"
rsync -a --include='*/' --include='*.py' --include='*.json' --exclude='*' "$SAE_ORIGINAL_ROOT/tools/utils" "$SAE_RUN_ROOT/code/tools/"
cp examples/reproduction/sae/ISOLATED_SAE_VALIDATION.txt "$SAE_RUN_ROOT/code/"
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR="$SAE_RUN_ROOT/cache/mpl" TRITON_CACHE_DIR="$SAE_RUN_ROOT/cache/triton" \
/data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/envs/70_RNAVerse/bin/python \
  examples/reproduction/sae/run_historical_smoke.py \
  --bundle examples/reproduction/sae \
  --isolated-eva-root "$SAE_RUN_ROOT/code" \
  --checkpoint /data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_13500 \
  --sae "$SAE_ORIGINAL_ROOT/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt" \
  --rnafold-bin /data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/bin/RNAfold \
  --output "$SAE_RUN_ROOT/result"
```

Coordinate GPU sharing before running; this environment/resource-specific command
is not a clean-install claim or a public resource release. It verifies one
historical example, not all 401 likelihood groups, all eight illustrations, or
the Table S30 cohort. The JSON report records full arguments, source hashes,
tensor checks, generated rows, archive comparisons and memory samples.
