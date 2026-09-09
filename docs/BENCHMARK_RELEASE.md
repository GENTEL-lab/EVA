# Benchmark reproduction: data, predictions, metrics and figures

This integrated component provides a CPU-only, standard-library evaluation layer and
a companion notebook. It is not a complete release of all model-inference
workflows. See RELEASE_STATUS.md for the current candidate publication status.
Original datasets, manuscript results and existing source files are unchanged.

## Installation and scope

Use Python 3.10 or later. The command-line tool needs no third-party packages,
GPU, network access or model weights. Jupyter is needed only for the interactive
notebook interface. Start in the full source checkout. Tests and the commands below work without installing the EVA package.

```bash
python -B -m unittest discover -s tests -p test_benchmark_release.py -v
python scripts/benchmark_release.py --help
```

Open `examples/notebooks/prediction/benchmark_reproduction.ipynb` to follow the same entry
points. Its three code cells have been executed both sequentially as Python and
in an isolated Jupyter kernel. The reference-analysis section ran; the optional
prediction-manifest section was explicitly unconfigured. This does not validate
model inference or arbitrary user-supplied prediction files.
The observed notebook-test dependency versions are recorded in
`examples/reproduction/benchmark_release/jupyter-validation-macos-py312.lock.txt`.
That lock is for the tested macOS/Python 3.12 environment, not a universal
installation recipe or an EVA GPU environment.

## A. Published summary tables to diagnostic figures

```bash
python scripts/benchmark_release.py audit-reference \
  --manifest examples/reproduction/benchmark_release/reference_manifest.json \
  --data-root examples/reproduction/benchmark_release/reference \
  --output results/reference_audit --plot
```

The frozen files are the public ncRNA, mRNA and protein summary tables from the
inspected source. Their SHA256 values are recorded in the manifest. Model and
dataset identifiers are transcribed from the existing plotting code, not inferred
from the set of successful runs. Current coverage is:

| Group | Existing metric cells | Plotter-declared metric cells |
|---|---:|---:|
| ncRNA | 195 (13 assays × 15 models) | 195 |
| mRNA | 75 (5 assays × 15 models) | 75 |
| Protein | 220 (20 assays × 11 models) | 240 (20 assays × 12 models) |

The protein table lacks the plotter-declared Evo2 7B row. Four ncRNA assays also
have different recorded sample counts across models (each differs by one):
Li_2016_tRNA, Kobori_2015_ribozyme_j12, Janzen_2022_fam1b1_ribozyme and
Domingo_2018_tRNA. The report includes the model-by-model counts. Whether these
differences reflect a documented exclusion, wild-type handling or a defect has
not been established. The command therefore writes a report and diagnostic SVGs
but returns **2**, exposing missing coverage and unresolved cohort differences.
Confirm the final manuscript's intended model scope before treating the plotting
declaration as a paper requirement. Do not invent scores or remove a required
model simply to make the check pass.

These inputs are already-computed per-assay correlations. Aggregation and
plotting cannot recover per-sequence predictions or reproduce inference.
SVGs use a linear axis and visibly show missing models. They are diagnostic
plots, not pixel-identical manuscript artwork or replacements for paper figures.

### Recovered protein-table candidate

A server archive contains the missing 20 Evo2 7B per-assay values. Every one of
the other 220 protein metric cells numerically matches the public table. The
recovered row gives mean absolute Spearman **0.5418927586400086**, matching the
archived summary. A separate candidate table and hashes are included; the public
snapshot is retained unchanged. This recovers an omitted archived result row,
not fresh inference or confirmation of its final manuscript/checkpoint lineage.

```bash
python scripts/benchmark_release.py audit-reference \
  --manifest examples/reproduction/benchmark_release/recovered_reference_manifest.json \
  --data-root examples/reproduction/benchmark_release/recovered_reference \
  --output results/recovered_reference_audit --plot
```

The candidate has **510/510 metric cells** under the plotter's declared scope,
including all 240 protein cells. Overall exit status remains 2 because the four
ncRNA sample-count differences remain unresolved. Coverage completion and
sample-count consistency are separate report fields. Confirm final paper scope
and source lineage before incorporating the candidate into a public release.

## B. Keyed predictions to correlations and plot-compatible tables

Prepare two ordinary CSVs per dataset/model combination:

* labels: `variant_id,sequence,label`
* predictions: `variant_id,sequence,score`

Use stable original IDs and actual source sequences. Preserve original labels.
Do not generate IDs after independently sorting files, join by row number, use
another model to fill missing predictions, or infer experimental readouts from
sequence names. The label and prediction ID sets must match exactly.

Complete `examples/reproduction/benchmark_release/evaluation_manifest.template.json` in a
new file. Its null fields are deliberately unusable until real values are
provided. Specify all intended models and assays; the required scope is their
Cartesian product. For genuinely different model scopes, use separately labeled
manifests, not an implicit intersection of successful rows.

The manifest specifies relative paths and hashes, an explicit sequence policy,
label-source status and the published Spearman as an exact decimal string.
`exact` preserves sequence text without RNA substitutions, including for protein
inputs. `rna_t_to_u` explicitly normalizes case and T/U and requires canonical
RNA. No sequence normalization is inferred from the filename.

```bash
python scripts/benchmark_release.py evaluate \
  --manifest evaluation_manifest.json --data-root downloaded_data \
  --output results/prediction_metrics --plot
```

The tool calculates signed Spearman using average ranks for ties. Across assays
it reports the mean absolute Spearman, matching the inspected plotting code's
aggregation; this is distinct from taking the absolute value after averaging.
It does not change score signs or select a scoring objective to match a table.

All required pairs must finish before `metrics_long.csv` and `metrics_wide.csv`
are exported. The long format has `Dataset,Model,Spearman,N_samples,Spearman_abs`
for RNA plotting; the wide format has `Model,<assay1>,...` for protein plotting.
An absent pair disables complete-cohort aggregation for the affected model.
Do not replace published summary files until the complete scope and provenance
have been reviewed. The output directory must not already exist.

The report records every completed pair, missing pair, malformed prediction,
input hash and script hash. Source declarations and inference provenance fields
are **not independently verified by this arithmetic tool**. They are retained
for review, not converted into a paper-reproduction claim.

Comparison uses half the last supplied published decimal unit. It does not use
an arbitrary 0.001 threshold. A different exact historical vector should be
compared directly when available; decimal agreement alone is weaker evidence.

Exit codes:

* **0:** reference-table audit complete, or all declared prediction pairs
  computed and compared with no declared label-source hold and no mismatch.
* **2:** missing coverage, invalid prediction jobs, absent comparison values,
  numerical mismatch or unresolved declared label sources. Inspect `report.json`.
* **Other nonzero:** malformed top-level inputs, missing dependencies or other
  execution failure. A traceback is not a successful or partial reproduction.

Even exit 0 does not certify fresh inference, complete manuscript scope, source
licensing, checkpoint lineage or scientific validity.

## C. Fresh model inference remains a separate upstream stage

The source contains these entry points; see REPRODUCTION.md for the new pinned 1.4B workflow:

| Workflow | Existing relative entry | Remaining requirement |
|---|---|---|
| EVA representative assay | `scripts/reproduce_dms.py` | Resolve original label/invocation mismatch; do not call the previous 21M run a numerical match |
| Historical EVA archive | `scripts/reproduce_historical_benchmark.py` | Arithmetic only; unresolved label provenance stays visible |
| Bounded RNA MLM / native ESM | `examples/reproduction/benchmark/run_competitor.py` | Correct per-model objective, weights, runtime and assay manifest; not interchangeable with all competing methods |
| Historical multi-model adapters | `examples/reproduction/benchmark/upstream/rna/` | Port original verified configs and external workers; legacy absolute paths are not reader instructions |
| Protein model adapters | `examples/reproduction/benchmark/upstream/protein/` | Original reference/codon/domain/ensemble conventions and exact model versions; not all methods are complete |

Those files belong to the existing full-source repair, not this small overlay.
Their presence is not a fresh execution result. This supplement performs no GPU
work, fine-tuning, training, sequence generation or optimization. It supplies the
downstream contract that their validated predictions should satisfy.

For each fresh run, preserve the exact checkpoint revision/hash, tokenizer hash,
code revision, environment, original command/config, scored-token exclusions,
reduction, conditioning, length policy, seed, output IDs and execution log.
Record elapsed time and peak device memory from an actual representative run.
Do not promise a runtime or hardware requirement based only on model size.

## Data release arrangement

Keep small source, manifests, notebook and tests on GitHub. Stage larger inputs,
keyed predictions and intermediate arrays in versioned data packages with a
file manifest and license/source information. Model weights may use a pinned
model-host revision or an archival deposit where redistribution is permitted.

Separate downloads for (1) cached-result analysis, (2) fresh inference inputs and
weights, and (3) optional training reproduction. Full training corpora must not
be mandatory for opening figures or evaluating already released weights.
Do not advertise a complete workflow until its required package is actually
accessible, downloaded into a clean directory, hash-verified and executed.

No deposit DOI or public download URL is fabricated here. Zenodo/GitHub
publication, checkpoint-to-paper binding, missing original inputs and complete
competitor inference are still release gates, not completed steps.
