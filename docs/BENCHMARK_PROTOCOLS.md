# Benchmark protocols and comparison models

The representative current-submission workflow is documented in
[REPRODUCTION.md](REPRODUCTION.md) and fixes the public EVA-1.4B model,
all 135 Milena samples, and the original sum-score convention. Its observed
output and numerical comparison are in `examples/reproduction/milena_14b/expected/`.
The current label file is retained as confirmed by the author. Historical
FASTA header metadata is not treated as an independent assay ground truth.

Historical source files and archived predictions remain unchanged with hashes
in `examples/reproduction/benchmark/provenance.json` and `historical_14b_manifest.json`.
[September 5 validation notes](HISTORICAL_BENCHMARK_VALIDATION.md) preserve
older findings, which must not be substituted for current validation status.

To recalculate the archived Milena metric without running a model:

```bash
python scripts/reproduce_historical_benchmark.py \
  --output results/milena_archived_metric --artifact-only
```

This exports all 135 archived prediction rows and recomputes Spearman
0.84 (rounded to two decimal places). The eight-binary64-ULP arithmetic check accounts for
summation differences on the same vector; it is not a fresh-inference
acceptance threshold. Without `--artifact-only`, exit 2 distinguishes this
archive calculation from a completed model reproduction.

## Comparison-model execution, not plotting

`examples/reproduction/benchmark/run_competitor.py` accepts locally prepared checkpoints
for its supported scoring interfaces. Models are loaded from `--model-dir`;
the RNA interface uses `local_files_only=True`. Prepare the model, tokenizer and
dependencies in the chosen model's compatible environment before running it.
For other model interfaces, use the upstream scorer and the common prediction
evaluator described below. Record checkpoint versions and scoring settings with
your results; a specific paper comparison uses that paper's protocol.

1. Snapshot an already acquired, version-pinned local model. Keep the manifest **outside** the model directory; subsequent scoring requires the exact file set and hashes.

```bash
python examples/reproduction/benchmark/run_competitor.py snapshot \
  --model-dir /models/rnafm-pinned --output /models/rnafm-manifest.json
```

2. RNA masked-LM scoring implements leave-one-token-out mean log likelihood, excluding tokenizer special tokens. It uses bounded mask batches, rejects unknown tokens/nonfinite logits, and has no fallback tokenizer. The original adapter's truncation is only reproduced when explicitly requested; the default is an error on an oversized sequence. `max-tokens` includes special tokens and must match the selected model.

```bash
python examples/reproduction/benchmark/run_competitor.py rna-mlm \
  --input data/test.fasta --model-dir /models/rnafm-pinned \
  --model-manifest /models/rnafm-manifest.json --max-tokens 1024 \
  --sequence-type rna --register-multimolecule --mask-batch-size 1 \
  --device cuda:0 --output outputs/rnafm_test --dry-run
```

Remove `--dry-run` for actual scoring. `--length-policy truncate` is an explicit protocol change recorded per sequence. For GENA-like DNA tokenizers select `--sequence-type dna`. `--trust-local-code` must be explicit if the pinned local snapshot contains required custom code. A dry run validates inputs/hashes only, not model load/forward.

3. Protein ESM scoring uses the recovered native wt-/masked-marginal implementation, preceded by strict reference/variant validation and followed by finite/count checks. It requires one matching ProteinGym reference entry; no inferred-WT fallback or out-of-range mutation skipping is permitted.

```bash
python examples/reproduction/benchmark/run_competitor.py protein-esm \
  --input /data/ProteinGym/ASSAY.csv --reference /data/ProteinGym/DMS_substitutions.csv \
  --model-dir /models/esm1v-pinned --model-manifest /models/esm1v-manifest.json \
  --weight-file esm1v_t33_650M_UR90S_1.pt --strategy wt-marginals \
  --device cuda:0 --output outputs/esm_ASSAY --dry-run
```

The snapshot must contain companion files required by `fair-esm`. Actual ESM execution requires `torch`, `fair-esm`, `pandas`, `numpy`, `scipy` and `tqdm`. Do not count a single ESM-1v checkpoint as an ensemble. Marginal methods are not EVA sequence likelihood and must keep separate protocol labels.

4. Metrics for RNA predictions require an independently verified label CSV with `variant_id,sequence,label`, not a bare positional vector:

```bash
python examples/reproduction/benchmark/run_competitor.py evaluate \
  --predictions outputs/rnafm_test/predictions.csv --labels data/verified_labels.csv \
  --output outputs/rnafm_metrics.json
```

The evaluator rejects missing/duplicate IDs, mismatched sequences, nonfinite values and constant vectors. It emits a model-independent Spearman report; plot/table generation is a separate downstream operation. For the current Milena example, the author-confirmed association is frozen in its manifest. Keep the original ID/sequence association when exporting labels; historical header values do not replace the selected experimental readouts.

The portable RNA scorer passed a real CPU forward/save/reload test with a randomly initialized tiny BERT and character tokenizer, including mask-batch equality and explicit length-policy checks. This tests the execution machinery only, not any released competitor's predictions.

### Using your own model outputs

You can run a third-party model in its own environment and evaluate its exported
predictions without loading that model in EVA. The metric step uses Python's
standard library and requires no network connection or model weights.

| File | Required CSV columns |
|---|---|
| Predictions | `variant_id,sequence,score` |
| Labels | `variant_id,sequence,label` |

Supply one row per assay variant in each file. IDs and sequences must match;
all scores and labels must be finite, and both vectors must vary. Record the
scoring convention, model version and environment alongside the CSV. The
evaluator calculates signed Spearman using the supplied scores.

```bash
python examples/reproduction/benchmark/run_competitor.py evaluate \
  --predictions results/my_model/predictions.csv \
  --labels data/verified_labels.csv --output results/my_model/metrics.json
```

Use the [benchmark notebook](../examples/notebooks/prediction/benchmark_reproduction.ipynb)
and its evaluation manifest when comparing keyed predictions across multiple
datasets. This gives local model deployments a common analysis entry point.

### Coverage and remaining external resources

| Family | Recovered execution source | Remaining requirement / verification boundary |
|---|---|---|
| RNAFM, RNABERT, RNAMSM, RiNALMo, GENA, GROVER | Native HF adapter + portable `rna-mlm` scorer | Model/tokenizer snapshots and compatible Transformers/multimolecule; native config contains candidate HF revisions, not established paper-wide linkage. New runner's inference is not yet validated on these weights. |
| ERNIE-RNA | Adapter + `compute_ernie_rna_scores_v2.py` | Original ERNIE/fairseq project and pretrained checkpoint; native absolute paths require migration. |
| Evo2 | Adapter + `score_fasta.py` | Evo2 runtime, explicit size/version checkpoint and compatible kernels; archived Docker orchestrator paths are not portable. |
| CodonFM | Adapter + `score_codonfm.py` | CodonFM project, model-specific safetensors and preprocessing configuration. |
| CodonGPT | Adapter + self-contained scoring/tokenizer worker | Original checkpoint, Transformers compatibility and explicit codon treatment. |
| AIDO.RNA | Adapter + `compute_aido_ll.py` | ModelGenerator source and checkpoint; unresolved external import must remain explicit. |
| GenerRNA | Adapter + wrapper | Its `calculate_loglikelihood_me/calculate_likelihood.py`, model and tokenizer remain external; wrapper alone is not a complete inference release. |
| Protein EVA / RNA reverse translation | Native single/batch workers, converter/codon helpers and historical EVA model/loader code | Confirm codon strategy, species/taxonomy conditioning, checkpoint and per-assay input provenance before rerun. Do not replace this with an arbitrary codon encoding. |
| ESM-1v/ESM-2 and ESM-C candidates | Native ESM and domain/ESM-C workers + strict ESM launcher | Exact weights, full/domain treatment and ensemble linkage; ESM-C requires its distinct runtime. No protein benchmark rerun is claimed here. |
| ProGen3 | Archived result-to-Spearman script | Scoring producer and exact weights remain missing; metric recalculation is not model evaluation. |

Original files under `upstream/` retain obsolete paths and historical fallback behavior for provenance. They are **not** the recommended clean-install CLI and must not be silently edited in place. The new parameterized runner avoids the HF tokenizer fallback and enforces score/variant validation. Full competitive-model numerical reproduction is still incomplete; obtaining code is not evidence of a successful run.

## Training configuration evidence

The recovered 21M pretrain `checkpoint-31006` and midtrain `checkpoint_86006`, and 1.4B pretrain `checkpoint_14500` and midtrain `checkpoint-25500`, all specify `mode: mixed`, `glm_probability: 0.333`, EP=4 and DP=4. These support mixed CLM/GLM and the parallelism settings for **those runs**, not automatically every released model.

Both recovered 21M stage configs set `use_lineage_prefix: true`. Therefore they do not substantiate a blanket statement that pretraining used RNA-type only and lineage began solely at midtraining. The final 1.4B mid config names a v31 pretraining checkpoint for which the matching pretraining config remains unresolved. Clarify the run-to-manuscript linkage rather than changing training objectives to match prose. The author's dense-control implementation has now been recovered: it uses a single eager FFN expert in the EVA transformer, not the historical `model_dense` API. See [Dense control](DENSE_CONTROL.md) for source identity and validation. Training data paths in historical YAMLs are provenance, not evidence that the original training datasets are publicly accessible.

The separately recovered `RNAVerse/checkpoint/clm` metadata supplies a fifth concrete configuration (v4 midtraining from v24 pretrain step 15000): it also specifies mixed mode, GLM probability 0.333, EP=4, DP=4, lineage and RNA-type prefixes. It establishes another observed historical run, not a reason to conflate the v24 and v31 checkpoint families.

## Resource recovery update: September 9, 2026

The [current resource index](OFFICIAL_MODEL_RESOURCES.md#recovered-comparison-model-files-september-9-2026)
records recovered RNA-FM/RNABERT/RNA-MSM snapshots and additional cached
CodonFM, ERNIE-RNA, AIDO.RNA and GenerRNA file hashes. These can identify matching
local files; the environment and scoring protocol must be recorded separately.
The records describe recovered artifacts, not additional benchmark runs.

For a model-by-model list of entry files, see [Benchmark scoring entry index](BENCHMARK_ENTRY_INDEX.md).
