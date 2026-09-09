# Paper workflows and artifact coverage

This index uses the current submission's six main figures and the September 7
Extended Data renumbering. It does not infer completed inference from stored
metric tables. `reproduction/release/benchmark_result_index.csv` indexes the
490 metric cells currently present in the repository's three benchmark tables;
this is not the complete set of manuscript experiments or all plotted models.

| Result or component | Source and execution entry | Artifact and validation boundary |
|---|---|---|
| Model training and context evaluation, Figure 1 | `training/`, `training/pretrain/README.md`, `notebooks/long_context_recall/` | Small training smoke is separate from full training. Exact manuscript training splits and some run/checkpoint links remain to be supplied. |
| RNA fitness, Figure 2 | `scripts/reproduce_milena.py`; `notebooks/prediction/data/`; `scripts/benchmark_release.py` | The representative Milena protocol and public checkpoint are frozen. Other assays retain their existing scores and sample counts. |
| Protein fitness, Figure 2 | `reproduction/benchmark/run_competitor.py`; `reproduction/benchmark/upstream/protein/` | Native ESM entry and archived metric calculations are included. Model-specific ensembles, codon handling and exact paper checkpoint bindings are not all recovered. |
| Essentiality, Figure 2 | `scripts/reproduce_essentiality.py`; `ESSENTIALITY_REPRODUCTION.md` | Recomputes archived predictions and ablation groups. Main inference/data-production provenance is incomplete. |
| Generated-sequence statistics, Figure 3 and ED6 | `notebooks/generation/`; existing deposited sequence/features | Exact final Figure 3a/b plotting inputs are not bound to a portable command in this candidate. |
| Aptamer and tRNA analyses, Figure 4 | `finetune/`; `notebooks/design/`; accompanying Source Data | Training execution tests do not reproduce laboratory measurements or validate predicted structures. |
| Computational design case studies, Figure 5 | Existing design notebooks and deposited results | Archived analyses are retained; exact final configurations are not all mapped. |
| Interpretability and steering, Figure 6 and ED8–10 | `scripts/reproduce_sae.py`; `SAE_REPRODUCTION.md` | Stored 401-case likelihood analysis and selected examples are distinct from the full generation cohort in Table S30. Current data remain as author-confirmed; full per-case S30 output is not bundled. |

## Competitor inputs and model versions

The preserved original configuration is
`reproduction/benchmark/upstream/rna/config.yaml`. Historical absolute paths
identify earlier resources; do not execute that configuration unchanged.
Use separate method environments and local checkpoint paths, or export
predictions from an upstream scorer to the
[common evaluator](BENCHMARK_PROTOCOLS.md#using-your-own-model-outputs).
The version identifiers below describe the original comparison records. New
benchmarks can use other locally prepared versions, recorded with their results.

| Method | Model resource identified in original source | Entry and remaining requirement |
|---|---|---|
| RNA-FM | `multimolecule/rnafm`, revision `2a98c38e8219d666ad35d04a689218ab50a36293` | `run_competitor.py rna-mlm`; Multimolecule registration and the method's original masking/length settings are required. Use a locally prepared model/tokenizer snapshot and record its file hashes. |
| RNABERT | `multimolecule/rnabert`, revision `7a9b5d5d5a96931ff056828c0968479098506cea` | Use the same bounded masked-LM entry with a local snapshot and compatible runtime. |
| RNA-MSM | `multimolecule/rnamsm`, revision `ad61a4915674596557b3c3c4fe4fd7162d36acc8` | Use a local snapshot; the original sequence/MSA protocol correspondence still requires confirmation for the paper comparison. |
| GENA-LM | Local snapshot names `AIRI-Institute/gena-lm-bert-base-t2t-multi`, revision `4633e5a1ada905bb7afee6877d71cc12578a95a5` | Original model-name field differs from local snapshot name. Resolve that binding before downloading or running a replacement. |
| ERNIE-RNA | `ERNIE-RNA_pretrain.pt` | Preserved ERNIE adapter and worker; cached weight and environment-file hashes are now recorded; the exact paper runtime has not been verified. |
| Evo2 1B, 7B, 40B | `evo2_1b_base`, `evo2_7b_base`, `evo2_40b_base` | Preserved Evo adapter; exact revisions and official compatible runtime still required. Do not replace the original scoring objective. |
| CodonFM 600M and 1B | Original 600m/1b model selectors | Preserved `score_codonfm.py` and adapter; cached weight/config identities are now recorded; the original runtime and paper checkpoint binding remain to be established. |
| AIDO.RNA | Original model-weight folder and ModelGenerator framework | Preserved AIDO worker/adapter; cached weight-shard/config hashes are now recorded; the paper snapshot and environment binding remain required. |
| GenerRNA | `model_updated.pt` and separate tokenizer | Preserved GenerRNA worker/adapter; cached weight and tokenizer-config hashes are now recorded; the complete tokenizer/runtime and paper checkpoint binding remain required. |
| CodonGPT | Original worker `compute_codongpt_ll.py` | Weight revision and original environment still required. |
| ESM protein models | `upstream/protein/evaluate_dms_esm_domain.py` | Strict native entry `run_competitor.py protein-esm`; exact reference/DMS files and model ensemble must be supplied. |
| ProGen3 and other unmatched protein rows | Existing summary-table entries | Missing producer/version must be recovered; a table row is not an inference implementation. |

Record the exact files used by a local model before scoring. A supplied
manifest identifies a matching historical snapshot; a newly generated manifest
identifies the checkpoint you have chosen for a new benchmark:

```bash
python reproduction/benchmark/run_competitor.py snapshot \
  --model-dir checkpoint/rnafm --output checkpoint/rnafm-manifest.json
```

[Official resource links](OFFICIAL_MODEL_RESOURCES.md) identify upstream model
and runtime sources. For an exact paper comparison, use the corresponding checkpoint and scoring
protocol from the original run.

See [BENCHMARK_PROTOCOLS.md](BENCHMARK_PROTOCOLS.md) for the parameterized
masked-LM and native ESM interfaces. The snapshot helper records identity; it
does not determine whether a checkpoint is the one used for a paper result.

The recovered Evo2 7B protein row remains a separate candidate table under
`reproduction/benchmark_release/recovered_reference/`. It is not substituted
into the current submission table. Four ncRNA model comparisons have recorded
sample-count differences of one; their original inclusion/WT rules still need
to be documented before those comparisons can be called complete.

## Resource recovery update: September 9, 2026

The [current resource index](OFFICIAL_MODEL_RESOURCES.md#recovered-comparison-model-files-september-9-2026)
records recovered RNA-FM/RNABERT/RNA-MSM snapshots and additional cached
CodonFM, ERNIE-RNA, AIDO.RNA and GenerRNA file hashes. These can identify matching
local files; the environment and scoring protocol must be recorded separately.
The records describe recovered artifacts, not additional benchmark runs.
