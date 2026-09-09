# Benchmark scoring entry index

This index maps comparison-model families to the released entry files. Scoring computes sequence or variant predictions; metric evaluation consumes those predictions. A source link identifies code availability, not evidence that every published comparison has been rerun.

## EVA and common evaluation

| Purpose | Released entry |
|---|---|
| EVA sequence scoring | [tools/predict.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/tools/predict.py) |
| Representative end-to-end benchmark | [scripts/reproduce_milena.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/scripts/reproduce_milena.py) |
| Metrics from predictions and reference-table audit | [scripts/benchmark_release.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/scripts/benchmark_release.py) |

## Comparison models

| Model family | Entry | Role |
|---|---|---|
| RNA-FM, RNABERT, RNA-MSM, GENA-LM | [run_competitor.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/run_competitor.py) | RNA masked-LM interface; see model-specific settings in the protocol guide. |
| ERNIE-RNA | [compute_ernie_rna_scores_v2.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_ernie_rna_scores_v2.py) | Original model-specific scoring worker. |
| Evo2 | [score_fasta.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/score_fasta.py) | Original Evo2 scoring worker; requires its own runtime. |
| CodonFM | [score_codonfm.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/score_codonfm.py) | Original CodonFM scoring worker. |
| CodonGPT | [compute_codongpt_ll.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_codongpt_ll.py) | Original CodonGPT scoring worker. |
| AIDO.RNA | [compute_aido_ll.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_aido_ll.py) | Original AIDO scoring worker; requires ModelGenerator. |
| GenerRNA | [compute_generrna_scores_conda.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_generrna_scores_conda.py) | Wrapper around the external GenerRNA likelihood implementation. |
| ESM-1v / ESM-2 | [evaluate_dms_esm_domain.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/protein/evaluate_dms_esm_domain.py) | Native protein scoring implementation; parameterized ESM interface is in run_competitor.py. |
| ESM-C | [evaluate_dms_esmc_domain.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/protein/evaluate_dms_esmc_domain.py) | Separate ESM-C scoring implementation and runtime. |
| ProGen3 | [calculate_progen3_spearman.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.0/examples/reproduction/benchmark/upstream/protein/scripts/calculate_progen3_spearman.py) | Metric calculation from existing scores; this file is not the model scoring producer. |

Historical workers retain their original paths and require the corresponding model-specific environment and checkpoints. The supported parameterized interfaces and expected prediction format are documented in [Benchmark protocols](BENCHMARK_PROTOCOLS.md). [Model resources](OFFICIAL_MODEL_RESOURCES.md) lists checkpoint sources.

The released protein summary tables also contain ProGen2 results; a corresponding scoring producer is not identified by this index. ProGen3 is listed above as metric-only. These cases must not be described as complete model-inference pipelines.
