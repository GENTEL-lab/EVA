# Benchmark scoring entry index

This index maps comparison-model families to the released entry files. Scoring computes sequence or variant predictions; metric evaluation consumes those predictions. A source link identifies code availability, not evidence that every published comparison has been rerun.

## EVA and common evaluation

| Purpose | Released entry |
|---|---|
| EVA sequence scoring | [tools/predict.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/tools/predict.py) |
| Representative end-to-end benchmark | [scripts/reproduce_milena.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/scripts/reproduce_milena.py) |
| Metrics from predictions and reference-table audit | [scripts/benchmark_release.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/scripts/benchmark_release.py) |

## Comparison models

| Model family | Entry | Role |
|---|---|---|
| RNA-FM, RNABERT, RNA-MSM, GENA-LM | [run_competitor.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/run_competitor.py) | RNA masked-LM interface; see model-specific settings in the protocol guide. |
| ERNIE-RNA | [compute_ernie_rna_scores_v2.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_ernie_rna_scores_v2.py) | Original model-specific scoring worker. |
| Evo2 | [score_fasta.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/score_fasta.py) | Original Evo2 scoring worker; requires its own runtime. |
| CodonFM | [score_codonfm.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/score_codonfm.py) | Original CodonFM scoring worker. |
| CodonGPT | [compute_codongpt_ll.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_codongpt_ll.py) | Original CodonGPT scoring worker. |
| AIDO.RNA | [compute_aido_ll.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_aido_ll.py) | Original AIDO scoring worker; requires ModelGenerator. |
| GenerRNA | [compute_generrna_scores_conda.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/rna/external_scripts/compute_generrna_scores_conda.py) | Wrapper around the external GenerRNA likelihood implementation. |
| ESM-1v / ESM-2 | [evaluate_dms_esm_domain.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/protein/evaluate_dms_esm_domain.py) | Native protein scoring implementation; parameterized ESM interface is in run_competitor.py. |
| ESM-C | [evaluate_dms_esmc_domain.py](https://github.com/GENTEL-lab/EVA/blob/v1.2.1/examples/reproduction/benchmark/upstream/protein/evaluate_dms_esmc_domain.py) | Separate ESM-C scoring implementation and runtime. |
| ProGen3 | [Original scoring CLI](https://github.com/Profluent-AI/progen3/blob/5c8c88c10b369a8c3044e11d586266f68dfca0bf/src/progen3/tools/score.py), [scorer](https://github.com/Profluent-AI/progen3/blob/5c8c88c10b369a8c3044e11d586266f68dfca0bf/src/progen3/scorer.py), [metric calculation](../examples/reproduction/benchmark/upstream/protein/scripts/calculate_progen3_spearman.py) | Fixed upstream model-scoring implementation followed by the released result-to-Spearman entry. |

Historical workers retain their original paths and require the corresponding model-specific environment and checkpoints. The supported parameterized interfaces and expected prediction format are documented in [Benchmark protocols](BENCHMARK_PROTOCOLS.md). [Model resources](OFFICIAL_MODEL_RESOURCES.md) lists checkpoint sources.

## ProGen3 source identity

The recovered scoring entry and scorer match upstream commit
`5c8c88c10b369a8c3044e11d586266f68dfca0bf` byte for byte.

| File | SHA256 |
|---|---|
| `src/progen3/tools/score.py` | `cd27545b3166108895b39ba66053d0275c158adad17fbd758693eb12b7a46e52` |
| `src/progen3/scorer.py` | `74777afb45b8144a9385da18b021b8516eaac6f97d9f51ea0d2d41b39ca07607` |

Use the [upstream environment and model documentation](https://github.com/Profluent-AI/progen3/blob/5c8c88c10b369a8c3044e11d586266f68dfca0bf/README.md). Code and model weights retain their respective upstream licences. This source-identity check does not constitute a new benchmark run.
