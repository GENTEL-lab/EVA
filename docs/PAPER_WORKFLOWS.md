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
Use separate method environments to avoid incompatible dependencies.
The entries below distinguish recovered version identifiers from missing ones.

| Method | Model resource identified in original source | Entry and remaining requirement |
|---|---|---|
| RNA-FM | `multimolecule/rnafm`, revision `2a98c38e8219d666ad35d04a689218ab50a36293` | `run_competitor.py rna-mlm`; Multimolecule registration and the method's original masking/length settings are required. The historical revision currently returns HTTP 404; no public version is substituted or newly run. |
| RNABERT | `multimolecule/rnabert`, revision `7a9b5d5d5a96931ff056828c0968479098506cea` | Same bounded masked-LM entry; historical revision currently returns HTTP 404. |
| RNA-MSM | `multimolecule/rnamsm`, revision `ad61a4915674596557b3c3c4fe4fd7162d36acc8` | Historical revision currently returns HTTP 404; sequence/MSA protocol correspondence is also unverified. |
| GENA-LM | Local snapshot names `AIRI-Institute/gena-lm-bert-base-t2t-multi`, revision `4633e5a1ada905bb7afee6877d71cc12578a95a5` | Original model-name field differs from local snapshot name. Resolve that binding before downloading or running a replacement. |
| ERNIE-RNA | `ERNIE-RNA_pretrain.pt` | Preserved ERNIE adapter and worker; exact release/checksum and fairseq environment still required. |
| Evo2 1B, 7B, 40B | `evo2_1b_base`, `evo2_7b_base`, `evo2_40b_base` | Preserved Evo adapter; exact revisions and official compatible runtime still required. Do not replace the original scoring objective. |
| CodonFM 600M and 1B | Original 600m/1b model selectors | Preserved `score_codonfm.py` and adapter; original container and immutable weight identities remain missing. |
| AIDO.RNA | Original model-weight folder and ModelGenerator framework | Preserved AIDO worker/adapter; exact model snapshot and environment still required. |
| GenerRNA | `model_updated.pt` and separate tokenizer | Preserved GenerRNA worker/adapter; checkpoint/tokenizer checksums and matching runtime still required. |
| CodonGPT | Original worker `compute_codongpt_ll.py` | Weight revision and original environment still required. |
| ESM protein models | `upstream/protein/evaluate_dms_esm_domain.py` | Strict native entry `run_competitor.py protein-esm`; exact reference/DMS files and model ensemble must be supplied. |
| ProGen3 and other unmatched protein rows | Existing summary-table entries | Missing producer/version must be recovered; a table row is not an inference implementation. |

The three historical Multimolecule revision IDs returned HTTP 404 in the
current public API check. They are retained as original configuration evidence,
not presented as working download commands. The GENA-LM revision resolves,
but its original model-name/snapshot mismatch still needs clarification.
Do not substitute current `main` weights and call them the paper model.
Once an original snapshot is recovered, its files can be recorded with:

```bash
python reproduction/benchmark/run_competitor.py snapshot \
  --model-dir checkpoint/rnafm --output checkpoint/rnafm-manifest.json
```

[Official resource links](OFFICIAL_MODEL_RESOURCES.md) identify upstream model
and runtime sources. These links do not establish which upstream version
produced an EVA paper result.

See [BENCHMARK_PROTOCOLS.md](BENCHMARK_PROTOCOLS.md) for the parameterized
masked-LM and native ESM interfaces. The snapshot helper records identity; it
does not determine whether a checkpoint is the one used for a paper result.
No unverified version is silently filled with a current `main` revision.

The recovered Evo2 7B protein row remains a separate candidate table under
`reproduction/benchmark_release/recovered_reference/`. It is not substituted
into the current submission table. Four ncRNA model comparisons have recorded
sample-count differences of one; their original inclusion/WT rules still need
to be documented before those comparisons can be called complete.
