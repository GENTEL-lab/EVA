# Generation case-level success table

Builds the paper table of SAE feature-steering success rates on structure-switch variants (213/225, 95% overall).

## Outputs

| File | Description |
|------|-------------|
| `data/cohort_cases.csv` | 225 evaluation cases (top-N by `bp_distance` per dataset) |
| `data/summary_by_dataset.csv` | Per-dataset success counts used in the table |
| `latex/generation_success_table.tex` | Standalone LaTeX document (auto-generated) |
| `latex/generation_success_table_fragment.tex` | Table-only fragment for `\input` into a paper |
| `generation_success_table.pdf` | Compiled table PDF |

## Rebuild

```bash
cd /path/to/yanjie/figures/table
python3 build_table.py
```

Requires `pdflatex` (TeX Live).

## Cohort definition

Cases are read from `../../rnafold_cases/{dataset}_structure_switch_cases.json`: for each benchmark, the top-N variants ranked by RNAfold base-pair distance between wild-type and mutant secondary structures (total N = 225). See `data/cohort_cases.csv`.

## Success criterion

For each structure-switch variant we have a WT sequence and a mutant sequence whose RNAfold folds differ. **Diagnostic base pairs** distinguish the two folds: pairs that appear in one reference structure but not the other (`diagnostic_wt_pairs` / `diagnostic_mutant_pairs` in the case files).

For a generated sequence:

1. Fold it with RNAfold.
2. Count **target hits**: diagnostic pairs of the desired fold recovered in the prediction.
3. Count **opposite hits**: diagnostic pairs of the undesired fold recovered.
4. **Structure score** = target hits − opposite hits (`net_target_score` in the scoring scripts).

A case **succeeds** if there exists at least one SAE steering setting (feature × strength) such that some steered sample, paired to an unsteered baseline on the **same random seed**, has a strictly higher structure score than that baseline (`paired_improved > 0`).

Aggregated counts per dataset are in `data/summary_by_dataset.csv`. Scoring implementation: `scripts/sae_feature_steering/score_rnafold_structure_switch_generation.py`.
