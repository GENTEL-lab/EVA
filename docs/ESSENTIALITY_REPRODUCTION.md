# Essentiality: archived-score recomputation and source audit

This workflow recomputes AUROC from historical predictions on CPU. It is **not**
a fresh EVA/Evo2 inference run, a complete regeneration of the original dataset,
or evidence that premature-stop likelihood changes establish a causal mechanism.
It leaves the historical files and manuscript figures unchanged.

## Run

Python 3.10+ and its standard library are sufficient; no GPU, PyTorch or package
installation is required. From the released source checkout, run:

```bash
python scripts/reproduce_essentiality.py \
  --source-root /path/to/historical/EVA1 \
  --output /path/to/new/essentiality-results
```

The original source root is currently available on `EVA_a100` at
`/data/yanjie_huang/enzyme1_server/eva/EVA1`. It is not a public dataset URL.
The required dataset checksum and directory layout are in
`examples/reproduction/essentiality/source_manifest.json`. The input contains 95,538 CDS
records from five species; these records must not be described as 95,538 unique
genes without resolving transcript and duplicate-annotation identifiers.

The input SHA256 is fixed. Missing files, mismatching metadata, different labels,
duplicate/missing row indices, incomplete shards, wrong positions or mutations,
nonfinite scores, and inconsistent `WT_LL - mutant_LL` fail explicitly. Existing
output directories are never overwritten. Tests need the optional dev dependency:

```bash
python -m pytest tests/test_essentiality_reproduction.py -q
```

## Outputs and checks

- `report.json`: full-precision metrics, input sizes/SHA256, runtime/code identity,
  protocol, legacy-output discrepancies, and limitations.
- `position_ablation.csv` and `position_ablation_per_species.csv`: 10 model/position
  rows and 50 species rows, generated from the same report object as `summary.md`.
- `gc_baseline_per_species.csv`: original-join and corrected-join estimates side by
  side; neither is substituted into the manuscript automatically.
- `record_identity.csv.gz`: one row per original record, with row index, full
  metadata, unmodified sequence SHA256 and stable record ID. This prevents
  same-gene transcript records from being silently collapsed.

The 5 September 2026 audit checked all 10 original JSONL groups: 95,538 records
in each, zero duplicate record indices, complete two-shard coverage and finite,
internally consistent scores. Mean per-species AUROC was:

| Position | EVA 1.4B | Historical small model (`eva30M`) |
| --- | ---: | ---: |
| 5% | 0.5513546348 | 0.5261235617 |
| 25% | 0.5676008323 | 0.5503912804 |
| 50% | 0.5903675411 | 0.5502150669 |
| 75% | 0.5639547957 | 0.5598797367 |
| 95% | 0.5841763810 | 0.6177058434 |

These agree with the historical final CSV and Markdown, but not the historical
`position_ablation_v2_summary.json`. That JSON puts the 50% results under the 5%
key and omits 50% entirely for both models. The corrected output uses explicit
IDs `p05`, `p25`, `p50`, `p75`, `p95`, validates the complete position set, and
generates all formats from one object. Archived files remain unmodified.

## GC baseline correction

The historical plotting script used `(organism, gene)` to attach sequences to
scored records. The 95,538 records have only 67,244 distinct values of that key,
including 14,625 repeated keys. The join therefore changed the sequence attached
to 23,500 rows: 16,094 Arabidopsis, 7,398 C. elegans and 8 S. pombe rows.

The corrected join preserves original row order **only after** verifying organism,
gene, locus tag, accession, lineage and essentiality label for every row. If a
scored record includes a sequence, its hash must also match. The stable record ID
adds the original index and source-sequence hash so identical gene names cannot
overwrite each other.

| GC definition | Mean per-species AUROC |
| --- | ---: |
| Replay of historical gene-name join | 0.5705016099 |
| Original sequence attached to its verified record | 0.5661303208 |

The GC numerator is G+C, and the denominator is the full original sequence
length, including ambiguous nucleotide symbols, matching the historical GC
definition. This correction is a data-association fix, not a change to EVA
scores. It requires transparent reconciliation of any manuscript/source-data
panel using the old GC baseline before publication.

## Keep protocols and artifacts distinct

1. **Main essentiality experiment:** existing `delta_ll_results_*_8192.json`
   scores are recomputed separately. The original data-building and main EVA
   inference entrypoints have not yet been recovered; the positional-ablation
   runner is not a substitute. The manuscript's near-5′ UAA×5 perturbation must
   not be conflated with the position-ablation cassette.
2. **Position ablation:** raw JSONL records explicitly use the 15-nt cassette
   `UAAUAAUAAUAGUGA`, inserted at `int(min(length,8192) * ratio)`, bounded away
   from the ends. WT and mutant scores are summed next-token log-likelihoods,
   including direction and end tokens. The recovered lineage-fixed source
   constructs `<bos>5|lineage|sequence3<eos>`, with no RNA-type token. Its comments
   and historical README describe a different ordering; do not copy that
   description as the actual implementation.
3. **Length limit:** the recovered runner truncates raw nucleotides to 8,192
   before adding lineage/direction tokens and the mutation. It does not enforce
   a final 8,192-token input length. There are 311 source records above 8,192 nt.
   Source records contain 11 non-ACGU characters (Y/K/S/W/M); no normalization is
   introduced in this recomputation.
4. **Weights:** historical `checkpoint/clm/model_weights.pt` hashes identically
   to the public EVA-1.4B CLM artifact. Historical
   `checkpoint/30M_1124/checkpoint-56844/model_weights.pt` does not hash identically
   to public EVA-21M. Both hashes are recorded in the manifest; “30M” versus “21M”
   is not adequate artifact identification. Archived per-record predictions lack
   checkpoint hashes or token-ID traces, so present-day source inspection alone
   cannot certify the exact historical execution.
5. **Interpretation:** AUROC is computed per species, then averaged equally.
   Pooled AUROC is provided as a diagnostic only. Species counts are not biological
   replicates. Negative labels' experimental validation and original DEG mapping
   policy remain to be documented. These ablations do not establish that NMD
   cannot operate at a particular site or rule out all RNA-level mechanisms.

The older `organized_position_ablation` results use a different scoring setup.
Directories named `*_full` contain partial early runs. Neither is selected by the
new recomputation entrypoint. Recovered original scripts under `archive/` retain
historical defects for audit only and must not be executed as release commands.
