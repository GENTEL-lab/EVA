# SAE validation ledger — 2026-09-05

| Issue | Resolution | Observed evidence | Remaining boundary |
| --- | --- | --- | --- |
| Historical likelihood inputs and selection were hard to locate | Archived eight raw JSONs, eight summaries, original source and SVG; added portable runner | 4,183 raw rows → 487 summary groups, all CSV fields/order match → original selected 401/401; W=80601, p=9.4e-68, d=1.70; exact SVG display-text match | CPU recomputation from saved raw predictions, not new full-model predictions |
| Selected structure illustrations lacked explicit raw bindings | Fixed eight author-selected source/run/case/feature/condition/sample/seed selectors | Eight unique unsteered/steered pairs, source context and diagnostic-pair counts checked; original SVG retained | WT-restoration illustrations are not a random sample or the direction-dependent S30 success criterion |
| Empty seed strings could produce false paired statistics | New portable pair helper rejects blank/missing seeds and ambiguous contexts/baselines | Regression tests cover blank, absent and mismatched seeds, contexts, duplicate cases/records and missing scores | Archived original code remains unchanged as evidence; its old summary must not be used as proof of pairing |
| 0x was easy to conflate with no steering | Explicitly retain 0x as a feature-ablation intervention in generation pairing | Regression test plus original hook semantics | The likelihood figure's original post-selection exclusion of 0x is retained separately |
| Table S30 lacked an executable reconstruction boundary | Rebuild numerical LaTeX rows from archived aggregate CSV and verify exact original text | 213/225 totals sum; CSV→TeX reconstruction is byte-for-byte identical | Upstream success-count aggregation is not recovered; 213/225 is not labeled raw-record reproduced |
| Fixed-cohort case identity could be confused with record ID | Verify stable top-N cohort against source array indices and metadata; keep runs separate | Saved 225 rows match source order; raw recorded-seed coverage and explicit missing/excluded cases exported | Available no-positive cases are not proven historical failures; coverage is not replacement paper data |
| Reproduction depended on private absolute paths | Source-checkout CPU command uses a local hashed bundle and pinned scientific Python dependencies | Existing Docker image, network disabled, checkout/deps read-only, no GPU; 20 regression tests pass | Clean image build and historical model inference are separate validations |
| The prior 21M/layer-1 smoke did not test historical SAE weights | Added a separately scoped original 1.4B/layer-13 single-case helper | Actual original weights verified by SHA; 238 final tensors match source casts; Pepper f2008 seed60 no_steer/0x sequences, structures and scores exactly match the archived example; peak process 3612 MiB | One 7-nt paired case, not all 401 groups/eight illustrations/225 cohort; existing RNAVerse environment, not a new environment build |

The complete CPU run is recorded locally under
`../validation_20260905_round2/sae/cpu_run_final` relative to the repository root,
with `cpu_execution_final.log` and `unit_tests_final.log` in the containing
`sae` directory. `run_report.json` records input/runner hashes and runtime versions;
`output_checksums.json` records the output hashes. The earlier `cpu_run` directory
is retained as a prior validation snapshot; cite `cpu_run_final` for this revision.

Missing original JSON argument metadata remains explicit for the glmS, Milena
and Myo `generation_v2` CSVs. Historical 1.4B and SAE checkpoint hashes/paths are
recorded in `protocol.json`; no checkpoint or training FASTA is shipped here.
The successful historical single-case report is in
`../validation_20260905_round2/sae/historical_gpu_run/run1/historical_smoke_report.json`;
its command/output and original-file before/after hashes are in
`historical_gpu_execution.log` in the containing `sae` evidence directory.
