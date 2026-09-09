# Historical benchmark validation notes from September 5

These notes preserve the earlier audit context. Statements about label
resolution are historical and are not conclusions about the current
author-confirmed submission. Use REPRODUCTION.md for the current workflow.

# Benchmark recovery and reproducibility boundaries

## What was recovered

`examples/reproduction/benchmark/provenance.json` identifies 95 unchanged historical files by source host, original path, byte size and SHA256. These include RNA adapters and their scoring workers, protein reverse-translation/ESM workers, the historical EVA model/tokenizer/loader support code, four training configurations, released Milena inputs, archived predictions and an earlier validation report. This is a source snapshot, not a claim that every external model/environment has been released or rerun. The original server was read-only.

All three released EVA-21M files at `GENTEL-Lab/EVA`, revision `514db6705637c1ec963b728768fc9b34728699ee`, match the historical `30M_1130_midtrain/checkpoint_86006` files byte-for-byte:

| File | SHA256 |
|---|---|
| config.json | `5432d6feca2adf7dc2a859099938b5073ea78d9c544dd389db761077011fc669` |
| tokenizer.json | `9315168d44e3d6df4cdd0c48f3ab6d5111beb13ac8b163a380c59a6fac5a021d` |
| model_weights.pt | `45d56a7399c4936429da149edf5003bbb5999490a64ec7d793bf5ac98116eb01` |

Weights are not duplicated in this bundle. A matching file hash establishes this identity; a *different* serialization hash would not alone prove different tensor values.

## Milena: arithmetic recovered, biological mapping unresolved

The 135 released FASTA sequences agree with the recovered score archive by both ID and sequence. However, **133/135 FASTA header fitness values differ from the positional label JSON**. The value multisets also differ, so reordering is not a justified fix. For example `G2U_fitness_0.03584` is paired positionally with `0.00197`. Neither header values nor positional labels have been selected as corrected ground truth.

| Evidence | Spearman against released positional labels | Status |
|---|---:|---|
| Recovered 1.4B no-condition archive, excluding direction/EOS targets | 0.8360456283218484 | Exact public-table arithmetic recovered; not new inference |
| Released 21M inference, first-round validation | 0.9037308669414685 | Executed; does not match public 21M table |
| Public 21M table | 0.5074679659247847 | Corresponding raw predictions not recovered |
| Earlier 21M archive, including direction/EOS targets | 0.9517764103767578 | Different historical protocol; not a replacement |

The first-round report is preserved unchanged even though its “checkpoint linkage not established” limitation is superseded by the new identity evidence. The sequence/readout mismatch remains unresolved. A targeted filename search of the supplied EVA and RNA benchmark trees found derivative inputs but not the original sequence-linked `cata.json` named by the label metadata. Do not adjust a protocol or substitute labels to chase the public value.

From the repository root, using Python 3.10/3.11 (no GPU or ML libraries):

```bash
python scripts/reproduce_historical_benchmark.py --output outputs/milena_archive_audit
```

This writes `report.json` and all 135 rows of `sequence_label_audit.csv`, then exits **2** because biological validation is blocked. To test *archived arithmetic only*:

```bash
python scripts/reproduce_historical_benchmark.py --output outputs/milena_arithmetic --artifact-only
python -m unittest discover -s tests -p test_benchmark_provenance.py -v
```

The arithmetic comparison allows eight binary64 ULPs for standard-library/scipy summation differences of the **same archived prediction vector**. This is not a scientific reproduction tolerance. New model-run metrics are never accepted using this check. All evidence directories must be new; existing output is not overwritten.

### Supplemental fresh 1.4B inference (2026-09-05)

The migrated `z_rnagym_70/RNAVerse/checkpoint/clm` directory was recovered with complete weights and metadata. Its training configuration identifies `mid_training_v4_from_v24_1.4B_checkpoint_15000_20251202`, not the later v31 run. `historical_14b_manifest.json` records 42 additional unchanged source/metadata files. The current weights have SHA256 `323c13d571d0be87e450cb7b103bf8b28396e55e877519e2d2fb3c434875d420`. This is a path-and-metadata match to the archive; the archive itself did **not** save a weight checksum, so immutable historical weight identity is not established.

All 135 sequences were scored using the recovered native model and worker, no condition, sum reduction, exclusion of `5`, `3`, `<eos>`, BF16 and batch 1 on a shared A100. No alternative protocol was tried. Source code was copied to an isolated directory; original code/checkpoints were not edited. Model construction used BF16 allocation (the original loader's final inference dtype) to avoid its transient FP32 GPU copy; weights loaded strictly, without architecture/backend replacement.

| Fresh-inference evidence | Value |
|---|---:|
| Sequences with bit-exact archived scores | 0 / 135 |
| Maximum absolute prediction difference | 0.302349328994751 |
| Mean absolute prediction difference | 0.06432948178715175 |
| Spearman, fresh versus archived prediction vectors | 0.977465613110916 |
| Spearman, fresh versus unresolved positional labels | 0.8394237924835843 |
| Spearman, archive versus the same positional labels | 0.8360456283218484 |
| Scoring time (not including loading/hash checks) | 48.135 s |
| Peak CUDA allocated / reserved memory | 3,169,734,656 / 3,181,379,584 bytes |

This completes a real forward-pass pipeline but **does not reproduce the archived vector** or resolve the assay labels. No `0.001` tolerance is applied. Original batch size/runtime and immutable weight binding remain missing; these are possible investigation targets, not demonstrated causes. The environment was the existing `70_RNAVerse` conda environment: Python 3.11, torch 2.7.0+cu126, Transformers 4.55.0, tokenizers 0.21.4, megablocks 0.10.0. The recovered attention uses torch SDPA; an external `flash-attn` distribution was absent. No package was installed.

To repeat on a single explicitly selected GPU with this compatible environment:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
python examples/reproduction/benchmark/run_historical_14b.py \
  --source-dir examples/reproduction/benchmark/historical_14b \
  --checkpoint /models/historical-rnaverse-clm \
  --archive examples/reproduction/benchmark/fixtures/milena/eva_1_4b_wotag_scores.json \
  --expected-weights-sha256 323c13d571d0be87e450cb7b103bf8b28396e55e877519e2d2fb3c434875d420 \
  --memory-limit-gib 4 --device cuda:0 --output outputs/milena_14b_fresh
```

The runner reports the exact source/weight hashes, all predictions and their per-row differences, and memory. Exit 0 means inference completed; inspect `status` to distinguish `EXACT_ARCHIVE_VECTOR_MATCH` from `FRESH_INFERENCE_COMPLETE_ARCHIVE_VECTOR_DIFFERS`. No labels are consumed. The initial validation attempt stopped before model loading because the metadata recorder incorrectly required the optional external `flash-attn` package; the recorder now explicitly records missing distribution metadata as null. No scoring fallback was introduced.

