# Observed reference run

All 135 sequences were scored with the pinned public EVA-1.4B CLM artifact.
The current reference table gives 0.8360456283218484; this run gives
0.8394237924835843 (difference 0.0033781641617359748).
Both values round to 0.84 at two decimal places. This is useful context,
but does not explain the difference in the underlying archived score vector.
The new vector is exactly equal to the separately recorded September 5 fresh
run under different PyTorch/MegaBlocks versions. The versions tested therefore
do not explain the historical mismatch. A recovered March 10 run log identifies the matching output directory
`scores_original_data_wotag`, 135 cata sequences, batch size 45 and three batches.
The current output file is byte-identical to the archived vector. See
`historical_run_evidence.json` and `historical_run_excerpt.txt`. A separate
batch-32 log targets a differently named directory and is not used to bind this
archive. This documents a batch-size difference from the declared batch-1
workflow; it does not prove that batching caused the metric delta. The original
log does not hash the checkpoint or executable source. No batch-size or
objective search was performed to fit the reported value.

The report's decimal-precision comparison is a diagnostic against the stored
CSV, not a demand for scientifically meaningful agreement to 16 decimal places.
The manuscript plot does not annotate an individual Milena coefficient.
Rounding to two decimals is descriptive context only. The observed 0.00338
difference is accepted for the representative example and is not a merge or
release blocker. Current execution status and optional strict comparison are
documented in [the reproduction guide](../../../docs/REPRODUCTION.md).
The dated JSON reports below preserve their original status strings and measurements.

The run used a cached weight file whose SHA256 matches the public revision's
LFS object, plus freshly retrieved public config and tokenizer files.
The complete 3,057,231,170-byte public weight file was independently downloaded
with the documented downloader in a new macOS Python environment; all three
file hashes exactly match the GPU input. `download_manifest.json` records this
check. Direct Hugging Face access from the GPU server reset the connection,
so that server used its byte-identical cached weights.
The label/sequence inputs are unchanged from the author-confirmed current
submission. Outputs are supplied for transparency, not as a replacement of the
manuscript table. GPU scoring took 11.45 seconds and peaked at 3,157,299,200
allocated bytes on one A100. Hashing/loading/plotting are separate overheads.

A second full run from the source archive in the freshly built Docker image
produced exactly the same 135 predictions. GPU scoring took 12.08 seconds with
the same peak allocated memory. See `clean_environment_report.json` and
`reproduction/release/clean_validation.json` for this final execution evidence.
