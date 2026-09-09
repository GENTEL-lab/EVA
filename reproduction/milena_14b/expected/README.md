# Observed reference run

All 135 sequences were scored with the pinned public EVA-1.4B CLM artifact.
The current reference table gives 0.8360456283218484; this run gives
0.8394237924835843 (difference 0.0033781641617359748).
Both values round to 0.84 at two decimal places. This is useful context,
but does not explain the difference in the underlying archived score vector.
The new vector is exactly equal to the separately recorded September 5 fresh
run under different PyTorch/MegaBlocks versions. The versions tested therefore
do not explain the historical mismatch. The historical prediction CLI defaults to batch size 128, while this declared
workflow uses batch size 1. The archive records neither its actual batch size
nor its dtype and exact command, so the defaults cannot establish the cause.
Original archive-producing command, model binding and execution settings are
still needed to resolve that question. No batch-size or objective search was
performed to fit the reported value.

The report's decimal-precision comparison is a diagnostic against the stored
CSV, not a demand for scientifically meaningful agreement to 16 decimal places.
The manuscript plot does not annotate an individual Milena coefficient.
Rounding to two decimals is descriptive context only; publication remains
pending the requested explanation of the different archived predictions.

The run used a cached weight file whose SHA256 matches the public revision's
LFS object, plus freshly retrieved public config and tokenizer files. Direct
Hugging Face download from the validation server failed with a connection reset;
this run is not evidence of a successful online download in that environment.
The label/sequence inputs are unchanged from the author-confirmed current
submission. Outputs are supplied for transparency, not as a replacement of the
manuscript table. GPU scoring took 11.45 seconds and peaked at 3,157,299,200
allocated bytes on one A100. Hashing/loading/plotting are separate overheads.
