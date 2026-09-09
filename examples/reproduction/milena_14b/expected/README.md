# Observed reference run

All 135 sequences were scored with the pinned public EVA-1.4B CLM artifact.
The end-to-end run achieved a Spearman correlation of **0.84**, matching the
reference result to two decimal places.

The directory contains saved predictions, environment records and execution
reports. [The reproduction guide](../../../../docs/REPRODUCTION.md) describes
the workflow, expected outputs and optional strict-reference check.

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
`examples/reproduction/release/clean_validation.json` for this final execution evidence.
