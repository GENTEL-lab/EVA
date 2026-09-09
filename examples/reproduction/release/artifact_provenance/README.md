# EVA artifact provenance

This supplement identifies the released EVA model files, OpenRNA data files, and bundled benchmark artifacts by immutable repository revision, file path, size and SHA256. EVA_Artifact_Provenance.csv is the human-readable inventory; the JSON contains the same records.

## Author confirmation

On 2026-09-09, the author confirmed that these released checkpoint and dataset files were used in the manuscript. This attestation supplies the paper-use correspondence; checksum evidence remains recorded separately. The comments below about historical reconstruction describe the scope of the independent file audit, not an absence of author confirmation.

## Connection to the manuscript

- EVA model family: the fixed Hugging Face model card identifies the released 21M, 145M, 437M and 1.4B variants. The 1.4B GLM checkpoint is described as the primary mixed-objective model; the 1.4B CLM checkpoint is a separate CLM-only variant. These are different files and must not be interchanged solely by parameter count.
- OpenRNA: the fixed dataset card identifies the collection used for the EVA model family. The corpus and separately published validation file are listed individually. Their release identity does not establish the exact historical training split, sampled order or run inputs.
- Representative Figure 2 reproduction: the released Milena manifest explicitly pins EVA_1.4B_CLM and the benchmark inputs. Its three model-file hashes were cross-checked against this inventory. Seven benchmark input/result files were downloaded from the fixed code release and matched their manifest hashes.
- Other Figure 2 results: the three reference tables identify released summary-score files. Historical per-result checkpoint and input bindings require the corresponding run records.

## Checksum verification

Use the SHA256SUMS file for the relevant repository from the root of your downloaded snapshot:

```sh
shasum -a 256 -c model_SHA256SUMS.txt
```

Use dataset_SHA256SUMS.txt for the dataset snapshot and benchmark_SHA256SUMS.txt for the code snapshot. An OK result indicates byte-for-byte agreement with the recorded digest. These commands verify existing local files and do not download data.

Large model/data hashes were read from the fixed Hugging Face LFS metadata. Small model components and benchmark files were downloaded and hashed. The inventory distinguishes these evidence sources. Git blob IDs and Xet IDs are not substituted for SHA256.

## Sources

- Model card: https://huggingface.co/GENTEL-Lab/EVA/blob/514db6705637c1ec963b728768fc9b34728699ee/README.md
- Dataset card: https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M/blob/103c79aab4d625828721a25ab98be51763c326bb/README.md
- Benchmark manifest: https://github.com/GENTEL-lab/EVA/blob/9e8d63bbe4ab4a0c5cf2646164751b52e8a71462/examples/reproduction/milena_14b/manifest.json
- Benchmark table manifest: https://github.com/GENTEL-lab/EVA/blob/9e8d63bbe4ab4a0c5cf2646164751b52e8a71462/examples/reproduction/benchmark_release/reference_manifest.json
