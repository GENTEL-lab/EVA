#!/usr/bin/env python3
"""Bounded fresh inference using an unchanged, explicitly supplied historical tree.

This compares predictions with an archive, not with corrected biological labels.
The old loader's device/dtype allocation order is replaced only to avoid a
transient full-float32 GPU copy. Architecture, weights and scoring stay explicit.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-limit-gib", type=float, default=5.0)
    args = parser.parse_args()
    if not 0 < args.memory_limit_gib <= 5:
        parser.error("memory-limit-gib must be in (0, 5]")
    if args.output.exists():
        parser.error("output must be a new directory")
    raw = json.loads(args.archive.read_text())
    protocol = dict(mode="rna", normalize=False, exclude_special_tokens=True,
                    length_normalize=False, condition=None)
    for key, value in protocol.items():
        if raw.get(key, "MISSING") != value:
            raise ValueError(f"Archive protocol mismatch: {key}")
    rows = raw["scores"]
    if len(rows) != 135 or len({row["header"] for row in rows}) != 135:
        raise ValueError("Expected the complete 135-row Milena archive with unique IDs")
    if any(len(row["sequence"]) != 54 or set(row["sequence"]) - set("ACGU")
           or not math.isfinite(row["log_likelihood"]) for row in rows):
        raise ValueError("Invalid sequence or archived prediction")
    checkpoint_hashes = {p.name: sha(p) for p in sorted(args.checkpoint.iterdir())
                         if p.is_file()}
    if checkpoint_hashes.get("model_weights.pt") != args.expected_weights_sha256:
        raise ValueError("Checkpoint SHA256 does not match the explicit expected identity")
    args.output.mkdir(parents=True)
    report = {"status": "PREFLIGHT_COMPLETE", "protocol": protocol,
              "batch_size": 1, "source_directory": str(args.source_dir),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hashes,
              "archive_path": str(args.archive), "archive_sha256": sha(args.archive),
              "runner_sha256": sha(__file__), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
              "biological_label_status": "Not assessed by this archive-vector comparison; labels are evaluated by the enclosing benchmark workflow",
              "archived_checkpoint_binding": "Migrated path and metadata correspondence; archive contains no checkpoint checksum",
              "limitations": ["Original inference batch size and exact historical runtime are not saved in archive.",
                              "A different current prediction vector is not accepted by an arbitrary tolerance."]}
    report_path = args.output / "report.json"
    try:
        isolated = args.output / "source"
        for subdir in ("eva", "tools"):
            shutil.copytree(args.source_dir / subdir, isolated / subdir,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        report["source_sha256"] = {str(p.relative_to(isolated)): sha(p)
                                   for p in sorted(isolated.rglob("*")) if p.is_file()}
        sys.path[:0] = [str(isolated), str(isolated / "tools")]
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        import torch
        from eva.config import EvaConfig
        from eva.causal_lm import EvaForCausalLM
        from eva.lineage_tokenizer import LineageRNATokenizer
        from utils.scorers.score_worker import compute_batch_likelihood
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the historical model implementation")
        torch.cuda.set_device(args.device)
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 1024 ** 3 / total,
                                                   args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        report["environment"] = {}
        for name in ("torch", "transformers", "tokenizers", "megablocks", "flash-attn"):
            try:
                report["environment"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # The recovered attention uses torch SDPA, not the optional
                # external flash-attn distribution. Missing metadata is explicit.
                report["environment"][name] = None
        report["gpu"] = torch.cuda.get_device_name(args.device)
        tokenizer = LineageRNATokenizer.from_pretrained(str(args.checkpoint))
        config_dict = json.loads((args.checkpoint / "config.json").read_text())
        config = EvaConfig(tokenizer=tokenizer, **config_dict)
        config.moe_world_size = 1  # Same single-GPU override as original ModelLoader.
        config.bf16 = True  # Allocation dtype only; original loader ends in bfloat16.
        print("Constructing historical architecture with bounded BF16 allocation", flush=True)
        weights = torch.load(args.checkpoint / "model_weights.pt", map_location="cpu",
                             weights_only=True)
        if "model" in weights and isinstance(weights["model"], dict):
            weights = weights["model"]
        elif "state_dict" in weights:
            weights = weights["state_dict"]
        config.vocab_size = weights["model.embed_tokens.weight"].shape[0]
        model = EvaForCausalLM(config).bfloat16()
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(args.device).eval()
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["load_protocol"] = "Unchanged historical classes; strict state_dict load; same BF16 inference, single-rank override. BF16 allocation avoids original loader's transient CUDA float32 weights."
        start = time.monotonic()
        values = compute_batch_likelihood(model, tokenizer, [r["sequence"] for r in rows],
                                          args.device, reduce_method="sum",
                                          exclude_special_tokens=True, batch_size=1,
                                          sequence_only=False)
        torch.cuda.synchronize(args.device)
        report["scoring_seconds"] = time.monotonic() - start
        if len(values) != len(rows) or not all(math.isfinite(x) for x in values):
            raise ValueError("Missing or nonfinite fresh predictions")
        differences = [score - row["log_likelihood"] for score, row in zip(values, rows)]
        report.update(status="EXACT_ARCHIVE_VECTOR_MATCH" if all(x == 0 for x in differences)
                      else "FRESH_INFERENCE_COMPLETE_ARCHIVE_VECTOR_DIFFERS",
                      n_sequences=len(rows), exact_equal_count=sum(x == 0 for x in differences),
                      max_absolute_difference=max(map(abs, differences)),
                      mean_absolute_difference=sum(map(abs, differences)) / len(differences),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        with (args.output / "predictions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["variant_id", "sequence", "archived_score",
                                                        "fresh_score", "fresh_minus_archived"])
            writer.writeheader()
            for row, score, difference in zip(rows, values, differences):
                writer.writerow(dict(variant_id=row["header"], sequence=row["sequence"],
                                     archived_score=row["log_likelihood"], fresh_score=score,
                                     fresh_minus_archived=difference))
        report["predictions_sha256"] = sha(args.output / "predictions.csv")
    except Exception as exc:
        report.update(status="FAILED_NO_PROTOCOL_FALLBACK", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "n_sequences", "exact_equal_count",
                      "max_absolute_difference", "cuda_peak_allocated_bytes")}), flush=True)


if __name__ == "__main__":
    main()
