#!/usr/bin/env python3
"""Bounded historical Pepper/1.4B/layer-13 inference, not a cohort benchmark.

Run only with an isolated copy of the historical eva/ and tools/ code. Original
checkpoints stay external/read-only. The legacy loader may copy tokenizer.json
into that isolated code copy; never point --isolated-eva-root at the original.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--isolated-eva-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--rnafold-bin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Output must be a new path")
    if not (options.isolated_eva_root / "ISOLATED_SAE_VALIDATION.txt").is_file():
        parser.error("Missing isolated-code marker; refusing to use an original checkout")
    options.output.mkdir(parents=True)
    report = {"scope": "One historical Pepper case, one feature, one paired seed, no_steer versus 0x ablation; not a full paper-result reproduction.",
              "status": "started", "pid": os.getpid(), "python": sys.version,
              "platform": platform.platform(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "runner_sha256": digest(__file__), "allocator_cap_gib": 4.0,
              "process_memory_stop_threshold_mib": 5120, "model_inference_performed": False}
    started = time.monotonic()
    stop, over_budget = threading.Event(), threading.Event()
    memory_samples = []

    def monitor():
        while not stop.is_set():
            try:
                text = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                for line in text.splitlines():
                    pid, value = (part.strip() for part in line.split(",", 1))
                    if pid == str(os.getpid()) and value.isdigit():
                        memory_samples.append({"seconds": round(time.monotonic() - started, 3), "mib": int(value)})
                        if int(value) > 5120:
                            over_budget.set()
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(.5)

    watcher = threading.Thread(target=monitor, daemon=True)
    try:
        manifest = json.loads((options.bundle / "manifest.json").read_text())
        protocol = json.loads((options.bundle / "protocol.json").read_text())
        selected = next(item for item in protocol["examples"]["cases"] if item["name"] == "pepper")
        csv_rel = "archive/data/sae_feature_steering/rnafold_cases/" + selected["generation_csv"]
        run = next(item for item in manifest["generation_runs"] if item["rows_path"] == csv_rel)
        if not run["args"]["paired_sample_seeds"]:
            raise ValueError("Selected original run did not record paired_sample_seeds")
        for label, path in (("historical_model", options.checkpoint / "model_weights.pt"), ("sae", options.sae)):
            actual = digest(path)
            if actual != protocol["checkpoint_provenance"][label]["sha256"]:
                raise ValueError(f"Unexpected {label} checkpoint hash")
            report[label + "_sha256"] = actual
        report["checkpoint_auxiliary_sha256"] = {
            p.name: digest(p) for p in options.checkpoint.iterdir()
            if p.is_file() and p.suffix in (".json", ".yaml")}
        report["model_code_sha256"] = {
            str(p.relative_to(options.isolated_eva_root)): digest(p)
            for root in (options.isolated_eva_root / "eva", options.isolated_eva_root / "tools")
            for p in sorted(root.rglob("*.py"))}
        report["original_args"] = run["args"]
        report["original_selection"] = selected
        args = argparse.Namespace(**run["args"])
        args.checkpoint, args.sae = str(options.checkpoint), str(options.sae)
        args.eva_root = str(options.isolated_eva_root)
        args.model_code_path = str(options.isolated_eva_root / "eva")
        args.extra_site_packages, args.device = "", "cuda:0"
        args.rnafold_bin = options.rnafold_bin
        args.num_samples = 1
        args.seed = selected["sample_seed"]
        report["sample_index_mapping"] = {"archived_sample_idx": selected["sample_idx"],
                                          "archived_seed": selected["sample_seed"],
                                          "smoke_sample_idx": 0,
                                          "note": "One sample is run directly with its archived RNG seed, instead of generating all preceding sample indices."}
        scripts = options.bundle / "archive/scripts/sae_feature_steering"
        sys.path.insert(0, str(scripts))
        sys.path.insert(0, str(options.isolated_eva_root))
        import torch
        import transformers
        import tokenizers
        import score_rnafold_structure_switch_generation as generation
        from tools.utils.model.loader import ModelLoader
        from eva.config import EvaConfig
        from eva.causal_lm import EvaForCausalLM
        from eva.lineage_tokenizer import LineageRNATokenizer
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info(0)
        report["initial_cuda_free_bytes"] = free
        if free < 8 * 1024**3:
            raise MemoryError("Less than 8 GiB is currently free; not starting shared inference")
        torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total, 0)
        watcher.start()
        # The original class initializes MegaBlocks experts on CUDA even when
        # the loader later requests CPU. Set its allocation-only bf16 flag to
        # avoid a transient FP32 expert copy, then validate every final tensor
        # against the source cast to the original loader's final BF16 dtype.
        loader = ModelLoader(args.checkpoint, model_code_path=args.model_code_path)
        shutil.copyfile(options.checkpoint / "tokenizer.json", options.isolated_eva_root / "eva/tokenizer.json")
        tokenizer = LineageRNATokenizer.from_pretrained(args.checkpoint)
        config_dict = json.loads((options.checkpoint / "config.json").read_text())
        config = EvaConfig(tokenizer=tokenizer, **config_dict)
        config.moe_world_size = 1
        config.bf16 = True
        raw = torch.load(options.checkpoint / "model_weights.pt", map_location="cpu", weights_only=False)
        weights = loader._extract_state_dict(raw)
        config.vocab_size = weights["model.embed_tokens.weight"].shape[0]
        model = EvaForCausalLM(config)
        model.load_state_dict(weights, strict=True)
        model.bfloat16()
        model.to(device=args.device)
        model.eval()
        checked = 0
        for name, value in model.state_dict().items():
            expected = weights[name].to(dtype=value.dtype, device="cpu")
            if not torch.equal(value.detach().cpu(), expected):
                raise ValueError(f"Loaded tensor differs from checkpoint after dtype cast: {name}")
            checked += 1
        report["loading_validation"] = {
            "strict_state_dict": True, "all_final_state_tensors_equal_source_cast": True,
            "checked_tensors": checked,
            "allocation_only_change": "config.bf16=True during original-class construction avoids transient FP32 CUDA experts; original loader final BF16 dtype retained.",
            "moe_implementation": config.moe_implementation,
        }
        del raw, weights, expected
        sae = generation.load_sae(args.sae, args.sae_mode, args.device)
        report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "tokenizers": tokenizers.__version__,
                              "rnafold": subprocess.check_output([args.rnafold_bin, "--version"], text=True).strip()}
        report["model_parameter_dtype"] = str(next(model.parameters()).dtype)
        report["model_parameters"] = sum(p.numel() for p in model.parameters())
        report["sae_mode"] = sae.mode

        def guard(_module, _inputs):
            if over_budget.is_set():
                raise MemoryError("Own process exceeded the 5 GiB monitoring threshold")
            if torch.cuda.mem_get_info(0)[0] < 6 * 1024**3:
                raise MemoryError("Free shared GPU memory fell below 6 GiB")

        model.register_forward_pre_hook(guard)
        source = options.bundle / "archive/data/sae_feature_steering/rnafold_cases/chen2019_pepper_structure_switch_cases.json"
        cases = json.loads(source.read_text())
        if isinstance(cases, dict):
            cases = cases["cases"]
        entry = {"case_index": selected["case_index"], "case": cases[selected["case_index"]]}
        print("Running one Pepper layer-13 f/2008 recorded-seed pair; GPU allocator cap 4 GiB", flush=True)
        rows = generation.run_one(model, tokenizer, sae, entry, "wt_state",
                                  {"feature_id": selected["feature_id"]}, args, [0.0])
        torch.cuda.synchronize()
        if len(rows) != 2:
            raise RuntimeError(f"Expected exactly 2 generated rows, got {len(rows)}")
        report["model_inference_performed"] = True
        report["rows"] = rows
        report["effective_args"] = vars(args)
        with (options.bundle / csv_rel).open(newline="") as stream:
            archived = [row for row in csv.DictReader(stream)
                        if row["case_index"] == str(selected["case_index"])
                        and row["feature_id"] == str(selected["feature_id"])
                        and row["direction"] == "wt_state"
                        and row["sample_seed"] == str(selected["sample_seed"])
                        and row["sample_idx"] == str(selected["sample_idx"])
                        and row["condition"] in ("no_steer", "0x")]
        compared = ("generated_span", "full_sequence", "structure", "target_hits", "opposite_hits",
                    "net_target_score", "anchor_pos1", "span_start0", "span_end0")
        comparison = []
        for row in rows:
            old = [item for item in archived if item["condition"] == row["condition"]]
            if len(old) != 1:
                raise ValueError("Ambiguous original selector")
            comparison.append({"condition": row["condition"],
                               "fields_match": {key: str(row[key]) == old[0][key] for key in compared},
                               "archive_anchor_one_x": float(old[0]["anchor_one_x"]),
                               "new_anchor_one_x": row["anchor_one_x"]})
        report["comparison_to_archived_sample"] = comparison
        report["all_compared_fields_match"] = all(all(item["fields_match"].values()) for item in comparison)
        report["status"] = "historical_model_sae_single_pair_inference_completed"
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        print("Historical pair completed; exact archived-field match:", report["all_compared_fields_match"], flush=True)
        return_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
        return_code = 1
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=2)
        report["elapsed_seconds"] = time.monotonic() - started
        report["process_gpu_memory_samples"] = memory_samples
        report["peak_process_gpu_mib"] = max((sample["mib"] for sample in memory_samples), default=None)
        (options.output / "historical_smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
