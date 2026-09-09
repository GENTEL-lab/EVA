#!/usr/bin/env python3
"""Lineage-fixed EVA eukaryote stop-cassette position ablation.

This wrapper keeps the original runner's data loading, sharding, mutation, and
analysis code, but fixes the scoring condition: both WT and mutant sequences are
scored with the lineage prefix used by the original essentiality experiment.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


BASE_SCRIPT = Path(__file__).with_name("eva_position_ablation_eukaryote.py")
spec = importlib.util.spec_from_file_location("position_ablation_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import base runner: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def prepare_inputs_with_optional_prefix(tokenizer, sequence: str, device: str) -> dict[str, torch.Tensor]:
    """Match EVA's official score_worker prefix/direction-token handling.

    Original notebook format: |lineage|5sequence3 (direction token after prefix)
    """
    sequence_with_direction = f"5{sequence}3"
    full_sequence = f"<bos>{sequence_with_direction}<eos>"
    token_ids = tokenizer.encode(full_sequence)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "position_ids": torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0),
        "sequence_ids": torch.zeros((1, len(token_ids)), dtype=torch.long, device=device),
    }


def add_lineage_prefix(sequence: str, item: dict[str, Any]) -> str:
    lineage = item.get("lineage")
    if lineage:
        return f"|{lineage}|{sequence}"
    return sequence


def main() -> None:
    args = base.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = base.load_data(args)
    wt_cache = base.load_wt_cache(args.wt_cache)
    done = base.load_done(output_dir, args.model_name)

    base.prepare_inputs = prepare_inputs_with_optional_prefix
    model, tokenizer = base.load_model(args.checkpoint, args.device, args.model_code_path)

    started = time.time()
    processed = 0
    for idx, item in enumerate(data, start=1):
        sequence = item["sequence"]
        if len(sequence) > args.max_seqlen:
            sequence = sequence[: args.max_seqlen]

        key = item["_key"]
        missing_positions = [(p, n, r) for p, n, r in base.POSITIONS if key not in done[p]]
        if not missing_positions:
            continue

        wt_ll = wt_cache.get(key)
        if wt_ll is None:
            wt_ll = base.compute_log_likelihood(
                model, tokenizer, add_lineage_prefix(sequence, item), args.device
            )

        base_row = {
            "key": key,
            "record_index": int(item["_record_index"]),
            "gene": item.get("gene", item.get("gene_name", "")),
            "locus_tag": item.get("locus_tag", ""),
            "organism": item["organism"],
            "nc": item.get("nc", item.get("refseq_id", "")),
            "label": int(item["label"]),
            "essential": bool(item["label"]),
            "length": len(sequence),
            "wt_ll": wt_ll,
            "lineage": item.get("lineage", ""),
            "scoring_condition": "lineage_prefixed",
        }

        for pos_id, pos_name, ratio in missing_positions:
            mutant, insert_pos = base.insert_mutation(sequence, ratio, base.MUTATION)
            mut_ll = base.compute_log_likelihood(
                model, tokenizer, add_lineage_prefix(mutant, item), args.device
            )
            row = dict(base_row)
            row.update(
                {
                    "position_id": pos_id,
                    "position": pos_name,
                    "position_ratio": ratio,
                    "insert_pos": insert_pos,
                    "mutation": base.MUTATION,
                    "mut_ll": mut_ll,
                    "delta_ll": wt_ll - mut_ll,
                }
            )
            base.append_jsonl(output_dir / f"{args.model_name}_{pos_id}.jsonl", row)
            done[pos_id].add(key)

        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            progress = {
                "model_name": args.model_name,
                "processed_new_records": processed,
                "input_index": idx,
                "total_records": len(data),
                "elapsed_seconds": elapsed,
                "records_per_second": rate,
                "done_by_position": {p: len(s) for p, s in done.items()},
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "scoring_condition": "lineage_prefixed",
            }
            base.write_progress(output_dir, args.model_name, progress)
            remaining = len(data) - idx
            eta = remaining / rate if rate > 0 else None
            print(
                f"{args.model_name}: input {idx}/{len(data)}, new {processed}, "
                f"rate={rate:.4f} rec/s, eta={eta:.0f}s" if eta else "",
                flush=True,
            )

    summary = base.analyze_all(output_dir, args.model_name)
    summary["num_shards"] = args.num_shards
    summary["shard_index"] = args.shard_index
    summary["scoring_condition"] = "lineage_prefixed"
    summary_path = output_dir / f"{args.model_name}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / f"{args.model_name}.done", "w") as handle:
        handle.write("done\n")
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
