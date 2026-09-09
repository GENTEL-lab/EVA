#!/usr/bin/env python3
"""Run EVA eukaryote stop-codon position ablation with optional WT LL reuse."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/eva")

from tools.utils.model.loader import ModelLoader  # noqa: E402


MUTATION = "UAAUAAUAAUAGUGA"
POSITIONS = [
    ("p05", "5%", 0.05),
    ("p25", "25%", 0.25),
    ("p50", "50%", 0.50),
    ("p75", "75%", 0.75),
    ("p95", "95%", 0.95),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--wt-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seqlen", type=int, default=8192)
    parser.add_argument("--sample-per-species", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--model-code-path", default="/eva/eva")
    return parser.parse_args()


def item_key(item: dict[str, Any], record_index: int | None = None) -> str:
    if record_index is None:
        record_index = item.get("_record_index", "")
    return "|".join(
        [
            str(record_index),
            str(item.get("organism", "")),
            str(item.get("gene", item.get("gene_name", ""))),
            str(item.get("locus_tag", "")),
            str(item.get("nc", item.get("refseq_id", ""))),
        ]
    )


def label_of(item: dict[str, Any]) -> int:
    if "label" in item:
        return int(item["label"])
    return int(bool(item["essential"]))


def load_json(path: str | Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def load_wt_cache(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = load_json(path)
    cache = {}
    for idx, item in enumerate(data):
        if "wt_ll" in item:
            cache[item_key(item, idx)] = float(item["wt_ll"])
    print(f"Loaded WT cache: {len(cache)} records from {path}", flush=True)
    return cache


def load_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    data = [x for x in load_json(args.dataset) if x.get("sequence")]
    for idx, item in enumerate(data):
        item["_record_index"] = idx
        item["label"] = label_of(item)
        item["_key"] = item_key(item, idx)

    if args.sample_per_species > 0:
        rng = random.Random(args.seed)
        sampled = []
        by_species = defaultdict(list)
        for item in data:
            by_species[item["organism"]].append(item)
        for items in by_species.values():
            pos = [x for x in items if x["label"] == 1]
            neg = [x for x in items if x["label"] == 0]
            n = min(args.sample_per_species // 2, len(pos), len(neg))
            if n > 0:
                sampled.extend(rng.sample(pos, n))
                sampled.extend(rng.sample(neg, n))
        data = sampled

    if args.limit > 0:
        data = data[: args.limit]

    if args.num_shards > 1:
        before = len(data)
        data = [x for x in data if x["_record_index"] % args.num_shards == args.shard_index]
        print(
            f"Applied shard filter: shard {args.shard_index}/{args.num_shards}, "
            f"{len(data)}/{before} records",
            flush=True,
        )

    by_species = defaultdict(lambda: [0, 0])
    for item in data:
        by_species[item["organism"]][item["label"]] += 1
    print(f"Loaded data: {len(data)} records from {args.dataset}", flush=True)
    for species, counts in sorted(by_species.items()):
        print(
            f"  {species}: essential={counts[1]}, non_essential={counts[0]}",
            flush=True,
        )
    return data


def load_model(checkpoint: str, device: str, model_code_path: str):
    print(f"Loading model: {checkpoint}", flush=True)
    loader = ModelLoader(checkpoint, model_code_path=model_code_path)
    model, tokenizer = loader.load(device=device)
    model.eval()
    print(f"Model loaded on {device}", flush=True)
    return model, tokenizer


def prepare_inputs(tokenizer, sequence: str, device: str) -> dict[str, torch.Tensor]:
    full_sequence = f"<bos>5{sequence}3<eos>"
    token_ids = tokenizer.encode(full_sequence)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "position_ids": torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0),
        "sequence_ids": torch.zeros((1, len(token_ids)), dtype=torch.long, device=device),
    }


def compute_log_likelihood(model, tokenizer, sequence: str, device: str) -> float:
    inputs = prepare_inputs(tokenizer, sequence, device)
    use_cuda_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda_amp
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_ctx:
            outputs = model(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                sequence_ids=inputs["sequence_ids"],
            )
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

    input_ids = inputs["input_ids"][0]
    total = 0.0
    for idx in range(len(input_ids) - 1):
        total += log_probs[0, idx, input_ids[idx + 1]].item()

    del inputs, outputs, log_probs
    return float(total)


def insert_mutation(sequence: str, ratio: float, mutation: str) -> tuple[str, int]:
    insert_pos = max(1, min(int(len(sequence) * ratio), len(sequence) - 1))
    return sequence[:insert_pos] + mutation + sequence[insert_pos:], insert_pos


def load_done(output_dir: Path, model_name: str) -> dict[str, set[str]]:
    done = {}
    for pos_id, _, _ in POSITIONS:
        path = output_dir / f"{model_name}_{pos_id}.jsonl"
        seen = set()
        if path.exists():
            with open(path, "r") as handle:
                for line in handle:
                    if line.strip():
                        seen.add(json.loads(line)["key"])
        done[pos_id] = seen
        print(f"Resume state {pos_id}: {len(seen)} records", flush=True)
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def rank_auc(labels: list[int], scores: list[float]) -> float | None:
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0 or n < 2:
        return None

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    rank_sum_pos = sum(ranks[i] for i, label in enumerate(labels) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def analyze_position(path: Path) -> dict[str, Any]:
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    by_species = defaultdict(list)
    for row in rows:
        by_species[row["organism"]].append(row)

    per_species = []
    for species, items in sorted(by_species.items()):
        labels = [int(x["label"]) for x in items]
        scores = [float(x["delta_ll"]) for x in items]
        auc = rank_auc(labels, scores)
        if auc is not None:
            per_species.append(
                {
                    "organism": species,
                    "auroc": auc,
                    "n_genes": len(items),
                    "n_essential": int(sum(labels)),
                    "n_nonessential": int(len(labels) - sum(labels)),
                }
            )

    labels = [int(x["label"]) for x in rows]
    scores = [float(x["delta_ll"]) for x in rows]
    aucs = [x["auroc"] for x in per_species]
    return {
        "n_genes": len(rows),
        "n_essential": int(sum(labels)),
        "n_nonessential": int(len(labels) - sum(labels)),
        "overall_auroc": rank_auc(labels, scores),
        "mean_auroc": float(np.mean(aucs)) if aucs else None,
        "std_auroc": float(np.std(aucs)) if aucs else None,
        "min_auroc": float(np.min(aucs)) if aucs else None,
        "max_auroc": float(np.max(aucs)) if aucs else None,
        "range_auroc": float(np.max(aucs) - np.min(aucs)) if aucs else None,
        "per_species": per_species,
    }


def analyze_all(output_dir: Path, model_name: str) -> dict[str, Any]:
    analysis = {}
    for pos_id, pos_name, _ in POSITIONS:
        path = output_dir / f"{model_name}_{pos_id}.jsonl"
        if path.exists():
            analysis[pos_name] = analyze_position(path)
    aurocs = [
        x["overall_auroc"]
        for x in analysis.values()
        if x.get("overall_auroc") is not None
    ]
    return {
        "model_name": model_name,
        "mutation": MUTATION,
        "positions": [{"id": x[0], "name": x[1], "ratio": x[2]} for x in POSITIONS],
        "analysis": analysis,
        "overall_position_std": float(np.std(aurocs)) if aurocs else None,
        "overall_position_range": float(max(aurocs) - min(aurocs)) if aurocs else None,
    }


def write_progress(output_dir: Path, model_name: str, payload: dict[str, Any]) -> None:
    with open(output_dir / f"{model_name}_progress.json", "w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    wt_cache = load_wt_cache(args.wt_cache)
    done = load_done(output_dir, args.model_name)
    model, tokenizer = load_model(args.checkpoint, args.device, args.model_code_path)

    started = time.time()
    processed = 0
    for idx, item in enumerate(data, start=1):
        sequence = item["sequence"]
        if len(sequence) > args.max_seqlen:
            sequence = sequence[: args.max_seqlen]

        key = item["_key"]
        missing_positions = [(p, n, r) for p, n, r in POSITIONS if key not in done[p]]
        if not missing_positions:
            continue

        wt_ll = wt_cache.get(key)
        if wt_ll is None:
            wt_ll = compute_log_likelihood(model, tokenizer, sequence, args.device)

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
        }

        for pos_id, pos_name, ratio in missing_positions:
            mutant, insert_pos = insert_mutation(sequence, ratio, MUTATION)
            mut_ll = compute_log_likelihood(model, tokenizer, mutant, args.device)
            row = dict(base_row)
            row.update(
                {
                    "position_id": pos_id,
                    "position": pos_name,
                    "position_ratio": ratio,
                    "insert_pos": insert_pos,
                    "mutation": MUTATION,
                    "mut_ll": mut_ll,
                    "delta_ll": wt_ll - mut_ll,
                }
            )
            append_jsonl(output_dir / f"{args.model_name}_{pos_id}.jsonl", row)
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
            }
            write_progress(output_dir, args.model_name, progress)
            remaining = len(data) - idx
            eta = remaining / rate if rate > 0 else None
            print(
                f"{args.model_name}: input {idx}/{len(data)}, new {processed}, "
                f"rate={rate:.4f} rec/s, eta={eta:.0f}s" if eta else "",
                flush=True,
            )

    summary = analyze_all(output_dir, args.model_name)
    summary["num_shards"] = args.num_shards
    summary["shard_index"] = args.shard_index
    summary_path = output_dir / f"{args.model_name}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / f"{args.model_name}.done", "w") as handle:
        handle.write("done\n")
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
