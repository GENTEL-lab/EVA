#!/usr/bin/env python3
"""Merge position-ablation JSONL shards and write combined AUROC summary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POSITIONS = [
    ("p05", "5%"),
    ("p25", "25%"),
    ("p50", "50%"),
    ("p75", "75%"),
    ("p95", "95%"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-model-name", required=True)
    return parser.parse_args()


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


def read_rows(input_dirs: list[Path], pos_id: str) -> list[dict[str, Any]]:
    rows_by_key = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob(f"*_{pos_id}.jsonl")):
            with open(path, "r") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_by_key[row["key"]] = row
    return list(rows_by_key.values())


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    aurocs = [x["auroc"] for x in per_species]
    return {
        "n_genes": len(rows),
        "n_essential": int(sum(labels)),
        "n_nonessential": int(len(labels) - sum(labels)),
        "overall_auroc": rank_auc(labels, scores),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "std_auroc": float(np.std(aurocs)) if aurocs else None,
        "min_auroc": float(np.min(aurocs)) if aurocs else None,
        "max_auroc": float(np.max(aurocs)) if aurocs else None,
        "range_auroc": float(np.max(aurocs) - np.min(aurocs)) if aurocs else None,
        "per_species": per_species,
    }


def main() -> None:
    args = parse_args()
    input_dirs = [Path(x) for x in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": args.merged_model_name,
        "input_dirs": [str(x) for x in input_dirs],
        "analysis": {},
    }

    for pos_id, pos_name in POSITIONS:
        rows = read_rows(input_dirs, pos_id)
        rows.sort(key=lambda x: int(x.get("record_index", 0)))
        merged_path = output_dir / f"{args.merged_model_name}_{pos_id}.jsonl"
        with open(merged_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        summary["analysis"][pos_name] = analyze(rows)
        print(f"Merged {pos_name}: {len(rows)} rows -> {merged_path}")

    aurocs = [
        x["overall_auroc"]
        for x in summary["analysis"].values()
        if x.get("overall_auroc") is not None
    ]
    summary["overall_position_std"] = float(np.std(aurocs)) if aurocs else None
    summary["overall_position_range"] = float(max(aurocs) - min(aurocs)) if aurocs else None

    summary_path = output_dir / f"{args.merged_model_name}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
