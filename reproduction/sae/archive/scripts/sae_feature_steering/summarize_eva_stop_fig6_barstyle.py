#!/usr/bin/env python3
"""Summarize and redraw EVA stop-cassette SAE steering in InterPLM Fig. 6 style."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CONDS = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]
COND_LABELS = {
    "no_steer": "No steer",
    "0x": "Steer 0x",
    "0.5x": "Steer 0.5x",
    "1x": "Steer 1x",
    "1.5x": "Steer 1.5x",
    "2x": "Steer 2x",
    "2.5x": "Steer 2.5x",
}
COLORS = {
    "no_steer": "#b8bdc2",
    "0x": "#f05b78",
    "0.5x": "#d98b1d",
    "1x": "#35b779",
    "1.5x": "#43b7ad",
    "2x": "#32a6d6",
    "2.5x": "#9b8cf4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--title", default="EVA stop-cassette feature steering")
    p.add_argument(
        "--group-a-title",
        default="p(target base) with steering stop-cassette feature f/4086",
    )
    p.add_argument(
        "--group-b-title",
        default="p(target base) with steering matched control f/776",
    )
    p.add_argument("--ylim", type=float, default=1.0)
    return p.parse_args()


def target_rows(row: dict) -> list[dict]:
    return [x for x in row["score"]["target_tokens"] if not x["is_eos_span"]]


def load_payload(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def summarize(payload: dict) -> tuple[list[dict], list[dict]]:
    rows = payload["rows"]
    flat_rows: list[dict] = []
    summary_rows: list[dict] = []

    by_group = {"A": {}, "B": {}}
    for row in rows:
        group = row["group"]
        if group not in by_group:
            continue
        by_group[group][row["condition"]] = row
        toks = target_rows(row)
        mean_prob = float(np.mean([t["prob"] for t in toks]))
        summary_rows.append(
            {
                "group": group,
                "group_label": row["group_label"],
                "condition": row["condition"],
                "feature_ids": ",".join(str(x) for x in row.get("feature_ids", [])),
                "mean_target_prob": mean_prob,
                "target_token_logp": float(row["score"]["target_token_logp"]),
            }
        )
        for tok in toks:
            flat_rows.append(
                {
                    "group": group,
                    "group_label": row["group_label"],
                    "condition": row["condition"],
                    "target_index": int(tok["target_index"]),
                    "token": tok["token"],
                    "prob": float(tok["prob"]),
                    "log_prob": float(tok["log_prob"]),
                    "target_token_logp": float(row["score"]["target_token_logp"]),
                }
            )

    for group in ["A", "B"]:
        base = by_group[group]["no_steer"]
        base_mean = float(np.mean([t["prob"] for t in target_rows(base)]))
        for item in summary_rows:
            if item["group"] == group:
                item["delta_mean_prob_vs_no_steer"] = item["mean_target_prob"] - base_mean

    return flat_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, summary_rows: list[dict]) -> None:
    ordered = [r for g in ["A", "B"] for c in CONDS for r in summary_rows if r["group"] == g and r["condition"] == c]
    with path.open("w") as handle:
        handle.write("| Panel | Feature | Condition | Mean target p | Delta vs no steer | Target logP |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for r in ordered:
            feature = r["group_label"]
            cond = COND_LABELS.get(r["condition"], r["condition"])
            handle.write(
                f"| {r['group']} | {feature} | {cond} | "
                f"{r['mean_target_prob']:.4f} | {r['delta_mean_prob_vs_no_steer']:+.4f} | "
                f"{r['target_token_logp']:.3f} |\n"
            )


def plot(payload: dict, flat_rows: list[dict], out_prefix: Path, args: argparse.Namespace) -> None:
    target = payload["target"]
    labels = [f"{i + 1}\n{base}" for i, base in enumerate(target)]
    panels = [("A", "a", args.group_a_title), ("B", "b", args.group_b_title)]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.8), sharey=True)
    x = np.arange(len(labels))
    width = 0.105
    offsets = (np.arange(len(CONDS)) - (len(CONDS) - 1) / 2.0) * width

    for ax, (group, letter, title) in zip(axes, panels):
        lookup = {
            (r["condition"], int(r["target_index"])): float(r["prob"])
            for r in flat_rows
            if r["group"] == group
        }
        for j, cond in enumerate(CONDS):
            values = [lookup[(cond, i)] for i in range(len(labels))]
            ax.bar(
                x + offsets[j],
                values,
                width=width * 0.92,
                color=COLORS[cond],
                edgecolor="white",
                linewidth=0.25,
                label=COND_LABELS[cond],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Masked RNA span position / target base", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.text(-0.11, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, args.ylim)
    axes[0].set_ylabel("Teacher-forced p(target base)", fontsize=11)
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 0.98), fontsize=8)
    fig.suptitle(args.title, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    payload = load_payload(args.input_json)
    flat_rows, summary_rows = summarize(payload)

    write_csv(out_prefix.with_name(out_prefix.name + "_per_token.csv"), flat_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_rows)
    write_markdown_table(out_prefix.with_name(out_prefix.name + "_summary.md"), summary_rows)
    plot(payload, flat_rows, out_prefix, args)

    compact = {
        "input_json": args.input_json,
        "target": payload["target"],
        "steer_token_index": payload.get("steer_token_index"),
        "summary": summary_rows,
        "interpretation": (
            "Panel A should show a dose-dependent increase for the stop-cassette feature; "
            "Panel B is the matched-control feature."
        ),
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(compact, handle, indent=2)

    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary.md')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_per_token.csv')}")


if __name__ == "__main__":
    main()
