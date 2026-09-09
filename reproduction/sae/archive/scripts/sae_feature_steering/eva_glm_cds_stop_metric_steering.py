#!/usr/bin/env python3
"""CDS-property steering metric for EVA GLM SAE features.

This tests a CDS feature with a CDS-specific metric instead of exact native-base
probability. At each in-frame codon in a masked CDS span, it estimates the
teacher-forced pseudo-probability mass assigned to stop codons (UAA/UAG/UGA).

Expected causal pattern for a CDS feature:
  steering CDS feature up -> in-frame stop-codon probability decreases.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    forward_with_steer,
    load_model,
    load_sae,
    parse_scales,
    token_ids,
)


STOP_CODONS = {"UAA", "UAG", "UGA"}
BASES = ["A", "C", "G", "U"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--probe-json", required=True)
    p.add_argument("--feature-max-json", default="")
    p.add_argument("--target-feature", type=int, default=4512)
    p.add_argument("--control-feature", type=int, default=5078)
    p.add_argument("--target-one-x", type=float, default=None)
    p.add_argument("--control-one-x", type=float, default=None)
    p.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    p.add_argument("--scale-source", choices=["feature_max_json", "constant", "current"], default="feature_max_json")
    p.add_argument("--clamp-value", type=float, default=10.0)
    p.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="decoder_delta")
    p.add_argument("--steer-offset", type=int, default=-3)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--title", default="EVA CDS feature steering: stop-codon metric")
    return p.parse_args()


def load_probes(path: str) -> tuple[list[dict[str, Any]], dict[int, float]]:
    with open(path) as handle:
        payload = json.load(handle)
    one_x = {int(k): float(v) for k, v in payload.get("feature_one_x_values", {}).items()}
    return payload["probes"], one_x


def load_feature_max(args: argparse.Namespace, probe_one_x: dict[int, float]) -> dict[int, float]:
    feature_max = dict(probe_one_x)
    if args.feature_max_json:
        with open(args.feature_max_json) as handle:
            payload = json.load(handle)
        raw = payload.get("feature_one_x_values", payload)
        feature_max.update({int(k): float(v) for k, v in raw.items()})
    if args.target_one_x is not None:
        feature_max[args.target_feature] = float(args.target_one_x)
    if args.control_one_x is not None:
        feature_max[args.control_feature] = float(args.control_one_x)
    feature_max.setdefault(args.target_feature, max(feature_max.get(args.target_feature, 0.0), 1e-6))
    feature_max.setdefault(args.control_feature, max(feature_max.get(args.control_feature, 0.0), 1e-6))
    return feature_max


def make_prompt_args(args: argparse.Namespace, probe: dict[str, Any]) -> argparse.Namespace:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prefix = probe["prefix"]
    prompt_args.target = probe["target"]
    prompt_args.suffix = probe["suffix"]
    prompt_args.steer_part = "prefix"
    prompt_args.steer_token_index = None
    prompt_args.steer_offset = int(probe.get("anchor_offset", args.steer_offset))
    return prompt_args


def base_token_ids(tokenizer: Any) -> dict[str, int]:
    ids = {}
    for base in BASES:
        enc = token_ids(tokenizer, base)
        if len(enc) != 1:
            raise ValueError(f"Base {base} did not encode to exactly one token: {enc}")
        ids[base] = enc[0]
    return ids


def codon_stop_masses(logits: torch.Tensor, prompt: Any, base_ids: dict[str, int]) -> list[dict[str, Any]]:
    probs = F.softmax(logits.float(), dim=-1)
    n_codons = len(prompt.target_ids) // 3
    rows = []
    for codon_idx in range(n_codons):
        pos_probs = []
        native = ""
        for offset in range(3):
            target_i = codon_idx * 3 + offset
            pred_pos = prompt.target_start + target_i - 1
            raw = torch.tensor(
                [float(probs[0, pred_pos, base_ids[b]].detach().cpu()) for b in BASES],
                dtype=torch.float64,
            )
            norm = raw / raw.sum().clamp_min(1e-12)
            pos_probs.append({b: float(norm[i]) for i, b in enumerate(BASES)})
            native += prompt.target_tokens[target_i]

        stop_mass = 0.0
        codon_mass = {}
        for b1 in BASES:
            for b2 in BASES:
                for b3 in BASES:
                    codon = b1 + b2 + b3
                    mass = pos_probs[0][b1] * pos_probs[1][b2] * pos_probs[2][b3]
                    codon_mass[codon] = mass
                    if codon in STOP_CODONS:
                        stop_mass += mass
        rows.append(
            {
                "codon_index": codon_idx,
                "native_codon": native,
                "stop_prob": float(stop_mass),
                "sense_prob": float(1.0 - stop_mass),
                "native_is_stop": native in STOP_CODONS,
                "top_codon": max(codon_mass, key=codon_mass.get),
                "top_codon_prob": float(max(codon_mass.values())),
            }
        )
    return rows


def run_group(
    group: str,
    label: str,
    feature_id: int,
    probes: list[dict[str, Any]],
    feature_max: dict[int, float],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    sae: Any,
    base_ids: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    conditions: list[tuple[str, float | None]] = [("no_steer", None)]
    conditions.extend((f"{scale:g}x", float(scale)) for scale in parse_scales(args.scales))
    for condition, scale in conditions:
        by_codon: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            prompt_args = make_prompt_args(args, probe)
            prompt = build_prompt(tokenizer, probe["prefix"], probe["target"], probe["suffix"], prompt_args, args.device)
            if len(prompt.target_ids) % 3 != 0:
                raise ValueError("target length must be a multiple of 3 tokens")
            if scale is None:
                logits, _ = forward_with_steer(model, prompt, sae, prompt_args, None, None, feature_max)
            else:
                logits, _ = forward_with_steer(model, prompt, sae, prompt_args, [feature_id], scale, feature_max)
            for row in codon_stop_masses(logits, prompt, base_ids):
                by_codon[int(row["codon_index"])].append(row)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        for codon_idx, codon_rows in sorted(by_codon.items()):
            rows.append(
                {
                    "group": group,
                    "group_label": label,
                    "feature_id": feature_id,
                    "condition": condition,
                    "scale": "" if scale is None else scale,
                    "codon_index": codon_idx + 1,
                    "native_codons": ",".join(r["native_codon"] for r in codon_rows),
                    "mean_stop_prob": float(np.mean([r["stop_prob"] for r in codon_rows])),
                    "std_stop_prob": float(np.std([r["stop_prob"] for r in codon_rows])),
                    "mean_sense_prob": float(np.mean([r["sense_prob"] for r in codon_rows])),
                    "top_codons": ",".join(r["top_codon"] for r in codon_rows),
                }
            )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], args: argparse.Namespace, out_prefix: Path) -> None:
    colors = {
        "no_steer": "#b8bdc2",
        "0x": "#f05b78",
        "0.5x": "#d98b1d",
        "1x": "#35b779",
        "1.5x": "#43b7ad",
        "2x": "#32a6d6",
        "2.5x": "#9b8cf4",
    }
    conds = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]
    panels = [
        ("A", "a", f"P(stop codon) with steering CDS feature f/{args.target_feature}"),
        ("B", "b", f"P(stop codon) with steering matched control f/{args.control_feature}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.5), sharey=True)
    width = 0.105
    for ax, (group, letter, title) in zip(axes, panels):
        panel_rows = [r for r in rows if r["group"] == group]
        codons = sorted({int(r["codon_index"]) for r in panel_rows})
        x = np.arange(len(codons))
        offsets = (np.arange(len(conds)) - (len(conds) - 1) / 2.0) * width
        lookup = {(r["condition"], int(r["codon_index"])): float(r["mean_stop_prob"]) for r in panel_rows}
        for j, cond in enumerate(conds):
            vals = [lookup.get((cond, c), 0.0) for c in codons]
            ax.bar(
                x + offsets[j],
                vals,
                width=width * 0.92,
                color=colors[cond],
                edgecolor="white",
                linewidth=0.25,
                label="No steer" if cond == "no_steer" else f"Steer {cond}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Codon {c}" for c in codons], fontsize=10)
        ax.set_xlabel("Position in masked CDS span", fontsize=12)
        ax.set_ylabel("Stop-codon probability", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.24, 0.92), fontsize=9)
    fig.suptitle(args.title, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for group in ["A", "B"]:
        group_rows = [r for r in rows if r["group"] == group]
        by_cond = defaultdict(list)
        for row in group_rows:
            by_cond[row["condition"]].append(float(row["mean_stop_prob"]))
        base = np.mean(by_cond["no_steer"])
        summary[group] = {}
        for cond, vals in by_cond.items():
            mean = float(np.mean(vals))
            summary[group][cond] = {
                "mean_stop_prob": mean,
                "delta_vs_no_steer": mean - float(base),
            }
    return summary


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    probes, probe_one_x = load_probes(args.probe_json)
    feature_max = load_feature_max(args, probe_one_x)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    base_ids = base_token_ids(tokenizer)

    rows = []
    rows.extend(
        run_group(
            "A",
            f"CDS feature f/{args.target_feature}",
            args.target_feature,
            probes,
            feature_max,
            args,
            model,
            tokenizer,
            sae,
            base_ids,
        )
    )
    rows.extend(
        run_group(
            "B",
            f"matched control f/{args.control_feature}",
            args.control_feature,
            probes,
            feature_max,
            args,
            model,
            tokenizer,
            sae,
            base_ids,
        )
    )
    summary = summarize(rows)
    payload = {
        "metric": "teacher-forced pseudo-probability mass of in-frame stop codons",
        "target_feature": args.target_feature,
        "control_feature": args.control_feature,
        "feature_one_x_values": {str(k): float(v) for k, v in feature_max.items()},
        "summary": summary,
        "rows": rows,
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot(rows, args, out_prefix)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
