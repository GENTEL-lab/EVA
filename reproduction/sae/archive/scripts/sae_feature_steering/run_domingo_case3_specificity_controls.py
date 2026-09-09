#!/usr/bin/env python3
"""Specificity controls for the Domingo case-3 f/5271 stem-rescue hit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    forward_with_steer,
    load_model,
    load_sae,
    score_target,
)


ROOT = Path("EVA1/data/sae_feature_steering/rnafold_cases")
FIG_DIR = Path("EVA1/data/sae_feature_steering/figures")
DEFAULT_CASES = ROOT / "domingo2018_trna_cases.json"
DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"
DEFAULT_OUT = FIG_DIR / "domingo_case3_f5271_specificity_controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-json", default=str(DEFAULT_CASES))
    parser.add_argument("--case-index", type=int, default=3)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--feature-id", type=int, default=5271)
    parser.add_argument("--one-x", type=float, default=3.9850058555603027)
    parser.add_argument("--base-scales", default="0,0.5,1,1.5,2,2.5")
    parser.add_argument("--anchor-scale", type=float, default=2.5)
    parser.add_argument(
        "--anchor-positions1",
        default="1,3,5,36,66,68,70",
        help="Comma-separated 1-based sequence positions for anchor specificity.",
    )
    parser.add_argument("--out-prefix", default=str(DEFAULT_OUT))
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")

    parser.set_defaults(
        prefix="",
        target="",
        suffix="",
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
        steer_part="prefix",
        steer_offset=0,
        steer_token_index=None,
        group_mode="joint",
        scale_source="feature_max_json",
        clamp_value=20.0,
        feature_max_json="",
    )
    return parser.parse_args()


def load_case(path: Path, case_index: int) -> dict[str, Any]:
    with path.open() as handle:
        cases = json.load(handle)
    return cases[case_index]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def configure_anchor(args: argparse.Namespace, mask_pos0: int, anchor_pos0: int) -> None:
    if anchor_pos0 < mask_pos0:
        args.steer_part = "prefix"
        args.steer_offset = anchor_pos0
    elif anchor_pos0 > mask_pos0:
        args.steer_part = "suffix"
        args.steer_offset = anchor_pos0 - mask_pos0 - 1
    else:
        raise ValueError("anchor position cannot be the masked target position")


def score_single_base(
    model: Any,
    tokenizer: Any,
    sae: Any,
    args: argparse.Namespace,
    case: dict[str, Any],
    target_base: str,
    anchor_pos0: int,
    scale: float | None,
    feature_max: dict[int, float],
) -> dict[str, Any]:
    rescue_pos0 = int(case["rescue_pos0"])
    configure_anchor(args, rescue_pos0, anchor_pos0)
    prefix = case["break_sequence"][:rescue_pos0]
    suffix = case["break_sequence"][rescue_pos0 + 1 :]
    prompt = build_prompt(tokenizer, prefix, target_base, suffix, args, args.device)
    if scale is None:
        logits, hook_info = forward_with_steer(model, prompt, sae, args, None, None, feature_max)
        condition = "No steer"
    else:
        logits, hook_info = forward_with_steer(model, prompt, sae, args, [args.feature_id], scale, feature_max)
        condition = f"Steer {scale:g}x"
    score = score_target(logits, prompt)
    tok = next(tok for tok in score["target_tokens"] if not tok["is_eos_span"])
    return {
        "condition": condition,
        "scale": "" if scale is None else scale,
        "target_base": target_base,
        "anchor_pos1": anchor_pos0 + 1,
        "anchor_base": case["break_sequence"][anchor_pos0],
        "mask_pos1": rescue_pos0 + 1,
        "prob": float(tok["prob"]),
        "log_prob": float(tok["log_prob"]),
        "hidden_diff": hook_info.get("hidden_diff", ""),
    }


def run_base_specificity(
    model: Any,
    tokenizer: Any,
    sae: Any,
    args: argparse.Namespace,
    case: dict[str, Any],
    feature_max: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    anchor_pos0 = int(case["break_pos0"])
    for base in ["A", "C", "G", "U"]:
        rows.append(score_single_base(model, tokenizer, sae, args, case, base, anchor_pos0, None, feature_max))
        for scale in parse_float_list(args.base_scales):
            rows.append(score_single_base(model, tokenizer, sae, args, case, base, anchor_pos0, scale, feature_max))
    return rows


def run_anchor_specificity(
    model: Any,
    tokenizer: Any,
    sae: Any,
    args: argparse.Namespace,
    case: dict[str, Any],
    feature_max: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    target_base = case["rescue_to"]
    for pos1 in parse_int_list(args.anchor_positions1):
        pos0 = pos1 - 1
        if pos0 == int(case["rescue_pos0"]):
            continue
        rows.append(score_single_base(model, tokenizer, sae, args, case, target_base, pos0, None, feature_max))
        rows.append(score_single_base(model, tokenizer, sae, args, case, target_base, pos0, args.anchor_scale, feature_max))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path_prefix: Path, base_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], case: dict[str, Any]) -> None:
    colors = {
        "No steer": "#b7b7b7",
        "Steer 0x": "#ff7d9a",
        "Steer 0.5x": "#d9932f",
        "Steer 1x": "#2da35a",
        "Steer 1.5x": "#2db7a3",
        "Steer 2x": "#3aa4cf",
        "Steer 2.5x": "#9b87f5",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))

    bases = ["A", "C", "G", "U"]
    conditions = list(colors.keys())
    x = np.arange(len(bases))
    width = 0.105
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * width
    base_lookup = {
        (row["target_base"], row["condition"]): float(row["prob"])
        for row in base_rows
    }
    for i, cond in enumerate(conditions):
        vals = [base_lookup[(base, cond)] for base in bases]
        axes[0].bar(x + offsets[i], vals, width=width, color=colors[cond], label=cond)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bases)
    axes[0].set_ylim(0, 0.5)
    axes[0].set_ylabel("Probability at rescue position 69")
    axes[0].set_xlabel("Candidate base")
    axes[0].set_title("Base specificity, anchor 3A, f/5271", fontsize=9.5)
    axes[0].text(-0.14, 1.05, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold")

    anchor_positions = sorted({int(row["anchor_pos1"]) for row in anchor_rows})
    anchor_labels = []
    deltas = []
    steered = []
    baseline = []
    for pos in anchor_positions:
        no = next(float(row["prob"]) for row in anchor_rows if int(row["anchor_pos1"]) == pos and row["condition"] == "No steer")
        st = next(float(row["prob"]) for row in anchor_rows if int(row["anchor_pos1"]) == pos and row["condition"] != "No steer")
        anchor_base = next(row["anchor_base"] for row in anchor_rows if int(row["anchor_pos1"]) == pos)
        anchor_labels.append(f"{pos}{anchor_base}")
        baseline.append(no)
        steered.append(st)
        deltas.append(st - no)
    x2 = np.arange(len(anchor_positions))
    axes[1].bar(x2 - 0.16, baseline, width=0.32, color="#b7b7b7", label="No steer")
    axes[1].bar(x2 + 0.16, steered, width=0.32, color="#9b87f5", label=f"Steer {2.5:g}x")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(anchor_labels)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_ylabel(f"P({case['rescue_to']}) at rescue position 69")
    axes[1].set_xlabel("Steered anchor position")
    axes[1].set_title("Anchor specificity, f/5271 at 2.5x", fontsize=9.5)
    axes[1].text(-0.14, 1.05, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper left", bbox_to_anchor=(1.0, 1.03), handlelength=1.0)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.84, wspace=0.42)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)
    case = load_case(Path(args.cases_json), args.case_index)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    feature_max = {int(args.feature_id): float(args.one_x)}
    base_rows = run_base_specificity(model, tokenizer, sae, args, case, feature_max)
    anchor_rows = run_anchor_specificity(model, tokenizer, sae, args, case, feature_max)
    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_name(out_prefix.name + "_base_specificity").with_suffix(".csv"), base_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_anchor_specificity").with_suffix(".csv"), anchor_rows)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump({"case": case, "base_specificity": base_rows, "anchor_specificity": anchor_rows}, handle, indent=2)
    plot(out_prefix, base_rows, anchor_rows, case)
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_base_specificity').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_anchor_specificity').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
