#!/usr/bin/env python3
"""InterPLM Fig. 6-style bar plot for the Domingo case-3 stem rescue."""

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
DEFAULT_OUT = FIG_DIR / "domingo_case3_interplm_style_span_bars"


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
    parser.add_argument("--span-flank", type=int, default=3)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    parser.add_argument("--target-feature", type=int, default=5271)
    parser.add_argument("--control-feature", type=int, default=5313)
    parser.add_argument("--target-one-x", type=float, default=3.9850058555603027)
    parser.add_argument("--control-one-x", type=float, default=3.9198832511901855)
    parser.add_argument("--out-prefix", default=str(DEFAULT_OUT))
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")

    # Fields expected by the shared steering helper.
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


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_window_probe(case: dict[str, Any], flank: int) -> dict[str, Any]:
    rescue_pos = int(case["rescue_pos0"])
    window_start = max(0, rescue_pos - flank)
    window_end = min(len(case["break_sequence"]), rescue_pos + flank + 1)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "positions0": list(range(window_start, window_end)),
    }


def configure_single_position_prompt(args: argparse.Namespace, case: dict[str, Any], pos0: int) -> tuple[str, str, str]:
    prefix = case["break_sequence"][:pos0]
    target = case["rescue_sequence"][pos0]
    suffix = case["break_sequence"][pos0 + 1 :]
    break_pos = int(case["break_pos0"])
    if break_pos < pos0:
        args.steer_part = "prefix"
        args.steer_offset = break_pos
    elif break_pos > pos0:
        args.steer_part = "suffix"
        args.steer_offset = break_pos - pos0 - 1
    else:
        raise ValueError("Cannot read the steering anchor position itself with the same anchor patch.")
    return prefix, target, suffix


def score_group(
    group: str,
    label: str,
    feature_id: int,
    model: Any,
    tokenizer: Any,
    sae: Any,
    case: dict[str, Any],
    probe: dict[str, Any],
    args: argparse.Namespace,
    feature_max: dict[int, float],
    scales: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    conditions: list[tuple[str, float | None]] = [("No steer", None)]
    conditions.extend((f"Steer {scale:g}x", scale) for scale in scales)

    for pos0 in probe["positions0"]:
        prefix, target, suffix = configure_single_position_prompt(args, case, int(pos0))
        prompt = build_prompt(tokenizer, prefix, target, suffix, args, args.device)
        for condition, scale in conditions:
            if scale is None:
                logits, hook_info = forward_with_steer(model, prompt, sae, args, None, None, feature_max)
            else:
                logits, hook_info = forward_with_steer(model, prompt, sae, args, [feature_id], scale, feature_max)
            score = score_target(logits, prompt)
            tok = next(tok for tok in score["target_tokens"] if not tok["is_eos_span"])
            rows.append(
                {
                    "group": group,
                    "group_label": label,
                    "feature_id": feature_id,
                    "condition": condition,
                    "scale": "" if scale is None else scale,
                    "seq_pos1": int(pos0) + 1,
                    "reference_base": case["rescue_sequence"][pos0],
                    "break_base": case["break_sequence"][pos0],
                    "prob": float(tok["prob"]),
                    "log_prob": float(tok["log_prob"]),
                    "hidden_diff": hook_info.get("hidden_diff", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path_prefix: Path, rows: list[dict[str, Any]], case: dict[str, Any], probe: dict[str, Any]) -> None:
    colors = {
        "No steer": "#b7b7b7",
        "Steer 0x": "#ff7d9a",
        "Steer 0.5x": "#d9932f",
        "Steer 1x": "#2da35a",
        "Steer 1.5x": "#2db7a3",
        "Steer 2x": "#3aa4cf",
        "Steer 2.5x": "#9b87f5",
    }
    groups = [
        ("target", "p(reference base) with steering f/5271"),
        ("control", "p(reference base) with steering matched control f/5313"),
    ]
    conditions = list(colors.keys())
    positions = sorted({int(row["seq_pos1"]) for row in rows})
    x_labels = []
    for pos in positions:
        base = case["rescue_sequence"][pos - 1]
        if pos == int(case["rescue_pos1"]):
            x_labels.append(f"{base}\n{pos}\n(mask)")
        else:
            x_labels.append(f"{base}\n{pos}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=False)
    bar_width = 0.105
    x = np.arange(len(positions))
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * bar_width

    for ax, (group, title) in zip(axes, groups):
        sub = [row for row in rows if row["group"] == group]
        lookup = {
            (int(row["seq_pos1"]), row["condition"]): float(row["prob"])
            for row in sub
        }
        for i, condition in enumerate(conditions):
            vals = [lookup[(pos, condition)] for pos in positions]
            ax.bar(x + offsets[i], vals, width=bar_width, color=colors[condition], label=condition, edgecolor="none")
        ax.set_title(title, fontsize=9.6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.2)
        ax.set_xlabel("Position in rescue-side span", fontsize=8.8)
        ax.set_ylim(0, max(0.5, max(float(row["prob"]) for row in sub) * 1.14))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    axes[0].set_ylabel("Probability (reference base)", fontsize=9)
    axes[0].text(-0.12, 1.06, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.12, 1.06, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.8, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=1.0)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.84, wspace=0.28)
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
    probe = build_window_probe(case, args.span_flank)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    scales = parse_scales(args.scales)
    feature_max = {
        int(args.target_feature): float(args.target_one_x),
        int(args.control_feature): float(args.control_one_x),
    }
    rows: list[dict[str, Any]] = []
    rows.extend(
        score_group(
            "target",
            "target f/5271",
            args.target_feature,
            model,
            tokenizer,
            sae,
            case,
            probe,
            args,
            feature_max,
            scales,
        )
    )
    rows.extend(
        score_group(
            "control",
            "matched active control f/5313",
            args.control_feature,
            model,
            tokenizer,
            sae,
            case,
            probe,
            args,
            feature_max,
            scales,
        )
    )
    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_suffix(".csv"), rows)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump({"case": case, "probe": probe, "rows": rows}, handle, indent=2)
    plot(out_prefix, rows, case, probe)
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
