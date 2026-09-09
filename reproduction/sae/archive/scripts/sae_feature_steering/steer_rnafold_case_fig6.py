#!/usr/bin/env python3
"""InterPLM Fig. 6-style steering for RNAfold WT/break/rescue cases.

For a selected case, we use the break mutant as context, mask the rescue
position, steer a feature at the already-mutated anchor side, and measure
teacher-forced P(rescue_base) at the masked position.
"""

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
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    forward_with_steer,
    load_feature_max,
    load_model,
    load_sae,
    parse_scales,
    score_target,
)


DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--target-feature", type=int, required=True)
    parser.add_argument("--control-feature", type=int, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5")
    parser.add_argument(
        "--scale-source",
        choices=["current", "constant", "feature_max_json"],
        default="constant",
    )
    parser.add_argument("--clamp-value", type=float, default=20.0)
    parser.add_argument("--feature-max-json", default="")
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--title", default="RNAfold case SAE steering")

    # Filled internally before calling shared helpers.
    parser.set_defaults(
        prefix="",
        target="",
        suffix="",
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
        steer_part="absolute",
        steer_offset=0,
        steer_token_index=None,
        group_mode="joint",
    )
    return parser.parse_args()


def load_case(path: Path, case_index: int) -> dict[str, Any]:
    with path.open() as handle:
        cases = json.load(handle)
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"case index {case_index} outside 0..{len(cases) - 1}")
    return cases[case_index]


def build_break_to_rescue_probe(case: dict[str, Any]) -> dict[str, Any]:
    break_seq = case["break_sequence"]
    rescue_pos = int(case["rescue_pos0"])
    break_pos = int(case["break_pos0"])
    target = case["rescue_to"]
    prefix = break_seq[:rescue_pos]
    suffix = break_seq[rescue_pos + 1 :]
    if break_pos < rescue_pos:
        steer_part = "prefix"
        steer_offset = break_pos
    elif break_pos > rescue_pos:
        steer_part = "suffix"
        steer_offset = break_pos - rescue_pos - 1
    else:
        raise ValueError("break and rescue positions are identical")
    return {
        "prefix": prefix,
        "target": target,
        "suffix": suffix,
        "steer_part": steer_part,
        "steer_offset": steer_offset,
        "break_seq": break_seq,
    }


def run_group(
    group: str,
    label: str,
    feature_id: int,
    model: Any,
    prompt: Any,
    sae: Any,
    args: argparse.Namespace,
    scales: list[float],
    feature_max: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    logits, hook_info = forward_with_steer(
        model, prompt, sae, args, feature_ids=None, scale=None, feature_max=feature_max
    )
    rows.append(
        {
            "group": group,
            "label": label,
            "feature_id": feature_id,
            "condition": "no_steer",
            "scale": None,
            "hook_info": hook_info,
            "score": score_target(logits, prompt),
        }
    )
    for scale in scales:
        logits, hook_info = forward_with_steer(
            model,
            prompt,
            sae,
            args,
            feature_ids=[feature_id],
            scale=scale,
            feature_max=feature_max,
        )
        rows.append(
            {
                "group": group,
                "label": label,
                "feature_id": feature_id,
                "condition": f"{scale:g}x",
                "scale": float(scale),
                "hook_info": hook_info,
                "score": score_target(logits, prompt),
            }
        )
    return rows


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tok = row["score"]["target_tokens"][0]
        out.append(
            {
                "group": row["group"],
                "label": row["label"],
                "feature_id": row["feature_id"],
                "condition": row["condition"],
                "scale": "" if row["scale"] is None else row["scale"],
                "token": tok["token"],
                "prob": tok["prob"],
                "log_prob": tok["log_prob"],
                "target_token_logp": row["score"]["target_token_logp"],
                "hidden_diff": row["hook_info"].get("hidden_diff", ""),
            }
        )
    return out


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = flatten(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def plot(path_prefix: Path, rows: list[dict[str, Any]], title: str) -> None:
    flat = flatten(rows)
    groups = sorted({row["group"] for row in flat})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for group in groups:
        subset = [row for row in flat if row["group"] == group]
        x = list(range(len(subset)))
        y = [row["prob"] for row in subset]
        label = subset[0]["label"]
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([row["condition"] for row in subset], rotation=35, ha="right")
    ax.set_ylabel("Teacher-forced P(rescue base)")
    ax.set_xlabel("Steering condition")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
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
    probe = build_break_to_rescue_probe(case)
    args.prefix = probe["prefix"]
    args.target = probe["target"]
    args.suffix = probe["suffix"]
    args.steer_part = probe["steer_part"]
    args.steer_offset = probe["steer_offset"]

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    prompt = build_prompt(tokenizer, args.prefix, args.target, args.suffix, args, args.device)
    scales = parse_scales(args.scales)
    feature_max = load_feature_max(args)

    rows = []
    rows.extend(
        run_group(
            "target",
            f"target f/{args.target_feature}",
            args.target_feature,
            model,
            prompt,
            sae,
            args,
            scales,
            feature_max,
        )
    )
    rows.extend(
        run_group(
            "control",
            f"control f/{args.control_feature}",
            args.control_feature,
            model,
            prompt,
            sae,
            args,
            scales,
            feature_max,
        )
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "probe": {
            **probe,
            "prompt_target": args.target,
            "steer_token_index": prompt.steer_token_index,
            "target_start": prompt.target_start,
        },
        "target_feature": args.target_feature,
        "control_feature": args.control_feature,
        "patch_mode": args.patch_mode,
        "scale_source": args.scale_source,
        "clamp_value": args.clamp_value,
        "feature_max": {str(k): float(v) for k, v in feature_max.items()},
        "rows": rows,
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot(out_prefix, rows, args.title)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
