#!/usr/bin/env python3
"""Directly scan RNAfold case features for steering effects.

The upstream feature scan ranks features by WT/rescue-vs-break activation
separation. That is only correlational. This script tests the next gate:
does clamping the feature at the break-side anchor actually change
teacher-forced P(rescue_base) at the masked paired site?
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
    load_model,
    load_sae,
    parse_scales,
    score_target,
)
from steer_rnafold_case_fig6 import build_break_to_rescue_probe  # noqa: E402


DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-scan-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--max-features-per-case", type=int, default=20)
    parser.add_argument("--case-indexes", default="")
    parser.add_argument("--feature-ranks", default="")
    parser.add_argument("--scales", default="0.5,1,1.5,2,2.5,5")
    parser.add_argument(
        "--scale-source",
        choices=["feature_scan", "constant", "current"],
        default="feature_scan",
        help="Use one_x_observed from the feature scan, a constant, or current activation.",
    )
    parser.add_argument("--clamp-value", type=float, default=20.0)
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")

    # Filled per case before calling shared helpers.
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
        feature_max_json="",
    )
    return parser.parse_args()


def parse_int_set(raw: str) -> set[int] | None:
    vals = {int(x.strip()) for x in raw.split(",") if x.strip()}
    return vals or None


def load_feature_scan(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    case_filter = parse_int_set(args.case_indexes)
    rank_filter = parse_int_set(args.feature_ranks)

    selected: list[dict[str, Any]] = []
    for entry in payload:
        case_index = int(entry["case_index"])
        if case_filter is not None and case_index not in case_filter:
            continue
        features = []
        for feature in entry.get("top_features", []):
            rank = int(feature.get("feature_rank", len(features) + 1))
            if rank_filter is not None and rank not in rank_filter:
                continue
            features.append(feature)
            if len(features) >= args.max_features_per_case:
                break
        if features:
            selected.append({"case_index": case_index, "case": entry["case"], "features": features})
        if case_filter is None and len(selected) >= args.max_cases:
            break
    return selected


def feature_max_for(feature: dict[str, Any], scale_source: str) -> dict[int, float]:
    fid = int(feature["feature_id"])
    if scale_source == "feature_scan":
        return {fid: float(feature["one_x_observed"])}
    return {}


def first_target_prob(logits: torch.Tensor, prompt: Any) -> dict[str, Any]:
    score = score_target(logits, prompt)
    tok = score["target_tokens"][0]
    return {
        "token": tok["token"],
        "prob": float(tok["prob"]),
        "log_prob": float(tok["log_prob"]),
        "target_token_logp": float(score["target_token_logp"]),
    }


def scan_feature(
    model: Any,
    prompt: Any,
    sae: Any,
    args: argparse.Namespace,
    feature: dict[str, Any],
    baseline: dict[str, Any],
    scales: list[float],
) -> dict[str, Any]:
    fid = int(feature["feature_id"])
    original_scale_source = args.scale_source
    feature_max = feature_max_for(feature, original_scale_source)
    if original_scale_source == "feature_scan":
        args.scale_source = "feature_max_json"
    points = []
    for scale in scales:
        logits, hook_info = forward_with_steer(
            model,
            prompt,
            sae,
            args,
            feature_ids=[fid],
            scale=scale,
            feature_max=feature_max,
        )
        scored = first_target_prob(logits, prompt)
        points.append(
            {
                "scale": float(scale),
                "prob": scored["prob"],
                "log_prob": scored["log_prob"],
                "delta_prob": scored["prob"] - baseline["prob"],
                "delta_log_prob": scored["log_prob"] - baseline["log_prob"],
                "hidden_diff": hook_info.get("hidden_diff", ""),
                "feature_info": hook_info.get("features", []),
            }
        )
    args.scale_source = original_scale_source

    best = max(points, key=lambda row: row["delta_prob"])
    worst = min(points, key=lambda row: row["delta_prob"])
    positive_steps = sum(1 for row in points if row["delta_prob"] > 0)
    return {
        "feature_id": fid,
        "feature_rank": int(feature.get("feature_rank", -1)),
        "combined_score": float(feature.get("combined_score", 0.0)),
        "pair_score": float(feature.get("pair_score", 0.0)),
        "window_score": float(feature.get("window_score", 0.0)),
        "one_x_observed": float(feature.get("one_x_observed", 0.0)),
        "baseline_prob": baseline["prob"],
        "baseline_log_prob": baseline["log_prob"],
        "best_scale": best["scale"],
        "best_prob": best["prob"],
        "best_delta_prob": best["delta_prob"],
        "best_delta_log_prob": best["delta_log_prob"],
        "worst_delta_prob": worst["delta_prob"],
        "positive_steps": positive_steps,
        "n_scales": len(points),
        "points": points,
    }


def flatten(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in payload:
        case = entry["case"]
        for result in entry["feature_results"]:
            rows.append(
                {
                    "case_index": entry["case_index"],
                    "record_id": case["record_id"],
                    "i1": case["i1"],
                    "j1": case["j1"],
                    "break_pos1": case["break_pos1"],
                    "break_from": case["break_from"],
                    "break_to": case["break_to"],
                    "rescue_pos1": case["rescue_pos1"],
                    "rescue_from": case["rescue_from"],
                    "rescue_to": case["rescue_to"],
                    "local_break_bp_distance": case["local_break_bp_distance"],
                    "feature_id": result["feature_id"],
                    "feature_rank": result["feature_rank"],
                    "combined_score": result["combined_score"],
                    "pair_score": result["pair_score"],
                    "window_score": result["window_score"],
                    "one_x_observed": result["one_x_observed"],
                    "baseline_prob": result["baseline_prob"],
                    "best_scale": result["best_scale"],
                    "best_prob": result["best_prob"],
                    "best_delta_prob": result["best_delta_prob"],
                    "best_delta_log_prob": result["best_delta_log_prob"],
                    "worst_delta_prob": result["worst_delta_prob"],
                    "positive_steps": result["positive_steps"],
                    "n_scales": result["n_scales"],
                }
            )
    rows.sort(key=lambda row: row["best_delta_prob"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_top(path_prefix: Path, payload: list[dict[str, Any]], top_n: int = 12) -> None:
    rows = flatten(payload)[:top_n]
    if not rows:
        return
    labels = [f"c{r['case_index']} f/{r['feature_id']}" for r in rows]
    values = [r["best_delta_prob"] for r in rows]
    colors = ["#3b7a78" if v >= 0 else "#b35c44" for v in values]
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.32 * len(rows) + 1.2)))
    y = list(range(len(rows)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Best delta P(rescue base) across scales")
    ax.set_title("Direct RNAfold steering-effect scan")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

    selected = load_feature_scan(Path(args.feature_scan_json), args)
    if not selected:
        raise RuntimeError("No case/features selected from feature scan")

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    scales = parse_scales(args.scales)

    payload = []
    for entry in selected:
        case = entry["case"]
        probe = build_break_to_rescue_probe(case)
        args.prefix = probe["prefix"]
        args.target = probe["target"]
        args.suffix = probe["suffix"]
        args.steer_part = probe["steer_part"]
        args.steer_offset = probe["steer_offset"]
        args.steer_token_index = None
        prompt = build_prompt(tokenizer, args.prefix, args.target, args.suffix, args, args.device)
        logits, _hook_info = forward_with_steer(
            model, prompt, sae, args, feature_ids=None, scale=None, feature_max={}
        )
        baseline = first_target_prob(logits, prompt)

        feature_results = []
        for feature in entry["features"]:
            feature_results.append(scan_feature(model, prompt, sae, args, feature, baseline, scales))
        payload.append(
            {
                "case_index": entry["case_index"],
                "case": case,
                "probe": {
                    "prompt_target": args.target,
                    "steer_part": args.steer_part,
                    "steer_offset": args.steer_offset,
                    "steer_token_index": prompt.steer_token_index,
                    "target_start": prompt.target_start,
                    "baseline": baseline,
                },
                "feature_results": feature_results,
            }
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    flat = flatten(payload)
    write_csv(out_prefix.with_suffix(".csv"), flat)
    plot_top(out_prefix, payload)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    if flat:
        top = flat[0]
        print(
            "Top direct effect: "
            f"case={top['case_index']} record={top['record_id']} "
            f"f/{top['feature_id']} delta={top['best_delta_prob']:.6f} "
            f"baseline={top['baseline_prob']:.6f} best={top['best_prob']:.6f} "
            f"scale={top['best_scale']}"
        )


if __name__ == "__main__":
    main()
