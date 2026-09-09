#!/usr/bin/env python3
"""Teacher-forced GLM span-likelihood gate for structure-switch features.

This is a lower-variance structure-switch readout than sampled span
generation. For each RNAfold switch case, it masks a local span around the
switch mutation, scores the target-state span and the opposite/context-state
span under identical steering, and reports whether feature clamping improves
the target-vs-opposite likelihood margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

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
from score_rnafold_structure_switch_generation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SAE,
    choose_span,
    direction_spec,
    feature_rows_for_direction,
    pair_positions_0,
    select_anchor,
)
from scan_rnafold_structure_switch_direct_effects import (  # noqa: E402
    load_feature_scan,
    parse_directions,
)


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
    parser.add_argument("--case-indexes", default="", help="Comma-separated original case indexes")
    parser.add_argument("--directions", default="mutant_state")
    parser.add_argument("--feature-ids", default="", help="Comma-separated feature IDs to test")
    parser.add_argument("--top-features-per-direction", type=int, default=4)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5,10")
    parser.add_argument("--span-flank", type=int, default=3)
    parser.add_argument("--max-span-len", type=int, default=12)
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")
    parser.add_argument("--min-anchor-activation", type=float, default=0.01)

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
        scale_source="feature_max_json",
        clamp_value=1.0,
        feature_max_json="",
    )
    return parser.parse_args()


def set_prompt_args(args: argparse.Namespace, prefix: str, suffix: str, span_start: int, span_end: int, anchor_pos0: int) -> None:
    args.prefix = prefix
    args.suffix = suffix
    if anchor_pos0 < span_start:
        args.steer_part = "prefix"
        args.steer_offset = anchor_pos0
    elif anchor_pos0 >= span_end:
        args.steer_part = "suffix"
        args.steer_offset = anchor_pos0 - span_end
    else:
        raise ValueError("Anchor position cannot be inside the masked span")
    args.steer_token_index = None


def score_span(
    model: Any,
    tokenizer: Any,
    sae: Any,
    args: argparse.Namespace,
    prefix: str,
    target_span: str,
    suffix: str,
    feature_id: int | None,
    scale: float | None,
    feature_max: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_prompt(tokenizer, prefix, target_span, suffix, args, args.device)
    logits, hook_info = forward_with_steer(
        model,
        prompt,
        sae,
        args,
        feature_ids=None if feature_id is None else [feature_id],
        scale=scale,
        feature_max=feature_max,
    )
    return score_target(logits, prompt), hook_info


def mutation_token_score(score: dict[str, Any], span_start: int, mutation_pos0: int) -> dict[str, Any]:
    idx = mutation_pos0 - span_start
    if idx < 0 or idx >= len(score["target_tokens"]):
        return {"prob": "", "log_prob": "", "token": ""}
    row = score["target_tokens"][idx]
    return {"prob": row["prob"], "log_prob": row["log_prob"], "token": row["token"]}


def run_feature_direction(
    model: Any,
    tokenizer: Any,
    sae: Any,
    entry: dict[str, Any],
    direction: str,
    feature_row: dict[str, Any],
    args: argparse.Namespace,
    scales: list[float],
) -> list[dict[str, Any]]:
    case = entry["case"]
    spec = direction_spec(case, direction)
    span_start, span_end = choose_span(case, args)
    span_len = span_end - span_start
    feature_id = int(feature_row["feature_id"])
    target_positions0 = pair_positions_0(spec["target_pairs_1"])
    anchor_pos0, one_x = select_anchor(
        model,
        tokenizer,
        sae,
        spec["target_sequence"],
        target_positions0,
        span_start,
        span_end,
        feature_id,
        args,
    )
    if one_x < args.min_anchor_activation:
        return []

    prefix = spec["context_sequence"][:span_start]
    suffix = spec["context_sequence"][span_end:]
    target_span = spec["target_sequence"][span_start:span_end]
    opposite_span = spec["context_sequence"][span_start:span_end]
    set_prompt_args(args, prefix, suffix, span_start, span_end, anchor_pos0)

    rows = []
    feature_max = {feature_id: one_x}
    conditions: list[tuple[str, float | None, int | None]] = [("no_steer", None, None)]
    conditions.extend((f"{scale:g}x", scale, feature_id) for scale in scales)

    baseline_margin = None
    baseline_target_logp = None
    baseline_opposite_logp = None
    mutation_pos0 = int(case["mutation_pos0"])
    for condition, scale, steer_fid in conditions:
        target_score, hook_info = score_span(
            model,
            tokenizer,
            sae,
            args,
            prefix,
            target_span,
            suffix,
            steer_fid,
            scale,
            feature_max,
        )
        opposite_score, _ = score_span(
            model,
            tokenizer,
            sae,
            args,
            prefix,
            opposite_span,
            suffix,
            steer_fid,
            scale,
            feature_max,
        )
        target_logp = float(target_score["target_token_logp"])
        opposite_logp = float(opposite_score["target_token_logp"])
        margin = target_logp - opposite_logp
        if condition == "no_steer":
            baseline_margin = margin
            baseline_target_logp = target_logp
            baseline_opposite_logp = opposite_logp
        target_mut = mutation_token_score(target_score, span_start, mutation_pos0)
        opposite_mut = mutation_token_score(opposite_score, span_start, mutation_pos0)
        rows.append(
            {
                "case_index": int(entry["case_index"]),
                "record_id": case["record_id"],
                "direction": direction,
                "target_state": spec["target_state"],
                "feature_id": feature_id,
                "feature_state_margin": float(feature_row.get("state_margin", 0.0)),
                "feature_positive_mean": float(feature_row.get("positive_mean", 0.0)),
                "condition": condition,
                "scale": "" if scale is None else float(scale),
                "span_start0": span_start,
                "span_end0": span_end,
                "span_len": span_len,
                "anchor_pos1": anchor_pos0 + 1,
                "anchor_one_x": one_x,
                "mutation_pos1": int(case["mutation_pos1"]),
                "mutation_from": case["mutation_from"],
                "mutation_to": case["mutation_to"],
                "context_span": opposite_span,
                "target_span": target_span,
                "target_logp": target_logp,
                "opposite_logp": opposite_logp,
                "target_vs_opposite_margin": margin,
                "target_logp_per_token": target_logp / max(span_len, 1),
                "opposite_logp_per_token": opposite_logp / max(span_len, 1),
                "margin_per_token": margin / max(span_len, 1),
                "delta_margin": "" if baseline_margin is None else margin - baseline_margin,
                "delta_target_logp": "" if baseline_target_logp is None else target_logp - baseline_target_logp,
                "delta_opposite_logp": "" if baseline_opposite_logp is None else opposite_logp - baseline_opposite_logp,
                "target_mutation_token": target_mut["token"],
                "target_mutation_prob": target_mut["prob"],
                "target_mutation_logp": target_mut["log_prob"],
                "opposite_mutation_token": opposite_mut["token"],
                "opposite_mutation_prob": opposite_mut["prob"],
                "opposite_mutation_logp": opposite_mut["log_prob"],
                "hook_info": json.dumps(hook_info),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["case_index"], row["record_id"], row["direction"], row["feature_id"])
        cur = by_feature.get(key)
        delta = row["delta_margin"]
        delta_f = float(delta) if delta != "" else 0.0
        if cur is None or delta_f > float(cur["best_delta_margin"]):
            by_feature[key] = {
                "case_index": row["case_index"],
                "record_id": row["record_id"],
                "direction": row["direction"],
                "target_state": row["target_state"],
                "feature_id": row["feature_id"],
                "feature_state_margin": row["feature_state_margin"],
                "feature_positive_mean": row["feature_positive_mean"],
                "best_condition": row["condition"],
                "best_scale": row["scale"],
                "best_delta_margin": delta_f,
                "best_margin": row["target_vs_opposite_margin"],
                "best_delta_target_logp": row["delta_target_logp"],
                "best_target_mutation_prob": row["target_mutation_prob"],
                "anchor_one_x": row["anchor_one_x"],
                "anchor_pos1": row["anchor_pos1"],
                "span_start0": row["span_start0"],
                "span_end0": row["span_end0"],
                "target_span": row["target_span"],
                "context_span": row["context_span"],
            }
    out = list(by_feature.values())
    out.sort(key=lambda row: float(row["best_delta_margin"]), reverse=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)

    entries = load_feature_scan(Path(args.feature_scan_json), args)
    directions = parse_directions(args.directions)
    scales = parse_scales(args.scales)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    rows = []
    for entry in entries:
        for direction in directions:
            for feature_row in feature_rows_for_direction(entry, direction, args):
                try:
                    rows.extend(run_feature_direction(model, tokenizer, sae, entry, direction, feature_row, args, scales))
                except ValueError as exc:
                    print(
                        f"Skipping case={entry['case_index']} direction={direction} "
                        f"feature={feature_row.get('feature_id')}: {exc}",
                        file=sys.stderr,
                    )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump({"rows": rows, "args": vars(args)}, handle, indent=2)
    write_csv(out_prefix.with_name(out_prefix.name + "_rows").with_suffix(".csv"), rows)
    summary = summarize(rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary").with_suffix(".csv"), summary)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_rows').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary').with_suffix('.csv')}")
    if summary:
        best = summary[0]
        print(
            "Best span-likelihood effect: "
            f"case={best['case_index']} f/{best['feature_id']} {best['best_condition']} "
            f"delta_margin={float(best['best_delta_margin']):.4f} "
            f"margin={float(best['best_margin']):.4f}"
        )


if __name__ == "__main__":
    main()
