#!/usr/bin/env python3
"""Direct GLM mutation-base scan for RNAfold structure-switch features.

This is a narrow causal gate before full generated-span fold scoring. For a
structure-switch case, it masks the mutation base, clamps selected SAE features
at a diagnostic anchor position, and measures whether the target structure
state's base becomes more likely.
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
from scan_rnafold_structure_switch_features import (  # noqa: E402
    feature_activations,
    pair_positions_0,
)


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
    parser.add_argument("--case-indexes", default="", help="Comma-separated original case indexes")
    parser.add_argument(
        "--directions",
        default="wt_state,mutant_state",
        help="Comma-separated directions to test: wt_state,mutant_state",
    )
    parser.add_argument(
        "--feature-ids",
        default="",
        help="Optional comma-separated feature IDs. If omitted, use top features from each direction.",
    )
    parser.add_argument("--top-features-per-direction", type=int, default=5)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5,10")
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


def parse_int_set(raw: str) -> set[int] | None:
    vals = {int(x.strip()) for x in raw.split(",") if x.strip()}
    return vals or None


def parse_feature_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_directions(raw: str) -> list[str]:
    directions = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in directions if x not in {"wt_state", "mutant_state"}]
    if bad:
        raise ValueError(f"Unknown directions: {bad}")
    return directions


def load_feature_scan(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    case_filter = parse_int_set(args.case_indexes)
    selected = []
    for entry in payload:
        case_index = int(entry["case_index"])
        if case_filter is not None and case_index not in case_filter:
            continue
        selected.append(entry)
    return selected


def feature_rows_for_direction(
    entry: dict[str, Any],
    direction: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    explicit = parse_feature_ids(args.feature_ids)
    if explicit:
        by_id: dict[int, dict[str, Any]] = {}
        for key in ("top_wt_state_features", "top_mutant_state_features"):
            for row in entry.get(key, []):
                by_id[int(row["feature_id"])] = row
        rows = []
        for fid in explicit:
            row = dict(by_id.get(fid, {}))
            row.setdefault("feature_id", fid)
            row.setdefault("direction", direction)
            rows.append(row)
        return rows

    key = "top_wt_state_features" if direction == "wt_state" else "top_mutant_state_features"
    return entry.get(key, [])[: args.top_features_per_direction]


def direction_probe(case: dict[str, Any], direction: str) -> dict[str, Any]:
    mutation_pos = int(case["mutation_pos0"])
    if direction == "mutant_state":
        context_seq = case["wt_sequence"]
        source_seq = case["mutant_sequence"]
        target_base = case["mutation_to"]
        source_pairs = case["diagnostic_mutant_pairs_1"]
    else:
        context_seq = case["mutant_sequence"]
        source_seq = case["wt_sequence"]
        target_base = case["mutation_from"]
        source_pairs = case["diagnostic_wt_pairs_1"]

    prefix = context_seq[:mutation_pos]
    suffix = context_seq[mutation_pos + 1 :]
    return {
        "context_sequence": context_seq,
        "source_sequence": source_seq,
        "prefix": prefix,
        "target": target_base,
        "suffix": suffix,
        "mutation_pos0": mutation_pos,
        "source_positions0": pair_positions_0(source_pairs),
    }


def choose_anchor(
    model: Any,
    tokenizer: Any,
    sae: Any,
    source_sequence: str,
    source_positions0: list[int],
    feature_id: int,
    mutation_pos0: int,
    args: argparse.Namespace,
) -> tuple[int, float]:
    candidate_positions = [pos for pos in source_positions0 if pos != mutation_pos0]
    if not candidate_positions:
        raise ValueError("No unmasked diagnostic anchor positions available")
    acts, seq_start = feature_activations(model, tokenizer, sae, source_sequence, args)
    best_pos = max(
        candidate_positions,
        key=lambda pos: float(acts[seq_start + pos, feature_id]),
    )
    best_activation = float(acts[seq_start + best_pos, feature_id])
    return best_pos, best_activation


def set_prompt_args(args: argparse.Namespace, probe: dict[str, Any], anchor_pos0: int) -> None:
    mutation_pos = int(probe["mutation_pos0"])
    args.prefix = probe["prefix"]
    args.target = probe["target"]
    args.suffix = probe["suffix"]
    if anchor_pos0 < mutation_pos:
        args.steer_part = "prefix"
        args.steer_offset = anchor_pos0
    elif anchor_pos0 > mutation_pos:
        args.steer_part = "suffix"
        args.steer_offset = anchor_pos0 - mutation_pos - 1
    else:
        raise ValueError("Anchor position cannot be the masked mutation position")
    args.steer_token_index = None


def first_target_prob(logits: torch.Tensor, prompt: Any) -> dict[str, Any]:
    score = score_target(logits, prompt)
    tok = score["target_tokens"][0]
    return {
        "token": tok["token"],
        "prob": float(tok["prob"]),
        "log_prob": float(tok["log_prob"]),
        "target_token_logp": float(score["target_token_logp"]),
    }


def run_feature_direction(
    model: Any,
    tokenizer: Any,
    sae: Any,
    entry: dict[str, Any],
    direction: str,
    feature_row: dict[str, Any],
    args: argparse.Namespace,
    scales: list[float],
) -> dict[str, Any] | None:
    case = entry["case"]
    feature_id = int(feature_row["feature_id"])
    probe = direction_probe(case, direction)
    anchor_pos0, one_x = choose_anchor(
        model,
        tokenizer,
        sae,
        probe["source_sequence"],
        probe["source_positions0"],
        feature_id,
        int(probe["mutation_pos0"]),
        args,
    )
    if one_x < args.min_anchor_activation:
        return None

    set_prompt_args(args, probe, anchor_pos0)
    prompt = build_prompt(tokenizer, args.prefix, args.target, args.suffix, args, args.device)
    baseline_logits, _ = forward_with_steer(
        model, prompt, sae, args, feature_ids=None, scale=None, feature_max={}
    )
    baseline = first_target_prob(baseline_logits, prompt)

    points = []
    feature_max = {feature_id: one_x}
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
    best = max(points, key=lambda row: row["delta_prob"])
    worst = min(points, key=lambda row: row["delta_prob"])
    return {
        "case_index": int(entry["case_index"]),
        "record_id": case["record_id"],
        "direction": direction,
        "feature_id": feature_id,
        "feature_state_margin": float(feature_row.get("state_margin", 0.0)),
        "feature_positive_mean": float(feature_row.get("positive_mean", 0.0)),
        "mutation_pos1": int(case["mutation_pos1"]),
        "mutation_from": case["mutation_from"],
        "mutation_to": case["mutation_to"],
        "target_base": probe["target"],
        "context_state": "wt_sequence" if direction == "mutant_state" else "mutant_sequence",
        "anchor_pos1": anchor_pos0 + 1,
        "anchor_one_x": one_x,
        "baseline_prob": baseline["prob"],
        "baseline_log_prob": baseline["log_prob"],
        "best_scale": best["scale"],
        "best_prob": best["prob"],
        "best_delta_prob": best["delta_prob"],
        "best_delta_log_prob": best["delta_log_prob"],
        "worst_delta_prob": worst["delta_prob"],
        "positive_steps": sum(1 for row in points if row["delta_prob"] > 0),
        "n_scales": len(points),
        "points": points,
    }


def flatten(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in payload:
        for point in result["points"]:
            rows.append(
                {
                    "case_index": result["case_index"],
                    "record_id": result["record_id"],
                    "direction": result["direction"],
                    "feature_id": result["feature_id"],
                    "feature_state_margin": result["feature_state_margin"],
                    "feature_positive_mean": result["feature_positive_mean"],
                    "mutation_pos1": result["mutation_pos1"],
                    "mutation_from": result["mutation_from"],
                    "mutation_to": result["mutation_to"],
                    "target_base": result["target_base"],
                    "context_state": result["context_state"],
                    "anchor_pos1": result["anchor_pos1"],
                    "anchor_one_x": result["anchor_one_x"],
                    "baseline_prob": result["baseline_prob"],
                    "scale": point["scale"],
                    "prob": point["prob"],
                    "delta_prob": point["delta_prob"],
                    "delta_log_prob": point["delta_log_prob"],
                    "hidden_diff": point["hidden_diff"],
                }
            )
    rows.sort(key=lambda row: row["delta_prob"], reverse=True)
    return rows


def summarize(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in payload:
        rows.append(
            {
                "case_index": result["case_index"],
                "record_id": result["record_id"],
                "direction": result["direction"],
                "feature_id": result["feature_id"],
                "feature_state_margin": result["feature_state_margin"],
                "feature_positive_mean": result["feature_positive_mean"],
                "mutation_pos1": result["mutation_pos1"],
                "mutation_from": result["mutation_from"],
                "mutation_to": result["mutation_to"],
                "target_base": result["target_base"],
                "context_state": result["context_state"],
                "anchor_pos1": result["anchor_pos1"],
                "anchor_one_x": result["anchor_one_x"],
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
    if not entries:
        raise RuntimeError("No feature-scan entries selected")

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    payload = []
    for entry in entries:
        for direction in directions:
            for feature_row in feature_rows_for_direction(entry, direction, args):
                result = run_feature_direction(model, tokenizer, sae, entry, direction, feature_row, args, scales)
                if result is not None:
                    payload.append(result)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    rows = flatten(payload)
    summary = summarize(payload)
    write_csv(out_prefix.with_name(out_prefix.name + "_rows").with_suffix(".csv"), rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary").with_suffix(".csv"), summary)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_rows').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary').with_suffix('.csv')}")
    if summary:
        top = summary[0]
        print(
            "Top structure-switch direct effect: "
            f"case={top['case_index']} {top['direction']} f/{top['feature_id']} "
            f"delta={top['best_delta_prob']:.6f} baseline={top['baseline_prob']:.6f} "
            f"best={top['best_prob']:.6f} scale={top['best_scale']}"
        )


if __name__ == "__main__":
    main()
