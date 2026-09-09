#!/usr/bin/env python3
"""Scan SAE features on RNAfold structure-switch candidates.

The upstream structure-switch search identifies WT and mutant structures with
different diagnostic base-pair sets. This script ranks features that separate
WT-state diagnostic positions from mutant-state diagnostic positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    encode_sae,
    load_model,
    load_sae,
    token_ids,
)


DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--top-n-cases", type=int, default=5)
    parser.add_argument("--case-indexes", default="", help="Comma-separated 0-based indexes")
    parser.add_argument("--top-k-features", type=int, default=30)
    parser.add_argument("--min-active", type=float, default=0.01)
    return parser.parse_args()


def load_cases(path: Path, args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        cases = json.load(handle)
    if args.case_indexes:
        indexes = [int(x.strip()) for x in args.case_indexes.split(",") if x.strip()]
    else:
        indexes = list(range(min(args.top_n_cases, len(cases))))
    return [(idx, cases[idx]) for idx in indexes if 0 <= idx < len(cases)]


def prepare_inputs(tokenizer: Any, sequence: str, device: str) -> dict[str, torch.Tensor]:
    full = f"<bos>5{sequence}3<eos>"
    ids = token_ids(tokenizer, full)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0),
        "sequence_ids": torch.zeros((1, len(ids)), dtype=torch.long, device=device),
        "full_ids": ids,
    }


def sequence_token_start(tokenizer: Any, full_ids: list[int]) -> int:
    bos_id = token_ids(tokenizer, "<bos>")[0]
    return full_ids.index(bos_id) + 2


def capture_hidden(model: Any, inputs: dict[str, torch.Tensor], layer: int, device: str) -> torch.Tensor:
    captured = None

    def hook(_module: Any, _inp: Any, out: Any) -> None:
        nonlocal captured
        captured = (out[0] if isinstance(out, tuple) else out).detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            sequence_ids=inputs["sequence_ids"],
        )
    handle.remove()
    if captured is None:
        raise RuntimeError("hook did not capture hidden states")
    return captured[0]


def pair_positions_0(pairs_1: list[list[int]]) -> list[int]:
    positions = set()
    for pair in pairs_1:
        if len(pair) != 2:
            continue
        positions.add(int(pair[0]) - 1)
        positions.add(int(pair[1]) - 1)
    return sorted(positions)


def feature_activations(
    model: Any,
    tokenizer: Any,
    sae: Any,
    sequence: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    inputs = prepare_inputs(tokenizer, sequence, args.device)
    seq_start = sequence_token_start(tokenizer, inputs["full_ids"])
    hidden = capture_hidden(model, inputs, args.layer, args.device)
    acts = encode_sae(hidden, sae).detach().float().cpu()
    return acts, seq_start


def mean_at_positions(acts: torch.Tensor, seq_start: int, positions0: list[int]) -> torch.Tensor:
    if not positions0:
        return torch.zeros(acts.shape[1], dtype=acts.dtype)
    token_positions = [seq_start + pos for pos in positions0]
    return acts[token_positions, :].mean(dim=0)


def max_at_positions(acts: torch.Tensor, seq_start: int, positions0: list[int]) -> torch.Tensor:
    if not positions0:
        return torch.zeros(acts.shape[1], dtype=acts.dtype)
    token_positions = [seq_start + pos for pos in positions0]
    return acts[token_positions, :].max(dim=0).values


def rank_direction(
    case: dict[str, Any],
    case_index: int,
    direction: str,
    positive: torch.Tensor,
    same_coords_negative: torch.Tensor,
    opposite_same_sequence: torch.Tensor,
    opposite_other_sequence: torch.Tensor,
    positive_max: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    n_features = int(positive.shape[0])
    for fid in range(n_features):
        pos = float(positive[fid])
        if pos < args.min_active:
            continue
        same = float(same_coords_negative[fid])
        opp_same = float(opposite_same_sequence[fid])
        opp_other = float(opposite_other_sequence[fid])
        strongest_negative = max(same, opp_same, opp_other)
        state_margin = pos - strongest_negative
        same_coord_contrast = pos - same
        opposite_state_contrast = pos - opp_other
        rows.append(
            {
                "case_index": case_index,
                "record_id": case["record_id"],
                "direction": direction,
                "feature_id": fid,
                "state_margin": state_margin,
                "same_coord_contrast": same_coord_contrast,
                "opposite_state_contrast": opposite_state_contrast,
                "positive_mean": pos,
                "same_coords_negative_mean": same,
                "opposite_same_sequence_mean": opp_same,
                "opposite_other_sequence_mean": opp_other,
                "positive_max": float(positive_max[fid]),
                "mutation_pos1": case["mutation_pos1"],
                "mutation_from": case["mutation_from"],
                "mutation_to": case["mutation_to"],
                "bp_distance": case["bp_distance"],
                "local_bp_distance": case["local_bp_distance"],
                "wt_only_pair_count": case["wt_only_pair_count"],
                "mutant_only_pair_count": case["mutant_only_pair_count"],
            }
        )
    rows.sort(key=lambda row: row["state_margin"], reverse=True)
    return rows[: args.top_k_features]


def scan_case(
    model: Any,
    tokenizer: Any,
    sae: Any,
    case_index: int,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    wt_positions0 = pair_positions_0(case["diagnostic_wt_pairs_1"])
    mutant_positions0 = pair_positions_0(case["diagnostic_mutant_pairs_1"])
    wt_acts, wt_start = feature_activations(model, tokenizer, sae, case["wt_sequence"], args)
    mutant_acts, mutant_start = feature_activations(model, tokenizer, sae, case["mutant_sequence"], args)

    wt_diag_wt = mean_at_positions(wt_acts, wt_start, wt_positions0)
    wt_diag_mut = mean_at_positions(mutant_acts, mutant_start, wt_positions0)
    mut_diag_wt = mean_at_positions(wt_acts, wt_start, mutant_positions0)
    mut_diag_mut = mean_at_positions(mutant_acts, mutant_start, mutant_positions0)
    wt_diag_wt_max = max_at_positions(wt_acts, wt_start, wt_positions0)
    mut_diag_mut_max = max_at_positions(mutant_acts, mutant_start, mutant_positions0)

    wt_ranked = rank_direction(
        case,
        case_index,
        "wt_state",
        positive=wt_diag_wt,
        same_coords_negative=wt_diag_mut,
        opposite_same_sequence=mut_diag_wt,
        opposite_other_sequence=mut_diag_mut,
        positive_max=wt_diag_wt_max,
        args=args,
    )
    mutant_ranked = rank_direction(
        case,
        case_index,
        "mutant_state",
        positive=mut_diag_mut,
        same_coords_negative=mut_diag_wt,
        opposite_same_sequence=wt_diag_mut,
        opposite_other_sequence=wt_diag_wt,
        positive_max=mut_diag_mut_max,
        args=args,
    )
    return {
        "case_index": case_index,
        "case": case,
        "wt_positions0": wt_positions0,
        "mutant_positions0": mutant_positions0,
        "top_wt_state_features": wt_ranked,
        "top_mutant_state_features": mutant_ranked,
    }


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
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(Path(args.cases_json), args)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    payload = []
    flat_rows = []
    for case_index, case in cases:
        entry = scan_case(model, tokenizer, sae, case_index, case, args)
        payload.append(entry)
        for row in entry["top_wt_state_features"] + entry["top_mutant_state_features"]:
            flat_rows.append(row)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    flat_rows.sort(key=lambda row: row["state_margin"], reverse=True)
    write_csv(out_prefix.with_suffix(".csv"), flat_rows)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    if flat_rows:
        top = flat_rows[0]
        print(
            "Top structure-switch feature: "
            f"case={top['case_index']} record={top['record_id']} "
            f"{top['direction']} f/{top['feature_id']} "
            f"margin={top['state_margin']:.6f} positive={top['positive_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
