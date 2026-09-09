#!/usr/bin/env python3
"""Scan SAE features on RNAfold WT/break/rescue candidate cases.

This is the bridge between RNAfold case discovery and InterPLM-style
steering. It loads the EVA model and an SAE, then ranks features whose
activations are high in WT/rescue but low in the break mutant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument(
        "--extra-site-packages",
        default="",
        help="Optional site-packages path appended before loading EVA, e.g. for megablocks.",
    )
    parser.add_argument("--top-n-cases", type=int, default=10)
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

    def hook(_module, _inp, out):
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


def feature_stats_for_sequence(
    model: Any,
    tokenizer: Any,
    sae: Any,
    sequence: str,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    inputs = prepare_inputs(tokenizer, sequence, args.device)
    seq_start = sequence_token_start(tokenizer, inputs["full_ids"])
    hidden = capture_hidden(model, inputs, args.layer, args.device)
    acts = encode_sae(hidden, sae).detach().float()

    i_pos = seq_start + int(case["i0"])
    j_pos = seq_start + int(case["j0"])
    break_pos = seq_start + int(case["break_pos0"])
    rescue_pos = seq_start + int(case["rescue_pos0"])
    window_start = seq_start + int(case["window_start0"])
    window_end = seq_start + int(case["window_end0"])

    pair_pos = [i_pos, j_pos]
    edit_pos = sorted(set([break_pos, rescue_pos]))
    return {
        "pair_mean": acts[pair_pos, :].mean(dim=0).cpu(),
        "pair_max": acts[pair_pos, :].max(dim=0).values.cpu(),
        "edit_mean": acts[edit_pos, :].mean(dim=0).cpu(),
        "window_mean": acts[window_start:window_end, :].mean(dim=0).cpu(),
        "window_max": acts[window_start:window_end, :].max(dim=0).values.cpu(),
    }


def rank_features(case: dict[str, Any], stats: dict[str, dict[str, torch.Tensor]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wt = stats["wt"]
    br = stats["break"]
    rescue = stats["rescue"]
    rows: list[dict[str, Any]] = []
    n_features = int(wt["pair_mean"].shape[0])
    for fid in range(n_features):
        wt_pair = float(wt["pair_mean"][fid])
        br_pair = float(br["pair_mean"][fid])
        re_pair = float(rescue["pair_mean"][fid])
        wt_window = float(wt["window_mean"][fid])
        br_window = float(br["window_mean"][fid])
        re_window = float(rescue["window_mean"][fid])
        wt_edit = float(wt["edit_mean"][fid])
        br_edit = float(br["edit_mean"][fid])
        re_edit = float(rescue["edit_mean"][fid])

        pair_active = max(wt_pair, re_pair)
        window_active = max(wt_window, re_window)
        if max(pair_active, window_active) < args.min_active:
            continue

        pair_score = min(wt_pair, re_pair) - br_pair
        window_score = min(wt_window, re_window) - br_window
        edit_score = min(wt_edit, re_edit) - br_edit
        similarity_penalty = 0.25 * (abs(wt_pair - re_pair) + abs(wt_window - re_window))
        combined_score = pair_score + window_score + 0.5 * edit_score - similarity_penalty
        one_x = max(
            float(wt["pair_max"][fid]),
            float(br["pair_max"][fid]),
            float(rescue["pair_max"][fid]),
            float(wt["window_max"][fid]),
            float(br["window_max"][fid]),
            float(rescue["window_max"][fid]),
        )
        rows.append(
            {
                "feature_id": fid,
                "combined_score": combined_score,
                "pair_score": pair_score,
                "window_score": window_score,
                "edit_score": edit_score,
                "wt_pair_mean": wt_pair,
                "break_pair_mean": br_pair,
                "rescue_pair_mean": re_pair,
                "wt_window_mean": wt_window,
                "break_window_mean": br_window,
                "rescue_window_mean": re_window,
                "wt_edit_mean": wt_edit,
                "break_edit_mean": br_edit,
                "rescue_edit_mean": re_edit,
                "one_x_observed": one_x,
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
            }
        )
    rows.sort(key=lambda x: x["combined_score"], reverse=True)
    return rows[: args.top_k_features]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)
    cases = load_cases(Path(args.cases_json), args)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    payload: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for case_index, case in cases:
        stats = {
            "wt": feature_stats_for_sequence(model, tokenizer, sae, case["wt_sequence"], case, args),
            "break": feature_stats_for_sequence(model, tokenizer, sae, case["break_sequence"], case, args),
            "rescue": feature_stats_for_sequence(model, tokenizer, sae, case["rescue_sequence"], case, args),
        }
        ranked = rank_features(case, stats, args)
        for rank, row in enumerate(ranked, start=1):
            row["case_index"] = case_index
            row["feature_rank"] = rank
            flat_rows.append(row)
        payload.append({"case_index": case_index, "case": case, "top_features": ranked})
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(out_prefix.with_suffix(".csv"), flat_rows)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    if flat_rows:
        top = max(flat_rows, key=lambda x: x["combined_score"])
        print(
            "Top feature: "
            f"case={top['case_index']} record={top['record_id']} "
            f"f/{top['feature_id']} combined={top['combined_score']:.4f} "
            f"pair_score={top['pair_score']:.4f} window_score={top['window_score']:.4f}"
        )


if __name__ == "__main__":
    main()
