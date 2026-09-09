#!/usr/bin/env python3
"""Score EVA GLM baseline probabilities for RNAfold steering cases.

Before feature steering, a case should pass a simpler gate: in the break
context with the paired site masked, does EVA already assign meaningful
probability to the compensatory rescue base? If not, an SAE intervention has
no strong model behavior to amplify.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    load_model,
    token_ids,
)


DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"
BASES = ("A", "U", "G", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-indexes", default="")

    # Fields required by build_prompt but not conceptually used here.
    parser.set_defaults(
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
        steer_part="absolute",
        steer_offset=0,
        steer_token_index=0,
    )
    return parser.parse_args()


def parse_int_set(raw: str) -> set[int] | None:
    vals = {int(x.strip()) for x in raw.split(",") if x.strip()}
    return vals or None


def load_cases(path: Path, args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        cases = json.load(handle)
    case_filter = parse_int_set(args.case_indexes)
    selected = []
    for idx, case in enumerate(cases):
        if case_filter is not None and idx not in case_filter:
            continue
        selected.append((idx, case))
        if case_filter is None and args.max_cases > 0 and len(selected) >= args.max_cases:
            break
    return selected


def make_probe(sequence: str, target_pos0: int, expected_base: str) -> tuple[str, str, str]:
    return sequence[:target_pos0], expected_base, sequence[target_pos0 + 1 :]


def score_probe(model: Any, tokenizer: Any, args: argparse.Namespace, prefix: str, target: str, suffix: str) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, prefix, target, suffix, args, args.device)
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        outputs = model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    pred_pos = prompt.target_start - 1
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    base_rows = []
    for base in BASES:
        ids = token_ids(tokenizer, base)
        if len(ids) != 1:
            raise RuntimeError(f"Base {base} tokenized to {ids}")
        tok_id = ids[0]
        base_rows.append(
            {
                "base": base,
                "token_id": int(tok_id),
                "prob": float(probs[0, pred_pos, tok_id].detach().cpu()),
                "log_prob": float(log_probs[0, pred_pos, tok_id].detach().cpu()),
            }
        )
    base_rows.sort(key=lambda row: row["prob"], reverse=True)
    expected = next(row for row in base_rows if row["base"] == target)
    return {
        "expected_base": target,
        "expected_prob": expected["prob"],
        "expected_log_prob": expected["log_prob"],
        "expected_rank": 1 + [row["base"] for row in base_rows].index(target),
        "top_base": base_rows[0]["base"],
        "top_prob": base_rows[0]["prob"],
        "margin_vs_best_other": expected["prob"]
        - max(row["prob"] for row in base_rows if row["base"] != target),
        "base_probs": {row["base"]: row["prob"] for row in base_rows},
        "prompt_len": prompt.prompt_len,
        "target_start": prompt.target_start,
    }


def score_case(model: Any, tokenizer: Any, args: argparse.Namespace, case_index: int, case: dict[str, Any]) -> list[dict[str, Any]]:
    target_pos0 = int(case["rescue_pos0"])
    contexts = [
        {
            "context": "wt_original",
            "sequence": case["wt_sequence"],
            "expected_base": case["rescue_from"],
            "description": "WT context, original paired base masked",
        },
        {
            "context": "break_rescue",
            "sequence": case["break_sequence"],
            "expected_base": case["rescue_to"],
            "description": "Break context, compensatory rescue base masked",
        },
    ]
    rows = []
    for context in contexts:
        prefix, target, suffix = make_probe(
            context["sequence"],
            target_pos0,
            context["expected_base"],
        )
        scored = score_probe(model, tokenizer, args, prefix, target, suffix)
        rows.append(
            {
                "case_index": case_index,
                "record_id": case["record_id"],
                "context": context["context"],
                "description": context["description"],
                "i1": case["i1"],
                "j1": case["j1"],
                "break_pos1": case["break_pos1"],
                "break_from": case["break_from"],
                "break_to": case["break_to"],
                "rescue_pos1": case["rescue_pos1"],
                "rescue_from": case["rescue_from"],
                "rescue_to": case["rescue_to"],
                "local_break_bp_distance": case["local_break_bp_distance"],
                "break_bp_distance": case["break_bp_distance"],
                "expected_base": scored["expected_base"],
                "expected_prob": scored["expected_prob"],
                "expected_log_prob": scored["expected_log_prob"],
                "expected_rank": scored["expected_rank"],
                "top_base": scored["top_base"],
                "top_prob": scored["top_prob"],
                "margin_vs_best_other": scored["margin_vs_best_other"],
                "prob_A": scored["base_probs"]["A"],
                "prob_U": scored["base_probs"]["U"],
                "prob_G": scored["base_probs"]["G"],
                "prob_C": scored["base_probs"]["C"],
                "prompt_len": scored["prompt_len"],
                "target_start": scored["target_start"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path_prefix: Path, rows: list[dict[str, Any]], top_n: int = 20) -> None:
    break_rows = [row for row in rows if row["context"] == "break_rescue"]
    break_rows.sort(key=lambda row: row["expected_prob"], reverse=True)
    top = break_rows[:top_n]
    if not top:
        return
    labels = [f"c{row['case_index']} {row['expected_base']} r{row['expected_rank']}" for row in top]
    values = [row["expected_prob"] for row in top]
    colors = ["#3b7a78" if row["expected_rank"] == 1 else "#8a8f98" for row in top]
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.32 * len(top) + 1.2)))
    y = list(range(len(top)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("No-steer P(compensatory rescue base)")
    ax.set_title("EVA GLM baseline gate for RNAfold rescue cases")
    ax.set_xlim(0, max(0.35, max(values) * 1.08))
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

    selected = load_cases(Path(args.cases_json), args)
    if not selected:
        raise RuntimeError("No cases selected")
    model, tokenizer = load_model(args)

    rows = []
    for case_index, case in selected:
        rows.extend(score_case(model, tokenizer, args, case_index, case))
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    write_csv(out_prefix.with_suffix(".csv"), rows)
    plot(out_prefix, rows)

    break_rows = [row for row in rows if row["context"] == "break_rescue"]
    break_rows.sort(key=lambda row: row["expected_prob"], reverse=True)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    if break_rows:
        top = break_rows[0]
        print(
            "Top break-rescue baseline: "
            f"case={top['case_index']} record={top['record_id']} "
            f"P({top['expected_base']})={top['expected_prob']:.6f} "
            f"rank={top['expected_rank']} top={top['top_base']}:{top['top_prob']:.6f}"
        )


if __name__ == "__main__":
    main()
