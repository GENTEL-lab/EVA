#!/usr/bin/env python3
"""Batch-scan SAE features by direct GLM-anchor steering effect.

The CLM WT/break/rescue activation scan is useful but indirect: it ranks
features in full-sequence scoring contexts, while steering is applied in a GLM
masked prompt. This script tests features directly in the actual GLM prompt at
the actual anchor token and ranks them by output change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
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
    decode_sae,
    encode_sae,
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
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument(
        "--feature-ids",
        default="",
        help="Comma-separated feature IDs to scan. If set, overrides --feature-start/--feature-end.",
    )
    parser.add_argument("--feature-start", type=int, default=0)
    parser.add_argument("--feature-end", type=int, default=0, help="Exclusive; 0 means all SAE features")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Use 1 for evidence-grade scans. Larger batches are faster but can "
            "change MoE routing/numerics for repeated GLM prompts."
        ),
    )
    parser.add_argument(
        "--target-activations",
        default="0,1,2,5,10",
        help="Absolute sparse activation values to clamp each feature to.",
    )
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")
    parser.add_argument("--top-n-plot", type=int, default=30)

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
        scale_source="constant",
        clamp_value=1.0,
        feature_max_json="",
    )
    return parser.parse_args()


def parse_feature_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_case(path: Path, case_index: int) -> dict[str, Any]:
    with path.open() as handle:
        cases = json.load(handle)
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"case index {case_index} outside 0..{len(cases) - 1}")
    return cases[case_index]


def batched_prompt(prompt: Any, batch_size: int) -> Any:
    return replace(
        prompt,
        input_ids=prompt.input_ids.repeat(batch_size, 1),
        position_ids=prompt.position_ids.repeat(batch_size, 1),
        sequence_ids=prompt.sequence_ids.repeat(batch_size, 1),
    )


def target_probs_from_logits(logits: torch.Tensor, prompt: Any) -> tuple[list[float], list[float]]:
    tok_id = int(prompt.target_ids[0])
    pred_pos = int(prompt.target_start - 1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    return (
        probs[:, pred_pos, tok_id].detach().cpu().tolist(),
        log_probs[:, pred_pos, tok_id].detach().cpu().tolist(),
    )


def forward_feature_batch(
    model: Any,
    prompt: Any,
    sae: Any,
    args: argparse.Namespace,
    feature_ids: list[int],
    target_activation: float,
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    batch_size = len(feature_ids)
    prompt_b = batched_prompt(prompt, batch_size)
    feat_tensor = torch.tensor(feature_ids, dtype=torch.long, device=args.device)
    diagnostics: dict[str, list[float]] = {}

    def hook(_module: Any, _inp: Any, out: Any):
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[:, prompt.steer_token_index, :].float()
        f = encode_sae(x, sae)
        row = torch.arange(batch_size, device=args.device)
        current_sparse = f[row, feat_tensor]
        pre = F.linear(x - sae.bias, sae.encoder_weight, sae.encoder_bias)
        current_pre = pre[row, feat_tensor]

        f_new = f.clone()
        target = torch.full_like(current_sparse, max(float(target_activation), 0.0))
        f_new[row, feat_tensor] = target

        if args.patch_mode == "reconstruct":
            x_recon_old = decode_sae(f, sae)
            x_recon_new = decode_sae(f_new, sae)
            x_new = x_recon_new + (x - x_recon_old)
        else:
            deltas = target - current_sparse
            directions = sae.decoder_weight[:, feat_tensor].T
            x_new = x + deltas.unsqueeze(1) * directions

        patched[:, prompt.steer_token_index, :] = x_new.to(dtype=patched.dtype)
        diagnostics["current_sparse"] = current_sparse.detach().cpu().tolist()
        diagnostics["current_pre"] = current_pre.detach().cpu().tolist()
        diagnostics["hidden_diff"] = (x_new - x).abs().mean(dim=1).detach().cpu().tolist()
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = model.model.layers[args.layer].register_forward_hook(hook)
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        outputs = model(
            input_ids=prompt_b.input_ids,
            position_ids=prompt_b.position_ids,
            sequence_ids=prompt_b.sequence_ids,
        )
    handle.remove()
    probs, log_probs = target_probs_from_logits(outputs.logits, prompt)
    return probs, log_probs, diagnostics


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[int, dict[str, Any]] = {}
    for row in rows:
        fid = int(row["feature_id"])
        current = by_feature.get(fid)
        if current is None or row["delta_prob"] > current["best_delta_prob"]:
            by_feature[fid] = {
                "feature_id": fid,
                "best_target_activation": row["target_activation"],
                "best_prob": row["prob"],
                "best_log_prob": row["log_prob"],
                "best_delta_prob": row["delta_prob"],
                "best_delta_log_prob": row["delta_log_prob"],
                "current_sparse_at_best": row["current_sparse"],
                "current_pre_at_best": row["current_pre"],
                "hidden_diff_at_best": row["hidden_diff"],
            }
    out = list(by_feature.values())
    out.sort(key=lambda row: row["best_delta_prob"], reverse=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path_prefix: Path, summary: list[dict[str, Any]], top_n: int) -> None:
    top = summary[:top_n]
    if not top:
        return
    labels = [f"f/{row['feature_id']} @ {row['best_target_activation']:g}" for row in top]
    values = [row["best_delta_prob"] for row in top]
    colors = ["#3b7a78" if val >= 0 else "#b35c44" for val in values]
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.30 * len(top) + 1.2)))
    y = list(range(len(top)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Best delta P(rescue base)")
    ax.set_title("Direct GLM-anchor all-feature scan")
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
    if args.batch_size != 1:
        print(
            "WARNING: --batch-size > 1 is exploratory only for EVA MoE models; "
            "confirm top hits with --batch-size 1 and a single-feature run.",
            file=sys.stderr,
        )

    case = load_case(Path(args.cases_json), args.case_index)
    probe = build_break_to_rescue_probe(case)
    args.prefix = probe["prefix"]
    args.target = probe["target"]
    args.suffix = probe["suffix"]
    args.steer_part = probe["steer_part"]
    args.steer_offset = probe["steer_offset"]
    args.steer_token_index = None

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    prompt = build_prompt(tokenizer, args.prefix, args.target, args.suffix, args, args.device)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        base_outputs = model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    baseline_score = score_target(base_outputs.logits.float(), prompt)
    baseline_token = baseline_score["target_tokens"][0]
    baseline_prob = float(baseline_token["prob"])
    baseline_log_prob = float(baseline_token["log_prob"])

    n_features = int(sae.encoder_weight.shape[0])
    if args.feature_ids:
        feature_ids = parse_feature_ids(args.feature_ids)
        invalid = [fid for fid in feature_ids if fid < 0 or fid >= n_features]
        if invalid:
            raise ValueError(f"Feature IDs outside 0..{n_features - 1}: {invalid}")
    else:
        feature_end = args.feature_end if args.feature_end > 0 else n_features
        feature_ids = list(range(args.feature_start, min(feature_end, n_features)))
    target_activations = parse_scales(args.target_activations)

    rows: list[dict[str, Any]] = []
    for target_activation in target_activations:
        for start in range(0, len(feature_ids), args.batch_size):
            batch = feature_ids[start : start + args.batch_size]
            probs, log_probs, diag = forward_feature_batch(
                model, prompt, sae, args, batch, target_activation
            )
            for idx, fid in enumerate(batch):
                rows.append(
                    {
                        "feature_id": int(fid),
                        "target_activation": float(target_activation),
                        "prob": float(probs[idx]),
                        "log_prob": float(log_probs[idx]),
                        "delta_prob": float(probs[idx] - baseline_prob),
                        "delta_log_prob": float(log_probs[idx] - baseline_log_prob),
                        "current_sparse": float(diag["current_sparse"][idx]),
                        "current_pre": float(diag["current_pre"][idx]),
                        "hidden_diff": float(diag["hidden_diff"][idx]),
                    }
                )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    rows.sort(key=lambda row: row["delta_prob"], reverse=True)
    summary = summarize(rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "case": case,
                "probe": {
                    **probe,
                    "prompt_target": args.target,
                    "steer_token_index": prompt.steer_token_index,
                    "target_start": prompt.target_start,
                },
                "baseline": {
                    "token": baseline_token["token"],
                    "prob": baseline_prob,
                    "log_prob": baseline_log_prob,
                },
                "patch_mode": args.patch_mode,
                "batch_size": args.batch_size,
                "target_activations": target_activations,
                "rows": rows,
                "summary": summary,
            },
            handle,
            indent=2,
        )
    write_csv(out_prefix.with_name(out_prefix.name + "_rows").with_suffix(".csv"), rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary").with_suffix(".csv"), summary)
    plot_summary(out_prefix, summary, args.top_n_plot)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_rows').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    if summary:
        top = summary[0]
        print(
            "Top GLM-anchor effect: "
            f"f/{top['feature_id']} delta={top['best_delta_prob']:.6f} "
            f"baseline={baseline_prob:.6f} best={top['best_prob']:.6f} "
            f"target_activation={top['best_target_activation']}"
        )


if __name__ == "__main__":
    main()
