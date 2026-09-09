#!/usr/bin/env python3
"""GLM-style SAE feature steering for EVA.

This script adapts the InterPLM/ESM-style feature steering figure to EVA:

  <bos_glm>5[prefix]<span_0>[suffix]3<eos><span_0>[target]<eos_span>

It steers SAE features at one existing context token, then measures the
teacher-forced probability of each target span token. Two feature groups are
plotted as Fig. 6a/6b-style panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# Recovered source SHA256: 5bc9bc821ed8efe2b214c05e40a2ee56c961e84c27cbbe053de65ab967a125f7
# Source: EVA1/scripts/sae_feature_steering/eva_glm_sae_steering_fig6.py
# Portable adaptation; a smoke run does not establish final paper feature provenance.
DEFAULT_EVA_ROOT = str(Path(__file__).resolve().parents[1])


@dataclass
class BuiltPrompt:
    prompt: str
    full: str
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    sequence_ids: torch.Tensor
    prompt_len: int
    target_start: int
    target_ids: list[int]
    target_with_eos_ids: list[int]
    target_tokens: list[str]
    steer_token_index: int


@dataclass
class SAEWeights:
    mode: str
    bias: torch.Tensor
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor
    decoder_weight: torch.Tensor
    k: int | None


def parse_feature_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make EVA GLM-style SAE steering Fig. 6a/6b panels."
    )
    p.add_argument("--checkpoint", required=True, help="EVA checkpoint directory")
    p.add_argument("--sae", required=True, help="Exact SAE checkpoint path")
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)

    seq = p.add_argument_group("probe sequence")
    seq.add_argument("--sequence", default="", help="Full RNA sequence; derives prefix/target/suffix")
    seq.add_argument("--span-start", type=int, default=None, help="0-based start in --sequence")
    seq.add_argument("--span-length", type=int, default=None)
    seq.add_argument("--prefix", default="")
    seq.add_argument("--target", default="")
    seq.add_argument("--suffix", default="")
    seq.add_argument("--max-suffix", type=int, default=200)
    seq.add_argument("--max-prefix", type=int, default=200)

    steer = p.add_argument_group("steering")
    steer.add_argument("--features-a", required=True, help="Comma-separated feature ids for panel A")
    steer.add_argument("--features-b", required=True, help="Comma-separated feature ids for panel B")
    steer.add_argument("--label-a", default="candidate feature group")
    steer.add_argument("--label-b", default="control feature group")
    steer.add_argument("--group-mode", choices=["average", "joint"], default="average")
    steer.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    steer.add_argument(
        "--scale-source",
        choices=["current", "constant", "feature_max_json"],
        default="current",
        help="What value defines 1x clamp for each feature.",
    )
    steer.add_argument("--clamp-value", type=float, default=10.0)
    steer.add_argument(
        "--feature-max-json",
        default="",
        help="Optional JSON mapping feature id to observed max/p99 activation.",
    )
    steer.add_argument(
        "--patch-mode",
        choices=["decoder_delta", "reconstruct"],
        default="decoder_delta",
        help="decoder_delta preserves the original hidden except for the feature direction.",
    )
    steer.add_argument(
        "--steer-part",
        choices=["prefix", "suffix", "absolute"],
        default="prefix",
        help="Context part containing the steered token.",
    )
    steer.add_argument(
        "--steer-offset",
        type=int,
        default=-1,
        help="Character offset in prefix/suffix. Negative counts from the end.",
    )
    steer.add_argument("--steer-token-index", type=int, default=None)

    out = p.add_argument_group("output")
    out.add_argument("--out-prefix", required=True)
    out.add_argument("--title", default="EVA GLM SAE feature steering")
    out.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_rna(s: str) -> str:
    return "".join(s.upper().replace("T", "U").split())


def derive_probe(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.sequence:
        seq = normalize_rna(args.sequence)
        if args.span_start is None or args.span_length is None:
            raise ValueError("--sequence requires --span-start and --span-length")
        start = args.span_start
        end = start + args.span_length
        if start < 0 or end > len(seq) or start >= end:
            raise ValueError("Invalid span range for --sequence")
        prefix = seq[:start]
        target = seq[start:end]
        suffix = seq[end:]
    else:
        prefix = normalize_rna(args.prefix)
        target = normalize_rna(args.target)
        suffix = normalize_rna(args.suffix)
        if not target:
            raise ValueError("Provide --target or --sequence with span coordinates")

    if args.max_prefix > 0 and len(prefix) > args.max_prefix:
        prefix = prefix[-args.max_prefix :]
    if args.max_suffix > 0 and len(suffix) > args.max_suffix:
        suffix = suffix[: args.max_suffix]
    return prefix, target, suffix


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def token_labels(tokenizer: Any, ids: list[int]) -> list[str]:
    labels = []
    for idx in ids:
        if hasattr(tokenizer, "id_to_token"):
            tok = tokenizer.id_to_token(int(idx))
        elif hasattr(tokenizer, "decode"):
            tok = tokenizer.decode([int(idx)])
        else:
            tok = str(idx)
        labels.append(str(tok))
    return labels


def resolve_steer_index(
    tokenizer: Any,
    prefix: str,
    target: str,
    suffix: str,
    args: argparse.Namespace,
) -> int:
    if args.steer_part == "absolute":
        if args.steer_token_index is None:
            raise ValueError("--steer-part absolute requires --steer-token-index")
        return args.steer_token_index

    if args.steer_part == "prefix":
        if not prefix:
            raise ValueError("Cannot steer prefix because prefix is empty")
        char_index = args.steer_offset if args.steer_offset >= 0 else len(prefix) + args.steer_offset
        if char_index < 0 or char_index >= len(prefix):
            raise ValueError("--steer-offset outside prefix")
        return len(token_ids(tokenizer, "<bos_glm>5" + prefix[:char_index]))

    if not suffix:
        raise ValueError("Cannot steer suffix because suffix is empty")
    char_index = args.steer_offset if args.steer_offset >= 0 else len(suffix) + args.steer_offset
    if char_index < 0 or char_index >= len(suffix):
        raise ValueError("--steer-offset outside suffix")
    before = "<bos_glm>5" + prefix + "<span_0>" + suffix[:char_index]
    return len(token_ids(tokenizer, before))


def build_glm_position_ids(
    tokenizer: Any,
    prefix: str,
    target: str,
    suffix: str,
    full_ids_len: int,
) -> list[int]:
    """EVA GLM jumping positions for a teacher-forced single span.

    This mirrors ``tools/utils/generators/glm.py``. The first span marker is
    placed at the masked position and then jumps over the span marker plus the
    hidden content. The second span marker reuses the first span position; the
    teacher-forced generated tokens then continue from the next position.
    """
    pre_ids = token_ids(tokenizer, "<bos_glm>5" + prefix)
    span1_ids = token_ids(tokenizer, "<span_0>")
    suffix_ids = token_ids(tokenizer, suffix + "3<eos>")
    span2_ids = token_ids(tokenizer, "<span_0>")
    target_eos_ids = token_ids(tokenizer, target + "<eos_span>")

    span_start = len(pre_ids)
    target_len = len(token_ids(tokenizer, target))

    pos: list[int] = []
    pos.extend(range(len(pre_ids)))
    pos.extend([span_start] * len(span1_ids))
    suffix_start = span_start + 1 + max(target_len, 1)
    pos.extend(range(suffix_start, suffix_start + len(suffix_ids)))
    pos.extend([span_start] * len(span2_ids))
    pos.extend(range(span_start + 1, span_start + 1 + len(target_eos_ids)))

    if len(pos) != full_ids_len:
        raise RuntimeError(f"Position ids length mismatch: {len(pos)} != {full_ids_len}")
    return pos


def build_prompt(tokenizer: Any, prefix: str, target: str, suffix: str, args: argparse.Namespace, device: str) -> BuiltPrompt:
    prompt = f"<bos_glm>5{prefix}<span_0>{suffix}3<eos><span_0>"
    full = f"{prompt}{target}<eos_span>"
    full_ids = token_ids(tokenizer, full)
    prompt_ids = token_ids(tokenizer, prompt)
    target_ids = token_ids(tokenizer, target)
    target_with_eos_ids = token_ids(tokenizer, target + "<eos_span>")
    position_ids = build_glm_position_ids(tokenizer, prefix, target, suffix, len(full_ids))
    steer_token_index = resolve_steer_index(tokenizer, prefix, target, suffix, args)
    if steer_token_index < 0 or steer_token_index >= len(prompt_ids):
        raise ValueError(
            f"Steer token index {steer_token_index} is outside context prompt length {len(prompt_ids)}"
        )
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    return BuiltPrompt(
        prompt=prompt,
        full=full,
        input_ids=input_ids,
        position_ids=torch.tensor([position_ids], dtype=torch.long, device=device),
        sequence_ids=torch.zeros((1, len(full_ids)), dtype=torch.long, device=device),
        prompt_len=len(prompt_ids),
        target_start=len(prompt_ids),
        target_ids=target_ids,
        target_with_eos_ids=target_with_eos_ids,
        target_tokens=token_labels(tokenizer, target_ids),
        steer_token_index=steer_token_index,
    )


def load_model(args: argparse.Namespace):
    eva_root = Path(args.eva_root)
    sys.path.insert(0, str(eva_root))
    from tools.utils.model.loader import ModelLoader  # noqa: E402

    model_code_path = args.model_code_path or str(eva_root / "eva")
    loader = ModelLoader(args.checkpoint, model_code_path=model_code_path)
    model, tokenizer = loader.load(device=args.device)
    model.eval()
    return model, tokenizer


def load_sae(path: str, mode: str, device: str) -> SAEWeights:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if mode == "auto":
        if "model_state_dict" in ckpt:
            cfg_mode = str(
                ckpt.get("cfg", {}).get("mode")
                or ckpt.get("config", {}).get("mode")
                or ""
            )
            if cfg_mode == "sae_l1_penalty":
                mode = "interplm"
                k = None
            else:
                mode = "batch_topk"
                k = int(ckpt.get("cfg", ckpt.get("config", {})).get("k", 32))
            state = ckpt["model_state_dict"]
        elif "sae" in ckpt:
            mode = "interplm"
            state = ckpt["sae"]
            k = None
        else:
            raise ValueError(f"Cannot infer SAE checkpoint format from keys: {list(ckpt.keys())}")
    elif mode == "batch_topk":
        state = ckpt["model_state_dict"]
        k = int(ckpt.get("cfg", ckpt.get("config", {})).get("k", 32))
    else:
        state = ckpt["sae"] if "sae" in ckpt else ckpt["model_state_dict"]
        k = None

    return SAEWeights(
        mode=mode,
        bias=state["bias"].to(device=device, dtype=torch.float32),
        encoder_weight=state["encoder.weight"].to(device=device, dtype=torch.float32),
        encoder_bias=state["encoder.bias"].to(device=device, dtype=torch.float32),
        decoder_weight=state["decoder.weight"].to(device=device, dtype=torch.float32),
        k=k,
    )


def encode_sae(x: torch.Tensor, sae: SAEWeights) -> torch.Tensor:
    pre = F.linear(x.float() - sae.bias, sae.encoder_weight, sae.encoder_bias)
    if sae.mode == "batch_topk":
        k = int(sae.k or 32)
        vals, idx = torch.topk(pre, k=k, dim=-1)
        vals = torch.where(vals > 0, vals, torch.zeros_like(vals))
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out
    return torch.relu(pre)


def decode_sae(f: torch.Tensor, sae: SAEWeights) -> torch.Tensor:
    return F.linear(f.float(), sae.decoder_weight, sae.bias)


def load_feature_max(args: argparse.Namespace) -> dict[int, float]:
    if not args.feature_max_json:
        return {}
    with open(args.feature_max_json, "r") as handle:
        raw = json.load(handle)
    return {int(k): float(v) for k, v in raw.items()}


def get_one_x_value(
    feature_id: int,
    current_activation: float,
    args: argparse.Namespace,
    feature_max: dict[int, float],
) -> float:
    if args.scale_source == "current":
        return max(float(current_activation), 1e-6)
    if args.scale_source == "constant":
        return float(args.clamp_value)
    if feature_id not in feature_max:
        raise KeyError(f"Feature {feature_id} missing from --feature-max-json")
    return float(feature_max[feature_id])


def steer_vector(
    x: torch.Tensor,
    sae: SAEWeights,
    feature_ids: list[int],
    scale: float,
    args: argparse.Namespace,
    feature_max: dict[int, float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Use pre-activation for reporting current values, but patch the actual
    # sparse SAE activation vector. For L1/ReLU SAE this matches InterPLM's
    # reconstruction-plus-residual intervention.
    pre = torch.nn.functional.linear(x.float() - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())
    f = encode_sae(x.unsqueeze(0), sae)[0]
    f_new = f.clone()
    feature_info = []
    for feature_id in feature_ids:
        current = float(pre[feature_id].detach().cpu())  # use pre-activation as current
        one_x = get_one_x_value(feature_id, current, args, feature_max)
        target = float(scale) * one_x
        f_new[feature_id] = max(target, 0.0)
        feature_info.append(
            {
                "feature_id": int(feature_id),
                "current_activation": current,
                "current_sparse_activation": float(f[feature_id].detach().cpu()),
                "one_x_value": one_x,
                "target_activation": target,
            }
        )

    if args.patch_mode == "reconstruct":
        x_recon_old = decode_sae(f.unsqueeze(0), sae)[0]
        x_recon_new = decode_sae(f_new.unsqueeze(0), sae)[0]
        x_new = x_recon_new + (x.float() - x_recon_old)
    else:
        x_new = x.float().clone()
        for info in feature_info:
            fid = info["feature_id"]
            delta = info["target_activation"] - info["current_sparse_activation"]
            x_new = x_new + float(delta) * sae.decoder_weight[:, fid]
    return x_new, {"features": feature_info}


def forward_with_steer(
    model: Any,
    prompt: BuiltPrompt,
    sae: SAEWeights,
    args: argparse.Namespace,
    feature_ids: list[int] | None,
    scale: float | None,
    feature_max: dict[int, float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    hook_info: dict[str, Any] = {}

    def hook(_module, _inp, out):
        nonlocal hook_info
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, prompt.steer_token_index, :].float()
        x_new, info = steer_vector(x, sae, feature_ids or [], float(scale), args, feature_max)
        patched[0, prompt.steer_token_index, :] = x_new.to(dtype=patched.dtype)
        # Diagnostic: check how much the hidden state changed
        diff = (x_new - x).abs().mean().item() if x.shape == x_new.shape else -1.0
        info["hidden_diff"] = diff
        hook_info = info
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = None
    if feature_ids is not None and scale is not None:
        handle = model.model.layers[args.layer].register_forward_hook(hook)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    try:
        with torch.inference_mode(), autocast_ctx:
            outputs = model(
                input_ids=prompt.input_ids,
                position_ids=prompt.position_ids,
                sequence_ids=prompt.sequence_ids,
            )
    finally:
        if handle is not None:
            handle.remove()
    if not torch.isfinite(outputs.logits).all():
        raise FloatingPointError("Non-finite steering logits")
    return outputs.logits.float(), hook_info


def score_target(logits: torch.Tensor, prompt: BuiltPrompt) -> dict[str, Any]:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    token_rows = []
    total_logp = 0.0
    base_logp = 0.0
    for j, tok_id in enumerate(prompt.target_with_eos_ids):
        pred_pos = prompt.target_start + j - 1
        logp = float(log_probs[0, pred_pos, tok_id].detach().cpu())
        prob = float(probs[0, pred_pos, tok_id].detach().cpu())
        total_logp += logp
        if j < len(prompt.target_ids):
            base_logp += logp
        token_rows.append(
            {
                "target_index": j,
                "is_eos_span": bool(j >= len(prompt.target_ids)),
                "token_id": int(tok_id),
                "token": prompt.target_tokens[j] if j < len(prompt.target_tokens) else "<eos_span>",
                "pred_pos": int(pred_pos),
                "prob": prob,
                "log_prob": logp,
            }
        )
    return {
        "target_token_logp": base_logp,
        "target_with_eos_logp": total_logp,
        "target_tokens": token_rows,
    }


def run_feature_group(
    name: str,
    label: str,
    features: list[int],
    model: Any,
    prompt: BuiltPrompt,
    sae: SAEWeights,
    args: argparse.Namespace,
    feature_max: dict[int, float],
    scales: list[float],
) -> list[dict[str, Any]]:
    rows = []
    logits, hook_info = forward_with_steer(
        model, prompt, sae, args, feature_ids=None, scale=None, feature_max=feature_max
    )
    base_score = score_target(logits, prompt)
    rows.append(
        {
            "group": name,
            "group_label": label,
            "condition": "no_steer",
            "scale": None,
            "feature_ids": [],
            "hook_info": hook_info,
            "score": base_score,
        }
    )

    for scale in scales:
        if args.group_mode == "joint":
            logits, hook_info = forward_with_steer(
                model, prompt, sae, args, feature_ids=features, scale=scale, feature_max=feature_max
            )
            rows.append(
                {
                    "group": name,
                    "group_label": label,
                    "condition": f"{scale:g}x",
                    "scale": float(scale),
                    "feature_ids": features,
                    "hook_info": hook_info,
                    "score": score_target(logits, prompt),
                }
            )
        else:
            per_feature = []
            for feature_id in features:
                logits, hook_info = forward_with_steer(
                    model,
                    prompt,
                    sae,
                    args,
                    feature_ids=[feature_id],
                    scale=scale,
                    feature_max=feature_max,
                )
                per_feature.append(
                    {
                        "feature_id": feature_id,
                        "hook_info": hook_info,
                        "score": score_target(logits, prompt),
                    }
                )
            rows.append(average_feature_scores(name, label, features, scale, per_feature))
    return rows


def average_feature_scores(
    name: str,
    label: str,
    features: list[int],
    scale: float,
    per_feature: list[dict[str, Any]],
) -> dict[str, Any]:
    if not per_feature:
        raise ValueError("Cannot average empty feature list")
    n_tok = len(per_feature[0]["score"]["target_tokens"])
    token_rows = []
    for i in range(n_tok):
        probs = [x["score"]["target_tokens"][i]["prob"] for x in per_feature]
        logps = [x["score"]["target_tokens"][i]["log_prob"] for x in per_feature]
        template = per_feature[0]["score"]["target_tokens"][i]
        token_rows.append(
            {
                **template,
                "prob": float(np.mean(probs)),
                "log_prob": float(np.mean(logps)),
                "prob_std": float(np.std(probs)),
                "log_prob_std": float(np.std(logps)),
            }
        )
    return {
        "group": name,
        "group_label": label,
        "condition": f"{scale:g}x",
        "scale": float(scale),
        "feature_ids": features,
        "per_feature": per_feature,
        "hook_info": {"averaged_features": features},
        "score": {
            "target_token_logp": float(np.mean([x["score"]["target_token_logp"] for x in per_feature])),
            "target_with_eos_logp": float(np.mean([x["score"]["target_with_eos_logp"] for x in per_feature])),
            "target_tokens": token_rows,
        },
    }


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        for tok in row["score"]["target_tokens"]:
            if tok["is_eos_span"]:
                continue
            flat.append(
                {
                    "group": row["group"],
                    "group_label": row["group_label"],
                    "condition": row["condition"],
                    "scale": "" if row["scale"] is None else row["scale"],
                    "feature_ids": ",".join(str(x) for x in row["feature_ids"]),
                    "target_index": tok["target_index"],
                    "token": tok["token"],
                    "prob": tok["prob"],
                    "log_prob": tok["log_prob"],
                    "target_token_logp": row["score"]["target_token_logp"],
                }
            )
    return flat


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = flatten_rows(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def plot_fig(path_prefix: Path, rows: list[dict[str, Any]], title: str, target_tokens: list[str]) -> None:
    groups = [("A", "a"), ("B", "b")]
    group_rows = {g: [r for r in rows if r["group"] == g] for g, _ in groups}
    colors = {
        "no_steer": "#737373",
        "0x": "#4c78a8",
        "0.5x": "#72b7b2",
        "1x": "#54a24b",
        "1.5x": "#eeca3b",
        "2x": "#f58518",
        "2.5x": "#e45756",
        "3x": "#b279a2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(1, len(target_tokens) + 1)
    xticks = [f"{i}\n{tok}" for i, tok in zip(x, target_tokens)]

    for ax, (group, panel) in zip(axes, groups):
        subset = group_rows[group]
        label = subset[0]["group_label"] if subset else group
        for row in subset:
            y = [
                tok["prob"]
                for tok in row["score"]["target_tokens"]
                if not tok["is_eos_span"]
            ]
            cond = row["condition"]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors.get(cond, None),
                label="No steer" if cond == "no_steer" else cond,
            )
        ax.set_title(f"Fig. 6{panel} adapted: {label}", fontsize=10)
        ax.set_xlabel("Generated span position / target token")
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Teacher-forced p(target token)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features_a = parse_feature_list(args.features_a)
    features_b = parse_feature_list(args.features_b)
    if not features_a or not features_b:
        raise ValueError("--features-a and --features-b must be non-empty")
    scales = parse_scales(args.scales)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    prefix, target, suffix = derive_probe(args)
    prompt = build_prompt(tokenizer, prefix, target, suffix, args, args.device)
    feature_max = load_feature_max(args)

    rows = []
    rows.extend(
        run_feature_group(
            "A", args.label_a, features_a, model, prompt, sae, args, feature_max, scales
        )
    )
    rows.extend(
        run_feature_group(
            "B", args.label_b, features_b, model, prompt, sae, args, feature_max, scales
        )
    )

    payload = {
        "checkpoint": args.checkpoint,
        "sae": args.sae,
        "sae_mode": sae.mode,
        "layer": args.layer,
        "prompt": prompt.prompt,
        "target": target,
        "prefix_len": len(prefix),
        "suffix_len": len(suffix),
        "target_start": prompt.target_start,
        "steer_token_index": prompt.steer_token_index,
        "scale_source": args.scale_source,
        "patch_mode": args.patch_mode,
        "group_mode": args.group_mode,
        "rows": rows,
    }
    with open(out_prefix.with_suffix(".json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot_fig(out_prefix, rows, args.title, prompt.target_tokens)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
