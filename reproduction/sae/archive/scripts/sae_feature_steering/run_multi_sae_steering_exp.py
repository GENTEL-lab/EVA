#!/usr/bin/env python3
"""Multi-SAE Multi-Case Steering Experiment.

This script tests steering effects across multiple SAE checkpoints and multiple
RNAfold WT/break/rescue cases to find positive steering candidates.

Usage:
    python run_multi_sae_steering_exp.py \
        --sae-candidates sae1.pt,sae2.pt \
        --cases-json cases.json \
        --case-indexes 0,1,2,3,4 \
        --top-features 10 \
        --scales 0,0.5,1,1.5,2,2.5,5,10 \
        --out-prefix results/multi_sae_exp
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================================
# SAE Configuration - Define your SAE checkpoints here
# ============================================================================

SAE_CANDIDATES = {
    "layer13_step200000": {
        "path": "notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt",
        "layer": 13,
        "mode": "auto",
    },
    "layer23_step60000": {
        "path": "notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23/checkpoints/checkpoint_step60000.pt",
        "layer": 23,
        "mode": "auto",
    },
    "layer23_stable_step160000": {
        "path": "notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23_stable_lr3e-5_l1x3_noresample/checkpoints/checkpoint_step160000.pt",
        "layer": 23,
        "mode": "auto",
    },
}

DEFAULT_EVA_ROOT = "/data/yanjie_huang/enzyme1_server/eva/EVA1"
DEFAULT_CHECKPOINT = "/data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_15000"
DEFAULT_EXTRA_SITE_PACKAGES = "/data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/envs/70_RNAVerse/lib/python3.11/site-packages"


# ============================================================================
# SAE Loading Utilities
# ============================================================================

def load_sae(path: str, mode: str, device: str):
    """Load SAE checkpoint with proper mode detection."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if mode == "auto":
        if "model_state_dict" in ckpt:
            cfg_mode = str(
                ckpt.get("cfg", {}).get("mode") or ckpt.get("config", {}).get("mode") or ""
            )
            if cfg_mode == "sae_l1_penalty":
                mode = "interplm"
                k = None
            else:
                mode = "batch_topk"
                k = int(ckpt.get("config", {}).get("k", 32))
            state = ckpt["model_state_dict"]
        elif "sae" in ckpt:
            mode = "interplm"
            state = ckpt["sae"]
            k = None
        else:
            raise ValueError(f"Cannot infer SAE format: {list(ckpt.keys())}")
    elif mode == "batch_topk":
        state = ckpt["model_state_dict"]
        k = int(ckpt.get("config", {}).get("k", 32))
    else:
        state = ckpt["sae"] if "sae" in ckpt else ckpt["model_state_dict"]
        k = None

    return {
        "mode": mode,
        "bias": state["bias"].to(device=device, dtype=torch.float32),
        "encoder_weight": state["encoder.weight"].to(device=device, dtype=torch.float32),
        "encoder_bias": state["encoder.bias"].to(device=device, dtype=torch.float32),
        "decoder_weight": state["decoder.weight"].to(device=device, dtype=torch.float32),
        "k": k,
    }


def encode_sae(x: torch.Tensor, sae: dict) -> torch.Tensor:
    """Encode through SAE (pre-activation -> sparse activation)."""
    pre = F.linear(x.float() - sae["bias"], sae["encoder_weight"], sae["encoder_bias"])
    if sae["mode"] == "batch_topk":
        k = int(sae["k"] or 32)
        vals, idx = torch.topk(pre, k=k, dim=-1)
        vals = torch.where(vals > 0, vals, torch.zeros_like(vals))
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out
    return torch.relu(pre)


def decode_sae(f: torch.Tensor, sae: dict) -> torch.Tensor:
    """Decode from sparse activation to reconstruction."""
    return F.linear(f.float(), sae["decoder_weight"], sae["bias"])


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str, eva_root: str, device: str):
    """Load EVA model and tokenizer."""
    sys.path.insert(0, eva_root)
    from tools.utils.model.loader import ModelLoader

    loader = ModelLoader(checkpoint_path, model_code_path=f"{eva_root}/eva")
    model, tokenizer = loader.load(device=device)
    model.eval()
    return model, tokenizer


def token_ids(tokenizer, text: str) -> list[int]:
    """Tokenize text."""
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


# ============================================================================
# Hidden State Capture
# ============================================================================

def capture_hidden_at_layer(model, input_ids, position_ids, sequence_ids, layer: int, device: str):
    """Capture hidden states at a specific layer."""
    captured = None

    def hook(_module, _inp, out):
        nonlocal captured
        hidden = out[0] if isinstance(out, tuple) else out
        captured = hidden.detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        model(input_ids=input_ids, position_ids=position_ids, sequence_ids=sequence_ids)

    handle.remove()
    return captured[0] if captured is not None else None


# ============================================================================
# Feature Analysis
# ============================================================================

def analyze_features_for_case(model, tokenizer, sae, case: dict, layer: int, device: str) -> dict:
    """Analyze SAE features for a WT/break/rescue case."""
    results = {}

    for state_name in ["wt", "break", "rescue"]:
        seq = case[f"{state_name}_sequence"]
        full = f"<bos>5{seq}3<eos>"
        ids = token_ids(tokenizer, full)

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
        sequence_ids = torch.zeros((1, len(ids)), dtype=torch.long, device=device)

        hidden = capture_hidden_at_layer(model, input_ids, position_ids, sequence_ids, layer, device)
        if hidden is None:
            continue

        acts = encode_sae(hidden, sae).detach().float()

        # Get positions
        seq_start = ids.index(token_ids(tokenizer, "<bos>")[0]) + 2
        i_pos = seq_start + int(case["i0"])
        j_pos = seq_start + int(case["j0"])

        # Compute statistics
        pair_acts = acts[[i_pos, j_pos], :].mean(dim=0)
        results[state_name] = {
            "pair_mean": pair_acts.cpu(),
            "pair_max": acts[[i_pos, j_pos], :].max(dim=0).values.cpu(),
        }

    return results


def rank_features(case: dict, stats: dict, top_k: int = 20) -> list[dict]:
    """Rank features by WT/rescue vs break activation separation."""
    wt = stats.get("wt", {})
    br = stats.get("break", {})
    re = stats.get("rescue", {})

    if not wt or not br or not re:
        return []

    rows = []
    n_features = int(wt["pair_mean"].shape[0])

    for fid in range(n_features):
        wt_pair = float(wt["pair_mean"][fid])
        br_pair = float(br["pair_mean"][fid])
        re_pair = float(re["pair_mean"][fid])

        # Score: high in WT/rescue, low in break
        pair_score = min(wt_pair, re_pair) - br_pair

        if pair_score <= 0:
            continue

        # One-x value (max activation across states)
        one_x = max(
            float(wt["pair_max"][fid]),
            float(br["pair_max"][fid]),
            float(re["pair_max"][fid]),
        )

        rows.append({
            "feature_id": fid,
            "pair_score": pair_score,
            "wt_pair_mean": wt_pair,
            "break_pair_mean": br_pair,
            "rescue_pair_mean": re_pair,
            "one_x_observed": one_x,
        })

    rows.sort(key=lambda x: x["pair_score"], reverse=True)
    return rows[:top_k]


# ============================================================================
# Steering Experiment
# ============================================================================

def steer_and_score(
    model, tokenizer, sae, case: dict, layer: int, device: str,
    feature_id: int, scale: float, one_x_value: float
) -> dict:
    """Perform steering and return P(rescue_base)."""
    break_seq = case["break_sequence"]
    rescue_pos = int(case["rescue_pos0"])
    break_pos = int(case["break_pos0"])
    rescue_target = case["rescue_to"]

    # Build prompt: break context, mask rescue position
    prefix = break_seq[:rescue_pos]
    target = rescue_target
    suffix = break_seq[rescue_pos + 1:]

    prompt = f"<bos_glm>5{prefix}<span_0>{suffix}3<eos><span_0>"
    full = f"{prompt}{target}<eos_span>"
    full_ids = token_ids(tokenizer, full)
    prompt_ids = token_ids(tokenizer, prompt)

    # Determine steer token index
    if break_pos < rescue_pos:
        # Steer in prefix
        char_idx = break_pos
        pre = token_ids(tokenizer, "<bos_glm>5" + prefix[:char_idx])
        steer_idx = len(pre)
    else:
        # Steer in suffix
        char_idx = break_pos - rescue_pos - 1
        pre = token_ids(tokenizer, "<bos_glm>5" + prefix + "<span_0>" + suffix[:char_idx])
        steer_idx = len(pre)

    # Build position_ids (GLM style)
    target_len = len(token_ids(tokenizer, target))
    position_ids = list(range(len(prompt_ids)))
    position_ids.extend([len(token_ids(tokenizer, "<bos_glm>5" + prefix))] * len(token_ids(tokenizer, "<span_0>")))
    suffix_ids = token_ids(tokenizer, suffix + "3<eos>")
    position_ids.extend(range(len(prompt_ids) + 1 + max(target_len, 1),
                              len(prompt_ids) + 1 + max(target_len, 1) + len(suffix_ids)))
    position_ids.extend([len(token_ids(tokenizer, "<bos_glm>5" + prefix))] * len(token_ids(tokenizer, "<span_0>")))
    position_ids.extend(range(len(prompt_ids) + 1, len(full_ids)))

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    position_ids = torch.tensor([position_ids], dtype=torch.long, device=device)
    sequence_ids = torch.zeros((1, len(full_ids)), dtype=torch.long, device=device)

    # Steering hook
    target_activation = scale * one_x_value
    hook_info = {}

    def steer_hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        x = hidden[0, steer_idx, :].float()

        # SAE encode
        pre_act = F.linear(x - sae["bias"], sae["encoder_weight"], sae["encoder_bias"])
        f = encode_sae(x.unsqueeze(0), sae)[0]
        f_new = f.clone()
        f_new[feature_id] = max(target_activation, 0.0)

        # Reconstruct with modification
        x_recon_old = decode_sae(f.unsqueeze(0), sae)[0]
        x_recon_new = decode_sae(f_new.unsqueeze(0), sae)[0]
        x_new = x_recon_new + (x - x_recon_old)

        hidden[0, steer_idx, :] = x_new.to(dtype=hidden.dtype)
        hook_info["hidden_diff"] = (x_new - x).abs().mean().item()

        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden

    handle = model.model.layers[layer].register_forward_hook(steer_hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        outputs = model(input_ids=input_ids, position_ids=position_ids, sequence_ids=sequence_ids)

    handle.remove()

    # Get probability of rescue target base
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    # Find target token id
    target_ids = token_ids(tokenizer, target)
    target_start = len(prompt_ids)
    pred_pos = target_start - 1

    if pred_pos < probs.shape[1] and target_ids:
        tok_id = target_ids[0]
        p_rescue = float(probs[0, pred_pos, tok_id].detach().cpu())
    else:
        p_rescue = 0.0

    return {
        "p_rescue": p_rescue,
        "hidden_diff": hook_info.get("hidden_diff", 0.0),
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    sae_configs: dict,
    cases_json: str,
    case_indexes: list[int],
    top_features: int,
    scales: list[float],
    out_prefix: str,
    device: str = "cuda:0",
):
    """Run multi-SAE multi-case steering experiment."""

    results = {
        "sae_configs": {k: {"path": v["path"], "layer": v["layer"]} for k, v in sae_configs.items()},
        "scales": scales,
        "cases": [],
        "summary": [],
    }

    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure proper suffix for CSV
    json_path = out_path.parent / f"{out_path.name}.json"
    csv_path = out_path.parent / f"{out_path.name}_summary.csv"
    plot_path = out_path.parent / out_path.name

    if DEFAULT_EXTRA_SITE_PACKAGES and DEFAULT_EXTRA_SITE_PACKAGES not in sys.path:
        sys.path.append(DEFAULT_EXTRA_SITE_PACKAGES)

    # Load all cases
    with open(cases_json) as f:
        all_cases = json.load(f)

    selected_cases = [all_cases[i] for i in case_indexes if i < len(all_cases)]

    # Process each SAE
    for sae_name, sae_config in sae_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing SAE: {sae_name}")
        print(f"  Path: {sae_config['path']}")
        print(f"  Layer: {sae_config['layer']}")
        print(f"{'='*60}")

        try:
            # Load SAE
            sae = load_sae(sae_config["path"], sae_config["mode"], device)
            layer = sae_config["layer"]

            # Load model (only once per SAE since it's the same model)
            if "model" not in results:
                print(f"Loading EVA model...")
                model, tokenizer = load_model(DEFAULT_CHECKPOINT, DEFAULT_EVA_ROOT, device)
                results["model_checkpoint"] = DEFAULT_CHECKPOINT
            else:
                model = results["model"]
                tokenizer = results["tokenizer"]

            if "model" not in results:
                results["model"] = model
                results["tokenizer"] = tokenizer

            # Process each case
            for case_idx, case in enumerate(selected_cases):
                print(f"\n  Case {case_idx}: {case['record_id']}")

                # Analyze features
                stats = analyze_features_for_case(model, tokenizer, sae, case, layer, device)
                top_features_list = rank_features(case, stats, top_k=top_features)

                if not top_features_list:
                    print(f"    No discriminative features found")
                    continue

                # Get baseline (no steer)
                case_result = {
                    "case_index": case_idx,
                    "record_id": case["record_id"],
                    "wt_pair": f"({case['i0']}, {case['j0']})",
                    "break_mut": f"{case['break_from']}->{case['break_to']}",
                    "rescue_target": case["rescue_to"],
                    "local_bp_distance": case["local_break_bp_distance"],
                    "features": [],
                }

                print(f"    Top features: {[f['feature_id'] for f in top_features_list[:5]]}")

                # Test each feature at each scale
                for feat in top_features_list:
                    fid = feat["feature_id"]
                    one_x = feat["one_x_observed"]

                    feat_result = {
                        "feature_id": fid,
                        "pair_score": feat["pair_score"],
                        "one_x_observed": one_x,
                        "scales": {},
                    }

                    # No steer baseline
                    no_steer = steer_and_score(model, tokenizer, sae, case, layer, device, fid, 0, one_x)
                    feat_result["scales"]["no_steer"] = {
                        "p_rescue": no_steer["p_rescue"],
                        "hidden_diff": no_steer["hidden_diff"],
                    }

                    # Steering at different scales
                    best_delta = 0
                    best_scale = 0
                    for scale in scales:
                        if scale == 0:
                            continue
                        result = steer_and_score(model, tokenizer, sae, case, layer, device, fid, scale, one_x)
                        delta = result["p_rescue"] - no_steer["p_rescue"]
                        feat_result["scales"][f"{scale}x"] = {
                            "p_rescue": result["p_rescue"],
                            "delta": delta,
                            "hidden_diff": result["hidden_diff"],
                        }
                        if delta > best_delta:
                            best_delta = delta
                            best_scale = scale

                    feat_result["best_delta"] = best_delta
                    feat_result["best_scale"] = best_scale
                    print(f"      f/{fid}: best delta={best_delta:.6f} at {best_scale}x")

                    case_result["features"].append(feat_result)

                results["cases"].append(case_result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary
    summary = generate_summary(results)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Generate plots
    plot_results(results, plot_path)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print_summary_table(summary)
    print(f"\nFiles saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

    return results, summary


def generate_summary(results: dict) -> list[dict]:
    """Generate summary table of results."""
    summary = []

    for case_result in results.get("cases", []):
        for feat_result in case_result.get("features", []):
            baseline = feat_result["scales"].get("no_steer", {}).get("p_rescue", 0)
            best_delta = feat_result.get("best_delta", 0)
            best_scale = feat_result.get("best_scale", 0)

            # Count positive scales
            positive_scales = sum(
                1 for k, v in feat_result["scales"].items()
                if k != "no_steer" and v.get("delta", 0) > 0
            )
            total_scales = sum(1 for k in feat_result["scales"].keys() if k != "no_steer")

            summary.append({
                "case": case_result["record_id"],
                "feature_id": feat_result["feature_id"],
                "pair_score": feat_result["pair_score"],
                "baseline_P": baseline,
                "best_delta": best_delta,
                "best_scale": best_scale,
                "positive_scales": f"{positive_scales}/{total_scales}",
            })

    # Sort by best_delta
    summary.sort(key=lambda x: x["best_delta"], reverse=True)
    return summary


def print_summary_table(summary: list[dict]):
    """Print summary table to console."""
    print(f"\n{'Case':<30} {'Feature':<10} {'PairScore':<12} {'Baseline':<10} {'BestDelta':<12} {'Scale':<8} {'Pos/Total'}")
    print("-" * 95)
    for row in summary[:20]:  # Top 20
        print(f"{row['case']:<30} f/{row['feature_id']:<8} {row['pair_score']:<12.4f} "
              f"{row['baseline_P']:<10.4f} {row['best_delta']:<12.6f} {row['best_scale']:<8.1f} {row['positive_scales']}")


def plot_results(results: dict, out_path):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("Skipping plots (matplotlib or numpy not available)")
        return

    import numpy as np
    out_path = Path(out_path)

    # Plot 1: Top steering effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect all positive effects
    all_effects = []
    for case_result in results.get("cases", []):
        for feat_result in case_result.get("features", []):
            all_effects.append({
                "case": case_result["record_id"],
                "feature_id": feat_result["feature_id"],
                "best_delta": feat_result.get("best_delta", 0),
                "pair_score": feat_result["pair_score"],
            })

    all_effects.sort(key=lambda x: x["best_delta"], reverse=True)

    # Top effects bar plot
    ax = axes[0, 0]
    top_n = min(20, len(all_effects))
    labels = [f"{e['case'][:15]}\nf/{e['feature_id']}" for e in all_effects[:top_n]]
    values = [e["best_delta"] for e in all_effects[:top_n]]
    colors = ["#3b7a78" if v >= 0 else "#b35c44" for v in values]
    ax.barh(range(top_n), values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Best Delta P(rescue_base)")
    ax.set_title("Top Steering Effects")

    # Dose response for top feature
    ax = axes[0, 1]
    if all_effects:
        top_case = all_effects[0]["case"]
        top_feat = all_effects[0]["feature_id"]

        for case_result in results.get("cases", []):
            if case_result["record_id"] == top_case:
                for feat_result in case_result.get("features", []):
                    if feat_result["feature_id"] == top_feat:
                        scales = []
                        p_vals = []
                        for k, v in sorted(feat_result["scales"].items(), key=lambda x: (x[0]=="no_steer", x[0])):
                            if k == "no_steer":
                                scales.append(0)
                            else:
                                scales.append(float(k.replace("x", "")))
                            p_vals.append(v["p_rescue"])

                        ax.plot(scales, p_vals, marker="o", linewidth=2)
                        ax.axhline(p_vals[0], color="gray", linestyle="--", alpha=0.5)
                        ax.set_xlabel("Steering Scale")
                        ax.set_ylabel("P(rescue_base)")
                        ax.set_title(f"Dose Response: {top_case[:20]} f/{top_feat}")
                        break

    # Pair score vs steering effect
    ax = axes[1, 0]
    pair_scores = [e["pair_score"] for e in all_effects]
    deltas = [e["best_delta"] for e in all_effects]
    ax.scatter(pair_scores, deltas, alpha=0.5, s=30)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Feature Pair Score (activation separation)")
    ax.set_ylabel("Best Steering Delta")
    ax.set_title("Activation Separation vs Steering Effect")

    # Distribution of effects
    ax = axes[1, 1]
    deltas = [e["best_delta"] for e in all_effects]
    ax.hist(deltas, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linewidth=2)
    ax.axvline(np.mean(deltas), color="blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Best Steering Delta")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Effects (mean={np.mean(deltas):.6f})")

    plt.tight_layout()
    fig.savefig(out_path.with_suffix("_effects.png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix("_effects.pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved plots: {out_path.with_suffix('_effects.png/pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Multi-SAE Steering Experiment")
    parser.add_argument("--sae-configs", default=",".join(SAE_CANDIDATES.keys()),
                        help="Comma-separated SAE names from SAE_CANDIDATES dict")
    parser.add_argument("--sae-paths", default="",
                        help="Comma-separated explicit SAE paths (overrides SAE_CANDIDATES)")
    parser.add_argument("--sae-layers", default="",
                        help="Comma-separated layer numbers for explicit paths")
    parser.add_argument("--cases-json", required=True,
                        help="Path to cases JSON file")
    parser.add_argument("--case-indexes", default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated case indexes")
    parser.add_argument("--top-features", type=int, default=10,
                        help="Number of top features to test per case")
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5,10",
                        help="Comma-separated steering scales")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extra-site-packages", default="")
    args = parser.parse_args()

    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)

    # Parse configs
    if args.sae_paths:
        sae_paths = args.sae_paths.split(",")
        sae_layers = [int(l) for l in args.sae_layers.split(",")] if args.sae_layers else [13] * len(sae_paths)
        sae_names = [f"custom_sae_{i}" for i in range(len(sae_paths))]
        sae_configs = {
            name: {"path": path, "layer": layer, "mode": "auto"}
            for name, path, layer in zip(sae_names, sae_paths, sae_layers)
        }
    else:
        selected = args.sae_configs.split(",")
        sae_configs = {k: v for k, v in SAE_CANDIDATES.items() if k in selected}

    # Parse indexes and scales
    case_indexes = [int(x) for x in args.case_indexes.split(",")]
    scales = [float(x) for x in args.scales.split(",")]

    print(f"Running experiment with {len(sae_configs)} SAEs, {len(case_indexes)} cases")
    print(f"SAEs: {list(sae_configs.keys())}")
    print(f"Cases: {case_indexes}")
    print(f"Scales: {scales}")

    run_experiment(
        sae_configs=sae_configs,
        cases_json=args.cases_json,
        case_indexes=case_indexes,
        top_features=args.top_features,
        scales=scales,
        out_prefix=args.out_prefix,
        device=args.device,
    )


if __name__ == "__main__":
    main()
