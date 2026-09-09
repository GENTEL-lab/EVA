#!/usr/bin/env python3
"""
Optimized Feature Selection with Output Score Filtering.

Uses existing activation separation results and focuses on computing
output scores and steering effects for top candidates only.

Usage:
    python filter_features_by_output_score_v2.py \
        --feature-scan-json data/xxx_feature_scan.json \
        --cases-json data/xxx_cases.json \
        --out-prefix results/filtered
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1')
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1/scripts/sae_feature_steering')

import torch
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np

from eva_glm_sae_steering_fig6 import (
    load_model, load_sae, build_prompt, forward_with_steer, score_target
)


DEFAULT_EVA_ROOT = "/data/yanjie_huang/enzyme1_server/eva/EVA1"
DEFAULT_CHECKPOINT = "/data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_15000"

SAE_CONFIGS = {
    "layer13_step200000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt",
        "layer": 13,
    },
}


def token_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-scan-json", required=True, help="Existing feature scan JSON")
    parser.add_argument("--cases-json", required=True, help="Cases JSON")
    parser.add_argument("--sae", default="layer13_step200000")
    parser.add_argument("--top-k", type=int, default=30, help="Top k features per case to test")
    parser.add_argument("--scales", default="0,1,2,2.5,5", help="Comma-separated scales")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def capture_hidden_at_layer(model, tokenizer, sequence, layer, device):
    """Capture hidden state at a specific layer."""
    full = f"<bos>5{sequence}3<eos>"
    ids = token_ids(tokenizer, full)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
    sequence_ids = torch.zeros((1, len(ids)), dtype=torch.long, device=device)

    captured = None
    def hook(module, inp, out):
        nonlocal captured
        hidden = out[0] if isinstance(out, tuple) else out
        captured = hidden.detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        model(input_ids=input_ids, position_ids=position_ids, sequence_ids=sequence_ids)

    handle.remove()
    return captured[0]


def compute_decoder_stats(sae, feature_id):
    """Compute decoder statistics for a feature."""
    decoder_vec = sae.decoder_weight[:, feature_id]
    return {
        "decoder_norm": float(torch.norm(decoder_vec).item()),
        "decoder_mean_abs": float(torch.mean(torch.abs(decoder_vec)).item()),
        "decoder_max_abs": float(torch.max(torch.abs(decoder_vec)).item()),
    }


def build_break_to_rescue_probe(case):
    """Build probe for steering test."""
    break_seq = case["break_sequence"]
    rescue_pos = int(case["rescue_pos0"])
    break_pos = int(case["break_pos0"])

    prefix = break_seq[:rescue_pos]
    target = case["rescue_to"]
    suffix = break_seq[rescue_pos + 1:]

    if break_pos < rescue_pos:
        steer_part = "prefix"
        steer_offset = break_pos
    else:
        steer_part = "suffix"
        steer_offset = break_pos - rescue_pos - 1

    return {
        "prefix": prefix,
        "target": target,
        "suffix": suffix,
        "steer_part": steer_part,
        "steer_offset": steer_offset,
    }


def run_steering_test(model, tokenizer, sae, case, layer, device, feature_id, scales):
    """Test steering effect of a feature."""
    probe = build_break_to_rescue_probe(case)

    args = argparse.Namespace(
        prefix=probe["prefix"],
        target=probe["target"],
        suffix=probe["suffix"],
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
        steer_part=probe["steer_part"],
        steer_offset=probe["steer_offset"],
        steer_token_index=None,
        group_mode="joint",
        feature_max_json="",
        scale_source="constant",
        clamp_value=20.0,
        patch_mode="reconstruct",
        device=device,
        layer=layer,
    )

    prompt = build_prompt(tokenizer, probe["prefix"], probe["target"], probe["suffix"], args, device)

    # Baseline
    logits, _ = forward_with_steer(model, prompt, sae, args, None, None, {})
    baseline = score_target(logits, prompt)
    p_baseline = baseline["target_tokens"][0]["prob"]

    # Test scales
    best_delta = 0
    best_scale = 0
    all_results = []

    for scale in scales:
        if scale == 0:
            continue
        logits, _ = forward_with_steer(
            model, prompt, sae, args,
            feature_ids=[feature_id],
            scale=scale,
            feature_max={}
        )
        score = score_target(logits, prompt)
        p = score["target_tokens"][0]["prob"]
        delta = p - p_baseline
        all_results.append({"scale": scale, "p": p, "delta": delta})

        if delta > best_delta:
            best_delta = delta
            best_scale = scale

    return {
        "baseline": p_baseline,
        "best_delta": best_delta,
        "best_scale": best_scale,
        "all_results": all_results,
    }


def main():
    args = parse_args()
    scales = [float(x) for x in args.scales.split(",")]

    print("=" * 70)
    print("FEATURE SELECTION WITH OUTPUT SCORE FILTERING (Optimized)")
    print("=" * 70)

    # Load feature scan results
    with open(args.feature_scan_json) as f:
        scan_data = json.load(f)

    # Load cases
    with open(args.cases_json) as f:
        all_cases = json.load(f)

    # Build case index lookup
    case_lookup = {case["record_id"]: case for case in all_cases}

    print(f"Loaded {len(scan_data)} cases from feature scan")

    # Load model and SAE
    sae_config = SAE_CONFIGS[args.sae]
    print(f"Loading model and SAE ({args.sae}, layer {sae_config['layer']})...")

    model_args = argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        eva_root=DEFAULT_EVA_ROOT,
        model_code_path=None,
        device=args.device,
    )
    model, tokenizer = load_model(model_args)
    sae = load_sae(sae_config["path"], "auto", args.device)
    layer = sae_config["layer"]

    print(f"SAE has {sae.encoder_weight.shape[0]} features")

    # Results storage
    all_results = []
    all_features_flat = []

    for entry in scan_data:
        case_id = entry.get("case_index", entry.get("record_id", "unknown"))
        case_data = entry.get("case", entry)
        record_id = case_data.get("record_id", str(case_id))

        if record_id not in case_lookup:
            print(f"Warning: case {record_id} not found in cases JSON")
            continue

        case = case_lookup[record_id]
        features = entry.get("top_features", [])[:args.top_k]

        if not features:
            continue

        print(f"\n{'='*60}")
        print(f"Case: {record_id}")
        print(f"  WT pair: ({case['i0']}, {case['j0']}) = {case.get('wt_i_base', '?')}-{case.get('wt_j_base', '?')}")
        print(f"  Break: {case['break_from']}->{case['break_to']}")
        print(f"  Rescue: {case['rescue_to']}")
        print(f"  Testing top {len(features)} features")

        case_result = {
            "case_index": case_id,
            "record_id": record_id,
            "features": [],
        }

        # Get hidden state for output score computation
        hidden = capture_hidden_at_layer(model, tokenizer, case["break_sequence"], layer, args.device)
        i_pos = int(case["i0"]) + 2
        j_pos = int(case["j0"]) + 2

        for feat in features:
            fid = int(feat["feature_id"])

            # Compute decoder stats
            decoder_stats = compute_decoder_stats(sae, fid)

            # Compute activation at pair positions
            centered = hidden - sae.bias
            pre_act = F.linear(centered, sae.encoder_weight, sae.encoder_bias)
            acts = F.relu(pre_act)
            activation_at_pair = float(acts[[i_pos, j_pos], fid].mean().item())

            # Output score proxy: decoder_norm * activation
            output_score = decoder_stats["decoder_norm"] * max(0, activation_at_pair)

            # Test steering effect
            steer_result = run_steering_test(
                model, tokenizer, sae, case, layer, args.device, fid, scales
            )

            feat_result = {
                "feature_id": fid,
                "pair_score": float(feat.get("pair_score", 0)),
                "combined_score": float(feat.get("combined_score", 0)),
                "decoder_norm": decoder_stats["decoder_norm"],
                "decoder_mean_abs": decoder_stats["decoder_mean_abs"],
                "activation_at_pair": activation_at_pair,
                "output_score": output_score,
                "steering_baseline": steer_result["baseline"],
                "steering_best_delta": steer_result["best_delta"],
                "steering_best_scale": steer_result["best_scale"],
                "steering_all_results": steer_result["all_results"],
            }

            case_result["features"].append(feat_result)
            all_features_flat.append({**feat_result, "record_id": record_id})

            # Print positive results
            if feat_result["steering_best_delta"] > 0.005:
                print(f"    f/{fid}: pair_score={feat_result['pair_score']:.4f}, "
                      f"output_score={output_score:.4f}, steering_Δ={steer_result['best_delta']:+.6f}")

        all_results.append(case_result)

    # Save results
    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Rankings
    print("\n" + "=" * 70)
    print("RANKING BY STEERING EFFECT")
    print("=" * 70)

    by_steering = sorted(all_features_flat, key=lambda x: x["steering_best_delta"], reverse=True)

    print(f"\n{'Rank':<6} {'Case':<22} {'Feature':<10} {'PairScore':<12} {'OutputScore':<14} {'SteerΔ':<12}")
    print("-" * 80)
    for i, f in enumerate(by_steering[:25]):
        print(f"{i+1:<6} {f['record_id'][:20]:<22} f/{f['feature_id']:<8} "
              f"{f['pair_score']:<12.4f} {f['output_score']:<14.4f} {f['steering_best_delta']:<+12.6f}")

    # Save top by steering
    with open(out_path.parent / f"{out_path.name}_top_by_steering.json", "w") as f:
        json.dump(by_steering[:30], f, indent=2)

    # Ranking by output score
    print("\n" + "=" * 70)
    print("RANKING BY OUTPUT SCORE")
    print("=" * 70)

    by_output = sorted(all_features_flat, key=lambda x: x["output_score"], reverse=True)

    print(f"\n{'Rank':<6} {'Case':<22} {'Feature':<10} {'PairScore':<12} {'OutputScore':<14} {'SteerΔ':<12}")
    print("-" * 80)
    for i, f in enumerate(by_output[:25]):
        print(f"{i+1:<6} {f['record_id'][:20]:<22} f/{f['feature_id']:<8} "
              f"{f['pair_score']:<12.4f} {f['output_score']:<14.4f} {f['steering_best_delta']:<+12.6f}")

    # Statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    positive = [f for f in all_features_flat if f["steering_best_delta"] > 0]
    negative = [f for f in all_features_flat if f["steering_best_delta"] < -0.005]

    print(f"Total features tested: {len(all_features_flat)}")
    print(f"Positive steering effect (>0): {len(positive)}")
    print(f"Negative steering effect (<-0.005): {len(negative)}")

    if positive:
        best = max(positive, key=lambda x: x["steering_best_delta"])
        print(f"\nBest steering effect: Δ={best['steering_best_delta']:.6f}")
        print(f"  Case: {best['record_id']}, Feature: f/{best['feature_id']}")
        print(f"  Pair score: {best['pair_score']:.4f}, Output score: {best['output_score']:.4f}")

    # Correlation analysis
    if len(all_features_flat) > 10:
        try:
            from scipy.stats import pearsonr
            pair_scores = [f["pair_score"] for f in all_features_flat]
            steering_deltas = [f["steering_best_delta"] for f in all_features_flat]
            corr, pval = pearsonr(pair_scores, steering_deltas)

            output_scores = [f["output_score"] for f in all_features_flat]
            corr_out, pval_out = pearsonr(output_scores, steering_deltas)

            print(f"\nCorrelation analysis:")
            print(f"  pair_score vs steering_Δ: r={corr:.4f}, p={pval:.4f}")
            print(f"  output_score vs steering_Δ: r={corr_out:.4f}, p={pval_out:.4f}")

            if abs(corr) < 0.3:
                print("  ⚠️ LOW correlation: activation separation does NOT predict steering!")
            if abs(corr_out) > abs(corr):
                print("  ✓ Output score is a better predictor than pair_score!")
        except ImportError:
            print("(scipy not available for correlation analysis)")

    print(f"\nResults saved to:")
    print(f"  {out_path.with_suffix('.json')}")
    print(f"  {out_path.parent / f'{out_path.name}_top_by_steering.json'}")


if __name__ == "__main__":
    main()
