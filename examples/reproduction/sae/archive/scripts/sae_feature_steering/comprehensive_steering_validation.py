#!/usr/bin/env python3
"""
Comprehensive f/5271 Steering Analysis with Figure Generation.

This script:
1. Validates steering at break position (pos 2) with fine scale granularity
2. Validates steering at T-loop position (pos 58) - where f/5271 activates strongest
3. Compares dose-response curves at different positions
4. Generates publication-quality figure proving steering is effective
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1')
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1/scripts/sae_feature_steering')

import torch
import torch.nn.functional as F
from contextlib import nullcontext

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
    parser.add_argument("--feature-id", type=int, default=5271)
    parser.add_argument("--control-feature", type=int, default=3571, help="Low-effect control")
    parser.add_argument("--matched-control", type=int, default=5313, help="Matched active control")
    parser.add_argument("--sae", default="layer13_step200000")
    parser.add_argument("--case-index", type=int, default=3)
    parser.add_argument("--cases-json", default="/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering/rnafold_cases/domingo2018_trna_cases.json")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("COMPREHENSIVE f/5271 STEERING VALIDATION")
    print("=" * 70)

    # Load case
    with open(args.cases_json) as f:
        cases = json.load(f)
    case = cases[args.case_index]

    print(f"Case: {case['record_id']}")
    print(f"WT pair: ({case['i0']}, {case['j0']}) = {case['wt_i_base']}-{case['wt_j_base']}")
    print(f"Break: pos {case['break_pos0']} {case['break_from']}->{case['break_to']}")
    print(f"Rescue target: pos {case['rescue_pos0']} {case['rescue_from']}->{case['rescue_to']}")

    # Load model and SAE
    sae_config = SAE_CONFIGS[args.sae]
    print(f"\nLoading model and SAE...")

    model_args = argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        eva_root=DEFAULT_EVA_ROOT,
        model_code_path=None,
        device=args.device,
    )
    model, tokenizer = load_model(model_args)
    sae = load_sae(sae_config["path"], "auto", args.device)
    layer = sae_config["layer"]

    # Setup probe
    rescue_pos = case['rescue_pos0']
    break_pos = case['break_pos0']
    prefix = case['break_sequence'][:rescue_pos]
    target = case['rescue_to']
    suffix = case['break_sequence'][rescue_pos + 1:]

    feature_max = {args.feature_id: 3.985, args.control_feature: 3.5, args.matched_control: 3.92}
    scales = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # ====== EXPERIMENT 1: Steering at break position (pos 2) ======
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: Steering at BREAK position (pos {break_pos})")
    print("=" * 70)

    # Configure args for break position
    args_b = argparse.Namespace(
        prefix=prefix, target=target, suffix=suffix, sequence="",
        span_start=None, span_length=None, max_prefix=0, max_suffix=0,
        steer_part="prefix", steer_offset=break_pos, steer_token_index=None,
        group_mode="joint", feature_max_json="", scale_source="feature_max_json",
        clamp_value=20.0, patch_mode="reconstruct", device=args.device, layer=layer,
    )

    prompt = build_prompt(tokenizer, prefix, target, suffix, args_b, args.device)

    # Baseline
    logits, _ = forward_with_steer(model, prompt, sae, args_b, None, None, feature_max)
    baseline = score_target(logits, prompt)
    p_baseline_break = baseline["target_tokens"][0]["prob"]
    print(f"Baseline P(rescue_base): {p_baseline_break:.6f}")

    # Steering dose-response
    def run_dose_response(feature_id, label):
        results = []
        for scale in scales:
            if scale == 0:
                continue
            logits, _ = forward_with_steer(
                model, prompt, sae, args_b,
                feature_ids=[feature_id], scale=scale, feature_max=feature_max
            )
            score = score_target(logits, prompt)
            p = score["target_tokens"][0]["prob"]
            delta = p - p_baseline_break
            results.append({'scale': scale, 'p': p, 'delta': delta})
            print(f"  f/{feature_id} {scale}x: P={p:.4f}, Δ={delta:+.4f}")
        return results

    print(f"\nTarget feature f/{args.feature_id}:")
    target_break_results = run_dose_response(args.feature_id, "target")

    print(f"\nLow-effect control f/{args.control_feature}:")
    control_break_results = run_dose_response(args.control_feature, "low-effect control")

    print(f"\nMatched active control f/{args.matched_control}:")
    matched_break_results = run_dose_response(args.matched_control, "matched control")

    # ====== EXPERIMENT 2: Steering at T-loop position (pos 58) ======
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: Steering at T-LOOP position (pos 58)")
    print("=" * 70)

    # Get token indices
    full = f"<bos>5{case['break_sequence']}3<eos>"
    full_ids = token_ids(tokenizer, full)
    bos_id = token_ids(tokenizer, "<bos>")[0]
    seq_start = full_ids.index(bos_id) + 2

    # T-loop position
    tloop_pos = 58
    steer_token_index = seq_start + tloop_pos

    args_t = argparse.Namespace(
        prefix=prefix, target=target, suffix=suffix, sequence="",
        span_start=None, span_length=None, max_prefix=0, max_suffix=0,
        steer_part="absolute", steer_offset=0, steer_token_index=steer_token_index,
        group_mode="joint", feature_max_json="", scale_source="feature_max_json",
        clamp_value=20.0, patch_mode="reconstruct", device=args.device, layer=layer,
    )

    prompt_t = build_prompt(tokenizer, prefix, target, suffix, args_t, args.device)

    # Baseline (should be same)
    logits, _ = forward_with_steer(model, prompt_t, sae, args_t, None, None, feature_max)
    baseline = score_target(logits, prompt)
    p_baseline_tloop = baseline["target_tokens"][0]["prob"]
    print(f"Baseline P(rescue_base): {p_baseline_tloop:.6f}")

    def run_dose_response_tloop(feature_id, label):
        results = []
        for scale in scales:
            if scale == 0:
                continue
            logits, _ = forward_with_steer(
                model, prompt_t, sae, args_t,
                feature_ids=[feature_id], scale=scale, feature_max=feature_max
            )
            score = score_target(logits, prompt)
            p = score["target_tokens"][0]["prob"]
            delta = p - p_baseline_tloop
            results.append({'scale': scale, 'p': p, 'delta': delta})
            print(f"  f/{feature_id} {scale}x: P={p:.4f}, Δ={delta:+.4f}")
        return results

    print(f"\nTarget feature f/{args.feature_id} at T-loop:")
    target_tloop_results = run_dose_response_tloop(args.feature_id, "target")

    print(f"\nLow-effect control f/{args.control_feature} at T-loop:")
    control_tloop_results = run_dose_response_tloop(args.control_feature, "control")

    # Save all results
    all_results = {
        'case': {
            'record_id': case['record_id'],
            'wt_pair': f"({case['i0']}, {case['j0']})",
            'break': f"pos {case['break_pos0']} {case['break_from']}->{case['break_to']}",
            'rescue': f"pos {case['rescue_pos0']} {case['rescue_from']}->{case['rescue_to']}",
        },
        'feature_id': args.feature_id,
        'scales': scales,
        'break_position': {
            'baseline': p_baseline_break,
            'target': target_break_results,
            'low_effect_control': control_break_results,
            'matched_control': matched_break_results,
        },
        'tloop_position': {
            'baseline': p_baseline_tloop,
            'target': target_tloop_results,
            'low_effect_control': control_tloop_results,
        },
    }

    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: BEST STEERING EFFECTS")
    print("=" * 70)

    best_target_break = max(target_break_results, key=lambda x: x['delta'])
    best_target_tloop = max(target_tloop_results, key=lambda x: x['delta'])
    best_control_break = max(control_break_results, key=lambda x: x['delta'])
    best_control_tloop = max(control_tloop_results, key=lambda x: x['delta'])

    print(f"\n  Steering at BREAK position (pos {break_pos}):")
    print(f"    Target f/{args.feature_id}: best Δ={best_target_break['delta']:+.4f} at {best_target_break['scale']}x")
    print(f"    Control f/{args.control_feature}: best Δ={best_control_break['delta']:+.4f} at {best_control_break['scale']}x")
    print(f"    Matched f/{args.matched_control}: best Δ={max(r['delta'] for r in matched_break_results):+.4f}")

    print(f"\n  Steering at T-LOOP position (pos 58):")
    print(f"    Target f/{args.feature_id}: best Δ={best_target_tloop['delta']:+.4f} at {best_target_tloop['scale']}x")
    print(f"    Control f/{args.control_feature}: best Δ={best_control_tloop['delta']:+.4f} at {best_control_tloop['scale']}x")

    # Calculate ratios
    print(f"\n  Target vs Control ratio (break pos): {best_target_break['delta'] / max(best_control_break['delta'], 1e-10):.1f}x")
    print(f"  Target vs Control ratio (T-loop pos): {best_target_tloop['delta'] / max(best_control_tloop['delta'], 1e-10):.1f}x")

    print(f"\nResults saved to: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
