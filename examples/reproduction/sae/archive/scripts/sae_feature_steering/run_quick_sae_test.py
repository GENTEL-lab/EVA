#!/usr/bin/env python3
"""
Quick Multi-SAE Steering Test Script.

Run this in your conda environment with:
  conda run -n 70_RNAVerse python run_quick_sae_test.py [options]

Or add your conda path explicitly:
  /home/huangyanjie/miniconda3/envs/70_RNAVerse/bin/python run_quick_sae_test.py [options]

This script tests multiple SAE checkpoints on Domingo case 3 to compare steering effects.
"""

import argparse
import json
import sys
from pathlib import Path

# Add EVA paths
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1')
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1/scripts/sae_feature_steering')
sys.path.insert(0, '/home/huangyanjie/miniconda3/envs/70_RNAVerse/lib/python3.11/site-packages')

from eva_glm_sae_steering_fig6 import (
    load_model, load_sae, build_prompt, forward_with_steer,
    parse_scales, score_target
)

# SAE Configurations to test
SAE_CONFIGS = {
    # Layer 13 SAEs
    "layer13_step200000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt",
        "layer": 13,
    },
    # Layer 23 SAEs
    "layer23_step60000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23/checkpoints/checkpoint_step60000.pt",
        "layer": 23,
    },
    "layer23_stable_step100000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23_stable_lr3e-5_l1x3_noresample/checkpoints/checkpoint_step100000.pt",
        "layer": 23,
    },
    "layer23_stable_step160000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23_stable_lr3e-5_l1x3_noresample/checkpoints/checkpoint_step160000.pt",
        "layer": 23,
    },
    # From rna_benchmark
    "rna_benchmark_layer13": {
        "path": "/data/yanjie_huang/enzyme1_server/rna_benchmark/interpretability/DSSR_sequences/sae_evo2_online_1400M/sae_final.pt",
        "layer": 13,
    },
}

# EVA Checkpoint
EVA_CHECKPOINT = "/data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_15000"

# Domingo Case 3 details
CASE_3 = {
    "record_id": "Domingo_2018_tRNA_68",
    "break_sequence": "AUUCCGUUGGCGUAAUGGUAACGCGUUUCCCUCCUAAGGAGAAGACUGCGGGUUCGAGUCCCGUAUGGAGAG",  # 3U->A
    "rescue_pos": 68,
    "rescue_target": "U",  # A->U
    "break_pos": 2,  # 3U->A
}


def run_steering_test(sae_name, sae_config, case, scales, device):
    """Test steering on a specific SAE."""
    print(f"\n{'='*60}")
    print(f"Testing SAE: {sae_name}")
    print(f"  Layer: {sae_config['layer']}")
    print(f"  Path: {sae_config['path']}")
    print(f"{'='*60}")

    try:
        # Load model and SAE
        print("Loading model...")
        model, tokenizer = load_model(
            argparse.Namespace(
                checkpoint=EVA_CHECKPOINT,
                eva_root="/data/yanjie_huang/enzyme1_server/eva/EVA1",
                model_code_path=None,
                device=device,
                extra_site_packages=""
            )
        )

        print("Loading SAE...")
        sae = load_sae(sae_config["path"], "auto", device)
        layer = sae_config["layer"]

        # Build prompt
        break_seq = case["break_sequence"]
        rescue_pos = case["rescue_pos"]
        prefix = break_seq[:rescue_pos]
        target = case["rescue_target"]
        suffix = break_seq[rescue_pos + 1:]

        args = argparse.Namespace(
            prefix=prefix,
            target=target,
            suffix=suffix,
            sequence="",
            span_start=None,
            span_length=None,
            max_prefix=0,
            max_suffix=0,
            steer_part="prefix",
            steer_offset=case["break_pos"],
            steer_token_index=None,
            group_mode="joint",
            feature_max_json="",
            device=device,
            layer=layer,
        )

        prompt = build_prompt(tokenizer, prefix, target, suffix, args, device)

        # Get feature max for f/5271 (from previous analysis)
        # In a real run, we'd compute this from activation analysis
        feature_max = {5271: 3.985, 3571: 3.5}  # approximate values

        # Test no-steer baseline
        print("Running no-steer baseline...")
        logits, _ = forward_with_steer(model, prompt, sae, args, None, None, feature_max)
        baseline_score = score_target(logits, prompt)
        baseline_p = baseline_score["target_tokens"][0]["prob"]
        print(f"  No-steer P(rescue_base): {baseline_p:.6f}")

        # Test features at different scales
        results = {"sae_name": sae_name, "layer": layer, "baseline": baseline_p, "features": {}}

        for feature_id in [5271, 3571]:
            print(f"\n  Feature f/{feature_id}:")
            feat_results = {"scales": {}}

            for scale in scales:
                logits, _ = forward_with_steer(
                    model, prompt, sae, args,
                    feature_ids=[feature_id],
                    scale=scale,
                    feature_max=feature_max
                )
                score = score_target(logits, prompt)
                p = score["target_tokens"][0]["prob"]
                delta = p - baseline_p
                feat_results["scales"][f"{scale}x"] = {"p": p, "delta": delta}
                print(f"    {scale}x: P={p:.6f}, delta={delta:+.6f}")

            # Find best scale
            best_scale = max(feat_results["scales"].items(),
                          key=lambda x: x[1]["delta"])[0]
            feat_results["best_scale"] = best_scale
            feat_results["best_delta"] = feat_results["scales"][best_scale]["delta"]
            results["features"][str(feature_id)] = feat_results

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"sae_name": sae_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Quick Multi-SAE Steering Test")
    parser.add_argument("--saes", default="layer13_step200000,layer23_stable_step160000",
                       help="Comma-separated SAE names to test")
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5,10",
                       help="Comma-separated scales")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="multi_sae_comparison_results.json")
    args = parser.parse_args()

    sae_names = args.saes.split(",")
    scales = [float(s) for s in args.scales.split(",")]

    print("="*70)
    print("MULTI-SAE STEERING COMPARISON")
    print("="*70)
    print(f"SAEs to test: {sae_names}")
    print(f"Scales: {scales}")

    all_results = []

    for sae_name in sae_names:
        if sae_name not in SAE_CONFIGS:
            print(f"Unknown SAE: {sae_name}")
            continue

        result = run_steering_test(sae_name, SAE_CONFIGS[sae_name], CASE_3, scales, args.device)
        all_results.append(result)

    # Save results
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for r in all_results:
        if "error" in r:
            print(f"{r['sae_name']}: ERROR - {r['error']}")
        else:
            f5271_delta = r['features']['5271']['best_delta']
            f3571_delta = r['features']['3571']['best_delta']
            print(f"{r['sae_name']} (layer {r['layer']}): f/5271 best delta={f5271_delta:+.6f}, f/3571 best delta={f3571_delta:+.6f}")

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
