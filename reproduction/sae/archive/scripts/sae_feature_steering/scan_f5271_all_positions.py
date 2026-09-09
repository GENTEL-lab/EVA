#!/usr/bin/env python3
"""
Multi-Position f/5271 Activation and Steering Scan.

This script tests:
1. f/5271 activation at EVERY position in the tRNA sequence
2. Steering effect of f/5271 from EACH position (clamping at different sites)
3. Identifies which positions are most important for steering
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
    parser.add_argument("--sae", default="layer13_step200000")
    parser.add_argument("--case-index", type=int, default=3, help="Domingo case 3")
    parser.add_argument("--cases-json", default="/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering/rnafold_cases/domingo2018_trna_cases.json")
    parser.add_argument("--scale", type=float, default=2.5, help="Steering scale")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-positions", type=int, default=20, help="Max positions to test for steering")
    return parser.parse_args()


def capture_hidden_at_layer(model, tokenizer, sequence, layer, device):
    """Capture hidden state at specific layer."""
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


def main():
    args = parse_args()

    print("=" * 70)
    print("MULTI-POSITION FEATURE ACTIVATION & STEERING SCAN")
    print("=" * 70)
    print(f"Feature: f/{args.feature_id}")
    print(f"Case: index {args.case_index}")

    # Load case
    with open(args.cases_json) as f:
        cases = json.load(f)
    case = cases[args.case_index]

    print(f"Record: {case['record_id']}")
    print(f"WT sequence: {case['wt_sequence']}")
    print(f"Break sequence: {case['break_sequence']}")
    print(f"Rescue position: {case['rescue_pos0']}, target: {case['rescue_to']}")

    # Identify tRNA structure regions
    seq = case['break_sequence']
    print()
    print("=" * 70)
    print("tRNA STRUCTURE REGIONS")
    print("=" * 70)
    print(f"  Acceptor stem: pos 0-7, 64-71 (5' and 3' ends)")
    print(f"  D-stem:        pos 8-12, 22-26")
    print(f"  D-loop:        pos 13-21")
    print(f"  Anticodon stem: pos 27-31, 39-43")
    print(f"  Anticodon loop: pos 32-38")
    print(f"  Variable loop:  pos 44-48")
    print(f"  T-stem:        pos 49-53, 60-64")
    print(f"  T-loop:        pos 54-60  <-- UUCG motif around pos 52-55")
    print(f"  Break site:    pos {case['break_pos0']}")
    print(f"  Rescue site:   pos {case['rescue_pos0']}")

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

    # Step 1: Compute f/5271 activation at EVERY position
    print(f"\n{'='*70}")
    print(f"STEP 1: f/{args.feature_id} activation at all positions")
    print("=" * 70)

    # Get hidden state
    hidden = capture_hidden_at_layer(model, tokenizer, case["break_sequence"], layer, args.device)

    # SAE encode
    centered = hidden - sae.bias
    pre_act = F.linear(centered, sae.encoder_weight, sae.encoder_bias)
    acts = F.relu(pre_act)

    # Find bos position
    bos_id = token_ids(tokenizer, "<bos>")[0]
    eos_id = token_ids(tokenizer, "<eos>")[0]
    full = f"<bos>5{case['break_sequence']}3<eos>"
    full_ids = token_ids(tokenizer, full)
    seq_start = full_ids.index(bos_id) + 2  # +2 for <bos>5

    # Get activations for f/5271 at all sequence positions
    feature_activations = []
    for i, tok_id in enumerate(full_ids):
        base = ""
        if seq_start <= i < seq_start + len(case['break_sequence']):
            base = case['break_sequence'][i - seq_start]
        feature_activations.append({
            'token_index': i,
            'position': i - seq_start if seq_start <= i < seq_start + len(case['break_sequence']) else None,
            'base': base,
            'activation': float(acts[i, args.feature_id].item())
        })

    # Print top positions
    sorted_pos = sorted([a for a in feature_activations if a['position'] is not None],
                       key=lambda x: x['activation'], reverse=True)

    print(f"\nTop 15 positions by f/{args.feature_id} activation:")
    print(f"{'Rank':<6} {'Pos':<5} {'Base':<6} {'Activation':<12} {'Region'}")
    print("-" * 70)
    for i, a in enumerate(sorted_pos[:15]):
        pos = a['position']
        region = "?"
        if pos is not None:
            if 0 <= pos <= 7 or 64 <= pos <= 71:
                region = "acceptor stem"
            elif 8 <= pos <= 12 or 22 <= pos <= 26:
                region = "D-stem"
            elif 13 <= pos <= 21:
                region = "D-loop"
            elif 27 <= pos <= 31 or 39 <= pos <= 43:
                region = "anticodon stem"
            elif 32 <= pos <= 38:
                region = "anticodon loop"
            elif 44 <= pos <= 48:
                region = "variable loop"
            elif 49 <= pos <= 53 or 60 <= pos <= 64:
                region = "T-stem"
            elif 54 <= pos <= 60:
                region = "T-loop (UUCG?)"
        print(f"{i+1:<6} {pos:<5} {a['base']:<6} {a['activation']:<12.4f} {region}")

    # Find UUCG positions
    print(f"\nUUCG motif positions in this tRNA:")
    for i in range(len(case['break_sequence']) - 3):
        if case['break_sequence'][i:i+4] == 'UUCG':
            print(f"  UUCG at pos {i}: activation={acts[seq_start+i, args.feature_id].item():.4f}")

    # Step 2: Steering from different positions
    print(f"\n{'='*70}")
    print(f"STEP 2: Steering f/{args.feature_id} from different positions")
    print("=" * 70)

    # Test steering at top activation positions
    test_positions = [a['position'] for a in sorted_pos[:args.max_positions] if a['position'] is not None]

    # Always include break position for reference
    if case['break_pos0'] not in test_positions:
        test_positions.append(case['break_pos0'])

    # Setup baseline
    rescue_pos = case['rescue_pos0']
    prefix = case['break_sequence'][:rescue_pos]
    target = case['rescue_to']
    suffix = case['break_sequence'][rescue_pos + 1:]

    feature_max = {args.feature_id: 3.985}

    print(f"\n  Testing steering at {len(test_positions)} positions")
    print(f"  Scale: {args.scale}x, Feature: f/{args.feature_id}")
    print()

    results = []

    for pos in test_positions:
        # Configure args for this position
        # If pos < rescue_pos, it's in prefix
        if pos < rescue_pos:
            steer_part = "prefix"
            steer_offset = pos
        else:
            steer_part = "suffix"
            steer_offset = pos - rescue_pos - 1

        # Use absolute token index for accuracy
        steer_token_index = seq_start + pos

        test_args = argparse.Namespace(
            prefix=prefix,
            target=target,
            suffix=suffix,
            sequence="",
            span_start=None,
            span_length=None,
            max_prefix=0,
            max_suffix=0,
            steer_part="absolute",
            steer_offset=0,
            steer_token_index=steer_token_index,
            group_mode="joint",
            feature_max_json="",
            scale_source="feature_max_json",
            clamp_value=20.0,
            patch_mode="reconstruct",
            device=args.device,
            layer=layer,
        )

        prompt = build_prompt(tokenizer, prefix, target, suffix, test_args, args.device)

        # Get activation at this position
        activation = float(acts[seq_start + pos, args.feature_id].item()) if pos < len(case['break_sequence']) else 0

        # Baseline
        logits, _ = forward_with_steer(model, prompt, sae, test_args, None, None, feature_max)
        baseline = score_target(logits, prompt)
        p_baseline = baseline["target_tokens"][0]["prob"]

        # Steering
        logits, _ = forward_with_steer(
            model, prompt, sae, test_args,
            feature_ids=[args.feature_id],
            scale=args.scale,
            feature_max=feature_max
        )
        score = score_target(logits, prompt)
        p_steered = score["target_tokens"][0]["prob"]
        delta = p_steered - p_baseline

        # Determine region
        region = "?"
        if 0 <= pos <= 7 or 64 <= pos <= 71:
            region = "acceptor stem"
        elif 8 <= pos <= 12 or 22 <= pos <= 26:
            region = "D-stem"
        elif 13 <= pos <= 21:
            region = "D-loop"
        elif 27 <= pos <= 31 or 39 <= pos <= 43:
            region = "anticodon stem"
        elif 32 <= pos <= 38:
            region = "anticodon loop"
        elif 44 <= pos <= 48:
            region = "variable loop"
        elif 49 <= pos <= 53 or 60 <= pos <= 64:
            region = "T-stem"
        elif 54 <= pos <= 60:
            region = "T-loop"

        results.append({
            'position': pos,
            'base': case['break_sequence'][pos] if pos < len(case['break_sequence']) else '?',
            'region': region,
            'activation': activation,
            'baseline': p_baseline,
            'p_steered': p_steered,
            'delta': delta
        })

        print(f"  Pos {pos:3d} ({case['break_sequence'][pos] if pos < len(case['break_sequence']) else '?'}, {region:20s}): "
              f"act={activation:6.3f}, baseline={p_baseline:.4f}, steered={p_steered:.4f}, Δ={delta:+.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    # Sort by delta
    results_by_delta = sorted(results, key=lambda x: x['delta'], reverse=True)

    print(f"\nTop 10 positions by steering effect:")
    print(f"{'Pos':<5} {'Base':<5} {'Region':<22} {'Activation':<12} {'Δ':<10}")
    print("-" * 60)
    for r in results_by_delta[:10]:
        print(f"{r['position']:<5} {r['base']:<5} {r['region']:<22} {r['activation']:<12.4f} {r['delta']:+.4f}")

    # Find best region
    best = results_by_delta[0]
    print(f"\nBest steering position: pos {best['position']} ({best['region']})")
    print(f"  Activation: {best['activation']:.4f}")
    print(f"  Δ: {best['delta']:+.4f}")

    # Check if T-loop positions are among the best
    tloop_results = [r for r in results if r['region'] == 'T-loop' or 'T-stem' in r['region']]
    if tloop_results:
        best_tloop = max(tloop_results, key=lambda x: x['delta'])
        print(f"\nBest T-stem/T-loop position: pos {best_tloop['position']}, Δ={best_tloop['delta']:+.4f}")

    # Save results
    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'case': {
            'record_id': case['record_id'],
            'wt_sequence': case['wt_sequence'],
            'break_sequence': case['break_sequence'],
            'rescue_position': case['rescue_pos0'],
            'break_position': case['break_pos0'],
        },
        'feature_id': args.feature_id,
        'scale': args.scale,
        'all_position_activations': feature_activations,
        'steering_results': results,
        'best_position': best,
    }

    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
