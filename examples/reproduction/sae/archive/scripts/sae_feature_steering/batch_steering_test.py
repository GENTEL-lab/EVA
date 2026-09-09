#!/usr/bin/env python3
"""
Batch Feature Selection Test on Multiple Datasets.

Tests top features from multiple case datasets to find the best steering candidates.
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
    parser.add_argument("--case-files", required=True, help="Comma-separated case JSON files")
    parser.add_argument("--sae", default="layer13_step200000")
    parser.add_argument("--top-k", type=int, default=10, help="Top k features per case")
    parser.add_argument("--cases-per-file", type=int, default=20, help="Max cases per file")
    parser.add_argument("--scales", default="0,1,2,2.5,5", help="Comma-separated scales")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_probe_for_case(case):
    """Build steering probe for a case."""
    # Handle different case formats
    if "break_sequence" in case:
        break_seq = case["break_sequence"]
        rescue_pos = int(case.get("rescue_pos0", case.get("mut_pos", 0)))
        break_pos = int(case.get("break_pos0", case.get("mut_pos", 0)))
        rescue_target = case.get("rescue_to", case.get("target_base", "A"))
    elif "mutant_sequence" in case:
        # Structure switch case format
        break_seq = case["mutant_sequence"]
        rescue_pos = int(case.get("diagnostic_pos1", case.get("mut_pos", 0)))
        break_pos = int(case.get("mut_pos", 0))
        rescue_target = case.get("rescue_base", case.get("target_base", "A"))
    else:
        return None

    prefix = break_seq[:rescue_pos]
    target = rescue_target
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


def capture_hidden(model, tokenizer, sequence, layer, device):
    """Capture hidden state."""
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


def test_feature(model, tokenizer, sae, case, layer, device, feature_id, scales):
    """Test a single feature on a case."""
    probe = build_probe_for_case(case)
    if probe is None:
        return None

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

    try:
        prompt = build_prompt(tokenizer, probe["prefix"], probe["target"], probe["suffix"], args, device)
    except Exception:
        return None

    # Baseline
    try:
        logits, _ = forward_with_steer(model, prompt, sae, args, None, None, {})
        baseline = score_target(logits, prompt)
        p_baseline = baseline["target_tokens"][0]["prob"]
    except Exception:
        return None

    # Test scales
    best_delta = 0
    best_scale = 0
    all_results = []

    for scale in scales:
        if scale == 0:
            continue
        try:
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
        except Exception:
            continue

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
    print("BATCH STEERING TEST ON MULTIPLE DATASETS")
    print("=" * 70)

    # Load case files
    case_files = args.case_files.split(",")
    all_results = []

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

    print(f"Model loaded. SAE has {sae.encoder_weight.shape[0]} features")

    for case_file in case_files:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(case_file).name}")
        print("=" * 60)

        try:
            with open(case_file) as f:
                cases = json.load(f)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        # Limit cases per file
        cases = cases[:args.cases_per_file]
        print(f"  {len(cases)} cases")

        # Collect top features (simplified: test all features that appear in scan results or use f/5929)
        # For this batch test, we'll use a set of known promising features
        test_features = [5929, 5271, 6778, 6514, 8153, 5313, 252, 6451, 2097, 7081, 2956, 3779]

        file_results = []

        for case_idx, case in enumerate(cases):
            case_id = case.get("record_id", f"case_{case_idx}")
            print(f"\n  Case {case_idx}: {case_id[:30]}")

            # Get hidden state for output score computation
            try:
                seq = case.get("break_sequence") or case.get("mutant_sequence")
                if seq:
                    hidden = capture_hidden(model, tokenizer, seq, layer, args.device)
                else:
                    continue
            except Exception:
                continue

            i_pos = int(case.get("i0", 0)) + 2
            j_pos = int(case.get("j0", 0)) + 2

            # Test each feature
            for fid in test_features:
                # Compute output score proxy
                centered = hidden - sae.bias
                pre_act = F.linear(centered, sae.encoder_weight, sae.encoder_bias)
                acts = F.relu(pre_act)
                decoder_vec = sae.decoder_weight[:, fid]
                decoder_norm = float(torch.norm(decoder_vec).item())
                activation_at_pos = float(acts[[i_pos, j_pos], fid].mean().item())
                output_score = decoder_norm * max(0, activation_at_pos)

                # Test steering
                result = test_feature(model, tokenizer, sae, case, layer, args.device, fid, scales)

                if result and result["best_delta"] > 0.001:
                    case_result = {
                        "case_file": Path(case_file).name,
                        "case_id": case_id,
                        "case_idx": case_idx,
                        "feature_id": fid,
                        "decoder_norm": decoder_norm,
                        "activation_at_pair": activation_at_pos,
                        "output_score": output_score,
                        "baseline": result["baseline"],
                        "best_delta": result["best_delta"],
                        "best_scale": result["best_scale"],
                        "all_results": result["all_results"],
                    }
                    file_results.append(case_result)
                    print(f"    f/{fid}: Δ={result['best_delta']:+.6f} at {result['best_scale']}x")

            all_results.extend(file_results)

    # Save results
    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Rankings
    print("\n" + "=" * 70)
    print("TOP 30 STEERING RESULTS (ALL DATASETS)")
    print("=" * 70)

    sorted_results = sorted(all_results, key=lambda x: x["best_delta"], reverse=True)

    print(f"\n{'Rank':<6} {'Case':<25} {'Feature':<10} {'Delta':<12} {'Scale':<8} {'Dataset':<20}")
    print("-" * 90)
    for i, r in enumerate(sorted_results[:30]):
        print(f"{i+1:<6} {r['case_id'][:22]:<25} f/{r['feature_id']:<8} "
              f"{r['best_delta']:<+12.6f} {r['best_scale']:<8.1f} {r['case_file'][:18]:<20}")

    # Save top results
    with open(out_path.parent / f"{out_path.name}_top30.json", "w") as f:
        json.dump(sorted_results[:30], f, indent=2)

    # Statistics by feature
    print("\n" + "=" * 70)
    print("BEST FEATURE PER DATASET")
    print("=" * 70)

    from collections import defaultdict
    by_file = defaultdict(list)
    for r in all_results:
        by_file[r["case_file"]].append(r)

    for filename, results in sorted(by_file.items(), key=lambda x: max(r["best_delta"] for r in x[1]) if x[1] else 0, reverse=True):
        if not results:
            continue
        best = max(results, key=lambda x: x["best_delta"])
        print(f"\n{filename}:")
        print(f"  Best: f/{best['feature_id']} Δ={best['best_delta']:+.6f} ({best['case_id'][:25]})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total tests: {len(all_results)}")
    print(f"Positive results (Δ>0.001): {sum(1 for r in all_results if r['best_delta'] > 0.001)}")

    if sorted_results:
        best = sorted_results[0]
        print(f"\nBEST OVERALL:")
        print(f"  Case: {best['case_id']}")
        print(f"  Feature: f/{best['feature_id']}")
        print(f"  Delta: {best['best_delta']:+.6f}")
        print(f"  Scale: {best['best_scale']}x")
        print(f"  Dataset: {best['case_file']}")

    print(f"\nResults saved to:")
    print(f"  {out_path.with_suffix('.json')}")
    print(f"  {out_path.parent / f'{out_path.name}_top30.json'}")


if __name__ == "__main__":
    main()
