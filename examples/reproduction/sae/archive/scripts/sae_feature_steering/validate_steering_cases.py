#!/usr/bin/env python3
"""
Validated Steering Case Discovery.

This script:
1. Uses existing batch results
2. For each case, computes background delta distribution (from all features)
3. Identifies features with delta significantly above background (Z-score > 2)
4. Outputs only validated candidates
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1')
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1/scripts/sae_feature_steering')

import torch
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-results-json", required=True, help="Existing batch results")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--min-delta", type=float, default=0.005, help="Minimum delta to consider")
    parser.add_argument("--z-score-threshold", type=float, default=2.0, help="Z-score threshold for significance")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("VALIDATED STEERING CASE DISCOVERY")
    print("=" * 70)

    # Load batch results
    with open(args.batch_results_json) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")

    # Group by case
    case_groups = defaultdict(list)
    for r in results:
        case_id = r['case_id']
        case_groups[case_id].append(r)

    print(f"Total unique cases: {len(case_groups)}")

    # Step 1: For each case, compute background delta distribution
    print(f"\n{'='*70}")
    print("STEP 1: Computing background delta distribution per case")
    print("=" * 70)

    validated_candidates = []

    for case_id, case_results in case_groups.items():
        case_file = case_results[0]['case_file']

        # Compute delta distribution for this case
        deltas = [r['best_delta'] for r in case_results]
        mean_delta = sum(deltas) / len(deltas)
        std_delta = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5

        if std_delta == 0:
            continue

        # Find features with delta significantly above background
        for r in case_results:
            delta = r['best_delta']
            z_score = (delta - mean_delta) / std_delta

            # Only keep if:
            # 1. Delta > min threshold
            # 2. Z-score > threshold (significantly above background)
            if delta > args.min_delta and z_score > args.z_score_threshold:
                validated_candidates.append({
                    **r,
                    'case_mean_delta': mean_delta,
                    'case_std_delta': std_delta,
                    'z_score': z_score,
                })

    print(f"Validated candidates (delta>{args.min_delta}, z-score>{args.z_score_threshold}): {len(validated_candidates)}")

    # Sort by z-score
    validated_candidates.sort(key=lambda x: x['z_score'], reverse=True)

    # Print top candidates
    print(f"\n{'='*70}")
    print("TOP VALIDATED CANDIDATES")
    print("=" * 70)

    print(f"\n{'Rank':<6} {'Case':<25} {'Feature':<10} {'Delta':<12} {'Z-score':<10} {'Dataset':<20}")
    print("-" * 85)

    for i, r in enumerate(validated_candidates[:30]):
        print(f"{i+1:<6} {r['case_id'][:22]:<25} f/{r['feature_id']:<8} "
              f"{r['best_delta']:<+12.6f} {r['z_score']:<10.2f} {r['case_file'][:18]:<20}")

    # Save results
    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(validated_candidates, f, indent=2)

    # Save top candidates separately
    with open(out_path.parent / f"{out_path.name}_top30.json", "w") as f:
        json.dump(validated_candidates[:30], f, indent=2)

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if validated_candidates:
        best = validated_candidates[0]
        print(f"\nBEST VALIDATED CANDIDATE:")
        print(f"  Case: {best['case_id']}")
        print(f"  Feature: f/{best['feature_id']}")
        print(f"  Delta: {best['best_delta']:+.6f}")
        print(f"  Z-score: {best['z_score']:.2f}")
        print(f"  Scale: {best['best_scale']}x")
        print(f"  Dataset: {best['case_file']}")

        # Group by dataset
        by_dataset = defaultdict(list)
        for r in validated_candidates:
            by_dataset[r['case_file']].append(r)

        print(f"\nValidated candidates per dataset:")
        for dataset, candidates in sorted(by_dataset.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {dataset[:30]}: {len(candidates)} candidates")

    print(f"\nResults saved to:")
    print(f"  {out_path.with_suffix('.json')}")
    print(f"  {out_path.parent / f'{out_path.name}_top30.json'}")

    # Additional validation: show cases with multiple validated features
    print(f"\n{'='*70}")
    print("CASES WITH MULTIPLE VALIDATED FEATURES")
    print("=" * 70)

    case_count = defaultdict(list)
    for r in validated_candidates:
        case_count[r['case_id']].append(r)

    multi_feature_cases = {k: v for k, v in case_count.items() if len(v) >= 3}
    for case_id, candidates in sorted(multi_feature_cases.items(), key=lambda x: max(r['z_score'] for r in x[1]), reverse=True)[:10]:
        print(f"\n{case_id[:35]} ({len(candidates)} features):")
        for r in candidates[:5]:
            print(f"    f/{r['feature_id']}: Δ={r['best_delta']:+.6f}, z={r['z_score']:.2f}")


if __name__ == "__main__":
    main()
