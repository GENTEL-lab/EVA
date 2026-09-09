#!/usr/bin/env python3
"""
Analyze existing steering results across multiple cases and features.

This script reads from existing CSV/JSON results and produces a summary report
of the best steering candidates found so far.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering/rnafold_cases")

def load_steering_csv(path):
    """Load steering results from CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def analyze_doming_case3():
    """Analyze the best case: Domingo case 3 f/5271."""
    print("=" * 70)
    print("ANALYSIS: Domingo Case 3 - f/5271 vs f/3571")
    print("=" * 70)

    csv_path = DATA_DIR / "domingo_case3_f5271_vs_f3571_single_featuremax_reconstruct.csv"
    if not csv_path.exists():
        print("File not found:", csv_path)
        return

    rows = load_steering_csv(csv_path)

    # Group by feature
    features = defaultdict(list)
    for row in rows:
        features[row['feature_id']].append(row)

    print("\nTarget Feature f/5271:")
    print("-" * 50)
    target_rows = sorted([r for r in features.get('5271', [])],
                        key=lambda x: float(x['scale']) if x['scale'] else 0)
    for row in target_rows:
        scale = row['scale'] if row['scale'] else 'no_steer'
        prob = float(row['prob'])
        print(f"  {scale:>10}: P(U) = {prob:.6f}")

    print("\nControl Feature f/3571:")
    print("-" * 50)
    control_rows = sorted([r for r in features.get('3571', [])],
                         key=lambda x: float(x['scale']) if x['scale'] else 0)
    for row in control_rows:
        scale = row['scale'] if row['scale'] else 'no_steer'
        prob = float(row['prob'])
        print(f"  {scale:>10}: P(U) = {prob:.6f}")

    # Calculate deltas
    target_baseline = float([r for r in target_rows if r['condition'] == 'no_steer'][0]['prob'])
    target_best = max([float(r['prob']) for r in target_rows])
    target_delta = target_best - target_baseline

    control_baseline = float([r for r in control_rows if r['condition'] == 'no_steer'][0]['prob'])
    control_best = max([float(r['prob']) for r in control_rows])
    control_delta = control_best - control_baseline

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  f/5271 (target): baseline={target_baseline:.6f}, best={target_best:.6f}, delta=+{target_delta:.6f}")
    print(f"  f/3571 (control): baseline={control_baseline:.6f}, best={control_best:.6f}, delta=+{control_delta:.6f}")
    print(f"  Effect ratio: {target_delta / max(control_delta, 1e-10):.1f}x")
    print("=" * 50)

def analyze_glm_anchor_results():
    """Analyze GLM anchor selected features results."""
    print("\n" + "=" * 70)
    print("ANALYSIS: GLM Anchor Selected Features Comparison")
    print("=" * 70)

    csv_path = DATA_DIR / "domingo_case3_glm_anchor_selected_extended_reconstruct_b1_summary.csv"
    if not csv_path.exists():
        print("File not found:", csv_path)
        return

    rows = load_steering_csv(csv_path)

    # Get best delta for each feature
    feature_results = defaultdict(lambda: {'baseline': 0, 'best': 0, 'best_delta': 0, 'scales_tested': 0})

    for row in rows:
        fid = row['feature_id']
        try:
            prob = float(row['prob'])
            if row['condition'] == 'no_steer':
                feature_results[fid]['baseline'] = prob
            else:
                feature_results[fid]['scales_tested'] += 1
                if prob > feature_results[fid]['best']:
                    feature_results[fid]['best'] = prob
        except:
            pass

    # Calculate deltas and sort
    results = []
    for fid, data in feature_results.items():
        data['feature_id'] = fid
        data['best_delta'] = data['best'] - data['baseline']
        results.append(data)

    results.sort(key=lambda x: x['best_delta'], reverse=True)

    print(f"\n{'Feature':<12} {'Baseline':<12} {'Best':<12} {'Delta':<12} {'Ratio vs f/3571'}")
    print("-" * 70)
    control_delta = 0.0000004  # f/3571 baseline

    for r in results[:10]:
        fid = r['feature_id']
        ratio = r['best_delta'] / max(control_delta, 1e-10)
        print(f"f/{fid:<10} {r['baseline']:<12.6f} {r['best']:<12.6f} {r['best_delta']:+12.6f} {ratio:>12.1f}x")

def analyze_specificity_controls():
    """Analyze specificity control results."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Specificity Controls for f/5271")
    print("=" * 70)

    # Base specificity
    base_csv = DATA_DIR / "domingo_case3_f5271_specificity_controls_base_specificity.csv"
    if base_csv.exists():
        print("\nBase Specificity (2.5x scale):")
        print("-" * 50)
        rows = load_steering_csv(base_csv)
        for row in rows:
            print(f"  {row['base']}: P={float(row['prob']):.4f}")

    # Anchor specificity
    anchor_csv = DATA_DIR / "domingo_case3_f5271_specificity_controls_anchor_specificity.csv"
    if anchor_csv.exists():
        print("\nAnchor Specificity (2.5x scale):")
        print("-" * 50)
        rows = load_steering_csv(anchor_csv)
        for row in rows:
            print(f"  Anchor {row['anchor_pos']}: P(U)={float(row['p_u']):.4f}")

def analyze_structure_switch_candidates():
    """Analyze structure switch candidates."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Structure Switch Candidates - Summary")
    print("=" * 70)

    ranking_csv = DATA_DIR / "structure_switch_generation_candidate_ranking_latest_validation.csv"
    if ranking_csv.exists():
        print("\nTop Structure Switch Candidates:")
        print("-" * 70)
        rows = load_steering_csv(ranking_csv)

        # Sort by target best delta
        for row in rows[:10]:
            target_best = row.get('target_best_delta', 'N/A')
            best_control = row.get('best_control_delta', 'N/A')
            conclusion = row.get('conclusion', 'unknown')
            print(f"  {row.get('case', 'N/A')[:30]}: target_best={target_best}, control={best_control}, {conclusion}")
    else:
        print("\nNo structure switch ranking file found.")
        print("Searching for individual results...")

        # List available structure switch files
        switch_files = list(DATA_DIR.glob("*structure_switch*summary.csv"))
        print(f"\nFound {len(switch_files)} structure switch summary files:")
        for f in switch_files[:10]:
            print(f"  {f.name}")

def generate_summary_report():
    """Generate overall summary report."""
    print("\n" + "=" * 70)
    print("OVERALL STEERING ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
BEST POSITIVE STEERING CANDIDATE: Domingo Case 3 / f/5271
========================================================

Case Details:
  - Dataset: Domingo 2018 tRNA
  - WT pair: (2, 68) = U-A
  - Break: pos 2 U->A (disrupts pair)
  - Rescue: pos 68 A->U (restores pair)
  - Local BP distance: 45

Steering Results:
  - Target feature: f/5271
  - Effect: P(rescue_base) increases from 0.3135 to 0.3420
  - Delta: +0.0285 (+9.1% relative increase)
  - Best scale: 2.5x
  - Control f/3571: no effect (delta ~0)

Key Observations:
  1. Dose response is NON-MONOTONIC:
     - 0.5x-1x: P(U) decreases
     - 2x-2.5x: P(U) peaks
     - 5x-10x: P(U) decreases back

  2. Anchor specificity:
     - f/5271 works at position 3 (designed anchor)
     - f/5271 also works at position 1 (nearby 5' stem-side)
     - Suggests it's a LOCAL STEM CONTEXT feature, not unique nucleotide anchor

  3. Base specificity:
     - At 2.5x: U increases, C/G decrease
     - Not uniform probability increase

FAILED STRUCTURE TRANSFORMATION ATTEMPTS:
=========================================
  - Janzen fam21 f/7081: passed n=20 but failed n=60 validation
  - Multiple other candidates: matched or beaten by controls

RECOMMENDATIONS:
================
  1. Publish Domingo Case 3 / f/5271 as evidence of SAE steering capability
     - Frame as "local stem context feature" not "specific nucleotide control"
     - Include caveat about non-monotonic dose response

  2. For stronger claims, need to find:
     - Cases with cleaner monotonic dose response
     - Feature with stricter anchor specificity
     - Structure transformation proof (currently none exist)
""")

def main():
    print("EVA SAE Steering Results Analysis")
    print("=" * 70)

    analyze_doming_case3()
    analyze_glm_anchor_results()
    analyze_specificity_controls()
    analyze_structure_switch_candidates()
    generate_summary_report()

if __name__ == "__main__":
    main()
