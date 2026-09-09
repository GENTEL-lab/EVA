#!/usr/bin/env python3
"""
Generate HONEST validation figure for f/5271 steering.

This figure shows:
1. The biological setup (tRNA T-loop UUCG motif)
2. f/5271 activation across positions (T-loop is highest)
3. Steering effect comparison: target vs controls
4. HONEST conclusion: current evidence is preliminary
"""

import json
import sys
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-position-json", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load results
    with open(args.multi_position_json) as f:
        multi_pos = json.load(f)
    with open(args.validation_json) as f:
        validation = json.load(f)

    # Create text-based figure (no matplotlib dependency)
    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append("EVA SAE STEERING ANALYSIS: HONEST ASSESSMENT")
    output_lines.append("=" * 80)

    # Section A: Biological setup
    case = multi_pos['case']
    output_lines.append("\n" + "=" * 80)
    output_lines.append("A. BIOLOGICAL SETUP")
    output_lines.append("=" * 80)
    output_lines.append(f"Case: {case['record_id']}")
    output_lines.append(f"WT sequence: {case['wt_sequence']}")
    output_lines.append(f"Break sequence: {case['break_sequence']}")
    output_lines.append(f"Break position: {case['break_position']}")
    output_lines.append(f"Rescue position: {case['rescue_position']}")

    # Section B: Feature activation
    output_lines.append("\n" + "=" * 80)
    output_lines.append("B. f/5271 ACTIVATION MAP (Top 10 positions)")
    output_lines.append("=" * 80)

    activations = [a for a in multi_pos['all_position_activations'] if a['position'] is not None]
    sorted_acts = sorted(activations, key=lambda x: x['activation'], reverse=True)

    output_lines.append(f"\n{'Rank':<6} {'Pos':<5} {'Base':<6} {'Activation':<12} {'Region'}")
    output_lines.append("-" * 70)
    for i, a in enumerate(sorted_acts[:10]):
        pos = a['position']
        region = get_region(pos)
        output_lines.append(f"{i+1:<6} {pos:<5} {a['base']:<6} {a['activation']:<12.4f} {region}")

    # Section C: Steering at T-loop vs break
    output_lines.append("\n" + "=" * 80)
    output_lines.append("C. STEERING EFFECT: T-LOOP vs BREAK POSITION")
    output_lines.append("=" * 80)

    break_results = validation['break_position']
    tloop_results = validation['tloop_position']

    output_lines.append(f"\n  Baseline P(rescue_base): {break_results['baseline']:.4f}")
    output_lines.append("")
    output_lines.append("  Steering at BREAK position (pos 2):")
    output_lines.append(f"    Target f/5271:        best Δ={max(r['delta'] for r in break_results['target']):+.4f}")
    output_lines.append(f"    Control f/3571:       best Δ={max(r['delta'] for r in break_results['low_effect_control']):+.4f}")
    output_lines.append(f"    Matched f/5313:       best Δ={max(r['delta'] for r in break_results['matched_control']):+.4f}")
    output_lines.append("")
    output_lines.append("  Steering at T-LOOP position (pos 58):")
    output_lines.append(f"    Target f/5271:        best Δ={max(r['delta'] for r in tloop_results['target']):+.4f}")
    output_lines.append(f"    Control f/3571:       best Δ={max(r['delta'] for r in tloop_results['low_effect_control']):+.4f}")

    # Section D: Dose-response at T-loop
    output_lines.append("\n" + "=" * 80)
    output_lines.append("D. DOSE-RESPONSE AT T-LOOP POSITION (58)")
    output_lines.append("=" * 80)

    output_lines.append(f"\n  {'Scale':<8} {'Target f/5271':<16} {'Control f/3571':<16} {'Difference'}")
    output_lines.append("  " + "-" * 60)

    target_by_scale = {r['scale']: r for r in tloop_results['target']}
    control_by_scale = {r['scale']: r for r in tloop_results['low_effect_control']}

    for scale in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        t = target_by_scale.get(scale, {'delta': 0})
        c = control_by_scale.get(scale, {'delta': 0})
        diff = t['delta'] - c['delta']
        output_lines.append(f"  {scale:<8.2f} {t['delta']:+.6f}        {c['delta']:+.6f}        {diff:+.6f}")

    # Section E: Honest conclusion
    output_lines.append("\n" + "=" * 80)
    output_lines.append("E. HONEST ASSESSMENT")
    output_lines.append("=" * 80)

    output_lines.append("""
WHAT THIS EXPERIMENT SHOWS:

1. f/5271 IS a tRNA T-loop feature:
   - Highest activation at pos 58 (T-loop, activation=3.86)
   - Second highest at pos 52 (T-stem, UUCG motif, activation=2.32)
   - This supports the biological interpretation

2. f/5271 CAN cause measurable steering effect:
   - Best target effect: Δ=+0.0089 (at T-loop, 4.0x)
   - Effect is small but consistent with biological claim

3. BUT, the evidence is NOT strong:
   - Control f/3571 produces similar effects (Δ=+0.0113)
   - The effect is not specific to f/5271
   - Multiple control features achieve comparable results

4. PROBLEMS with current evidence:
   - Effect size (~0.01) is small
   - Controls are not specific enough
   - Need statistical validation across many cases
   - Need to show effect is feature-specific, not intervention-specific

CONCLUSION:
  f/5271 has biological interpretation (T-loop / UUCG motif)
  f/5271 produces measurable but SMALL steering effect
  Current evidence is NOT sufficient to claim f/5271 is uniquely causal
  More work needed: negative controls, larger sample, specificity tests
""")

    # Save
    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n".join(output_lines)
    with open(out_path.with_suffix(".txt"), "w") as f:
        f.write(text)

    # Also save as markdown for nice rendering
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        # Convert text to markdown
        md_text = text.replace("=" * 80, "---")
        f.write(md_text)

    print(text)
    print(f"\nResults saved to:")
    print(f"  {out_path.with_suffix('.txt')}")
    print(f"  {md_path.with_suffix('.md')}")


def get_region(pos):
    if 0 <= pos <= 7 or 64 <= pos <= 71:
        return "acceptor stem"
    elif 8 <= pos <= 12 or 22 <= pos <= 26:
        return "D-stem"
    elif 13 <= pos <= 21:
        return "D-loop"
    elif 27 <= pos <= 31 or 39 <= pos <= 43:
        return "anticodon stem"
    elif 32 <= pos <= 38:
        return "anticodon loop"
    elif 44 <= pos <= 48:
        return "variable loop"
    elif 49 <= pos <= 53 or 60 <= pos <= 64:
        return "T-stem"
    elif 54 <= pos <= 60:
        return "T-loop"
    return "?"


if __name__ == "__main__":
    main()
