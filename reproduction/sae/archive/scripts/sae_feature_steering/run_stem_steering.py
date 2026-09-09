#!/usr/bin/env python3
"""
Simple RNA stem/hairpin feature steering experiment.
Parallel to InterPLM Fig 6: steer stem feature at position i,
measure P(complement) at paired/masked position.

Prompt structure (GLM):
  <bos_glm>5{stem5}<span_0>{stem3}3<eos><span_0>

We steer at stem5 positions, measure P(expected_complement) at <span_0>.
"""
import sys
sys.path.insert(0, '/eva/scripts/sae_feature_steering')

import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from eva_glm_sae_steering_fig6 import (
    load_model, load_sae, build_prompt, forward_with_steer,
    token_ids, parse_feature_list, parse_scales, SAEWeights
)


COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A"}


def run_one_condition(model, tokenizer, prompt, sae, args,
                      feature_ids, scale, feature_max):
    """Run forward pass with or without steering."""
    logits, hook_info = forward_with_steer(
        model, prompt, sae, args,
        feature_ids if scale is not None else None,
        scale,
        feature_max
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    # Get base probabilities at the target prediction position
    target_start = prompt.target_start
    pred_pos = target_start - 1

    base_probs = {}
    for base, complement in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]:
        tok_ids = token_ids(tokenizer, base)
        if tok_ids:
            tok_id = tok_ids[0]
            base_probs[base] = float(probs[0, pred_pos, tok_id].detach().cpu())

    expected_comp = prompt.target_tokens[0] if prompt.target_tokens else "X"
    return {
        "expected_complement": expected_comp,
        "complement_prob": base_probs.get(expected_comp, 0.0),
        "all_base_probs": base_probs,
        "hook_info": hook_info,
    }


def build_hairpin_stem_probe(tokenizer, stem5, stem3, loop, steer_char_idx, args, device):
    """
    Build a probe for steering on stem5, predicting complement in stem3.

    Structure: 5'-{stem5}-{loop}-{stem3}-3'
    Steer at position steer_char_idx in stem5
    Predict complement at corresponding position in stem3

    GLM prompt:
      <bos_glm>5{prefix}<span_0>{suffix}3<eos><span_0>

    The target character is placed AFTER the span close, and the suffix
    contains the remaining stem3. The model predicts the target token
    at position (target_start - 1) = len(prompt_ids) - 1.
    """
    # The expected complement at the paired position
    steer_base = stem5[steer_char_idx]
    expected_comp = COMPLEMENTS.get(steer_base, "?")

    # Find the paired position in stem3 (antiparallel)
    paired_stem3_idx = len(stem5) - 1 - steer_char_idx

    # Build prefix: FULL stem5 + loop
    prefix = stem5 + loop

    # The paired position becomes the TARGET (first token after span close)
    # The remaining stem3 chars form the suffix (context in the span)
    target = expected_comp
    suffix = stem3[paired_stem3_idx + 1:]  # stem3 after the paired position

    # steer_offset: from end of prefix, point to stem5[steer_char_idx]
    steer_offset = -(len(loop) + len(stem5) - steer_char_idx)

    # Build the prompt
    prompt_args = argparse.Namespace(
        prefix=prefix,
        target=target,
        suffix=suffix,
        steer_part="prefix",
        steer_token_index=None,
        steer_offset=steer_offset,
        device=device,
    )

    prompt = build_prompt(tokenizer, prefix, target, suffix, prompt_args, device)

    return {
        "stem5": stem5,
        "loop": loop,
        "stem3": stem3,
        "steer_char_idx": steer_char_idx,
        "steer_base": steer_base,
        "expected_comp": expected_comp,
        "paired_stem3_idx": paired_stem3_idx,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/eva/checkpoint/glm")
    parser.add_argument("--sae", default="/data/yanjie_huang/rna_benchmark/interpretability/DSSR_sequences/sae_evo2_online_1400M/sae_final.pt")
    parser.add_argument("--eva-root", default="/eva")
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--stem-features", default="507")
    parser.add_argument("--token-features", default="135")
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    parser.add_argument("--num-probes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="/data/yanjie_huang/eva/EVA1/data/sae_feature_steering/stem_steering/result")
    parser.add_argument("--patch-mode", default="delta", choices=["delta", "reconstruct"])
    parser.add_argument("--scale-source", default="feature_max", choices=["current", "constant", "feature_max"])
    parser.add_argument("--clamp-value", type=float, default=1.0)
    parser.add_argument("--feature-max-json", default=None)
    args = parser.parse_args()

    # Default feature_max_json if not provided
    if args.feature_max_json is None:
        import os
        args.feature_max_json = os.path.dirname(args.out_prefix) + "/feature_max.json"

    print("Loading model and SAE...")
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, "batch_topk", args.device)

    stem_features = parse_feature_list(args.stem_features)
    token_features = parse_feature_list(args.token_features)
    scales = parse_scales(args.scales)

    # Generate diverse hairpin probes
    rng = random.Random(args.seed)
    bases = ["A", "U", "G", "C"]

    probes = []
    seen = set()
    attempts = 0
    while len(probes) < args.num_probes and attempts < args.num_probes * 5:
        attempts += 1
        stem5 = "".join(rng.choices(bases, k=4))
        if stem5 in seen:
            continue
        seen.add(stem5)
        stem3 = "".join(COMPLEMENTS[b] for b in stem5)
        loop = "".join(rng.choices(bases, k=4))

        # Try steering at different positions
        for steer_idx in range(4):
            try:
                probe_info = build_hairpin_stem_probe(
                    tokenizer, stem5, stem3, loop, steer_idx, args, args.device
                )
                probes.append(probe_info)
                if len(probes) >= args.num_probes:
                    break
            except Exception as e:
                print(f"  Warning: failed probe: {e}")
                continue

    print(f"Generated {len(probes)} probes")

    # Compute feature maxima from all probes using pre-activation values (not topk)
    feature_max = {}
    print("Computing feature maxima (pre-activation, not topk)...")
    for pi, probe_info in enumerate(probes):
        prompt = probe_info["prompt"]
        cap = [None]
        def hook(mod, inp, out):
            cap[0] = (out[0] if isinstance(out, tuple) else out)
        h = model.model.layers[args.layer].register_forward_hook(hook)
        with torch.inference_mode():
            model(input_ids=prompt.input_ids,
                 position_ids=prompt.position_ids,
                 sequence_ids=prompt.sequence_ids)
        h.remove()
        hidden = cap[0][0, prompt.steer_token_index, :].float()

        # Use pre-activation (before topk) for feature maxima
        pre = torch.nn.functional.linear(hidden - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())
        pre_cpu = pre.detach().cpu()

        for f in stem_features + token_features:
            if f < 1024:
                feature_max[f] = max(feature_max.get(f, 0), float(pre_cpu[f]))
        torch.cuda.empty_cache()

    for k in feature_max:
        feature_max[k] = max(feature_max[k], 1e-6)
    print(f"Feature maxima (pre-act): {feature_max}")
    print(f"  f/507 max pre-act: {feature_max.get(507, 'N/A')}")
    print(f"  f/135 max pre-act: {feature_max.get(135, 'N/A')}")

    # Run steering experiments
    results = []

    conditions = [("no_steer", None, None)]
    for s in scales:
        conditions.append((f"{s:g}x", stem_features, s))

    # Skip diagnostic, go straight to experiments

    # Quick sanity check: verify prediction position and complement probs
    print("\n=== Sanity check (first 3 probes) ===")
    for pi in range(min(3, len(probes))):
        probe_info = probes[pi]
        prompt = probe_info["prompt"]
        logits_baseline, _ = forward_with_steer(model, prompt, sae, args, None, None, feature_max)
        log_probs = F.log_softmax(logits_baseline.float(), dim=-1)
        target_start = prompt.target_start
        pred_pos = target_start - 1

        # Check actual tokens at key positions
        full_ids = prompt.input_ids[0].tolist()
        pred_tok_id = full_ids[pred_pos]
        steer_tok_id = full_ids[prompt.steer_token_index]

        # Get probs at pred_pos
        base_probs = {}
        for base in ["A", "U", "G", "C"]:
            ids = token_ids(tokenizer, base)
            if ids:
                base_probs[base] = float(torch.exp(log_probs[0, pred_pos, ids[0]]).detach().cpu())

        print(f"  Probe {pi}: stem5={probe_info['stem5']}, steer={probe_info['steer_base']}→{probe_info['expected_comp']}")
        print(f"    steer_pos={prompt.steer_token_index}(tok={steer_tok_id}), pred_pos={pred_pos}(tok={pred_tok_id})")
        print(f"    P(A)={base_probs.get('A',0):.4f} P(U)={base_probs.get('U',0):.4f} P(G)={base_probs.get('G',0):.4f} P(C)={base_probs.get('C',0):.4f}")
        print(f"    expected_comp={probe_info['expected_comp']}, P(expected)={base_probs.get(probe_info['expected_comp'], 0):.4f}")
    print("================================\n")

    # Run steering experiments
    diag_printed = [False]
    for cond_name, feat_ids, scale in conditions:
        for pi, probe_info in enumerate(probes):
            prompt = probe_info["prompt"]

            logits, hook_info = forward_with_steer(
                model, prompt, sae, args,
                feat_ids, scale, feature_max
            )

            # Diagnostic: print hook_info for first few runs
            if not diag_printed[0] and hook_info:
                print(f"  [DIAG] {cond_name}: hook_info = {hook_info}")
                diag_printed[0] = True

            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = torch.exp(log_probs)

            target_start = prompt.target_start
            pred_pos = target_start - 1  # Target token is right after span close (not target_start-1)

            base_probs = {}
            for base in ["A", "U", "G", "C"]:
                tok = token_ids(tokenizer, base)
                if tok:
                    base_probs[base] = float(probs[0, pred_pos, tok[0]].detach().cpu())

            expected = probe_info["expected_comp"]
            comp_prob = base_probs.get(expected, 0.0)

            results.append({
                "condition": cond_name,
                "scale": 0.0 if scale is None else scale,
                "probe_idx": pi,
                "stem5": probe_info["stem5"],
                "steer_base": probe_info["steer_base"],
                "expected_comp": expected,
                "complement_prob": comp_prob,
                "p_A": base_probs.get("A", 0),
                "p_U": base_probs.get("U", 0),
                "p_G": base_probs.get("G", 0),
                "p_C": base_probs.get("C", 0),
                "steer_feature": feat_ids if feat_ids else [],
            })

            if (pi + 1) % 10 == 0:
                print(f"  {cond_name}: {pi+1}/{len(probes)} probes done")

    # Save raw results
    import os
    out_dir = os.path.dirname(args.out_prefix)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({
            "results": results,
            "feature_max": {str(k): v for k, v in feature_max.items()},
            "stem_features": stem_features,
            "token_features": token_features,
            "num_probes": len(probes),
        }, f, indent=2)

    # Aggregate results
    print("\nAggregating results...")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["complement_prob"])

    print("\nComplement probability by condition:")
    for cond in sorted(by_cond.keys(), key=lambda x: (x == "no_steer", float(x.replace("x","")) if "x" in x else 999)):
        vals = by_cond[cond]
        print(f"  {cond:12s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")

    # Plot Fig 6 style
    print("\nGenerating figure...")
    plot_fig6_style(results, args.out_prefix, stem_features, token_features)

    print(f"\nWrote {args.out_prefix}.json")
    print("Done!")


def plot_fig6_style(results, out_prefix, stem_features, token_features):
    """Generate InterPLM Fig 6a/6b style plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Group by condition
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    # Get all conditions and sort
    all_conds = sorted(by_cond.keys(), key=lambda x: (
        x != "no_steer", float(x.replace("x","")) if "x" in x else 999
    ))

    # Aggregate
    cond_means = {}
    cond_stds = {}
    cond_sems = {}
    for cond in all_conds:
        vals = [r["complement_prob"] for r in by_cond[cond]]
        cond_means[cond] = np.mean(vals)
        cond_stds[cond] = np.std(vals)
        cond_sems[cond] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

    # Color scheme (like Fig 6)
    colors = {
        "no_steer": "#404040",
        "0.5x": "#21918C",
        "1x": "#5EC962",
        "1.5x": "#FDE725",
        "2x": "#FF9C19",
        "2.5x": "#E63946",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Stem feature steering (f/507)
    ax = axes[0]
    x_vals = list(range(len(all_conds)))
    y_vals = [cond_means[c] for c in all_conds]
    y_errs = [cond_sems[c] for c in all_conds]
    bar_colors = [colors.get(c, "#999999") for c in all_conds]

    bars = ax.bar(x_vals, y_vals, 0.6, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='black', capsize=5, capthick=1.5, elinewidth=1.5)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(all_conds, fontsize=11)
    ax.set_xlabel("Steering scale (f/507)", fontsize=12)
    ax.set_ylabel(f"P(complementary base)", fontsize=12)
    ax.set_title("A. Stem feature steering (f/507)\nsteer at stem_5, predict complement at stem_3",
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val, err in zip(bars, y_vals, y_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Compare stem vs token feature
    ax2 = axes[1]

    # Get stem and token results separately
    stem_conds = defaultdict(list)
    token_conds = defaultdict(list)
    for r in results:
        feats = r["steer_feature"]
        if feats and feats[0] == stem_features[0]:
            stem_conds[r["condition"]].append(r["complement_prob"])
        elif feats and feats[0] == token_features[0]:
            token_conds[r["condition"]].append(r["complement_prob"])

    x = np.arange(len(all_conds))
    w = 0.35
    stem_vals = [np.mean(stem_conds.get(c, [0])) for c in all_conds]
    stem_errs = [np.std(stem_conds.get(c, [0])) / np.sqrt(len(stem_conds.get(c, [1]))) for c in all_conds]
    token_vals = [np.mean(token_conds.get(c, [0])) for c in all_conds]
    token_errs = [np.std(token_conds.get(c, [0])) / np.sqrt(len(token_conds.get(c, [1]))) for c in all_conds]

    ax2.bar(x - w/2, stem_vals, w, label=f'Stem feature (f/{stem_features[0]})',
            color='#E94F37', alpha=0.85)
    ax2.bar(x + w/2, token_vals, w, label=f'Loop/Token feature (f/{token_features[0]})',
            color='#2E86AB', alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_conds, fontsize=11)
    ax2.set_xlabel("Steering scale", fontsize=12)
    ax2.set_ylabel(f"P(complementary base)", fontsize=12)
    ax2.set_title(f"B. Stem feature vs Token control\nf/{stem_features[0]} vs f/{token_features[0]}",
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    # Add annotation for "no_steer" baseline
    no_steer_val = cond_means.get("no_steer", 0)
    ax2.axhline(no_steer_val, color='gray', linestyle='--', alpha=0.5, label='no_steer baseline')

    plt.tight_layout()
    plt.savefig(out_prefix + "_fig6_style.png", dpi=150, bbox_inches='tight')
    plt.savefig(out_prefix + "_fig6_style.pdf", bbox_inches='tight')
    print(f"Saved figure to {out_prefix}_fig6_style.png/pdf")

    # Also save a summary
    summary = {
        "conditions": all_conds,
        "stem_feature": stem_features[0],
        "token_feature": token_features[0],
        "cond_means": {c: float(cond_means[c]) for c in all_conds},
        "cond_stds": {c: float(cond_stds[c]) for c in all_conds},
    }
    with open(out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
