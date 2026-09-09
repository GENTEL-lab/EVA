#!/usr/bin/env python3
"""
Direct MLM steering: predict stem3 complement given stem5+loop context.
Uses the full hairpin sequence without GLM spans.

Full sequence: <bos>5{stem5}{loop}{stem3}3<eos>
We steer at stem5 positions, measure P(complement) at stem3 positions.
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
    load_model, load_sae, forward_with_steer,
    token_ids, parse_feature_list, parse_scales, SAEWeights
)


# RNA complement: A↔U, G↔C (for RNA sequences with U instead of T)
COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}

# RNA complement rules: A↔U, G↔C
COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}


class SimplePrompt:
    """Simple prompt for full-sequence MLM."""
    def __init__(self, input_ids, position_ids, sequence_ids,
                 steer_token_index, stem3_positions, expected_comps,
                 stem5_positions=None):
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.sequence_ids = sequence_ids
        self.steer_token_index = steer_token_index
        self.stem3_positions = stem3_positions
        self.expected_comps = expected_comps
        self.stem5_positions = stem5_positions


def build_hairpin_mlm_probe(tokenizer, stem5, stem3, loop, steer_char_idx, device):
    """
    Build a probe for direct MLM: steer at stem5, predict stem3.

    Full sequence: <bos>5{stem5}{loop}{stem3}3<eos>
    Token positions:
      0: <bos>, 1: 5
      2..2+len(stem5)-1: stem5
      2+len(stem5)..2+len(stem5)+len(loop)-1: loop
      2+len(stem5)+len(loop)..end-2: stem3
      end-1: 3, end: <eos>
    """
    bos_token = token_ids(tokenizer, "<bos>")[0]
    seq = stem5 + loop + stem3
    full = f"<bos>5{seq}3<eos>"
    ids = token_ids(tokenizer, full)

    bos_pos = ids.index(bos_token)
    stem5_start = bos_pos + 2  # after <bos> and '5'
    stem5_end = stem5_start + len(stem5)
    loop_start = stem5_end
    loop_end = loop_start + len(loop)
    stem3_start = loop_end
    stem3_end = len(ids) - 2  # before '3' and <eos>

    stem5_positions = list(range(stem5_start, stem5_end))
    stem3_positions = list(range(stem3_start, stem3_end))

    # Expected complement at each stem3 position
    # Stem3 is generated using RNA rules: stem3[i] = COMPLEMENTS[stem5[i]]
    # So stem3[i] pairs with stem5[i]. The model predicts COMPLEMENTS[stem3[i]].
    expected_comps = []
    for i in range(len(stem3)):
        if i < len(stem5):
            pair_base = stem5[i]  # stem3[i] pairs with stem5[i]
        else:
            pair_base = stem5[0]
        expected_comps.append(COMPLEMENTS.get(pair_base, "?"))

    # Steer position: stem3[steer_char_idx] pairs with stem5[steer_char_idx]
    steer_pos = stem5_start + steer_char_idx

    prompt = SimplePrompt(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        position_ids=torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0),
        sequence_ids=torch.zeros((1, len(ids)), dtype=torch.long, device=device),
        steer_token_index=steer_pos,
        stem3_positions=stem3_positions,
        expected_comps=expected_comps,
        stem5_positions=stem5_positions,
    )
    return prompt, {
        "stem5": stem5, "loop": loop, "stem3": stem3,
        "steer_char_idx": steer_char_idx, "steer_base": stem5[steer_char_idx],
        "expected_comps": expected_comps,
        "stem5_positions": stem5_positions,
        "stem3_positions": stem3_positions,
        "steer_pos": steer_pos,
    }


def forward_mlm_steer(model, prompt, sae, args, feature_ids, scale, feature_max):
    """Run forward with steering at the steer position."""
    hook_info = {}

    def hook(_module, _inp, out):
        nonlocal hook_info
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, prompt.steer_token_index, :].float()

        # Apply steering using the steer_vector approach
        if feature_ids and scale is not None:
            from eva_glm_sae_steering_fig6 import steer_vector
            x_new, info = steer_vector(x, sae, feature_ids, float(scale), args, feature_max)
            info["hidden_diff"] = float((x_new - x).abs().mean().item())
            patched[0, prompt.steer_token_index, :] = x_new.to(dtype=patched.dtype)
            hook_info.update(info)
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = None
    if feature_ids is not None and scale is not None:
        handle = model.model.layers[args.layer].register_forward_hook(hook)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else None
    with torch.inference_mode(), (autocast_ctx if autocast_ctx else torch.inference_mode()):
        outputs = model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    if handle is not None:
        handle.remove()
    return outputs.logits.float(), hook_info


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
    args = parser.parse_args()

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
        stem3 = "".join(COMPLEMENTS[b] for b in stem5)  # RNA complement rules
        loop = "".join(rng.choices(bases, k=4))

        for steer_idx in range(4):
            try:
                prompt, info = build_hairpin_mlm_probe(
                    tokenizer, stem5, stem3, loop, steer_idx, args.device
                )
                probes.append((prompt, info))
                if len(probes) >= args.num_probes:
                    break
            except Exception as e:
                print(f"  Warning: failed probe: {e}")
                continue

    print(f"Generated {len(probes)} probes")

    # Verify probe structure with diagnostic
    print("\n=== Probe structure diagnostic ===")
    p0, i0 = probes[0]
    ids = p0.input_ids[0].tolist()
    print(f"  Full sequence IDs: {ids}")
    print(f"  Steer position: {p0.steer_token_index} (token={ids[p0.steer_token_index]})")
    print(f"  Stem3 positions: {p0.stem3_positions}")
    print(f"  Expected comps: {p0.expected_comps}")
    print(f"  Stem5: {i0['stem5']}, Stem3: {i0['stem3']}, Loop: {i0['loop']}")

    # Print token at each position
    base_names = {}
    for base in ['A', 'U', 'G', 'C']:
        tid = token_ids(tokenizer, base)
        if tid:
            base_names[tid[0]] = base
    special_names = {1: '<bos>', 86: '5', 87: '3', 2: '<eos>'}
    print("  Token sequence:")
    for i, tid in enumerate(ids):
        name = special_names.get(tid, base_names.get(tid, f'id{tid}'))
        marker = ''
        if i == p0.steer_token_index:
            marker = ' [STEER]'
        if i in p0.stem3_positions:
            marker += ' [stem3]'
        print(f"    pos {i:2d}: tok={tid:3d} {name:6s}{marker}")
    print()

    # Compute feature maxima
    feature_max = {}
    print("Computing feature maxima (pre-activation)...")
    for pi, (prompt, info) in enumerate(probes):
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
        pre = torch.nn.functional.linear(hidden - sae.bias.float(),
                                          sae.encoder_weight.float(),
                                          sae.encoder_bias.float())
        for f in stem_features + token_features:
            if f < 1024:
                feature_max[f] = max(feature_max.get(f, 0), float(pre[f].item()))
        torch.cuda.empty_cache()

    for k in feature_max:
        feature_max[k] = max(feature_max[k], 1e-6)
    print(f"Feature maxima: {feature_max}")
    print()

    # Run steering experiments
    results = []
    conditions = [("no_steer", None, None)]
    for s in scales:
        conditions.append((f"{s:g}x", stem_features, s))

    print("Running steering experiments...")
    diag_printed = [False]
    for cond_name, feat_ids, scale in conditions:
        for pi, (prompt, info) in enumerate(probes):
            if feat_ids is not None and scale is not None:
                logits, hook_info = forward_mlm_steer(
                    model, prompt, sae, args, feat_ids, scale, feature_max
                )
            else:
                logits, hook_info = forward_mlm_steer(
                    model, prompt, sae, args, None, None, feature_max
                )

            if not diag_printed[0] and hook_info:
                print(f"  [DIAG] {cond_name}: {hook_info}")
                diag_printed[0] = True

            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get base probabilities at stem3 positions
            base_probs_at_stem3 = {}
            for base in ["A", "U", "G", "C"]:
                tid = token_ids(tokenizer, base)
                if tid:
                    tid = tid[0]
                    vals = [float(probs[0, pos, tid].item()) for pos in prompt.stem3_positions]
                    base_probs_at_stem3[base] = vals

            # Average complement probability across stem3 positions
            comp_probs = []
            for j, (pos, exp_comp) in enumerate(zip(prompt.stem3_positions, prompt.expected_comps)):
                comp_p = base_probs_at_stem3.get(exp_comp, [0])[j] if j < len(base_probs_at_stem3.get(exp_comp, [])) else 0
                comp_probs.append(comp_p)

            avg_comp_prob = np.mean(comp_probs) if comp_probs else 0

            results.append({
                "condition": cond_name,
                "scale": 0.0 if scale is None else scale,
                "probe_idx": pi,
                "stem5": info["stem5"],
                "loop": info["loop"],
                "stem3": info["stem3"],
                "steer_char_idx": info["steer_char_idx"],
                "steer_base": info["steer_base"],
                "expected_comps": prompt.expected_comps,
                "avg_complement_prob": avg_comp_prob,
                "stem3_probs": {b: base_probs_at_stem3.get(b, [0]) for b in ["A", "U", "G", "C"]},
                "steer_pos": info["steer_pos"],
                "stem3_positions": prompt.stem3_positions,
                "steer_feature": feat_ids if feat_ids else [],
            })

            if (pi + 1) % 20 == 0:
                print(f"  {cond_name}: {pi+1}/{len(probes)} probes done")

    # Save results
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
    print("\nComplement probability by condition:")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["avg_complement_prob"])

    for cond in sorted(by_cond.keys(), key=lambda x: (x != "no_steer", float(x.replace("x","")) if "x" in x else 999)):
        vals = by_cond[cond]
        print(f"  {cond:12s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")

    # Plot
    print("\nGenerating figure...")
    plot_results(results, args.out_prefix, stem_features, token_features)
    print(f"\nWrote {args.out_prefix}.json and figures")
    print("Done!")


def plot_results(results, out_prefix, stem_features, token_features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    all_conds = sorted(by_cond.keys(), key=lambda x: (
        x != "no_steer", float(x.replace("x","")) if "x" in x else 999
    ))

    cond_means = {}
    cond_stds = {}
    cond_sems = {}
    for cond in all_conds:
        vals = [r["avg_complement_prob"] for r in by_cond[cond]]
        cond_means[cond] = np.mean(vals)
        cond_stds[cond] = np.std(vals)
        cond_sems[cond] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

    colors = {
        "no_steer": "#404040",
        "0.5x": "#21918C",
        "1x": "#5EC962",
        "1.5x": "#FDE725",
        "2x": "#FF9C19",
        "2.5x": "#E63946",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Main result
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
    ax.set_ylabel("P(complementary base) at stem3", fontsize=12)
    ax.set_title(f"A. Stem feature steering (f/{stem_features[0]})\nsteer at stem5, predict complement at stem3",
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    for bar, val, err in zip(bars, y_vals, y_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Compare stem vs token
    ax2 = axes[1]
    stem_conds = defaultdict(list)
    token_conds = defaultdict(list)
    for r in results:
        feats = r["steer_feature"]
        if feats and feats[0] == stem_features[0]:
            stem_conds[r["condition"]].append(r["avg_complement_prob"])
        elif feats and feats[0] == token_features[0]:
            token_conds[r["condition"]].append(r["avg_complement_prob"])

    x = np.arange(len(all_conds))
    w = 0.35
    stem_vals = [np.mean(stem_conds.get(c, [0])) for c in all_conds]
    stem_errs = [np.std(stem_conds.get(c, [0])) / np.sqrt(len(stem_conds.get(c, [1]))) for c in all_conds]
    token_vals = [np.mean(token_conds.get(c, [0])) for c in all_conds]
    token_errs = [np.std(token_conds.get(c, [0])) / np.sqrt(len(token_conds.get(c, [1]))) for c in all_conds]

    ax2.bar(x - w/2, stem_vals, w, label=f'Stem feature (f/{stem_features[0]})',
            color='#E94F37', alpha=0.85)
    ax2.bar(x + w/2, token_vals, w, label=f'Loop feature (f/{token_features[0]})',
            color='#2E86AB', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_conds, fontsize=11)
    ax2.set_xlabel("Steering scale", fontsize=12)
    ax2.set_ylabel("P(complementary base) at stem3", fontsize=12)
    ax2.set_title(f"B. Stem feature vs Loop control\nf/{stem_features[0]} vs f/{token_features[0]}",
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    no_steer_val = cond_means.get("no_steer", 0)
    ax2.axhline(no_steer_val, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_prefix + "_mlm_fig6_style.png", dpi=150, bbox_inches='tight')
    plt.savefig(out_prefix + "_mlm_fig6_style.pdf", bbox_inches='tight')
    print(f"Saved figure to {out_prefix}_mlm_fig6_style.png/pdf")

    summary = {
        "conditions": all_conds,
        "stem_feature": stem_features[0],
        "token_feature": token_features[0],
        "cond_means": {c: float(cond_means[c]) for c in all_conds},
        "cond_stds": {c: float(cond_stds[c]) for c in all_conds},
    }
    with open(out_prefix + "_mlm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
