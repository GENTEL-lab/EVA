#!/usr/bin/env python3
"""
SAE feature steering for EVA 244M + L1 Constrained SAE.
Mimics InterPLM Fig. 6: steer at stem5, predict complement at stem3.

Model: EVA 244M (d_in=448, 16 layers)
SAE: L1 Constrained (d_hidden=3584, ReLU, no topk)
Steering: InterPLM style - SAE_recon + (x - SAE_recon_old)

Full sequence: <bos>5{stem5}{loop}{stem3}3<eos>
We steer at stem5 positions, measure P(complement) at stem3 positions.
"""
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path

# ----- paths -----
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (
    load_model, load_sae, token_ids,
    parse_feature_list, parse_scales,
)


# RNA complement: A↔U, G↔C
COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}


class SimplePrompt:
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
    """Build a probe: steer at stem5, predict complement at stem3."""
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
    # stem3[i] = COMPLEMENTS[stem5[i]], pairs with stem5[i]
    # The model predicts COMPLEMENTS[stem3[i]] = stem5[i]
    expected_comps = [stem5[i] if i < len(stem5) else stem5[0]
                     for i in range(len(stem3))]

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


def steer_interplm_style(x, sae, feature_ids, scale, feature_max):
    """
    InterPLM-style steering:
    x_new = SAE_recon_modified + (x - SAE_recon_old)

    x: (d_in,) hidden state at steer position
    Returns (d_in,) modified hidden state
    """
    # SAE pre-activation (d_hidden,)
    pre = F.linear(x.float() - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())

    # Current sparse activations (ReLU for L1 SAE)
    f_old = F.relu(pre)

    # Reconstruct current x
    x_recon_old = F.linear(f_old.float(), sae.decoder_weight.float(), sae.bias.float())

    # Residual
    residual = x.float() - x_recon_old

    # Modify features
    f_new = f_old.clone()
    feature_info = []
    for fid in feature_ids:
        current = float(pre[fid].detach().cpu())
        one_x = float(feature_max.get(fid, max(abs(current), 1e-6)))
        target = float(scale) * one_x
        f_new[fid] = max(target, 0.0)  # ReLU: can't go negative
        feature_info.append({
            "feature_id": int(fid),
            "current_pre": current,
            "current_f": float(f_old[fid].item()),
            "one_x": one_x,
            "target_f": target,
        })

    # Reconstruct with modified features
    x_recon_new = F.linear(f_new.float(), sae.decoder_weight.float(), sae.bias.float())

    # Add residual back (InterPLM style)
    x_new = x_recon_new + residual

    return x_new, {"features": feature_info, "residual_norm": float(residual.norm().item())}


def forward_mlm_steer(model, prompt, sae, args, feature_ids, scale, feature_max):
    """Run forward with InterPLM-style steering."""
    hook_info = {}

    def hook(_module, _inp, out):
        nonlocal hook_info
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, prompt.steer_token_index, :].float()

        if feature_ids and scale is not None:
            x_new, info = steer_interplm_style(x, sae, feature_ids, scale, feature_max)
            diff = float((x_new - x).abs().mean().item())
            info["hidden_diff"] = diff
            patched[0, prompt.steer_token_index, :] = x_new.to(dtype=patched.dtype)
            hook_info = info
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = None
    if feature_ids is not None and scale is not None:
        handle = model.model.layers[args.layer].register_forward_hook(hook)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        outputs = model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    if handle is not None:
        handle.remove()
    return outputs.logits.float(), hook_info


def scan_features(model, tokenizer, probes, sae, args, n_samples=200):
    """Scan features: compare mean activation at stem5 vs loop positions."""
    rng = random.Random(42)
    stem_features = []
    loop_features = []

    # Collect activations from multiple probes
    all_stem_acts = []
    all_loop_acts = []

    for pi, (prompt, info) in enumerate(rng.choices(probes, k=n_samples)):
        cap = [None]
        def hook(mod, inp, out):
            cap[0] = (out[0] if isinstance(out, tuple) else out)
        h = model.model.layers[args.layer].register_forward_hook(hook)
        with torch.inference_mode():
            model(input_ids=prompt.input_ids,
                 position_ids=prompt.position_ids,
                 sequence_ids=prompt.sequence_ids)
        h.remove()
        hidden = cap[0][0, :, :].float()

        for pos in prompt.stem5_positions:
            x = hidden[pos, :]
            pre = F.linear(x - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())
            all_stem_acts.append(F.relu(pre).cpu())
        for pos in prompt.stem3_positions:
            x = hidden[pos, :]
            pre = F.linear(x - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())
            all_loop_acts.append(F.relu(pre).cpu())
        torch.cuda.empty_cache()

    stem_acts = torch.stack(all_stem_acts)  # (N_stem, d_hidden)
    loop_acts = torch.stack(all_loop_acts)  # (N_loop, d_hidden)

    stem_mean = stem_acts.mean(dim=0)
    loop_mean = loop_acts.mean(dim=0)

    # Find features with high stem activation, low loop
    diff = stem_mean - loop_mean
    top_stem = torch.argsort(diff, descending=True)[:50]
    top_loop = torch.argsort(diff, descending=False)[:50]

    results = {
        "n_samples": n_samples,
        "n_stem_positions": len(all_stem_acts),
        "n_loop_positions": len(all_loop_acts),
        "d_hidden": diff.shape[0],
        "top_stem_features": [(int(i), float(diff[i]), float(stem_mean[i]), float(loop_mean[i])) for i in top_stem],
        "top_loop_features": [(int(i), float(diff[i]), float(stem_mean[i]), float(loop_mean[i])) for i in top_loop],
    }
    return results


def run_steering(model, tokenizer, probes, sae, args,
                  feature_ids, scales, feature_max):
    """Run steering across conditions."""
    results = []

    conditions = [("no_steer", None, None)]
    for s in scales:
        conditions.append((f"{s:g}x", feature_ids, s))

    for cond_name, feat_ids, scale in conditions:
        for pi, (prompt, info) in enumerate(probes):
            if feat_ids is not None and scale is not None:
                logits, hook_info = forward_mlm_steer(model, prompt, sae, args, feat_ids, scale, feature_max)
            else:
                logits, hook_info = forward_mlm_steer(model, prompt, sae, args, None, None, feature_max)

            probs = torch.softmax(logits, dim=-1)

            base_ids = {}
            for base in ['A', 'U', 'G', 'C']:
                t = token_ids(tokenizer, base)
                if t:
                    base_ids[base] = t[0]

            comp_probs = []
            for i, (pos, exp_comp) in enumerate(zip(prompt.stem3_positions, prompt.expected_comps)):
                tid = base_ids.get(exp_comp)
                if tid is not None and pos < probs.shape[1] and tid < probs.shape[2]:
                    p = float(probs[0, pos, tid])
                else:
                    p = 0.0
                comp_probs.append(p)

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
                "avg_complement_prob": avg_comp_prob,
                "stem3_probs": comp_probs,
                "steer_pos": info["steer_pos"],
                "stem3_positions": prompt.stem3_positions,
                "hook_info": hook_info,
            })

            if (pi + 1) % 20 == 0:
                print(f"  {cond_name}: {pi+1}/{len(probes)} probes done")

    return results


def compute_feature_max(model, tokenizer, probes, sae, args):
    """Compute feature maxima from pre-activation values."""
    feature_max = {}
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
        pre = F.linear(hidden - sae.bias.float(), sae.encoder_weight.float(), sae.encoder_bias.float())

        for fid in range(pre.shape[0]):
            v = float(pre[fid].item())
            if v > 0:
                feature_max[fid] = max(feature_max.get(fid, 0), v)
        torch.cuda.empty_cache()

    for k in feature_max:
        feature_max[k] = max(feature_max[k], 1e-6)
    return feature_max


def plot_results(results, out_prefix, top_feature, control_feature, d_hidden):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["avg_complement_prob"])

    all_conds = sorted(by_cond.keys(), key=lambda x: (
        x != "no_steer", float(x.replace("x","")) if "x" in x else 999
    ))

    cond_means = {}
    cond_sems = {}
    for cond in all_conds:
        vals = by_cond[cond]
        cond_means[cond] = np.mean(vals)
        cond_sems[cond] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

    colors = {
        "no_steer": "#404040",
        "-2x": "#21918C",
        "-1x": "#31688E",
        "-0.5x": "#35B779",
        "0x": "#5EC962",
        "0.5x": "#FDE725",
        "1x": "#FF9C19",
        "1.5x": "#FB8861",
        "2x": "#F4445E",
        "2.5x": "#E63946",
        "5x": "#9D1C5E",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Main result with target feature
    ax = axes[0]
    x_vals = list(range(len(all_conds)))
    y_vals = [cond_means[c] for c in all_conds]
    y_errs = [cond_sems[c] for c in all_conds]
    bar_colors = [colors.get(c, "#999999") for c in all_conds]

    bars = ax.bar(x_vals, y_vals, 0.6, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='black', capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(all_conds, fontsize=11)
    ax.set_xlabel("Steering scale", fontsize=12)
    ax.set_ylabel("P(complementary base) at stem3", fontsize=12)
    ax.set_title(f"A. Stem feature steering (f/{top_feature})\nsteer at stem5, predict complement at stem3\nL1 SAE d_hidden={d_hidden}",
                 fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    for bar, val, err in zip(bars, y_vals, y_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel B: Delta vs no_steer
    ax2 = axes[1]
    no_steer_val = cond_means.get("no_steer", 0)
    deltas = [cond_means[c] - no_steer_val for c in all_conds]
    delta_colors = []
    for c, d in zip(all_conds, deltas):
        if c == "no_steer":
            delta_colors.append("#999999")
        elif d > 0:
            delta_colors.append("#2A9D8F")
        else:
            delta_colors.append("#E76F51")

    ax2.bar(x_vals, deltas, 0.6, color=delta_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(all_conds, fontsize=11)
    ax2.set_xlabel("Steering scale", fontsize=12)
    ax2.set_ylabel("Delta P(complement) vs no_steer", fontsize=12)
    ax2.set_title(f"B. Effect size (f/{top_feature} vs control f/{control_feature})\n244M + L1 SAE",
                  fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_prefix + "_244M_L1_steering.png", dpi=150, bbox_inches='tight')
    plt.savefig(out_prefix + "_244M_L1_steering.pdf", bbox_inches='tight')
    print(f"Saved: {out_prefix}_244M_L1_steering.png/pdf")

    return {
        "conditions": all_conds,
        "cond_means": {c: float(cond_means[c]) for c in all_conds},
        "cond_sems": {c: float(cond_sems[c]) for c in all_conds},
        "top_feature": top_feature,
        "control_feature": control_feature,
        "d_hidden": d_hidden,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/yanjie_huang/eva/EVA_checkpoint/244M_1125/checkpoint-14765")
    parser.add_argument("--sae", default="/data/yanjie_huang/rna_benchmark/interpretability/DSSR_sequences/sae_interplm_v2_1M/sae_layer13.pt")
    parser.add_argument("--sae-mode", default="interplm", choices=["auto", "batch_topk", "interplm"])
    parser.add_argument("--eva-root", default="/data/yanjie_huang/eva/EVA1")
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=12)  # 0-indexed: layer 13 in 1-indexed
    parser.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,1.5,2,2.5,5")
    parser.add_argument("--num-probes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--top-features", default="")
    parser.add_argument("--control-features", default="")
    args = parser.parse_args()

    print(f"=== EVA 244M + L1 SAE Steering ===")
    print(f"Model: {args.checkpoint}")
    print(f"SAE: {args.sae}")
    print(f"Layer: {args.layer} (1-indexed: {args.layer+1})")
    print(f"SAE mode: {args.sae_mode}")
    print()

    # Load model and SAE
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    d_hidden = sae.encoder_weight.shape[0]
    print(f"Model loaded. SAE d_hidden={d_hidden}")

    scales = parse_scales(args.scales)

    # Generate hairpin probes
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
        for steer_idx in range(4):
            try:
                prompt, info = build_hairpin_mlm_probe(tokenizer, stem5, stem3, loop, steer_idx, args.device)
                probes.append((prompt, info))
                if len(probes) >= args.num_probes:
                    break
            except Exception as e:
                continue

    print(f"Generated {len(probes)} probes")

    # Feature scan to find stem-selective features
    print("\n=== Feature scan ===")
    scan_results = scan_features(model, tokenizer, probes, sae, args, n_samples=min(200, len(probes)))
    print(f"d_hidden: {scan_results['d_hidden']}")
    print(f"Top stem features (stem5 > loop):")
    for fid, diff, stem_m, loop_m in scan_results["top_stem_features"][:10]:
        print(f"  f/{fid}: diff={diff:.3f} (stem={stem_m:.4f}, loop={loop_m:.4f})")

    # Use top stem feature and a control (low diff)
    if args.top_features:
        top_features = parse_feature_list(args.top_features)
    else:
        top_features = [scan_results["top_stem_features"][0][0]]

    if args.control_features:
        control_features = parse_feature_list(args.control_features)
    else:
        # Pick a control: high activation but low stem-loop selectivity
        # Use the scan results to find a non-stem-selective feature with high activation
        top_stem_ids = {f[0] for f in scan_results["top_stem_features"][:20]}
        top_loop_ids = {f[0] for f in scan_results["top_loop_features"][:20]}
        # Collect all features with their stem+loop activation
        all_feats = []
        for fid, diff_val, stem_m, loop_m in scan_results["top_stem_features"] + scan_results["top_loop_features"]:
            if fid not in top_stem_ids and fid not in top_loop_ids:
                all_feats.append((fid, stem_m + loop_m, abs(diff_val)))
        all_feats.sort(key=lambda x: -x[1])  # Sort by activation desc
        # Pick one with moderate selectivity
        control_candidates = [f for f in all_feats if f[2] < 0.5][:50]
        if control_candidates:
            control_features = [control_candidates[len(control_candidates)//2][0]]
        else:
            control_features = [scan_results["top_stem_features"][20][0]]

    print(f"\nTop feature: f/{top_features[0]}")
    print(f"Control feature: f/{control_features[0]}")

    # Compute feature maxima
    print("\nComputing feature maxima...")
    feature_max = compute_feature_max(model, tokenizer, probes, sae, args)
    print(f"Top feature f/{top_features[0]} max pre-act: {feature_max.get(top_features[0], 'N/A'):.4f}")
    print(f"Control feature f/{control_features[0]} max pre-act: {feature_max.get(control_features[0], 'N/A'):.4f}")

    # Sanity check: verify baseline predictions
    print("\n=== Sanity check ===")
    logits_base, _ = forward_mlm_steer(model, probes[0][0], sae, args, None, None, {})
    probs_base = torch.softmax(logits_base, dim=-1)
    base_ids = {}
    for b in ['A', 'U', 'G', 'C']:
        t = token_ids(tokenizer, b)
        if t:
            base_ids[b] = t[0]
    p0, i0 = probes[0]
    print(f"Probe 0: stem5={i0['stem5']}, stem3={i0['stem3']}, loop={i0['loop']}")
    print(f"  steer_pos={i0['steer_pos']}, stem3_positions={i0['stem3_positions']}")
    print(f"  Expected comps: {i0['expected_comps']}")
    for pos, exp in zip(i0['stem3_positions'], i0['expected_comps']):
        vals = {b: float(probs_base[0, pos, base_ids[b]]) for b in ['A', 'U', 'G', 'C']}
        top_base = max(vals, key=vals.get)
        marker = " <-- TOP" if top_base == exp else ""
        print(f"  pos {pos} (expect={exp}): P(A)={vals['A']:.3f} P(U)={vals['U']:.3f} P(G)={vals['G']:.3f} P(C)={vals['C']:.3f}{marker}")

    # Run steering experiments
    print("\n=== Running steering ===")
    all_results = []

    # Test target feature
    print(f"\nTarget feature f/{top_features[0]}:")
    results = run_steering(model, tokenizer, probes, sae, args, top_features, scales, feature_max)
    for r in results:
        r["feature"] = top_features[0]
        r["feature_type"] = "stem"
    all_results.extend(results)

    # Test control feature
    print(f"\nControl feature f/{control_features[0]}:")
    results = run_steering(model, tokenizer, probes, sae, args, control_features, scales, feature_max)
    for r in results:
        r["feature"] = control_features[0]
        r["feature_type"] = "control"
    all_results.extend(results)

    # Aggregate
    from collections import defaultdict
    by_cond_type = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        key = f"{r['feature_type']}_{r['condition']}"
        by_cond_type[r['feature_type']][r['condition']].append(r['avg_complement_prob'])

    print("\n=== Results ===")
    for feat_type in ["stem", "control"]:
        print(f"\n{feat_type} feature (f/{top_features[0] if feat_type=='stem' else control_features[0]}):")
        for cond in sorted(by_cond_type[feat_type].keys(), key=lambda x: (x != "no_steer", float(x.replace("x","")) if "x" in x else 999)):
            vals = by_cond_type[feat_type][cond]
            print(f"  {cond:10s}: mean={np.mean(vals):.4f} +/- {np.std(vals):.4f} (n={len(vals)})")

    # Save
    import os
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    summary = {
        "top_feature": top_features[0],
        "control_feature": control_features[0],
        "d_hidden": d_hidden,
        "num_probes": len(probes),
        "scan_results": {k: v for k, v in scan_results.items() if k != "detailed"},
        "feature_max": {str(k): float(v) for k, v in feature_max.items()},
    }

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2)
    print(f"\nSaved: {args.out_prefix}.json")

    # Plot
    print("\nGenerating figure...")
    summary_data = plot_results(all_results, args.out_prefix, top_features[0], control_features[0], d_hidden)

    with open(args.out_prefix + "_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
