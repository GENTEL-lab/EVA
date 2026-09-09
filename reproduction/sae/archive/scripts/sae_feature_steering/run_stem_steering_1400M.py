#!/usr/bin/env python3
"""
SAE feature steering for EVA 1400M + L1 Constrained SAE.
Replicates InterPLM Fig. 6: steer at stem5, predict complement at stem3.

Model: EVA 1400M (d_in=1024, 26 layers)
SAE: L1 Constrained (d_hidden=8192, ReLU, no topk)
Steering: InterPLM style - SAE_recon_modified + (x - SAE_recon_old)

Full sequence: <bos>5{stem5}{loop}{stem3}3<eos>
We steer at stem5 positions, measure P(complement) at stem3 positions.

InterPLM method:
1. Train L1 SAE on RNA model
2. Identify features selective for hairpin stems (stem5 >> loop)
3. Steer those features → observe causal control of complement prediction
4. Plot P(complement) vs steering scale
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
from collections import defaultdict

# ----- paths -----
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/data/yanjie_huang/eva/EVA1')

from tools.utils.model.loader import ModelLoader

# RNA complement: A↔U, G↔C
COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}


def load_model(checkpoint_dir, device):
    """Load EVA 1400M model."""
    loader = ModelLoader(checkpoint_dir, model_code_path='/data/yanjie_huang/eva/EVA1/eva')
    model, tokenizer = loader.load(device=device)
    model.eval()
    return model, tokenizer


def load_sae_interplm(sae_path, device):
    """Load L1 constrained SAE (sae_repro_release format)."""
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    # Try different key formats
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "sae" in ckpt:
        state = ckpt["sae"]
    else:
        raise ValueError(f"Unknown SAE format. Keys: {list(ckpt.keys())}")
    return {
        "bias": state["bias"].to(device=device, dtype=torch.float32),
        "encoder_weight": state["encoder.weight"].to(device=device, dtype=torch.float32),
        "encoder_bias": state["encoder.bias"].to(device=device, dtype=torch.float32),
        "decoder_weight": state["decoder.weight"].to(device=device, dtype=torch.float32),
        "d_hidden": state["encoder.weight"].shape[0],
        "d_in": state["encoder.weight"].shape[1],
    }


def token_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


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
    """Build a probe: steer at stem5, predict complement at stem3.

    Full sequence: <bos>5{stem5}{loop}{stem3}3<eos>
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
    """InterPLM-style steering:
    x_new = SAE_recon_modified + (x - SAE_recon_old)

    This preserves the residual (information not captured by SAE),
    only modifying the SAE-reconstructed part.
    """
    # SAE pre-activation (d_hidden,)
    pre = F.linear(x.float() - sae["bias"].float(),
                   sae["encoder_weight"].float(),
                   sae["encoder_bias"].float())

    # Current sparse activations (ReLU for L1 SAE)
    f_old = F.relu(pre)

    # Reconstruct current x
    x_recon_old = F.linear(f_old.float(),
                           sae["decoder_weight"].float(),
                           sae["bias"].float())

    # Residual: information not captured by SAE
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
    x_recon_new = F.linear(f_new.float(),
                            sae["decoder_weight"].float(),
                            sae["bias"].float())

    # Add residual back (InterPLM style - key difference from naive delta)
    x_new = x_recon_new + residual

    return x_new, {"features": feature_info, "residual_norm": float(residual.norm().item())}


def forward_mlm_steer(model, prompt, sae, layer, device, feature_ids, scale, feature_max):
    """Run forward with InterPLM-style steering."""
    hook_info = {}

    def hook(_module, _inp, out):
        nonlocal hook_info
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, prompt.steer_token_index, :].float()

        if feature_ids is not None and scale is not None:
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
        handle = model.model.layers[layer].register_forward_hook(hook)

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
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


def scan_features(model, tokenizer, probes, sae, layer, device, n_samples=500):
    """Scan features: compare mean activation at stem5 vs loop positions.

    InterPLM method: find features that fire more at stem positions than loop positions.
    These are the "hairpin structure" features.
    """
    rng = random.Random(42)

    all_stem5_acts = []  # stem5 = hairpin stem
    all_stem3_acts = []  # stem3 = complementary strand
    all_loop_acts = []    # loop = non-pairing region (control)
    all_random_acts = []  # random positions (baseline)

    for pi, (prompt, info) in enumerate(rng.choices(probes, k=min(n_samples, len(probes)))):
        cap = [None]
        def hook(mod, inp, out):
            cap[0] = (out[0] if isinstance(out, tuple) else out)
        h = model.model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            model(input_ids=prompt.input_ids,
                 position_ids=prompt.position_ids,
                 sequence_ids=prompt.sequence_ids)
        h.remove()
        hidden = cap[0][0, :, :].float()

        for pos in prompt.stem5_positions:
            x = hidden[pos, :]
            pre = F.linear(x - sae["bias"].float(),
                          sae["encoder_weight"].float(),
                          sae["encoder_bias"].float())
            all_stem5_acts.append(F.relu(pre).cpu())

        for pos in prompt.stem3_positions:
            x = hidden[pos, :]
            pre = F.linear(x - sae["bias"].float(),
                          sae["encoder_weight"].float(),
                          sae["encoder_bias"].float())
            all_stem3_acts.append(F.relu(pre).cpu())

        # Loop positions (between stem5 and stem3)
        loop_start = prompt.stem5_positions[-1] + 1
        loop_end = prompt.stem3_positions[0]
        for pos in range(loop_start, loop_end):
            if 0 <= pos < hidden.shape[0]:
                x = hidden[pos, :]
                pre = F.linear(x - sae["bias"].float(),
                              sae["encoder_weight"].float(),
                              sae["encoder_bias"].float())
                all_loop_acts.append(F.relu(pre).cpu())

        torch.cuda.empty_cache()

    # Compute mean activation per feature
    stem5_tensor = torch.stack(all_stem5_acts) if all_stem5_acts else None
    stem3_tensor = torch.stack(all_stem3_acts) if all_stem3_acts else None
    loop_tensor = torch.stack(all_loop_acts) if all_loop_acts else None

    if stem5_tensor is not None:
        stem5_mean = stem5_tensor.mean(dim=0)
        stem3_mean = stem3_tensor.mean(dim=0)
        loop_mean = loop_tensor.mean(dim=0)

        # Key metric: stem selectivity = stem5_mean - loop_mean
        stem_selectivity = stem5_mean - loop_mean

        # Find top stem-selective features (high at stem, low at loop)
        top_stem = torch.argsort(stem_selectivity, descending=True)

        # Find top loop features (high at loop, low at stem)
        top_loop = torch.argsort(stem_selectivity, descending=False)

        # Find features with high activation at BOTH stem positions (paired)
        paired = (stem5_mean + stem3_mean) / 2
        top_paired = torch.argsort(paired, descending=True)

        results = {
            "n_samples": len(all_stem5_acts),
            "n_stem5": len(all_stem5_acts),
            "n_stem3": len(all_stem3_acts),
            "n_loop": len(all_loop_acts),
            "d_hidden": stem5_mean.shape[0],
            "top_stem_selective": [(int(i), float(stem_selectivity[i]),
                                   float(stem5_mean[i]), float(loop_mean[i]),
                                   float(stem3_mean[i]))
                                  for i in top_stem[:30]],
            "top_loop_selective": [(int(i), float(stem_selectivity[i]),
                                   float(stem5_mean[i]), float(loop_mean[i]),
                                   float(stem3_mean[i]))
                                  for i in top_loop[:10]],
            "top_paired": [(int(i), float(paired[i]),
                           float(stem5_mean[i]), float(stem3_mean[i]))
                          for i in top_paired[:10]],
            "stem5_mean": stem5_mean,
            "stem3_mean": stem3_mean,
            "loop_mean": loop_mean,
            "stem_selectivity": stem_selectivity,
        }
    else:
        results = {"d_hidden": 8192, "error": "no samples"}

    return results


def compute_feature_max(model, tokenizer, probes, sae, layer, device, feature_ids):
    """Compute feature maxima (observed max pre-activation) for scaling."""
    feature_max = {}
    for pi, (prompt, info) in enumerate(probes):
        cap = [None]
        def hook(mod, inp, out):
            cap[0] = (out[0] if isinstance(out, tuple) else out)
        h = model.model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            model(input_ids=prompt.input_ids,
                 position_ids=prompt.position_ids,
                 sequence_ids=prompt.sequence_ids)
        h.remove()
        hidden = cap[0][0, prompt.steer_token_index, :].float()
        pre = F.linear(hidden - sae["bias"].float(),
                       sae["encoder_weight"].float(),
                       sae["encoder_bias"].float())

        for fid in feature_ids:
            v = float(pre[fid].item())
            if v > 0:
                feature_max[fid] = max(feature_max.get(fid, 0), v)
        torch.cuda.empty_cache()

    for k in feature_max:
        feature_max[k] = max(feature_max[k], 1e-6)
    return feature_max


def run_steering(model, tokenizer, probes, sae, layer, device,
                  feature_ids, scales, feature_max):
    """Run steering across conditions (scales)."""
    results = []

    conditions = [("no_steer", None, None)]
    for s in scales:
        conditions.append((f"{s:g}x", feature_ids, s))

    for cond_name, feat_ids, scale in conditions:
        for pi, (prompt, info) in enumerate(probes):
            logits, hook_info = forward_mlm_steer(
                model, prompt, sae, layer, device, feat_ids, scale, feature_max
            )
            probs = torch.softmax(logits, dim=-1)

            base_ids = {}
            for base in ['A', 'U', 'G', 'C']:
                t = token_ids(tokenizer, base)
                if t:
                    base_ids[base] = t[0]

            comp_probs = []
            comp_correct = 0
            for i, (pos, exp_comp) in enumerate(zip(prompt.stem3_positions, prompt.expected_comps)):
                tid = base_ids.get(exp_comp)
                if tid is not None and pos < probs.shape[1] and tid < probs.shape[2]:
                    p = float(probs[0, pos, tid])
                    # Is this the top prediction?
                    top_tok = int(probs[0, pos, :].argmax().item())
                    if top_tok == tid:
                        comp_correct += 1
                else:
                    p = 0.0
                comp_probs.append(p)

            avg_comp_prob = np.mean(comp_probs) if comp_probs else 0
            comp_accuracy = comp_correct / len(comp_probs) if comp_probs else 0

            # Per-position probabilities
            pos_probs = {}
            for base in ['A', 'U', 'G', 'C']:
                tid = base_ids.get(base)
                if tid is not None:
                    pos_probs[base] = [float(probs[0, pos, tid]) for pos in prompt.stem3_positions]

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
                "comp_accuracy": comp_accuracy,
                "stem3_probs": comp_probs,
                "pos_probs": pos_probs,
                "steer_pos": info["steer_pos"],
                "stem3_positions": prompt.stem3_positions,
                "hook_info": hook_info,
            })

        if (pi + 1) % 20 == 0:
            print(f"  {cond_name}: {pi+1}/{len(probes)} probes done")

    return results


def plot_fig6_style(all_results, out_prefix, top_feature, control_feature, d_hidden, d_in):
    """Plot InterPLM Fig 6a/6b style figures.

    Fig 6a: P(complement) at stem3 vs steering scale for stem feature
    Fig 6b: Same for control feature + comparison
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Organize results by feature type and condition
    by_type_cond = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        feat_type = r.get("feature_type", "stem")
        by_type_cond[feat_type][r["condition"]].append(r)

    # Get all conditions
    all_conds = sorted(set(r["condition"] for r in all_results),
                      key=lambda x: (x != "no_steer", float(x.replace("x","")) if "x" in x else 999))

    # Compute stats
    cond_stats = {}
    for feat_type in ["stem", "control"]:
        cond_stats[feat_type] = {}
        for cond in all_conds:
            vals = [r["avg_complement_prob"] for r in by_type_cond[feat_type][cond]]
            if vals:
                cond_stats[feat_type][cond] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "sem": np.std(vals) / np.sqrt(len(vals)),
                    "n": len(vals),
                    "acc": np.mean([r["comp_accuracy"] for r in by_type_cond[feat_type][cond]]),
                }

    # Colors
    colors = {
        "no_steer": "#404040",
        "-2x": "#21918C", "-1x": "#31688E", "-0.5x": "#35B779",
        "0.5x": "#5EC962", "1x": "#FDE725", "1.5x": "#FF9C19",
        "2x": "#F4445E", "2.5x": "#E63946", "5x": "#9D1C5E",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Stem feature P(complement) vs scale
    ax = axes[0]
    x_vals = list(range(len(all_conds)))
    y_vals = [cond_stats["stem"].get(c, {}).get("mean", 0) for c in all_conds]
    y_errs = [cond_stats["stem"].get(c, {}).get("sem", 0) for c in all_conds]
    bar_colors = [colors.get(c, "#999999") for c in all_conds]

    bars = ax.bar(x_vals, y_vals, 0.6, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='black',
                 capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(all_conds, fontsize=10, rotation=45)
    ax.set_xlabel("Steering scale", fontsize=12)
    ax.set_ylabel("P(complementary base) at stem3", fontsize=12)
    ax.set_title(f"A. Stem feature f/{top_feature}\nsteer at stem5, predict complement",
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(y_vals) * 1.3 if y_vals else 1.0)

    for bar, val, err in zip(bars, y_vals, y_errs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel B: Delta P(complement) vs no_steer
    ax2 = axes[1]
    no_steer = cond_stats["stem"].get("no_steer", {}).get("mean", 0)
    deltas = [cond_stats["stem"].get(c, {}).get("mean", no_steer) - no_steer for c in all_conds]
    delta_colors = []
    for c, d in zip(all_conds, deltas):
        if c == "no_steer":
            delta_colors.append("#999999")
        elif d > 0:
            delta_colors.append("#2A9D8F")
        else:
            delta_colors.append("#E76F51")

    ax2.bar(x_vals, deltas, 0.6, color=delta_colors, alpha=0.85,
            edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(all_conds, fontsize=10, rotation=45)
    ax2.set_xlabel("Steering scale", fontsize=12)
    ax2.set_ylabel("Δ P(complement) vs no_steer", fontsize=12)
    ax2.set_title(f"B. Effect size (f/{top_feature})", fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    # Panel C: Stem vs Control comparison
    ax3 = axes[2]
    stem_vals = [cond_stats["stem"].get(c, {}).get("mean", 0) for c in all_conds]
    ctrl_vals = [cond_stats["control"].get(c, {}).get("mean", 0) for c in all_conds]
    stem_errs = [cond_stats["stem"].get(c, {}).get("sem", 0) for c in all_conds]
    ctrl_errs = [cond_stats["control"].get(c, {}).get("sem", 0) for c in all_conds]

    w = 0.35
    ax3.bar([x - w/2 for x in x_vals], stem_vals, w, label=f'Stem f/{top_feature}',
             color='#E94F37', alpha=0.85, yerr=stem_errs, capsize=3)
    ax3.bar([x + w/2 for x in x_vals], ctrl_vals, w, label=f'Control f/{control_feature}',
             color='#2E86AB', alpha=0.85, yerr=ctrl_errs, capsize=3)
    ax3.set_xticks(x_vals)
    ax3.set_xticklabels(all_conds, fontsize=10, rotation=45)
    ax3.set_xlabel("Steering scale", fontsize=12)
    ax3.set_ylabel("P(complement) at stem3", fontsize=12)
    ax3.set_title(f"C. Stem vs Control\nd_in={d_in}, d_hidden={d_hidden}",
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle(f"EVA 1400M + L1 SAE Steering (layer 13)\nInterPLM Fig.6 Replication",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_prefix + "_1400M_L1_steering.png", dpi=150, bbox_inches='tight')
    plt.savefig(out_prefix + "_1400M_L1_steering.pdf", bbox_inches='tight')
    print(f"Saved: {out_prefix}_1400M_L1_steering.png/pdf")

    # Also save a summary
    summary = {
        "conditions": all_conds,
        "top_feature": top_feature,
        "control_feature": control_feature,
        "d_hidden": d_hidden,
        "d_in": d_in,
        "stem": {c: float(cond_stats["stem"].get(c, {}).get("mean", 0)) for c in all_conds},
        "stem_err": {c: float(cond_stats["stem"].get(c, {}).get("sem", 0)) for c in all_conds},
        "control": {c: float(cond_stats["control"].get(c, {}).get("mean", 0)) for c in all_conds},
        "control_err": {c: float(cond_stats["control"].get(c, {}).get("sem", 0)) for c in all_conds},
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                       default="/data/yanjie_huang/eva/EVA_checkpoint/1400M_1129/checkpoint_13500")
    parser.add_argument("--sae",
                       default="/data/yanjie_huang/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--layer", type=int, default=13)  # layer 13 (1-indexed)
    parser.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,1.5,2,2.5,5")
    parser.add_argument("--num-probes", type=int, default=100)
    parser.add_argument("--scan-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix",
                       default="/data/yanjie_huang/eva/EVA1/data/sae_feature_steering/stem_steering_1400M/result")
    parser.add_argument("--top-features", default="")
    parser.add_argument("--control-features", default="")
    parser.add_argument("--feature-max-json", default="")
    args = parser.parse_args()

    print("=" * 60)
    print("EVA 1400M + L1 SAE Feature Steering")
    print("=" * 60)
    print(f"Model: {args.checkpoint}")
    print(f"SAE: {args.sae}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print()

    # Load model and SAE
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.device)
    print("Loading SAE...")
    sae = load_sae_interplm(args.sae, args.device)
    print(f"SAE: d_in={sae['d_in']}, d_hidden={sae['d_hidden']}")
    print()

    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]

    # Generate hairpin probes
    print("Generating hairpin probes...")
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
        stem3 = "".join(COMPLEMENTS[b] for b in stem5)  # RNA complement
        loop = "".join(rng.choices(bases, k=4))
        # One probe per stem5 (steer at position 0)
        try:
            prompt, info = build_hairpin_mlm_probe(tokenizer, stem5, stem3, loop, 0, args.device)
            probes.append((prompt, info))
        except Exception as e:
            continue

    print(f"Generated {len(probes)} unique hairpin probes")
    p0, i0 = probes[0]
    print(f"Example: stem5={i0['stem5']}, loop={i0['loop']}, stem3={i0['stem3']}")
    print(f"  steer_pos={i0['steer_pos']}, stem3_positions={i0['stem3_positions']}")
    print(f"  expected_comps={i0['expected_comps']}")
    print()

    # ============================================
    # STEP 1: Feature scan - find stem-selective features
    # ============================================
    print("=" * 60)
    print("STEP 1: Feature Scan")
    print("=" * 60)
    print(f"Comparing stem5 (hairpin stem) vs loop (non-pairing) positions")
    print(f"Scanning {args.scan_samples} samples...")

    scan = scan_features(model, tokenizer, probes, sae, args.layer, args.device,
                         n_samples=args.scan_samples)

    print(f"\nd_hidden: {scan['d_hidden']}")
    print(f"Samples: {scan['n_stem5']} stem5, {scan['n_stem3']} stem3, {scan['n_loop']} loop positions")

    print(f"\nTop 15 STEM-SELECTIVE features (high at stem5, low at loop):")
    print(f"{'Feature':>10} | {'Sel(Δ)':>8} | {'Stem5':>8} | {'Loop':>8} | {'Stem3':>8}")
    print("-" * 60)
    for fid, sel, s5, loop, s3 in scan["top_stem_selective"][:15]:
        print(f"  f/{fid:>5} | {sel:>8.4f} | {s5:>8.4f} | {loop:>8.4f} | {s3:>8.4f}")

    print(f"\nTop PAIRED features (high at both stem5 and stem3):")
    for fid, paired, s5, s3 in scan["top_paired"][:5]:
        print(f"  f/{fid}: paired={paired:.4f} (stem5={s5:.4f}, stem3={s3:.4f})")
    print()

    # Select features
    if args.top_features:
        top_features = [int(x) for x in args.top_features.split(",")]
    else:
        top_features = [scan["top_stem_selective"][0][0]]

    if args.control_features:
        control_features = [int(x) for x in args.control_features.split(",")]
    else:
        # Control: feature with similar activation but low selectivity
        top_stem_ids = {f[0] for f in scan["top_stem_selective"][:20]}
        all_feats = []
        for fid, sel, s5, loop, s3 in scan["top_stem_selective"]:
            if fid not in top_stem_ids and s5 > 0.01:
                all_feats.append((fid, s5, abs(sel)))
        all_feats.sort(key=lambda x: -x[1])
        if len(all_feats) > 5:
            control_features = [all_feats[5][0]]  # pick one with moderate activation
        else:
            control_features = [scan["top_stem_selective"][20][0]]

    print(f"Selected features:")
    print(f"  Top (stem-selective): f/{top_features[0]}")
    fid_top = top_features[0]
    for item in scan["top_stem_selective"]:
        if item[0] == fid_top:
            print(f"    selectivity={item[1]:.4f}, stem5={item[2]:.4f}, loop={item[3]:.4f}")
            break
    print(f"  Control: f/{control_features[0]}")
    fid_ctrl = control_features[0]
    for item in scan["top_stem_selective"]:
        if item[0] == fid_ctrl:
            print(f"    selectivity={item[1]:.4f}, stem5={item[2]:.4f}, loop={item[3]:.4f}")
            break
    print()

    # ============================================
    # STEP 2: Compute feature maxima
    # ============================================
    print("=" * 60)
    print("STEP 2: Compute Feature Maxima")
    print("=" * 60)

    if args.feature_max_json:
        with open(args.feature_max_json) as f:
            feature_max = {int(k): float(v) for k, v in json.load(f).items()}
        print(f"Loaded from {args.feature_max_json}")
    else:
        print("Computing from probes...")
        all_feature_ids = list(range(sae["d_hidden"]))
        feature_max = compute_feature_max(model, tokenizer, probes, sae, args.layer,
                                         args.device, all_feature_ids)

    print(f"  Top feature f/{top_features[0]}: max pre-act = {feature_max.get(top_features[0], 0):.4f}")
    print(f"  Control feature f/{control_features[0]}: max pre-act = {feature_max.get(control_features[0], 0):.4f}")
    print()

    # ============================================
    # STEP 3: Sanity check - baseline predictions
    # ============================================
    print("=" * 60)
    print("STEP 3: Baseline Sanity Check")
    print("=" * 60)

    # Show multiple probes to check baseline
    for pi in range(min(3, len(probes))):
        prompt, info = probes[pi]
        logits_base, _ = forward_mlm_steer(model, prompt, sae, args.layer, args.device,
                                           None, None, feature_max)
        probs_base = torch.softmax(logits_base, dim=-1)

        base_ids = {}
        for base in ['A', 'U', 'G', 'C']:
            t = token_ids(tokenizer, base)
            if t:
                base_ids[base] = t[0]

        print(f"\nProbe {pi}: stem5={info['stem5']}, loop={info['loop']}, stem3={info['stem3']}")
        correct = 0
        for i, (pos, exp) in enumerate(zip(info['stem3_positions'], info['expected_comps'])):
            vals = {b: float(probs_base[0, pos, base_ids[b]]) for b in ['A', 'U', 'G', 'C']}
            top_base = max(vals, key=vals.get)
            marker = " ✓" if top_base == exp else " ✗"
            if top_base == exp:
                correct += 1
            print(f"  pos {pos}: expect={exp}{marker} P(A)={vals['A']:.3f} P(U)={vals['U']:.3f} P(G)={vals['G']:.3f} P(C)={vals['C']:.3f}")
        print(f"  Baseline accuracy: {correct}/{len(info['stem3_positions'])}")
    print()

    # ============================================
    # STEP 4: Run steering experiments
    # ============================================
    print("=" * 60)
    print("STEP 4: Steering Experiments")
    print("=" * 60)

    all_results = []

    # Stem feature
    print(f"\n>>> Stem feature f/{top_features[0]}:")
    results = run_steering(model, tokenizer, probes, sae, args.layer, args.device,
                           top_features, scales, feature_max)
    for r in results:
        r["feature"] = top_features[0]
        r["feature_type"] = "stem"
    all_results.extend(results)

    # Control feature
    print(f"\n>>> Control feature f/{control_features[0]}:")
    results = run_steering(model, tokenizer, probes, sae, args.layer, args.device,
                           control_features, scales, feature_max)
    for r in results:
        r["feature"] = control_features[0]
        r["feature_type"] = "control"
    all_results.extend(results)

    # ============================================
    # STEP 5: Results summary
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 5: Results")
    print("=" * 60)

    by_type_cond = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        by_type_cond[r["feature_type"]][r["condition"]].append(r["avg_complement_prob"])

    print(f"\n{'Condition':<12} | {'Stem f/' + str(top_features[0]):<25} | {'Control f/' + str(control_features[0]):<25}")
    print("-" * 70)
    for cond in sorted(all_results[0]["condition"] if all_results else []):
        pass

    all_conds = sorted(set(r["condition"] for r in all_results),
                       key=lambda x: (x != "no_steer", float(x.replace("x","")) if "x" in x else 999))
    for cond in all_conds:
        s_vals = by_type_cond["stem"][cond]
        c_vals = by_type_cond["control"][cond]
        s_mean = np.mean(s_vals) if s_vals else 0
        s_std = np.std(s_vals) if s_vals else 0
        c_mean = np.mean(c_vals) if c_vals else 0
        c_std = np.std(c_vals) if c_vals else 0
        print(f"  {cond:<10} | {s_mean:.4f} ± {s_std:.4f} (n={len(s_vals):3d}) | {c_mean:.4f} ± {c_std:.4f} (n={len(c_vals):3d})")

    # ============================================
    # STEP 6: Plot
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 6: Generating Figures")
    print("=" * 60)

    import os
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    summary = plot_fig6_style(all_results, args.out_prefix, top_features[0],
                               control_features[0], sae["d_hidden"], sae["d_in"])

    # Save all results
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "top_feature": top_features[0],
                "control_feature": control_features[0],
                "d_hidden": sae["d_hidden"],
                "d_in": sae["d_in"],
                "num_probes": len(probes),
                "scan": {k: v for k, v in scan.items()
                        if k not in ["stem5_mean", "stem3_mean", "loop_mean", "stem_selectivity"]},
                "feature_max": {str(k): float(v) for k, v in feature_max.items()},
            },
        }, f, indent=2)

    with open(args.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved:")
    print(f"  {args.out_prefix}.json")
    print(f"  {args.out_prefix}_summary.json")
    print(f"  {args.out_prefix}_1400M_L1_steering.png/pdf")
    print("\nDone!")


if __name__ == "__main__":
    main()
