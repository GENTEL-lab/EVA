#!/usr/bin/env python3
"""
Minimal Multi-SAE Steering Test - No matplotlib dependency.
Uses existing eva_glm_sae_steering_fig6 functions for correct prompt building.
"""

import argparse
import json
import sys
from pathlib import Path

# Add EVA paths
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1')
sys.path.insert(0, '/data/yanjie_huang/enzyme1_server/eva/EVA1/scripts/sae_feature_steering')

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from eva_glm_sae_steering_fig6 import (
    load_model, load_sae, build_prompt, forward_with_steer, score_target, parse_scales
)

# SAE Configurations
SAE_CONFIGS = {
    "layer13_step200000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt",
        "layer": 13,
    },
    "layer23_stable_step160000": {
        "path": "/data/yanjie_huang/enzyme1_server/eva/EVA1/notebooks/interpretability_analysis/sae_repro_release/outputs/sae_l1_penalty_1400M_layer23_stable_lr3e-5_l1x3_noresample/checkpoints/checkpoint_step160000.pt",
        "layer": 23,
    },
}

EVA_CHECKPOINT = "/data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_15000"

CASE_3 = {
    "record_id": "Domingo_2018_tRNA_68",
    "break_sequence": "AUUCCGUUGGCGUAAUGGUAACGCGUUUCCCUCCUAAGGAGAAGACUGCGGGUUCGAGUCCCGUAUGGAGAG",
    "rescue_pos": 68,
    "rescue_target": "U",
    "break_pos": 2,
}


def load_model(checkpoint_path, eva_root, device):
    sys.path.insert(0, eva_root)
    from tools.utils.model.loader import ModelLoader
    loader = ModelLoader(checkpoint_path, model_code_path=f"{eva_root}/eva")
    model, tokenizer = loader.load(device=device)
    model.eval()
    return model, tokenizer


def load_sae(path, mode, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if mode == "auto":
        if "model_state_dict" in ckpt:
            cfg_mode = str(ckpt.get("cfg", {}).get("mode") or "")
            if cfg_mode == "sae_l1_penalty":
                mode = "interplm"
                state = ckpt["model_state_dict"]
            else:
                mode = "batch_topk"
                state = ckpt["model_state_dict"]
        elif "sae" in ckpt:
            mode = "interplm"
            state = ckpt["sae"]
        else:
            raise ValueError(f"Unknown format: {list(ckpt.keys())}")
    else:
        state = ckpt.get("sae", ckpt.get("model_state_dict", {}))

    return {
        "mode": mode,
        "bias": state["bias"].to(device=device, dtype=torch.float32),
        "encoder_weight": state["encoder.weight"].to(device=device, dtype=torch.float32),
        "encoder_bias": state["encoder.bias"].to(device=device, dtype=torch.float32),
        "decoder_weight": state["decoder.weight"].to(device=device, dtype=torch.float32),
    }


def encode_sae(x, sae):
    pre = F.linear(x.float() - sae["bias"], sae["encoder_weight"], sae["encoder_bias"])
    return torch.relu(pre)


def decode_sae(f, sae):
    return F.linear(f.float(), sae["decoder_weight"], sae["bias"])


def token_ids(tokenizer, text):
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def steer_and_score(model, tokenizer, sae, case, layer, device, feature_id, scale, one_x_value):
    """Perform steering and return P(rescue_base)."""
    break_seq = case["break_sequence"]
    rescue_pos = case["rescue_pos"]
    break_pos = case["break_pos"]
    rescue_target = case["rescue_target"]

    prefix = break_seq[:rescue_pos]
    target = rescue_target
    suffix = break_seq[rescue_pos + 1:]

    prompt = f"<bos_glm>5{prefix}<span_0>{suffix}3<eos><span_0>"
    full = f"{prompt}{target}<eos_span>"
    full_ids = token_ids(tokenizer, full)
    prompt_ids = token_ids(tokenizer, prompt)

    # Determine steer token index
    char_idx = break_pos
    pre = token_ids(tokenizer, "<bos_glm>5" + prefix[:char_idx])
    steer_idx = len(pre)

    # Build position_ids
    target_len = len(token_ids(tokenizer, target))
    position_ids = list(range(len(prompt_ids)))
    position_ids.extend([len(token_ids(tokenizer, "<bos_glm>5" + prefix))] * len(token_ids(tokenizer, "<span_0>")))
    suffix_ids = token_ids(tokenizer, suffix + "3<eos>")
    position_ids.extend(range(len(prompt_ids) + 1 + max(target_len, 1),
                              len(prompt_ids) + 1 + max(target_len, 1) + len(suffix_ids)))
    position_ids.extend([len(token_ids(tokenizer, "<bos_glm>5" + prefix))] * len(token_ids(tokenizer, "<span_0>")))
    position_ids.extend(range(len(prompt_ids) + 1, len(full_ids)))

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    position_ids = torch.tensor([position_ids], dtype=torch.long, device=device)
    sequence_ids = torch.zeros((1, len(full_ids)), dtype=torch.long, device=device)

    target_activation = scale * one_x_value

    def steer_hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        x = hidden[0, steer_idx, :].float()

        pre_act = F.linear(x - sae["bias"], sae["encoder_weight"], sae["encoder_bias"])
        f = torch.relu(pre_act)
        f_new = f.clone()
        f_new[feature_id] = max(target_activation, 0.0)

        x_recon_old = decode_sae(f.unsqueeze(0), sae)[0]
        x_recon_new = decode_sae(f_new.unsqueeze(0), sae)[0]
        x_new = x_recon_new + (x - x_recon_old)

        hidden[0, steer_idx, :] = x_new.to(dtype=hidden.dtype)
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden

    handle = model.model.layers[layer].register_forward_hook(steer_hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        outputs = model(input_ids=input_ids, position_ids=position_ids, sequence_ids=sequence_ids)

    handle.remove()

    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    target_ids = token_ids(tokenizer, target)
    target_start = len(prompt_ids)
    pred_pos = target_start - 1

    if pred_pos < probs.shape[1] and target_ids:
        tok_id = target_ids[0]
        p_rescue = float(probs[0, pred_pos, tok_id].detach().cpu())
    else:
        p_rescue = 0.0

    return p_rescue


def run_steering_test(sae_name, sae_config, case, scales, device):
    print(f"\n{'='*60}")
    print(f"Testing SAE: {sae_name} (layer {sae_config['layer']})")
    print(f"{'='*60}")

    try:
        print("Loading model and SAE...")
        from eva_glm_sae_steering_fig6 import load_model as lm, load_sae as ls
        model, tokenizer = lm(argparse.Namespace(
            checkpoint="/data/yanjie_huang/enzyme1_server/eva/EVA_checkpoint/1400M_1129/checkpoint_15000",
            eva_root="/data/yanjie_huang/enzyme1_server/eva/EVA1",
            model_code_path=None,
            device=device
        ))
        sae = ls(sae_config["path"], "auto", device)
        layer = sae_config["layer"]

        # Build probe for Domingo case 3
        break_seq = case["break_sequence"]
        rescue_pos = case["rescue_pos"]
        break_pos = case["break_pos"]

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
            steer_offset=break_pos,
            steer_token_index=None,
            group_mode="joint",
            feature_max_json="",
            device=device,
            layer=layer,
        )

        prompt = build_prompt(tokenizer, prefix, target, suffix, args, device)

        # Feature max values
        feature_max = {5271: 3.985, 3571: 3.5}

        # Get baseline
        print("Running no-steer baseline...")
        logits, _ = forward_with_steer(model, prompt, sae, args, None, None, feature_max)
        baseline_score = score_target(logits, prompt)
        baseline_p = baseline_score["target_tokens"][0]["prob"]
        print(f"  Baseline P(rescue_base): {baseline_p:.6f}")

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
                print(f"    {scale:>5}x: P={p:.6f}, delta={delta:+.6f}")

            best_scale = max(feat_results["scales"].items(), key=lambda x: x[1]["delta"])[0]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--saes", default="layer13_step200000,layer23_stable_step160000")
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,5,10")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="multi_sae_results.json")
    args = parser.parse_args()

    sae_names = args.saes.split(",")
    scales = [float(s) for s in args.scales.split(",")]

    print("="*70)
    print("MULTI-SAE STEERING COMPARISON")
    print("="*70)
    print(f"Testing on Domingo Case 3 (rescue_base=U)")

    all_results = []
    for sae_name in sae_names:
        if sae_name not in SAE_CONFIGS:
            print(f"Unknown SAE: {sae_name}")
            continue
        result = run_steering_test(sae_name, SAE_CONFIGS[sae_name], CASE_3, scales, args.device)
        all_results.append(result)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'SAE':<30} {'Layer':<8} {'Baseline':<12} {'f/5271 Δ':<12} {'f/3571 Δ':<12}")
    print("-"*70)

    for r in all_results:
        if "error" in r:
            print(f"{r['sae_name']:<30} ERROR: {r['error'][:40]}")
        else:
            f5271_delta = r['features']['5271']['best_delta']
            f3571_delta = r['features']['3571']['best_delta']
            print(f"{r['sae_name']:<30} {r['layer']:<8} {r['baseline']:<12.6f} {f5271_delta:<+12.6f} {f3571_delta:<+12.6f}")

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
