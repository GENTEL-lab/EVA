#!/usr/bin/env python3
"""Standalone SAE training entrypoint for EVA interpretability release."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "eva").is_dir() and (p / "notebooks" / "interpretability_analysis" / "sae_repro_release").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate EVA repo root containing eva/ and notebooks/interpretability_analysis/sae_repro_release")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from eva.causal_lm import EvaForCausalLM
    from eva.config import EvaConfig
    from eva.lineage_tokenizer import LineageRNATokenizer


def _import_eva_components():
    try:
        from eva.causal_lm import EvaForCausalLM
        from eva.config import EvaConfig
        from eva.lineage_tokenizer import LineageRNATokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing EVA training dependencies. Install "
            "'notebooks/interpretability_analysis/sae_repro_release/requirements-sae-training.txt' "
            "or use the provided Docker image."
        ) from exc
    return EvaForCausalLM, EvaConfig, LineageRNATokenizer


class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.bias = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=np.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=np.sqrt(5))

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x - self.bias, self.encoder.weight, self.encoder.bias)

    def encode_topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pre = self.pre_activation(x)
        flat = pre.view(-1, pre.shape[-1])
        vals, idx = torch.topk(flat, k=k, dim=-1)
        masked = torch.where(vals > 0, vals, torch.zeros_like(vals))
        out = torch.zeros_like(flat)
        out.scatter_(-1, idx, masked)
        return out.view_as(pre)

    def forward_topk(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode_topk(x, k)
        x_hat = self.decoder(f) + self.bias
        return x_hat, f

    def forward_relu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.pre_activation(x)
        f = torch.relu(pre)
        x_hat = self.decoder(f) + self.bias
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        w = self.decoder.weight
        n = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        w.div_(n)


class ConstrainedAdam(torch.optim.Adam):
    def __init__(self, params, constrained, **kwargs):
        super().__init__(params, **kwargs)
        self._constrained = constrained

    @torch.no_grad()
    def step(self, closure=None):
        for p in self._constrained:
            if p.grad is None:
                continue
            normed = p / p.norm(dim=0, keepdim=True).clamp(min=1e-8)
            p.grad -= (p.grad * normed).sum(dim=0, keepdim=True) * normed
        super().step(closure)
        for p in self._constrained:
            p /= p.norm(dim=0, keepdim=True).clamp(min=1e-8)


def set_global_reproducibility(seed: int, deterministic: bool) -> None:
    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def compute_aux_loss_topk(
    sae: SAE,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    dead_mask: torch.Tensor,
    aux_coef: float,
) -> torch.Tensor:
    if aux_coef <= 0 or not dead_mask.any():
        return torch.zeros((), device=x.device)
    resid = (x - x_hat.detach()).detach()
    x_c = x - sae.bias
    w_enc = sae.encoder.weight[dead_mask]
    b_enc = sae.encoder.bias[dead_mask]
    pre_d = x_c @ w_enc.T + b_enc
    acts_d = torch.relu(pre_d)
    w_dec = sae.decoder.weight[:, dead_mask]
    recon_d = F.linear(acts_d, w_dec)
    return aux_coef * (recon_d - resid).pow(2).mean()


def read_fasta_seqs(path: Path, max_seqs: int) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).upper().replace("T", "U")
                    if s:
                        seqs.append(s)
                    cur = []
                    if len(seqs) >= max_seqs:
                        break
                continue
            if line and all(c in "ACGUTNacgutn" for c in line):
                cur.append(line.upper().replace("T", "U"))
        if cur and len(seqs) < max_seqs:
            s = "".join(cur).upper().replace("T", "U")
            if s:
                seqs.append(s)
    return seqs[:max_seqs]


@torch.no_grad()
def batch_extract_hidden(
    model: "EvaForCausalLM",
    tok: "LineageRNATokenizer",
    seqs: list[str],
    layer: int,
    device: torch.device,
    max_len: int,
) -> torch.Tensor:
    all_rows = []
    for s in seqs:
        ids = tok.encode(s)
        if len(ids) > max_len:
            ids = ids[:max_len]
        if not ids:
            continue

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        _, l = input_ids.shape
        pos = torch.arange(l, dtype=torch.long, device=device).unsqueeze(0)
        seq_ids = torch.zeros((1, l), dtype=torch.long, device=device)

        captured = None

        def hook(_module, _inp, out):
            nonlocal captured
            t = out[0] if isinstance(out, tuple) else out
            captured = t.detach().float()

        h = model.model.layers[layer].register_forward_hook(hook)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(input_ids, position_ids=pos, sequence_ids=seq_ids)
        h.remove()

        if captured is None:
            continue
        all_rows.append(captured[0, : len(ids), :].reshape(-1, captured.shape[-1]))

    if not all_rows:
        return torch.empty(0, model.config.hidden_size, device="cpu")
    return torch.cat(all_rows, dim=0).cpu()


def lr_trapezoid(
    step: int,
    total_steps: int,
    max_lr: float,
    warmup_ratio: float = 0.05,
    decay_ratio: float = 0.05,
) -> float:
    if total_steps <= 0:
        return max_lr
    warmup = max(1, int(total_steps * warmup_ratio))
    decay_len = max(1, int(total_steps * decay_ratio))
    decay_start = total_steps - decay_len
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= decay_start:
        if step >= total_steps:
            return 0.0
        return max_lr * max(0.0, 1.0 - (step - decay_start) / max(decay_len, 1))
    return max_lr


def interplm_lr_multiplier(step: int, total_steps: int, warmup_steps: int, decay_start: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    if step >= decay_start:
        return max(0.0, (total_steps - step) / max(total_steps - decay_start, 1))
    return 1.0


def save_checkpoint(out_dir: Path, step: int, sae: SAE, cfg: dict, extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": sae.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, out_dir / f"checkpoint_step{step}.pt")


def _load_eva_model_and_tokenizer(cfg: dict, device: torch.device):
    EvaForCausalLM, EvaConfig, LineageRNATokenizer = _import_eva_components()
    ckpt_dir = Path(cfg["paths"]["ckpt"])
    ckpt_cfg = EvaConfig.from_json_file(str(ckpt_dir / "config.json"))
    ckpt_cfg.moe_world_size = 1

    state_dict = torch.load(ckpt_dir / "model_weights.pt", map_location="cpu", weights_only=False)
    if "model.embed_tokens.weight" in state_dict:
        ckpt_cfg.vocab_size = state_dict["model.embed_tokens.weight"].shape[0]

    model = EvaForCausalLM(ckpt_cfg)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    tokenizer = LineageRNATokenizer.from_pretrained(str(ckpt_dir))
    return model, tokenizer, ckpt_cfg


def train_batch_topk(cfg: dict, device: torch.device, log: logging.Logger) -> Path:
    model, tok, ckpt_cfg = _load_eva_model_and_tokenizer(cfg, device)

    sae = SAE(ckpt_cfg.hidden_size, cfg["d_hidden"]).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))

    seqs = read_fasta_seqs(Path(cfg["paths"]["data"]), cfg["n_seqs"])
    log.info("Loaded %d sequences from %s", len(seqs), cfg["paths"]["data"])

    buffer = deque()
    buf_tokens = 0
    step = 0
    dead_steps = torch.zeros(cfg["d_hidden"], dtype=torch.long, device=device)
    rng = random.Random(cfg["seed"])

    out_dir = Path(cfg["paths"]["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_batches_per_epoch = max(1, len(seqs) // cfg["seq_batch"])
    for epoch in range(cfg["n_epochs"]):
        log.info("Epoch %d/%d", epoch + 1, cfg["n_epochs"])
        rng.shuffle(seqs)
        for bi in range(n_batches_per_epoch):
            batch_seqs = seqs[bi * cfg["seq_batch"] : (bi + 1) * cfg["seq_batch"]]
            if not batch_seqs:
                continue

            try:
                h_cpu = batch_extract_hidden(model, tok, batch_seqs, cfg["layer"], device, cfg["max_len"])
            except Exception as exc:
                log.error("Hidden extraction failed on batch %d: %s", bi, exc)
                continue

            if h_cpu.shape[0] == 0:
                continue

            h_cpu = h_cpu[torch.randperm(h_cpu.shape[0])]
            buffer.append(h_cpu)
            buf_tokens += h_cpu.shape[0]
            while buf_tokens > cfg["buffer_size"]:
                buf_tokens -= buffer.popleft().shape[0]

            while buf_tokens >= cfg["batch_size"] and step < cfg["max_steps"]:
                flat = torch.cat(list(buffer), dim=0)
                idx = torch.randperm(flat.shape[0], device="cpu")[: cfg["batch_size"]]
                x = flat[idx].to(device, dtype=torch.float32)

                lr = lr_trapezoid(
                    step,
                    cfg["max_steps"],
                    cfg["lr"],
                    cfg["trapezoid_warmup_ratio"],
                    cfg["trapezoid_decay_ratio"],
                )
                for g in opt.param_groups:
                    g["lr"] = lr

                opt.zero_grad()
                x_hat, f = sae.forward_topk(x, cfg["k"])
                mse = (x - x_hat).pow(2).mean()
                fired = (f.abs() > 0).float().sum(0)
                dead_mask = dead_steps >= cfg["dead_feature_window"]
                aux = compute_aux_loss_topk(sae, x, x_hat, dead_mask, cfg["aux_coef"])
                loss = mse + aux

                loss.backward()
                clip_grad_norm_(sae.parameters(), cfg["grad_clip"])
                opt.step()
                if cfg.get("normalize_decoder", True):
                    sae.normalize_decoder()

                dead_steps = dead_steps + (fired == 0).long()
                dead_steps = torch.where(fired > 0, torch.zeros_like(dead_steps), dead_steps)

                step += 1
                if step % cfg["log_every"] == 0:
                    l0 = (f.abs() > 0).float().sum(-1).mean().item()
                    log.info("Step %d | loss=%.4f | L0=%.2f | lr=%.2e", step, loss.item(), l0, lr)
                if step % cfg["save_every"] == 0:
                    save_checkpoint(out_dir, step, sae, cfg)

    save_checkpoint(out_dir, step, sae, cfg)
    log.info("Batch-TopK training done. step=%d output=%s", step, out_dir)
    return out_dir


def train_sae_l1_penalty(cfg: dict, device: torch.device, log: logging.Logger) -> Path:
    model, tok, ckpt_cfg = _load_eva_model_and_tokenizer(cfg, device)

    sae = SAE(ckpt_cfg.hidden_size, cfg["d_hidden"]).to(device)
    opt = ConstrainedAdam(
        sae.parameters(),
        constrained=[sae.decoder.weight],
        lr=cfg["l1_lr"],
        betas=(0.9, 0.999),
    )

    l1_warmup_steps = cfg.get("l1_warmup_steps", 7500)
    decay_start = cfg.get("l1_lr_decay_start", int(0.8 * cfg["max_steps"]))

    seqs = read_fasta_seqs(Path(cfg["paths"]["data"]), cfg["n_seqs"])
    log.info("Loaded %d sequences from %s", len(seqs), cfg["paths"]["data"])

    buffer = deque()
    buf_tokens = 0
    step = 0
    steps_since_active = torch.zeros(cfg["d_hidden"], dtype=torch.long, device=device)

    out_dir = Path(cfg["paths"]["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_batches_per_epoch = max(1, len(seqs) // cfg["seq_batch"])
    for epoch in range(cfg["n_epochs"]):
        random.shuffle(seqs)
        for bi in range(n_batches_per_epoch):
            batch_seqs = seqs[bi * cfg["seq_batch"] : (bi + 1) * cfg["seq_batch"]]
            if not batch_seqs:
                continue

            h_cpu = batch_extract_hidden(model, tok, batch_seqs, cfg["layer"], device, cfg["max_len"])
            if h_cpu.shape[0] == 0:
                continue

            h_cpu = h_cpu[torch.randperm(h_cpu.shape[0])]
            buffer.append(h_cpu)
            buf_tokens += h_cpu.shape[0]
            while buf_tokens > cfg["buffer_size"]:
                buf_tokens -= buffer.popleft().shape[0]

            while buf_tokens >= cfg["batch_size"] and step < cfg["max_steps"]:
                flat = torch.cat(list(buffer), dim=0)
                idx = torch.randperm(flat.shape[0], device="cpu")[: cfg["batch_size"]]
                x = flat[idx].to(device, dtype=torch.float32)

                lr = cfg["l1_lr"] * interplm_lr_multiplier(
                    step,
                    cfg["max_steps"],
                    cfg["warmup_steps"],
                    decay_start,
                )
                for g in opt.param_groups:
                    g["lr"] = lr

                opt.zero_grad()
                x_hat, hid = sae.forward_relu(x)
                mse = (x - x_hat).pow(2).mean()
                l1 = hid.abs().mean()
                warmup_scale = min(1.0, step / max(l1_warmup_steps, 1))
                loss = mse + cfg["l1_penalty"] * warmup_scale * l1

                loss.backward()
                clip_grad_norm_(sae.parameters(), cfg["grad_clip"])
                opt.step()

                with torch.no_grad():
                    active = (hid.abs() > 1e-7).float().amax(dim=0) > 0
                    steps_since_active = torch.where(active, torch.zeros_like(steps_since_active), steps_since_active + 1)

                step += 1
                resample_steps = cfg.get("resample_steps", 5000)
                resample_after = cfg.get("resample_dead_after", 2500)
                if step % resample_steps == 0:
                    with torch.no_grad():
                        dead = (steps_since_active > resample_after).nonzero(as_tuple=True)[0]
                        if dead.numel() > 0:
                            sample_idx = torch.randint(0, x.shape[0], (dead.numel(),), device=device)
                            new_w = x[sample_idx] - sae.bias
                            new_w = new_w + 0.01 * torch.randn_like(new_w)
                            new_w = new_w / new_w.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            sae.encoder.weight.data[dead] = new_w
                            sae.decoder.weight.data[:, dead] = new_w.T
                            steps_since_active[dead] = 0

                if step % cfg["log_every"] == 0:
                    log.info("Step %d | loss=%.4f mse=%.4f l1=%.4f lr=%.2e", step, loss.item(), mse.item(), l1.item(), lr)
                if step % cfg["save_every"] == 0:
                    save_checkpoint(out_dir, step, sae, cfg)

    save_checkpoint(out_dir, step, sae, cfg)
    log.info("sae_L1_penalty training done. step=%d output=%s", step, out_dir)
    return out_dir


def resolve_runtime_paths(cfg: dict, mode: str) -> tuple[str, str, str]:
    ckpt = os.environ.get("SAE_CKPT_DIR")
    data = os.environ.get("SAE_DATA_FASTA")
    if not ckpt or not data:
        raise EnvironmentError(
            "SAE_CKPT_DIR and SAE_DATA_FASTA must be set. Use scripts/check_hf_paths.sh first."
        )

    mode_dir = "batch_topk_sae" if mode == "batch_topk" else "sae_l1_penalty"
    tag = os.environ.get("SAE_OUTPUT_TAG", "").strip()
    if tag:
        mode_dir = f"{mode_dir}_{tag}"

    out_root = os.environ.get(
        "SAE_OUTPUT_ROOT",
        str(REPO_ROOT / "notebooks" / "interpretability_analysis" / "sae_repro_release" / "outputs" / mode_dir / "checkpoints"),
    )

    cfg["paths"]["base"] = str(REPO_ROOT)
    cfg["paths"]["ckpt"] = ckpt
    cfg["paths"]["data"] = data
    cfg["paths"]["output_root"] = out_root

    return ckpt, data, out_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone SAE training runner")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--mode",
        required=True,
        choices=["batch_topk", "sae_l1_penalty"],
        help="Training mode",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path, data_path, out_dir = resolve_runtime_paths(cfg, args.mode)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_global_reproducibility(seed, deterministic)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    log_dir = REPO_ROOT / "notebooks" / "interpretability_analysis" / "sae_repro_release" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_file = log_dir / f"{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    log = logging.getLogger("sae_repro")

    log.info("=" * 72)
    log.info("EVA SAE training")
    log.info("repo_root: %s", REPO_ROOT)
    log.info("mode: %s", args.mode)
    log.info("config: %s", args.config)
    log.info("checkpoint: %s", ckpt_path)
    log.info("fasta: %s", data_path)
    log.info("output: %s", out_dir)
    log.info("seed: %d deterministic: %s", seed, deterministic)
    log.info("device: %s", device)
    log.info("=" * 72)

    if args.mode == "batch_topk":
        final_out = train_batch_topk(cfg, device, log)
    else:
        final_out = train_sae_l1_penalty(cfg, device, log)

    log.info("done: %s", final_out)
    log.info("log file: %s", log_file)


if __name__ == "__main__":
    main()
