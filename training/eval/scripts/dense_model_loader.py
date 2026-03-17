#!/usr/bin/env python3
"""
Dense模型加载器
支持从DCP和PyTorch格式的checkpoint加载Dense模型
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.eval.scripts.common.model_loader_base import (
    detect_checkpoint_format,
    validate_checkpoint as validate_dense_checkpoint,
    resolve_training_config,
    load_model_weights_from_dcp,
    load_model_weights_from_pytorch,
)
from model_dense.causal_lm import RNAGenDenseForCausalLM
from model_dense.config import RNAGenDenseConfig
from model.lineage_tokenizer import get_lineage_rna_tokenizer

logger = logging.getLogger(__name__)

# MoE 参数列表（Dense 模型不需要）
_MOE_KEYS = ['num_experts', 'num_experts_per_tok', 'router_aux_loss_coef',
             'moe_world_size', 'mlp_impl', 'memory_optimized_mlp']


def _create_dense_model(checkpoint_path: Path, device: str = 'cuda:0'):
    """创建 Dense 模型实例"""
    config_file = checkpoint_path / "config.json"
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    _, use_direction_tokens = resolve_training_config(checkpoint_path)
    if not use_direction_tokens:
        use_direction_tokens = config_dict.get('use_direction_tokens', False)

    tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)

    dense_config_dict = {k: v for k, v in config_dict.items() if k not in _MOE_KEYS}
    config = RNAGenDenseConfig(tokenizer=tokenizer, **dense_config_dict)
    model = RNAGenDenseForCausalLM(config)

    return model, tokenizer, config


def load_dense_model_from_dcp(checkpoint_path: str, device: str = 'cuda:0') -> Tuple[torch.nn.Module, any, RNAGenDenseConfig]:
    model, tokenizer, config = _create_dense_model(Path(checkpoint_path), device)
    model = load_model_weights_from_dcp(model, Path(checkpoint_path), device)
    return model, tokenizer, config


def load_dense_model_from_pytorch(checkpoint_path: str, device: str = 'cuda:0') -> Tuple[torch.nn.Module, any, RNAGenDenseConfig]:
    model, tokenizer, config = _create_dense_model(Path(checkpoint_path), device)
    model = load_model_weights_from_pytorch(model, Path(checkpoint_path), device)
    return model, tokenizer, config


def load_dense_model(
    checkpoint_path: str,
    device: str = 'cuda:0',
    force_format: Optional[str] = None
) -> Tuple[torch.nn.Module, any, RNAGenDenseConfig]:
    """加载Dense模型（自动检测格式）"""
    checkpoint_path = Path(checkpoint_path)

    if force_format:
        checkpoint_format = force_format
    else:
        checkpoint_format = detect_checkpoint_format(str(checkpoint_path))
    logger.info(f"使用格式: {checkpoint_format}")

    if checkpoint_format == 'model_only':
        return load_dense_model_from_pytorch(str(checkpoint_path / "model_only"), device=device)
    elif checkpoint_format == 'pytorch':
        return load_dense_model_from_pytorch(str(checkpoint_path), device=device)
    elif checkpoint_format == 'dcp':
        return load_dense_model_from_dcp(str(checkpoint_path), device=device)
    else:
        raise ValueError(f"不支持的checkpoint格式: {checkpoint_format}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试Dense模型加载")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if validate_dense_checkpoint(args.checkpoint):
        model, tokenizer, config = load_dense_model(args.checkpoint, device=args.device)
        print(f"Dense模型加载成功 - vocab_size={config.vocab_size}, layers={config.num_hidden_layers}")
    else:
        print("Checkpoint验证失败")
