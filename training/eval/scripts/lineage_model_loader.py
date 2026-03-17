#!/usr/bin/env python3
"""
Lineage模型加载器
支持从DCP和PyTorch格式的checkpoint加载模型
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
    validate_checkpoint as validate_lineage_checkpoint,
    resolve_training_config,
    load_model_weights_from_dcp,
    load_model_weights_from_pytorch,
)
from model.causal_lm import create_rnagen_model
from model.lineage_tokenizer import get_lineage_rna_tokenizer
from model.config import RNAGenConfig

logger = logging.getLogger(__name__)


def _create_lineage_model(checkpoint_path: Path, device: str = 'cuda:0'):
    """创建 Lineage MoE 模型实例"""
    config_file = checkpoint_path / "config.json"
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    # 调整MoE配置以支持单GPU加载
    original_moe_world_size = config_dict.get('moe_world_size')
    if original_moe_world_size and original_moe_world_size > 1:
        config_dict['moe_world_size'] = 1

    _, use_direction_tokens = resolve_training_config(checkpoint_path)
    # 回退到 config.json
    if not use_direction_tokens:
        use_direction_tokens = config_dict.get('use_direction_tokens', False)

    tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)
    config = RNAGenConfig(tokenizer=tokenizer, **config_dict)
    model = create_rnagen_model(config)

    return model, tokenizer, config


def load_lineage_model_from_dcp(checkpoint_path: str, device: str = 'cuda:0') -> Tuple[torch.nn.Module, any, RNAGenConfig]:
    model, tokenizer, config = _create_lineage_model(Path(checkpoint_path), device)
    model = load_model_weights_from_dcp(model, Path(checkpoint_path), device)
    return model, tokenizer, config


def load_lineage_model_from_pytorch(checkpoint_path: str, device: str = 'cuda:0') -> Tuple[torch.nn.Module, any, RNAGenConfig]:
    model, tokenizer, config = _create_lineage_model(Path(checkpoint_path), device)
    model = load_model_weights_from_pytorch(model, Path(checkpoint_path), device)
    return model, tokenizer, config


def load_lineage_model(
    checkpoint_path: str,
    device: str = 'cuda:0',
    force_format: Optional[str] = None
) -> Tuple[torch.nn.Module, any, RNAGenConfig]:
    """加载Lineage模型（自动检测格式）"""
    checkpoint_path = Path(checkpoint_path)

    if force_format:
        checkpoint_format = force_format
    else:
        checkpoint_format = detect_checkpoint_format(str(checkpoint_path))
    logger.info(f"使用格式: {checkpoint_format}")

    if checkpoint_format == 'model_only':
        return load_lineage_model_from_pytorch(str(checkpoint_path / "model_only"), device=device)
    elif checkpoint_format == 'pytorch':
        return load_lineage_model_from_pytorch(str(checkpoint_path), device=device)
    elif checkpoint_format == 'dcp':
        return load_lineage_model_from_dcp(str(checkpoint_path), device=device)
    else:
        raise ValueError(f"不支持的checkpoint格式: {checkpoint_format}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试模型加载")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if validate_lineage_checkpoint(args.checkpoint):
        model, tokenizer, config = load_lineage_model(args.checkpoint, device=args.device)
        print(f"模型加载成功 - vocab_size={config.vocab_size}, layers={config.num_hidden_layers}")
    else:
        print("Checkpoint验证失败")
