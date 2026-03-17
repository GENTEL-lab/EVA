"""
统一 FLOPs 计算，通过 is_moe 参数区分 Dense 和 MoE 模型
"""

import logging
from typing import Dict, Any

from .constants import FLOPS_MULTIPLIER

logger = logging.getLogger(__name__)


def calculate_model_flops(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any],
    world_size: int,
    distributed_config: Dict[str, Any] = None,
    is_moe: bool = False,
) -> int:
    """
    计算模型每个 optimizer step 的 FLOPs

    Args:
        model_config: 模型配置
        training_config: 训练配置
        data_config: 数据配置
        world_size: 总进程数
        distributed_config: 分布式配置（MoE 模型需要）
        is_moe: 是否为 MoE 模型

    Returns:
        每个 optimizer step 的 FLOPs
    """
    if distributed_config is None:
        distributed_config = {}

    batch_size = max(1, int(training_config.get('per_device_train_batch_size', 2)))
    seq_length = int(data_config.get('max_seq_length', model_config.get('max_position_embeddings', 1024)))
    hidden_size = int(model_config.get('hidden_size', 512))
    num_layers = int(model_config.get('num_hidden_layers', 8))
    intermediate_size = int(model_config.get('intermediate_size', 2048))
    num_heads = max(1, int(model_config.get('num_attention_heads', 8)))
    num_kv_heads = max(1, int(model_config.get('num_key_value_heads', num_heads)))
    gated_mlp = bool(model_config.get('gated_mlp', False))
    grad_accum_steps = max(1, int(training_config.get('gradient_accumulation_steps', 1)))

    tokens_per_device = batch_size * seq_length
    head_dim = hidden_size // num_heads
    kv_ratio = num_kv_heads / num_heads

    # Attention FLOPs: QKV/O投影 + QK^T + Attn·V
    attn_proj_flops = tokens_per_device * 4 * hidden_size * hidden_size * (1 + kv_ratio)
    attn_scores = 2 * batch_size * num_heads * seq_length * seq_length * head_dim
    attn_values = attn_scores
    attn_flops = attn_proj_flops + attn_scores + attn_values

    # MLP FLOPs
    mlp_per_token = 4 * hidden_size * intermediate_size  # up + down投影
    if gated_mlp:
        mlp_per_token += 2 * hidden_size * intermediate_size  # 额外的门控投影
        mlp_per_token += intermediate_size  # 元素级门控乘法

    mlp_flops = tokens_per_device * mlp_per_token
    router_flops = 0

    if is_moe:
        experts_per_tok = max(1, int(model_config.get('num_experts_per_tok', 1)))
        num_experts = int(model_config.get('num_experts', 1))
        if num_experts > 1:
            mlp_flops *= experts_per_tok
            router_flops = tokens_per_device * 2 * hidden_size * num_experts
        mlp_flops += router_flops

    flops_per_layer = attn_flops + mlp_flops
    flops_per_forward_replica = flops_per_layer * num_layers

    # 计算 data_parallel_size
    if is_moe:
        moe_world_size = model_config.get(
            'moe_world_size',
            distributed_config.get('expert_parallel_size', 1)
        )
        weight_parallel_size = distributed_config.get('weight_parallel_size', 1)
        denom = max(1, moe_world_size * weight_parallel_size)
        data_parallel_size = max(1, world_size // denom) if world_size else 1
    else:
        data_parallel_size = world_size

    total_flops = flops_per_forward_replica * int(data_parallel_size) * grad_accum_steps * FLOPS_MULTIPLIER
    return total_flops
