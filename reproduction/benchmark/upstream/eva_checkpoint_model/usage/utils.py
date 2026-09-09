#!/usr/bin/env python3
"""
RNAGen模型使用工具函数
提供模型加载、条件前缀构建、序列验证等通用功能
"""

import sys
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model1018.config import RNAGenConfig
from model1018.causal_lm import RNAGenForCausalLM
from model1018.lineage_tokenizer import LineageRNATokenizer


def load_model(
    checkpoint_path: str,
    device: str = 'auto'
) -> Tuple[RNAGenForCausalLM, LineageRNATokenizer, RNAGenConfig]:
    """
    加载RNAGen模型

    Args:
        checkpoint_path: checkpoint目录路径
        device: 设备类型 ('auto', 'cuda', 'cpu')

    Returns:
        (model, tokenizer, config)
    """
    checkpoint_path = Path(checkpoint_path)

    # 确定设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"正在加载模型从: {checkpoint_path}")

    # 加载配置
    config = RNAGenConfig.from_json_file(str(checkpoint_path / "config.json"))

    # 适配单GPU推理（移除distributed配置）
    config.moe_world_size = 1

    # 加载tokenizer
    tokenizer = LineageRNATokenizer.from_pretrained(str(checkpoint_path))

    # 创建模型
    model = RNAGenForCausalLM(config)

    # 加载权重
    weights_file = checkpoint_path / "model_weights.pt"
    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"✓ 模型已加载到 {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  词汇表大小: {tokenizer.vocab_size}")

    return model, tokenizer, config


def build_conditional_prefix(
    rna_type: Optional[str] = None,
    lineage: Optional[str] = None
) -> str:
    """
    构建条件前缀 - 统一用|包围

    Args:
        rna_type: RNA类型，如 'mRNA', 'rRNA', 'tRNA' 等
        lineage: 谱系信息，如 'd__bacteria;p__proteobacteria;...'

    Returns:
        条件前缀字符串，统一格式：|内容|

    Examples:
        >>> build_conditional_prefix(rna_type='mRNA')
        '|<rna_mRNA>|'

        >>> build_conditional_prefix(lineage='d__bacteria;p__proteobacteria')
        '|d__bacteria;p__proteobacteria|'

        >>> build_conditional_prefix(rna_type='mRNA', lineage='d__bacteria')
        '|d__bacteria;<rna_mRNA>|'
    """
    parts = []

    # 添加谱系信息
    if lineage:
        # 确保谱系格式正确（小写，使用分号分隔）
        lineage = lineage.lower().strip()
        parts.append(lineage)

    # 添加RNA类型
    if rna_type:
        # 标准化RNA类型名称
        rna_type = rna_type.strip()
        if not rna_type.startswith('<rna_'):
            rna_type = f'<rna_{rna_type}>'
        parts.append(rna_type)

    # 组合前缀：统一用|包围，多个部分用分号连接
    if parts:
        return '|' + ';'.join(parts) + '|'

    return ''


def validate_rna_sequence(sequence: str) -> bool:
    """
    验证RNA序列是否有效

    Args:
        sequence: RNA序列字符串

    Returns:
        是否为有效的RNA序列
    """
    # 移除空白字符
    sequence = sequence.strip().upper()

    # 检查是否为空
    if not sequence:
        return False

    # 检查是否只包含AUGC
    valid_bases = set('AUGC')
    return all(base in valid_bases for base in sequence)


def prepare_input_ids(
    tokenizer: LineageRNATokenizer,
    sequence: str,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    准备模型输入

    Args:
        tokenizer: tokenizer实例
        sequence: 输入序列（可包含条件前缀）
        device: 设备

    Returns:
        包含input_ids, position_ids, sequence_ids的字典
    """
    # 添加BOS和EOS标记
    full_sequence = f"<bos>{sequence}<eos>"

    # 编码
    token_ids = tokenizer.encode(full_sequence)

    # 创建tensors
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    sequence_ids = torch.zeros((1, len(token_ids)), dtype=torch.long, device=device)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'sequence_ids': sequence_ids
    }


def decode_sequence(
    tokenizer: LineageRNATokenizer,
    token_ids: torch.Tensor,
    remove_special_tokens: bool = True
) -> str:
    """
    解码token序列

    Args:
        tokenizer: tokenizer实例
        token_ids: token ID tensor
        remove_special_tokens: 是否移除特殊标记

    Returns:
        解码后的字符串
    """
    # 转换为列表
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # 解码
    sequence = tokenizer.decode(token_ids)

    # 移除特殊标记
    if remove_special_tokens:
        special_tokens = ['<bos>', '<eos>', '<pad>', '<bos_glm>', '<eos_span>']
        for token in special_tokens:
            sequence = sequence.replace(token, '')

    return sequence


def extract_rna_sequence(full_sequence: str) -> str:
    """
    从完整序列中提取纯RNA序列部分

    Args:
        full_sequence: 包含条件前缀的完整序列

    Returns:
        纯RNA序列（只包含AUGC）
    """
    # 移除所有特殊标记
    import re

    # 移除谱系信息（|...|格式）
    sequence = re.sub(r'\|[^|]*\|', '', full_sequence)

    # 移除RNA类型标记
    sequence = re.sub(r'<rna_[^>]+>', '', sequence)

    # 移除其他特殊标记
    sequence = re.sub(r'<[^>]+>', '', sequence)

    # 只保留AUGC
    sequence = ''.join(c for c in sequence if c in 'AUGC')

    return sequence


def format_output(
    sequence: str,
    score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    格式化输出结果

    Args:
        sequence: 序列
        score: 分数（可选）
        metadata: 元数据（可选）

    Returns:
        格式化的字符串
    """
    lines = []
    lines.append(f"序列: {sequence}")

    if score is not None:
        lines.append(f"分数: {score:.4f}")

    if metadata:
        lines.append("元数据:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")

    return '\n'.join(lines)


def get_available_rna_types() -> list:
    """
    获取支持的RNA类型列表

    Returns:
        RNA类型列表
    """
    return [
        'mRNA',      # 信使RNA
        'rRNA',      # 核糖体RNA
        'tRNA',      # 转运RNA
        'sRNA',      # 小RNA
        'lncRNA',    # 长链非编码RNA
        'circRNA',   # 环状RNA
        'viral_RNA', # 病毒RNA
        'miRNA',     # 微小RNA
        'snoRNA',    # 小核仁RNA
        'snRNA',     # 小核RNA
        'piRNA',     # PIWI结合RNA
        'ribozyme',  # 核酶
        'scaRNA',    # 小Cajal体RNA
        'Y_RNA',     # Y RNA
        'vault_RNA', # Vault RNA
    ]


def print_usage_header(script_name: str, description: str):
    """
    打印脚本使用说明头部

    Args:
        script_name: 脚本名称
        description: 脚本描述
    """
    print("=" * 80)
    print(f"{script_name}")
    print(f"{description}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # 测试工具函数
    print("=== RNAGen工具函数测试 ===\n")

    # 测试条件前缀构建
    print("1. 测试条件前缀构建:")
    print(f"  仅RNA类型: {build_conditional_prefix(rna_type='mRNA')}")
    print(f"  仅谱系: {build_conditional_prefix(lineage='d__bacteria;p__proteobacteria')}")
    print(f"  两者都有: {build_conditional_prefix(rna_type='mRNA', lineage='d__bacteria')}")
    print(f"  都没有: {build_conditional_prefix()}")
    print()

    # 测试序列验证
    print("2. 测试序列验证:")
    print(f"  'AUGCAUGC' 有效: {validate_rna_sequence('AUGCAUGC')}")
    print(f"  'ATGCATGC' 有效: {validate_rna_sequence('ATGCATGC')}")  # DNA序列
    print(f"  'AUGXAUGC' 有效: {validate_rna_sequence('AUGXAUGC')}")
    print()

    # 测试RNA序列提取
    print("3. 测试RNA序列提取:")
    test_seq = "|d__bacteria;<rna_mRNA>|AUGCAUGC"
    print(f"  输入: {test_seq}")
    print(f"  提取: {extract_rna_sequence(test_seq)}")
    print()

    # 显示支持的RNA类型
    print("4. 支持的RNA类型:")
    for rna_type in get_available_rna_types():
        print(f"  - {rna_type}")

    print("\n测试完成！")
