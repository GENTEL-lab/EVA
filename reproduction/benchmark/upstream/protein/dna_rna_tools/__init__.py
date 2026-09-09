"""
DNA/RNA转换工具包

基于Evo 2模型技术规范实现的DNA与RNA序列双向转换工具。

主要功能：
- DNA到RNA格式转换（符合Evo 2统一编码规范）
- 密码子分析与翻译（支持多种遗传密码表）
- 序列拼接与增广（支持@/#特殊符号）
- 反向互补（支持50%概率随机生成训练样本）

快速使用：
    from dna_rna_tools import quick_dna_to_rna, analyze_codon_sequence

    # DNA到RNA转换
    rna = quick_dna_to_rna("ATGCGATCG")

    # 密码子分析
    result = analyze_codon_sequence("ATGAAACCCGGGTAA")
"""

from .dna_rna_converter import (
    # 主工具类
    DNAToRNAConverter,

    # 功能类
    CodonAnalyzer,
    TranscriptBuilder,
    SequenceConverter,
    GeneticCodeTable,

    # 枚举类型
    GeneticCodeType,

    # 便捷函数
    quick_dna_to_rna,
    analyze_codon_sequence,
    reverse_translate_protein,
)

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    # 主工具类
    "DNAToRNAConverter",

    # 功能类
    "CodonAnalyzer",
    "TranscriptBuilder",
    "SequenceConverter",
    "GeneticCodeTable",

    # 枚举类型
    "GeneticCodeType",

    # 便捷函数
    "quick_dna_to_rna",
    "analyze_codon_sequence",
    "reverse_translate_protein",
]
