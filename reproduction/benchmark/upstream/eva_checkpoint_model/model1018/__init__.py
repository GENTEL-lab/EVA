"""
RNAGen - RNA生成模型

基于先进的MoE（Mixture of Experts）架构的RNA序列生成和理解模型
专为RNA序列分析、生成和预测任务设计

核心组件：
- RNAGenConfig: 模型配置
- RNAGenModel: 基础模型架构  
- RNAGenForCausalLM: 因果语言模型
- LineageRNATokenizer: 基于谱系的RNA专用分词器

技术特点：
- 支持专家并行和权重并行
- 优化的注意力机制
- 高效的批处理和数据加载
- 支持Greengenes谱系系统编码
- 统一的token配置管理
"""

from .config import RNAGenConfig
from .lineage_tokenizer import LineageRNATokenizer, get_lineage_rna_tokenizer
from .token_config import (
    RNA_SPECIAL_TOKENS,
    GLM_SPAN_TOKENS,
    RNA_TYPE_TOKENS,
    RNA_BASES,
    END_OF_SPAN_TOKEN,
    PAD_TOKEN_ID,
)

__version__ = "1.0.0"
__all__ = [
    "RNAGenConfig",
    "LineageRNATokenizer",
    "get_lineage_rna_tokenizer",
    # Token 配置常量
    "RNA_SPECIAL_TOKENS",
    "GLM_SPAN_TOKENS",
    "RNA_TYPE_TOKENS",
    "RNA_BASES",
    "END_OF_SPAN_TOKEN",
    "PAD_TOKEN_ID",
]