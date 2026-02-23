"""
Eva - RNA Generation Model

RNA sequence generation and understanding model based on advanced MoE (Mixture of Experts) architecture
Designed for RNA sequence analysis, generation and prediction tasks

Core Components:
- EvaConfig: Model configuration
- EvaModel: Base model architecture
- EvaForCausalLM: Causal language model
- LineageRNATokenizer: RNA-specific tokenizer

Technical Features:
- Support for expert parallelism and weight parallelism
- Optimized attention mechanism
- Efficient batching and data loading
"""

from .config import EvaConfig
from .lineage_tokenizer import LineageRNATokenizer, get_lineage_rna_tokenizer

__version__ = "1.0.0"
__all__ = [
    "EvaConfig",
    "LineageRNATokenizer",
    "get_lineage_rna_tokenizer",
]