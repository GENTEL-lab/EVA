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

from ._version import __version__

__all__ = [
    "__version__",
    "EvaConfig",
    "LineageRNATokenizer",
    "get_lineage_rna_tokenizer",
]


def __getattr__(name):
    if name == "EvaConfig":
        from .config import EvaConfig

        return EvaConfig
    if name in {"LineageRNATokenizer", "get_lineage_rna_tokenizer"}:
        from .lineage_tokenizer import LineageRNATokenizer, get_lineage_rna_tokenizer

        return {
            "LineageRNATokenizer": LineageRNATokenizer,
            "get_lineage_rna_tokenizer": get_lineage_rna_tokenizer,
        }[name]
    raise AttributeError(f"module 'eva' has no attribute {name!r}")
