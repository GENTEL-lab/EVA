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

from . import _runtime_env as _runtime_env

# Self-heal the Triton CUDA driver environment before anything can import
# torch/megablocks/triton (see eva/_runtime_env.py for details). This also
# covers `singularity exec/shell` sessions where container ENTRYPOINTs do
# not run. Set EVA_SKIP_RUNTIME_ENV=1 to disable.
_runtime_env.apply()

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
