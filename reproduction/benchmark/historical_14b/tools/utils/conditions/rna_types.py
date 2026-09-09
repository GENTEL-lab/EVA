"""
RNA Type Definition and Validation

Defines the 15 supported RNA types and their corresponding tokens.
Strict reference to RNA_TOKENS definition in the original script.
"""

from typing import List, Dict


# Supported RNA types and their descriptions
# Reference: lines 209-214 of ORIGINAL_SCRIPT_ANALYSIS.md
RNA_TYPES: Dict[str, str] = {
    "mRNA": "Messenger RNA",
    "rRNA": "Ribosomal RNA",
    "tRNA": "Transfer RNA",
    "sRNA": "Small RNA",
    "lncRNA": "Long non-coding RNA",
    "circRNA": "Circular RNA",
    "viral_RNA": "Viral RNA",
    "miRNA": "MicroRNA",
    "snoRNA": "Small nucleolar RNA",
    "snRNA": "Small nuclear RNA",
    "piRNA": "PIWI-interacting RNA",
    "ribozyme": "Ribozyme",
    "scaRNA": "Small Cajal body RNA",
    "Y_RNA": "Y RNA",
    "vault_RNA": "Vault RNA",
}


def validate_rna_type(rna_type: str) -> bool:
    """
    Validate if RNA type is supported

    Args:
        rna_type: RNA type name, e.g., "mRNA", "tRNA"

    Returns:
        True if supported, False otherwise
    """
    return rna_type in RNA_TYPES


def get_rna_token(rna_type: str) -> str:
    """
    Get the token corresponding to an RNA type

    Reference: lines 209-214 of ORIGINAL_SCRIPT_ANALYSIS.md
    Format: <rna_TYPE>, e.g., <rna_mRNA>, <rna_tRNA>

    Args:
        rna_type: RNA type name, e.g., "mRNA", "tRNA"

    Returns:
        Token string, e.g., "<rna_mRNA>"

    Raises:
        ValueError: If RNA type is not supported
    """
    if not validate_rna_type(rna_type):
        raise ValueError(
            f"Unsupported RNA type: {rna_type}\n"
            f"Supported types: {list(RNA_TYPES.keys())}"
        )
    return f"<rna_{rna_type}>"


def list_rna_types() -> List[str]:
    """
    List all supported RNA types

    Returns:
        List of RNA type names
    """
    return list(RNA_TYPES.keys())


def get_rna_type_description(rna_type: str) -> str:
    """
    Get the description of an RNA type

    Args:
        rna_type: RNA type name

    Returns:
        Description string

    Raises:
        ValueError: If RNA type is not supported
    """
    if not validate_rna_type(rna_type):
        raise ValueError(f"Unsupported RNA type: {rna_type}")
    return RNA_TYPES[rna_type]
