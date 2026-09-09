"""
Codon table definitions

Provides codon mappings required for reverse translation from protein to RNA.
Supports universal codon tables and species-specific optimal codon tables.
"""

from typing import Dict, Optional


# =============================================================================
# Universal codon table (standard genetic code, first codon for each amino acid)
# =============================================================================

CODON_TABLE_FIRST = {
    'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAA',
    'F': 'UUU', 'G': 'GGU', 'H': 'CAU', 'I': 'AUU',
    'K': 'AAA', 'L': 'UUA', 'M': 'AUG', 'N': 'AAU',
    'P': 'CCU', 'Q': 'CAA', 'R': 'CGU', 'S': 'UCU',
    'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
    '*': 'UAA'  # Stop codon
}

# Default codon table
DEFAULT_CODON_TABLE = CODON_TABLE_FIRST


# =============================================================================
# Species-specific codon tables (optimal codons)
# =============================================================================

# E. coli optimal codon table
ECOLI_CODON_TABLE = {
    'A': 'GCG', 'C': 'UGC', 'D': 'GAU', 'E': 'GAA',
    'F': 'UUU', 'G': 'GGC', 'H': 'CAU', 'I': 'AUU',
    'K': 'AAA', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
    'P': 'CCG', 'Q': 'CAG', 'R': 'CGU', 'S': 'AGC',
    'T': 'ACC', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
    '*': 'UAA'
}

# Human optimal codon table
HUMAN_CODON_TABLE = {
    'A': 'GCC', 'C': 'UGC', 'D': 'GAC', 'E': 'GAG',
    'F': 'UUC', 'G': 'GGC', 'H': 'CAC', 'I': 'AUC',
    'K': 'AAG', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
    'P': 'CCC', 'Q': 'CAG', 'R': 'AGG', 'S': 'AGC',
    'T': 'ACC', 'V': 'GUG', 'W': 'UGG', 'Y': 'UAC',
    '*': 'UGA'
}

# Yeast (Saccharomyces cerevisiae) optimal codon table
YEAST_CODON_TABLE = {
    'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAA',
    'F': 'UUU', 'G': 'GGU', 'H': 'CAU', 'I': 'AUU',
    'K': 'AAG', 'L': 'UUG', 'M': 'AUG', 'N': 'AAU',
    'P': 'CCU', 'Q': 'CAA', 'R': 'AGA', 'S': 'UCU',
    'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
    '*': 'UAA'
}

# Pseudomonas aeruginosa optimal codon table
PSEUDOMONAS_CODON_TABLE = {
    'A': 'GCC', 'C': 'UGC', 'D': 'GAC', 'E': 'GAA',
    'F': 'UUC', 'G': 'GGC', 'H': 'CAC', 'I': 'AUC',
    'K': 'AAG', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
    'P': 'CCG', 'Q': 'CAG', 'R': 'CGC', 'S': 'UCC',
    'T': 'ACC', 'V': 'GUC', 'W': 'UGG', 'Y': 'UAC',
    '*': 'UAA'
}

# TaxID → codon table mapping
TAXID_TO_CODON_TABLE = {
    '562': ECOLI_CODON_TABLE,       # E. coli
    '9606': HUMAN_CODON_TABLE,      # Homo sapiens
    '10090': HUMAN_CODON_TABLE,     # Mus musculus (uses human table)
    '4932': YEAST_CODON_TABLE,      # S. cerevisiae
    '287': PSEUDOMONAS_CODON_TABLE  # P. aeruginosa
}


# =============================================================================
# Functions
# =============================================================================

def get_codon_table(
    taxid: Optional[str] = None,
    species: Optional[str] = None,
    optimization: str = 'first'
) -> Dict[str, str]:
    """
    Get codon table

    Args:
        taxid: NCBI Taxonomy ID
        species: Species name
        optimization: Optimization strategy
            - 'first': Use universal codon table (first codon for each amino acid)
            - 'most_frequent': Use species-specific optimal codon table

    Returns:
        Codon table dictionary {amino acid: codon}
    """
    if optimization == 'first':
        return DEFAULT_CODON_TABLE

    # most_frequent strategy: try to use species-specific table
    if taxid and taxid in TAXID_TO_CODON_TABLE:
        return TAXID_TO_CODON_TABLE[taxid]

    # Match by species name
    if species:
        species_lower = species.lower()
        if 'escherichia' in species_lower or 'e. coli' in species_lower:
            return ECOLI_CODON_TABLE
        elif 'homo sapiens' in species_lower or 'human' in species_lower:
            return HUMAN_CODON_TABLE
        elif 'saccharomyces' in species_lower or 'yeast' in species_lower:
            return YEAST_CODON_TABLE
        elif 'pseudomonas' in species_lower:
            return PSEUDOMONAS_CODON_TABLE

    # Default to universal table
    return DEFAULT_CODON_TABLE


def reverse_translate(
    protein_seq: str,
    codon_table: Optional[Dict[str, str]] = None
) -> str:
    """
    Reverse translate protein sequence to RNA sequence

    Args:
        protein_seq: Protein sequence (single-letter amino acid codes)
        codon_table: Codon table (defaults to universal table)

    Returns:
        RNA sequence (nucleotides)

    Example:
        >>> reverse_translate("MKT")
        'AUGAAAACA'
    """
    if codon_table is None:
        codon_table = DEFAULT_CODON_TABLE

    rna_seq = []
    for aa in protein_seq.upper():
        if aa in codon_table:
            rna_seq.append(codon_table[aa])
        else:
            # Unknown amino acid, use N (Asparagine) codon
            rna_seq.append('AAU')

    return ''.join(rna_seq)
