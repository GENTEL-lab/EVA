# IO utilities
"""
Input/output processing

- FASTA file reading and writing
"""

from .fasta import (
    read_fasta,
    iter_fasta,
    write_fasta,
    FastaWriter,
    count_fasta,
)

__all__ = [
    'read_fasta',
    'iter_fasta',
    'write_fasta',
    'FastaWriter',
    'count_fasta',
]
