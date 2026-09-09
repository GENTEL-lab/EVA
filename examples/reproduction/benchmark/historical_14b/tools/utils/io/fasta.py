"""
FASTA file reading and writing

Provides reading and writing functionality for FASTA format files.
Supports streaming writes and file locks (for multi-process concurrent writes).
"""

import fcntl
from pathlib import Path
from typing import List, Tuple, Optional, Iterator


def read_fasta(filepath: str) -> List[Tuple[str, str]]:
    """
    Read FASTA file

    Args:
        filepath: FASTA file path

    Returns:
        List of sequences, each element is a (header, sequence) tuple
        header does not include the '>' prefix

    Example:
        sequences = read_fasta("input.fa")
        for header, seq in sequences:
            print(f">{header}")
            print(seq)
    """
    sequences = []
    current_header = None
    current_seq_parts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous sequence
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq_parts)))
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Save last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq_parts)))

    return sequences


def iter_fasta(filepath: str) -> Iterator[Tuple[str, str]]:
    """
    Iteratively read FASTA file (memory-friendly)

    Args:
        filepath: FASTA file path

    Yields:
        (header, sequence) tuple
    """
    current_header = None
    current_seq_parts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_header is not None:
                    yield (current_header, ''.join(current_seq_parts))
                current_header = line[1:]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        if current_header is not None:
            yield (current_header, ''.join(current_seq_parts))


def write_fasta(
    filepath: str,
    sequences: List[Tuple[str, str]],
    line_width: int = 80
) -> None:
    """
    Write FASTA file

    Args:
        filepath: Output file path
        sequences: List of sequences, each element is a (header, sequence) tuple
        line_width: Maximum width of each sequence line, 0 means no line breaks
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            if line_width > 0:
                # Wrap lines at specified width
                for i in range(0, len(seq), line_width):
                    f.write(seq[i:i + line_width] + '\n')
            else:
                f.write(seq + '\n')


class FastaWriter:
    """
    FASTA streaming writer

    Supports writing sequences one by one, suitable for generating large numbers of sequences.
    Supports file locks, can be used for multi-process concurrent writes.

    Example:
        with FastaWriter("output.fa", use_lock=True) as writer:
            for seq in generated_sequences:
                writer.write(header, seq)
    """

    def __init__(
        self,
        filepath: str,
        line_width: int = 80,
        use_lock: bool = False,
        append: bool = False
    ):
        """
        Initialize writer

        Args:
            filepath: Output file path
            line_width: Maximum width of each sequence line, 0 means no line breaks
            use_lock: Whether to use file lock (for multi-process writes)
            append: Whether to use append mode
        """
        self.filepath = Path(filepath)
        self.line_width = line_width
        self.use_lock = use_lock
        self.append = append
        self._file = None
        self._count = 0

    def __enter__(self):
        mode = 'a' if self.append else 'w'
        self._file = open(self.filepath, mode, encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False

    def write(self, header: str, sequence: str) -> None:
        """
        Write a single sequence

        Args:
            header: Sequence header (without '>')
            sequence: Sequence content
        """
        if self._file is None:
            raise RuntimeError("FastaWriter not opened, please use with statement")

        content = f">{header}\n"
        if self.line_width > 0:
            for i in range(0, len(sequence), self.line_width):
                content += sequence[i:i + self.line_width] + '\n'
        else:
            content += sequence + '\n'

        if self.use_lock:
            # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 559-568
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            try:
                self._file.write(content)
                self._file.flush()
            finally:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        else:
            self._file.write(content)

        self._count += 1

    @property
    def count(self) -> int:
        """Number of sequences written"""
        return self._count


def count_fasta(filepath: str) -> int:
    """
    Count the number of sequences in a FASTA file

    Args:
        filepath: FASTA file path

    Returns:
        Number of sequences
    """
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count
