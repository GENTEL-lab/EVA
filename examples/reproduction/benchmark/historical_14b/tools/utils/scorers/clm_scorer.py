"""
CLM Model Scorer

Calls computation scripts inside Docker containers to calculate sequence log-likelihood.
Supports direct scoring of RNA sequences and scoring of protein sequences after reverse translation.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseScorer
from ..conditions import GenerationCondition, LineageDatabase
from ..data.codon_tables import reverse_translate, get_codon_table


class CLMScorer(BaseScorer):
    """
    CLM (Causal Language Model) Scorer

    Calls computation scripts inside Docker containers to calculate sequence log-likelihood.

    Supports two modes:
    - RNA mode: Directly score RNA sequences
    - Protein mode: Reverse translate to RNA first, then score

    Example:
        scorer = CLMScorer(
            checkpoint_path="/path/to/model",
            device="cuda:0"
        )

        # RNA sequence scoring
        scores = scorer.score_rna(["AUGCUAGCUA", "GCUAGCUAGC"])

        # Protein sequence scoring
        scores = scorer.score_protein(["MKTAY", "AKQRQ"])

        # With species condition
        condition = GenerationCondition(taxid="9606")
        scores = scorer.score_rna(sequences, condition=condition)

        # Custom Docker path mapping
        scorer = CLMScorer(
            checkpoint_path="/path/to/model",
            docker_path_mapping=("/host/path", "/container/path")
        )
    """

    # Docker default configuration (can be overridden by parameters)
    DEFAULT_DOCKER_CONTAINER = "eva"
    DEFAULT_DOCKER_PYTHON = "/composer-python/python"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        docker_container: str = None,
        docker_python: str = None,
        docker_path_mapping: Optional[Tuple[str, str]] = None,
        lineage_db: Optional[LineageDatabase] = None
    ):
        """
        Initialize scorer

        Args:
            checkpoint_path: Model checkpoint path (host path or container path)
            device: Compute device
            docker_container: Docker container name (defaults to eva)
            docker_python: Python interpreter path inside Docker container
            docker_path_mapping: Path mapping tuple (host_path, container_path)
                Example: ("/home/user/data", "/data")
                If not specified, will attempt auto-detection or use original path
            lineage_db: Lineage database instance (optional)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.docker_container = docker_container or self.DEFAULT_DOCKER_CONTAINER
        self.docker_python = docker_python or self.DEFAULT_DOCKER_PYTHON
        self.lineage_db = lineage_db or LineageDatabase()

        # Path mapping configuration
        self._path_mapping = docker_path_mapping  # (host_path, docker_path)

        # Get worker script path
        self._script_dir = Path(__file__).parent.parent.parent  # tools/
        self._worker_script = self._script_dir / "utils" / "scorers" / "score_worker.py"

    def score(
        self,
        sequences: List[str],
        condition: Optional[GenerationCondition] = None,
        normalize: bool = False
    ) -> List[float]:
        """
        Calculate log-likelihood of RNA sequences

        Args:
            sequences: List of RNA sequences
            condition: Generation condition (species, RNA type, etc.)
            normalize: Whether to normalize (divide by sequence length)

        Returns:
            List of log-likelihood values
        """
        return self.score_rna(sequences, condition, normalize)

    def score_rna(
        self,
        sequences: List[str],
        condition: Optional[GenerationCondition] = None,
        normalize: bool = False
    ) -> List[float]:
        """
        Score RNA sequences

        Args:
            sequences: List of RNA sequences
            condition: Species condition (optional)
            normalize: Whether to normalize

        Returns:
            List of log-likelihood values
        """
        if not sequences:
            return []

        # Format sequences (add lineage prefix)
        formatted_sequences = [
            self._format_sequence(seq, condition)
            for seq in sequences
        ]

        # Compute via Docker
        reduce_method = 'mean' if normalize else 'sum'
        return self._compute_via_docker(formatted_sequences, reduce_method)

    def score_protein(
        self,
        sequences: List[str],
        condition: Optional[GenerationCondition] = None,
        normalize: bool = False,
        codon_optimization: str = 'first'
    ) -> List[float]:
        """
        Score protein sequences

        Automatically reverse translates protein sequences to RNA, then calculates log-likelihood.

        Args:
            sequences: List of protein sequences (single-letter amino acid codes)
            condition: Species condition (optional)
            normalize: Whether to normalize
            codon_optimization: Codon optimization strategy
                - 'first': Use universal codon table
                - 'most_frequent': Use species-specific optimal codon table

        Returns:
            List of log-likelihood values
        """
        if not sequences:
            return []

        # Get codon table
        taxid = condition.taxid if condition else None
        species = condition.species if condition else None
        codon_table = get_codon_table(taxid, species, codon_optimization)

        # Reverse translate
        rna_sequences = [
            reverse_translate(protein_seq, codon_table)
            for protein_seq in sequences
        ]

        # Score RNA sequences
        return self.score_rna(rna_sequences, condition, normalize)

    def _format_sequence(
        self,
        sequence: str,
        condition: Optional[GenerationCondition] = None
    ) -> str:
        """
        Format sequence (add condition prefix)

        Dynamically constructs prefix based on lineage and rna_type in condition:
        - Only rna_type:  |<rna_TYPE>|{sequence}
        - Only lineage:   |{lineage}|{sequence}
        - Both:           |{lineage};<rna_TYPE>|{sequence}
        - Neither:        {sequence}

        Args:
            sequence: RNA sequence
            condition: Generation condition (species, RNA type, etc.)

        Returns:
            Formatted sequence
        """
        if condition is None:
            return sequence

        # Resolve lineage
        lineage = condition.resolve_lineage(self.lineage_db)

        # Resolve rna_type token
        rna_token = None
        if condition.rna_type is not None:
            from ..conditions.rna_types import get_rna_token
            rna_token = get_rna_token(condition.rna_type)

        # Note: Direction markers 5/3 are added by Docker Worker
        if lineage and rna_token:
            return f"|{lineage};{rna_token}|{sequence}"
        elif lineage:
            return f"|{lineage}|{sequence}"
        elif rna_token:
            return f"|{rna_token}|{sequence}"
        else:
            return sequence

    def _convert_path_to_docker(self, host_path: str) -> str:
        """
        Convert host path to Docker container path

        Args:
            host_path: Host machine path

        Returns:
            Container path
        """
        if self._path_mapping is not None:
            host_base, docker_base = self._path_mapping
            if host_path.startswith(host_base):
                return host_path.replace(host_base, docker_base)
        # If no path mapping configured, return original path
        return host_path

    def _check_docker_container(self) -> bool:
        """Check if Docker container is running"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                check=True
            )
            return self.docker_container in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _compute_via_docker(
        self,
        sequences: List[str],
        reduce_method: str = 'mean'
    ) -> List[float]:
        """
        Compute log-likelihood via Docker

        Args:
            sequences: List of formatted sequences
            reduce_method: Reduction method ('mean' or 'sum')

        Returns:
            List of log-likelihood values
        """
        # 1. Check Docker container
        if not self._check_docker_container():
            raise RuntimeError(
                f"Docker container '{self.docker_container}' is not running. "
                f"Please start the container first: docker start {self.docker_container}"
            )

        # 2. Create temporary directory
        temp_dir = self._script_dir / "tmp"
        temp_dir.mkdir(exist_ok=True)

        # 3. Create temporary JSON file (containing sequences)
        pid = os.getpid()
        temp_input = temp_dir / f"sequences_{pid}.json"
        temp_output = temp_dir / f"output_{pid}.json"

        try:
            # Write input file
            with open(temp_input, 'w') as f:
                json.dump({'sequences': sequences}, f)

            # 4. Convert paths
            docker_input = self._convert_path_to_docker(str(temp_input))
            docker_output = self._convert_path_to_docker(str(temp_output))
            docker_checkpoint = self._convert_path_to_docker(self.checkpoint_path)
            docker_worker = self._convert_path_to_docker(str(self._worker_script))

            # 5. Build Docker command
            cmd = [
                'docker', 'exec', self.docker_container,
                self.docker_python,
                docker_worker,
                docker_input,
                docker_checkpoint,
                self.device,
                docker_output,
                reduce_method
            ]

            # 6. Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            # 7. Read results
            if not temp_output.exists():
                raise RuntimeError(
                    f"Docker computation failed, output file not generated.\n"
                    f"stderr: {result.stderr}"
                )

            with open(temp_output, 'r') as f:
                output = json.load(f)

            if not output.get('success', False):
                error_msg = output.get('error', 'Unknown error')
                raise RuntimeError(f"Docker computation failed: {error_msg}")

            return output['log_likelihoods']

        finally:
            # 8. Clean up temporary files
            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                temp_output.unlink()
