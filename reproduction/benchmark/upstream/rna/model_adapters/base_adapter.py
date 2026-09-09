"""Base adapter class for model scoring"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAdapter(ABC):
    """Base class for model adapters"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration

        Args:
            config: Model configuration dictionary
        """
        self.config = config

    @abstractmethod
    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute log-likelihood scores for sequences in FASTA file

        Args:
            fasta_path: Path to FASTA file
            device: GPU device (e.g., 'cuda:0')
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of scores, one per sequence in FASTA file
        """
        pass

    @abstractmethod
    def setup_environment(self):
        """
        Setup runtime environment (load model, activate conda, etc.)
        """
        pass

    def validate_sequences(self, sequences: List[str]) -> bool:
        """
        Validate sequence format

        Args:
            sequences: List of sequences

        Returns:
            True if valid, False otherwise
        """
        return True

    def get_model_name(self) -> str:
        """Get model name from config"""
        return self.config.get('model_name', 'unknown')
