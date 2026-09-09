"""
Generator Abstract Base Class

Defines a unified interface for all generators. Both CLM and GLM generators inherit from this class,
implementing their respective prompt construction and generation logic.

Design Pattern: Strategy Pattern
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any

import torch
import torch.nn as nn

from ..model.sampler import Sampler
from ..conditions.condition import GenerationCondition
from ..conditions.lineage import LineageDatabase


class BaseGenerator(ABC):
    """
    Generator Abstract Base Class

    Defines a unified interface for all generators. Both CLM and GLM generators inherit from this class,
    implementing their respective prompt construction and generation logic.

    Design Pattern: Strategy Pattern
    Eliminates if-else branching at the application layer through a unified interface.

    Attributes:
        model: Loaded model
        tokenizer: Tokenizer
        sampler: Sampler
        device: Computing device
        lineage_db: Lineage database (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        sampler: Sampler,
        device: str = "cuda:0",
        lineage_db: Optional[LineageDatabase] = None
    ):
        """
        Initialize generator

        Args:
            model: Loaded model
            tokenizer: Tokenizer
            sampler: Sampler
            device: Computing device
            lineage_db: Lineage database (for parsing species conditions)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.device = device
        self.lineage_db = lineage_db if lineage_db is not None else LineageDatabase()

        # Cache commonly used token IDs
        self._eos_id = self.tokenizer.token_to_id('<eos>')
        self._eos_span_id = self.tokenizer.token_to_id('<eos_span>')

        # Direction marker token IDs
        self._3prime_id = self.tokenizer.token_to_id('3')  # Forward end marker
        self._5prime_id = self.tokenizer.token_to_id('5')  # Reverse end marker

        # Nucleotide set
        self._nucleotides = {'A', 'U', 'G', 'C'}

    @abstractmethod
    def generate(
        self,
        condition: Optional[GenerationCondition] = None,
        num_seqs: int = 1,
        max_length: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate RNA sequences

        Args:
            condition: Generation conditions (RNA type, species, etc.), None for unconditional generation
            num_seqs: Number of sequences to generate
            max_length: Maximum sequence length
            batch_size: Batch size
            **kwargs: Subclass-specific parameters

        Returns:
            List of generated RNA sequences
        """
        pass

    @abstractmethod
    def build_prompt(
        self,
        condition: Optional[GenerationCondition] = None,
        **kwargs
    ) -> str:
        """
        Build generation prompt

        Args:
            condition: Generation conditions
            **kwargs: Subclass-specific parameters

        Returns:
            Prompt string
        """
        pass

    def _is_end_token(self, token_id: int, token_str: str, direction: str = "forward") -> bool:
        """
        Check if token is an end token

        Determines end marker based on generation direction:
        - Forward generation: end marker is <eos> or 3
        - Reverse generation: end marker is <eos> or 5

        This is because during model training:
        - Forward sequences: mask the first direction label 5, compute loss for the last direction label 3
        - Reverse sequences: mask the first direction label 3, compute loss for the last direction label 5

        Args:
            token_id: Token ID
            token_str: String representation of token
            direction: Generation direction, "forward" or "reverse"

        Returns:
            Whether it is an end token
        """
        # Universal end marker
        if token_id == self._eos_id or token_str in ['<eos>']:
            return True

        # Determine direction marker based on direction
        if direction == "forward":
            # Forward generation: 3 is the end marker
            return token_id == self._3prime_id or token_str == '3'
        else:
            # Reverse generation: 5 is the end marker
            return token_id == self._5prime_id or token_str == '5'

    def _is_nucleotide(self, token_str: str) -> bool:
        """Check if token is a nucleotide"""
        return token_str in self._nucleotides
