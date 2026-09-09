"""
Scorer Abstract Base Class

Defines the unified interface for all scorers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

# Avoid circular imports, use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..conditions import GenerationCondition


class BaseScorer(ABC):
    """
    Scorer Abstract Base Class

    Defines the unified interface for all scorers. Subclasses need to implement the score method.

    Example:
        class MyScorer(BaseScorer):
            def score(self, sequences, condition=None, normalize=False):
                # Implement scoring logic
                return [0.0] * len(sequences)
    """

    @abstractmethod
    def score(
        self,
        sequences: List[str],
        condition: Optional['GenerationCondition'] = None,
        normalize: bool = False
    ) -> List[float]:
        """
        Calculate the log-likelihood of sequences

        Args:
            sequences: List of RNA sequences
            condition: Generation condition (species, RNA type, etc.)
            normalize: Whether to normalize (divide by sequence length)

        Returns:
            List of log-likelihood values, corresponding one-to-one with input sequences
        """
        pass
