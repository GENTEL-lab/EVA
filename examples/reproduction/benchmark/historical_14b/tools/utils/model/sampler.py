"""
Sampling strategies

Encapsulates various sampling methods to sample the next token from model output logits.
Supports temperature scaling, top-k, and top-p (nucleus) sampling.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class Sampler:
    """
    Sampling strategy

    Encapsulates various sampling methods to sample the next token from model output logits.
    Supports temperature scaling, top-k, and top-p (nucleus) sampling.

    Example:
        sampler = Sampler(temperature=0.8, top_k=50)
        next_token = sampler.sample(logits)

        # Use top-p sampling
        sampler = Sampler(temperature=1.0, top_p=0.9)
        next_token = sampler.sample(logits)

        # Combine top-k and top-p
        sampler = Sampler(temperature=0.8, top_k=50, top_p=0.95)
        next_token = sampler.sample(logits)

    Attributes:
        temperature: Temperature parameter
        top_k: Top-K sampling parameter
        top_p: Top-P (Nucleus) sampling parameter
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None
    ):
        """
        Initialize sampler

        Args:
            temperature: Temperature parameter, controls generation randomness
                - temperature < 1.0: More deterministic, favors high-probability tokens
                - temperature > 1.0: More random, flatter distribution
                - temperature = 1.0: Original distribution
            top_k: Top-K sampling, only sample from the K highest probability tokens
                - None or 0 means do not use top-k
            top_p: Top-P (Nucleus) sampling, sample from the smallest token set with cumulative probability reaching P
                - None means do not use top-p
                - Can be used together with top_k

        Raises:
            ValueError: If parameters are invalid
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be greater than 0, current value: {temperature}")
        if top_k is not None and top_k < 0:
            raise ValueError(f"top_k must be non-negative, current value: {top_k}")
        if top_p is not None and (top_p <= 0 or top_p > 1):
            raise ValueError(f"top_p must be in range (0, 1], current value: {top_p}")

        self.temperature = temperature
        self.top_k = top_k if top_k and top_k > 0 else None
        self.top_p = top_p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from logits

        Args:
            logits: Model output logits, shape is (batch_size, vocab_size)

        Returns:
            Sampled token ID, shape is (batch_size, 1)
        """
        # Ensure logits is 2D
        if logits.dim() == 3:
            # If (batch_size, seq_len, vocab_size), take the last position
            logits = logits[:, -1, :]

        # Temperature scaling
        scaled_logits = logits.float() / self.temperature

        # Top-K filtering
        if self.top_k is not None:
            scaled_logits = self._top_k_filter(scaled_logits, self.top_k)

        # Top-P filtering
        if self.top_p is not None:
            scaled_logits = self._top_p_filter(scaled_logits, self.top_p)

        # Calculate probability distribution
        probs = F.softmax(scaled_logits, dim=-1)

        # Multinomial sampling
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Top-K filtering: only keep the K highest probability tokens

        Args:
            logits: Input logits, shape (batch_size, vocab_size)
            k: Number of tokens to keep

        Returns:
            Filtered logits
        """
        # Get top-k values and indices
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        # Get the k-th largest value as threshold
        threshold = top_k_values[:, -1:]
        # Set logits below threshold to negative infinity
        filtered_logits = torch.where(
            logits < threshold,
            torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
            logits
        )
        return filtered_logits

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Top-P (Nucleus) filtering: keep the smallest token set with cumulative probability reaching P

        Args:
            logits: Input logits, shape (batch_size, vocab_size)
            p: Cumulative probability threshold

        Returns:
            Filtered logits
        """
        # Sort by probability in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Calculate cumulative probability
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find positions where cumulative probability exceeds p
        # Keep the first token that exceeds p (ensure at least one token)
        sorted_indices_to_remove = cumulative_probs > p
        # Shift right by one to ensure the first token exceeding threshold is kept
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Set positions to remove to negative infinity
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Restore original order
        # Create reverse indices
        batch_size, vocab_size = logits.shape
        original_indices = torch.zeros_like(sorted_indices)
        original_indices.scatter_(1, sorted_indices, torch.arange(vocab_size, device=logits.device).expand(batch_size, -1))

        # Use reverse indices to restore order
        filtered_logits = sorted_logits.gather(1, original_indices)

        return filtered_logits

    def greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding: select the highest probability token

        Args:
            logits: Model output logits, shape is (batch_size, vocab_size)

        Returns:
            Selected token ID, shape is (batch_size, 1)
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits.argmax(dim=-1, keepdim=True)

    def __repr__(self) -> str:
        params = [f"temperature={self.temperature}"]
        if self.top_k is not None:
            params.append(f"top_k={self.top_k}")
        if self.top_p is not None:
            params.append(f"top_p={self.top_p}")
        return f"Sampler({', '.join(params)})"
