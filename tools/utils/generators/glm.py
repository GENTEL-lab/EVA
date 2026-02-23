"""
GLM (General Language Model) Generator - Span Infilling

Fills masked spans in existing sequences.
Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 304-393
"""

import random
from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGenerator
from ..model.sampler import Sampler
from ..conditions.condition import GenerationCondition
from ..conditions.lineage import LineageDatabase


class GLMGenerator(BaseGenerator):
    """
    GLM (General Language Model) Generator - Span Infilling

    Fills masked spans in existing sequences. Requires input sequences,
    and the model generates masked content based on context.

    Prompt format (reference ORIGINAL_SCRIPT_ANALYSIS.md lines 66-67):
        <bos_glm>|<condition>|5[prefix]<span_X>[suffix]3<eos><span_X>

    Where <span_X> is a span marker (X = 0-49), the model starts generating
    masked content after the second <span_X>, until it outputs <eos_span>.

    Example:
        generator = GLMGenerator(model, tokenizer, sampler)

        # Read input sequences from FASTA file
        sequences = [("seq1", "AUGCAUGCAUGC...")]

        # Generate
        condition = GenerationCondition(rna_type="mRNA")
        results = generator.generate(
            condition=condition,
            input_sequences=sequences,
            span_ratio=0.1,
            span_position="random"
        )
    """

    def build_prompt(
        self,
        condition: Optional[GenerationCondition] = None,
        prefix: str = "",
        suffix: str = "",
        span_id: int = 0,
        **kwargs
    ) -> str:
        """
        Build GLM generation prompt

        Args:
            condition: Generation conditions
            prefix: Sequence before span
            suffix: Sequence after span
            span_id: Span marker ID (0-49)

        Returns:
            GLM prompt string
        """
        if condition is None:
            condition = GenerationCondition()
        return condition.build_glm_prompt(
            prefix=prefix,
            suffix=suffix,
            span_id=span_id,
            lineage_db=self.lineage_db
        )

    def generate(
        self,
        condition: Optional[GenerationCondition] = None,
        input_sequences: List[Tuple[str, str]] = None,
        span_length: Optional[int] = None,
        span_ratio: Optional[float] = None,
        span_position: str = "random",
        span_id: str = "random",
        num_seqs: int = 1,
        max_length: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict]:
        """
        GLM Span Infilling generation

        Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 307-341

        Args:
            condition: Generation conditions
            input_sequences: Input sequence list, format: [(header, sequence), ...]
            span_length: Fixed span length (choose one between span_length and span_ratio)
            span_ratio: Span length as ratio of sequence (choose one between span_length and span_ratio)
            span_position: Span start position, "random" or specific value
            span_id: Span marker ID, "random" or integer 0-49
            num_seqs: Number of samples to generate per input sequence
            max_length: Maximum generation length
            batch_size: Ignored (GLM generates one at a time)

        Returns:
            List of generation results, each element contains:
            {
                "header": Original sequence header,
                "original": Original sequence,
                "prefix": Sequence before span,
                "generated": Generated span content,
                "suffix": Sequence after span,
                "full": Complete generated sequence (prefix + generated + suffix),
                "ground_truth": Original masked content,
                "span_start": Span start position,
                "span_length": Span length,
                "span_id": Span ID used,
                "sample_id": Sample number (starting from 1)
            }
        """
        if input_sequences is None or len(input_sequences) == 0:
            raise ValueError("GLM mode requires input_sequences")

        if span_length is None and span_ratio is None:
            raise ValueError("Must specify span_length or span_ratio")

        results = []

        for header, sequence in input_sequences:
            # Calculate span length
            if span_length is not None:
                actual_span_length = span_length
            else:
                actual_span_length = max(1, int(len(sequence) * span_ratio))

            # Ensure span length does not exceed sequence length
            actual_span_length = min(actual_span_length, len(sequence) - 2)

            # Calculate span position
            if span_position == "random":
                max_start = len(sequence) - actual_span_length
                actual_span_start = random.randint(0, max_start) if max_start > 0 else 0
            else:
                actual_span_start = int(span_position)

            # Select span ID
            if span_id == "random":
                actual_span_id = random.randint(0, 49)
            else:
                actual_span_id = int(span_id)

            # Extract parts
            prefix = sequence[:actual_span_start]
            ground_truth = sequence[actual_span_start:actual_span_start + actual_span_length]
            suffix = sequence[actual_span_start + actual_span_length:]

            # Build prompt
            prompt = self.build_prompt(
                condition=condition,
                prefix=prefix,
                suffix=suffix,
                span_id=actual_span_id
            )

            # Generate num_seqs samples for each input sequence
            for sample_id in range(1, num_seqs + 1):
                # Generate
                generated_span = self._generate_single(
                    prompt=prompt,
                    span_id=actual_span_id,
                    span_length=actual_span_length,
                    max_length=max_length
                )

                results.append({
                    "header": header,
                    "original": sequence,
                    "prefix": prefix,
                    "generated": generated_span,
                    "suffix": suffix,
                    "full": prefix + generated_span + suffix,
                    "ground_truth": ground_truth,
                    "span_start": actual_span_start,
                    "span_length": actual_span_length,
                    "span_id": actual_span_id,
                    "sample_id": sample_id
                })

        return results

    def _calculate_position_ids(
        self,
        input_ids: List[int],
        span_id: int,
        span_length: int
    ) -> List[int]:
        """
        Calculate jumping position_ids for GLM prompt

        Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 304-335

        Prompt format: <bos_glm>|<rna_type>|5[prefix]<span_X>[suffix]3<eos><span_X>
        Two <span_X> share the same position, first <span_X> jumps span_length positions

        Args:
            input_ids: Encoded token ID list
            span_id: Span marker ID
            span_length: Span length

        Returns:
            position_ids list
        """
        position_ids = []
        current_pos = 0

        span_token_id = self.tokenizer.token_to_id(f"<span_{span_id}>")
        span_start_position = None

        for i, token_id in enumerate(input_ids):
            if token_id == span_token_id:
                if span_start_position is None:
                    # First <span_X>: record position and jump
                    # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 323-326
                    span_start_position = current_pos
                    position_ids.append(current_pos)
                    current_pos += 1 + max(1, span_length)  # Skip span marker + span content
                else:
                    # Second <span_X>: reset to first position
                    # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 328-329
                    position_ids.append(span_start_position)
            else:
                position_ids.append(current_pos)
                current_pos += 1

        return position_ids

    def _generate_single(
        self,
        prompt: str,
        span_id: int,
        span_length: int,
        max_length: int
    ) -> str:
        """
        Single GLM span infilling generation

        Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 340-393

        Args:
            prompt: GLM prompt
            span_id: Span marker ID
            span_length: Expected span length
            max_length: Maximum generation length

        Returns:
            Generated span content
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Pre-calculate jumping position_ids for prompt
        # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 350-354
        prompt_pos_ids = self._calculate_position_ids(input_ids, span_id, span_length)
        current_pos = prompt_pos_ids[-1] + 1
        pos_ids = torch.tensor([prompt_pos_ids], dtype=torch.long, device=self.device)

        generated_span = ''

        with torch.no_grad():
            # Reference: ORIGINAL_SCRIPT_ANALYSIS.md line 357
            for step in range(max_length + 100):
                # Build sequence_ids
                seq_ids = torch.zeros_like(current_ids)

                # Forward pass
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 359-361
                outputs = self.model(
                    input_ids=current_ids,
                    position_ids=pos_ids,
                    sequence_ids=seq_ids,
                    use_cache=False
                )
                logits = outputs.logits

                # Sample next token
                next_token = self.sampler.sample(logits[:, -1, :])

                tok_id = next_token[0].item()
                tok_str = self.tokenizer.id_to_token(tok_id)

                # Check if finished
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 377-378
                if tok_id == self._eos_span_id or tok_str == '<eos_span>':
                    break

                # Only collect nucleotides
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 381-384
                if self._is_nucleotide(tok_str):
                    generated_span += tok_str
                    if len(generated_span) >= max_length:
                        break

                # Append new token and new position
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 387-390
                current_ids = torch.cat([current_ids, next_token], dim=1)
                next_pos = torch.tensor([[current_pos]], dtype=torch.long, device=self.device)
                pos_ids = torch.cat([pos_ids, next_pos], dim=1)
                current_pos += 1

        return generated_span
