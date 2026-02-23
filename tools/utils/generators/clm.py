"""
CLM (Causal Language Model) Generator

Generates complete RNA sequences using autoregressive left-to-right generation.
Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 248-302, generate_batch_clm function

Supports three modes:
1. Unconditional/Conditional generation: Generate complete sequences starting from <bos>
2. Forward continuation: Keep left half of sequence (5' end), generate right half (3' end)
3. Reverse continuation: Keep right half of sequence (3' end), generate left half (5' end)
"""

from typing import List, Optional, Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGenerator
from ..model.sampler import Sampler
from ..conditions.condition import GenerationCondition
from ..conditions.lineage import LineageDatabase


class CLMGenerator(BaseGenerator):
    """
    CLM (Causal Language Model) Generator

    Generates complete RNA sequences using autoregressive left-to-right generation.

    Supports three modes:
    1. Unconditional/Conditional generation: Generate complete sequences starting from <bos>
    2. Forward continuation: Keep left half of sequence (5' end), generate right half (3' end)
    3. Reverse continuation: Keep right half of sequence (3' end), generate left half (5' end)

    Prompt format (reference ORIGINAL_SCRIPT_ANALYSIS.md lines 53-64):
        - Unconditional: <bos>5
        - RNA type only: <bos>|<rna_mRNA>|5
        - Species only: <bos>|d__eukaryota;...;s__homo_sapiens|5
        - RNA type + species: <bos>|d__...;s__homo_sapiens;<rna_mRNA>|5

    Technical background (bidirectional continuation):
        Model training uses both forward (5'→3') and reverse (3'→5') sequences:
        - Forward training: <bos> + sequence + <eos>
        - Reverse training: <bos> + physically reversed sequence + <eos> (no special direction marker)

    Example:
        generator = CLMGenerator(model, tokenizer, sampler)

        # Unconditional generation
        seqs = generator.generate(num_seqs=100)

        # Conditional generation
        condition = GenerationCondition(rna_type="mRNA", taxid="9606")
        seqs = generator.generate(condition=condition, num_seqs=100)

        # Forward continuation (generate 3' end)
        results = generator.generate(
            input_sequences=[("seq1", "AUGCAUGC...")],
            direction="forward",
            split_ratio=0.5
        )

        # Reverse continuation (generate 5' end)
        results = generator.generate(
            input_sequences=[("seq1", "AUGCAUGC...")],
            direction="reverse",
            split_pos=100
        )
    """

    def build_prompt(
        self,
        condition: Optional[GenerationCondition] = None,
        **kwargs
    ) -> str:
        """
        Build CLM generation prompt

        Args:
            condition: Generation conditions

        Returns:
            CLM prompt string
        """
        if condition is None:
            condition = GenerationCondition()
        return condition.build_clm_prompt(lineage_db=self.lineage_db)

    def generate(
        self,
        condition: Optional[GenerationCondition] = None,
        num_seqs: int = 1,
        max_length: int = 8192,
        batch_size: int = 1,
        min_length: int = 10,
        input_sequences: Optional[List[Tuple[str, str]]] = None,
        direction: str = "forward",
        split_pos: Optional[int] = None,
        split_ratio: Optional[float] = None,
        **kwargs
    ) -> Union[List[str], List[Dict[str, str]]]:
        """
        Generate RNA sequences

        Supports two modes:
        1. No input sequences: Standard unconditional/conditional generation, returns List[str]
        2. With input sequences: Continuation mode, returns List[Dict] with detailed information

        Args:
            condition: Generation conditions
            num_seqs: Number of sequences to generate
                - Without input sequences: Total number of sequences to generate
                - With input sequences: Number of samples to generate per input sequence
            max_length: Maximum sequence length (including prompt part)
            batch_size: Batch size
            min_length: Minimum sequence length, sequences shorter than this will be discarded
            input_sequences: Input sequence list, format: [(header, sequence), ...]
            direction: Continuation direction, "forward" (generate 3' end) or "reverse" (generate 5' end)
            split_pos: Split position (fixed position)
            split_ratio: Split ratio (choose one between split_pos and split_ratio)

        Returns:
            Without input sequences: List of generated RNA sequences List[str]
            With input sequences: List of dictionaries with detailed information List[Dict], each containing:
                - header: Original sequence header
                - prompt: Retained sequence part (biological 5'→3')
                - ground_truth: Masked sequence part (biological 5'→3')
                - generated: Generated sequence part (biological 5'→3')
                - full: Complete reconstructed sequence (biological 5'→3')
                - sample_id: Sample number (starting from 1)
        """
        # If no input sequences, use standard generation mode
        if input_sequences is None:
            return self._generate_standard(
                condition=condition,
                num_seqs=num_seqs,
                max_length=max_length,
                batch_size=batch_size,
                min_length=min_length
            )

        # Continuation mode
        return self._generate_continuation(
            condition=condition,
            input_sequences=input_sequences,
            num_seqs=num_seqs,
            direction=direction,
            split_pos=split_pos,
            split_ratio=split_ratio,
            max_length=max_length,
            batch_size=batch_size
        )

    def _generate_standard(
        self,
        condition: Optional[GenerationCondition] = None,
        num_seqs: int = 1,
        max_length: int = 8192,
        batch_size: int = 1,
        min_length: int = 10
    ) -> List[str]:
        """
        Standard generation mode (unconditional/conditional generation)

        Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 248-302

        Args:
            condition: Generation conditions
            num_seqs: Number of sequences to generate
            max_length: Maximum sequence length
            batch_size: Batch size
            min_length: Minimum sequence length

        Returns:
            List of generated RNA sequences
        """
        # Build prompt
        prompt = self.build_prompt(condition)

        all_sequences = []
        remaining = num_seqs

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch_seqs = self._generate_batch(
                prompt=prompt,
                batch_size=current_batch,
                max_length=max_length
            )

            # Filter out sequences that are too short
            valid_seqs = [s for s in batch_seqs if len(s) >= min_length]
            all_sequences.extend(valid_seqs)
            remaining -= current_batch

        return all_sequences[:num_seqs]

    def _generate_continuation(
        self,
        condition: Optional[GenerationCondition] = None,
        input_sequences: List[Tuple[str, str]] = None,
        num_seqs: int = 1,
        direction: str = "forward",
        split_pos: Optional[int] = None,
        split_ratio: Optional[float] = None,
        max_length: int = 8192,
        batch_size: int = 1
    ) -> List[Dict[str, str]]:
        """
        Continuation mode generation

        Args:
            condition: Generation conditions
            input_sequences: Input sequence list [(header, sequence), ...]
            num_seqs: Number of samples to generate per input sequence
            direction: Continuation direction "forward" or "reverse"
            split_pos: Split position
            split_ratio: Split ratio
            max_length: Maximum sequence length
            batch_size: Batch size

        Returns:
            List of dictionaries with detailed information
        """
        results = []

        for header, sequence in input_sequences:
            # Calculate split point
            seq_len = len(sequence)
            if split_pos is not None:
                pos = min(split_pos, seq_len)
            elif split_ratio is not None:
                pos = int(seq_len * split_ratio)
            else:
                # No split point specified, use entire sequence as prompt (pure continuation)
                pos = seq_len if direction == "forward" else 0

            # Split sequence based on direction
            if direction == "forward":
                # Forward: left is prompt, right is ground_truth
                prompt_seq = sequence[:pos]  # 5' end retained
                ground_truth = sequence[pos:]  # 3' end masked
                model_prompt_seq = prompt_seq  # Use directly, no reversal needed
            else:
                # Reverse: right is prompt, left is ground_truth
                prompt_seq = sequence[pos:]  # 3' end retained (biological order)
                ground_truth = sequence[:pos]  # 5' end masked (biological order)
                # Key: physically reverse prompt sequence before feeding to model
                model_prompt_seq = prompt_seq[::-1]

            # Build complete prompt (including <bos> and conditions)
            base_prompt = self.build_prompt(condition)
            # base_prompt format like "<bos>5" or "<bos>|<rna_mRNA>|5"
            # Need to append sequence after 5
            full_prompt = base_prompt + model_prompt_seq

            # Check prompt length
            prompt_tokens = self.tokenizer.encode(full_prompt)
            if len(prompt_tokens) >= max_length:
                print(f"Warning: Sequence {header} prompt length ({len(prompt_tokens)}) "
                      f"exceeds max_length ({max_length}), skipping")
                continue

            # Generate num_seqs samples for each input sequence
            for sample_id in range(1, num_seqs + 1):
                # Generate
                generated_seqs = self._generate_batch(
                    prompt=full_prompt,
                    batch_size=1,
                    max_length=max_length,
                    direction=direction
                )

                if not generated_seqs:
                    generated_raw = ""
                else:
                    generated_raw = generated_seqs[0]

                # Post-processing: Reverse mode needs to reverse generation result
                if direction == "forward":
                    generated = generated_raw
                    # Reconstruct complete sequence: prompt + generated
                    full_seq = prompt_seq + generated
                else:
                    # Key: physically reverse generation result back
                    generated = generated_raw[::-1]
                    # Reconstruct complete sequence: generated + prompt (biological 5'→3' order)
                    full_seq = generated + prompt_seq

                results.append({
                    "header": header,
                    "prompt": prompt_seq,  # Biological 5'→3'
                    "ground_truth": ground_truth,  # Biological 5'→3'
                    "generated": generated,  # Biological 5'→3'
                    "full": full_seq,  # Biological 5'→3'
                    "direction": direction,
                    "split_pos": pos,
                    "sample_id": sample_id
                })

        return results

    def _generate_batch(
        self,
        prompt: str,
        batch_size: int,
        max_length: int,
        direction: str = "forward"
    ) -> List[str]:
        """
        Batch generate RNA sequences

        Strictly references: ORIGINAL_SCRIPT_ANALYSIS.md lines 251-301

        Args:
            prompt: Generation prompt
            batch_size: Batch size
            max_length: Maximum sequence length
            direction: Generation direction, "forward" or "reverse", used to determine end marker

        Returns:
            List of generated sequences
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        current_ids = torch.tensor(
            [input_ids] * batch_size,
            dtype=torch.long,
            device=self.device
        )

        # Initialize sequences and completion status
        seqs = ['' for _ in range(batch_size)]
        finished = [False] * batch_size

        with torch.no_grad():
            # Reference original script: for step in range(max_length + 100)
            for step in range(max_length + 100):
                if all(finished):
                    break

                # Build position_ids and sequence_ids
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 267-269
                seq_len = current_ids.shape[1]
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
                pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
                seq_ids = torch.zeros_like(current_ids)

                # Forward pass
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 272-274
                outputs = self.model(
                    input_ids=current_ids,
                    position_ids=pos_ids,
                    sequence_ids=seq_ids,
                    use_cache=False
                )
                logits = outputs.logits

                # Sample next token
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 277-284
                next_tokens = self.sampler.sample(logits[:, -1, :])

                # Update sequences
                # Reference: ORIGINAL_SCRIPT_ANALYSIS.md lines 287-298
                for i in range(batch_size):
                    if finished[i]:
                        continue

                    tok_id = next_tokens[i].item()
                    tok_str = self.tokenizer.id_to_token(tok_id)

                    # Check if finished
                    if self._is_end_token(tok_id, tok_str, direction):
                        finished[i] = True
                    # Only collect nucleotides
                    elif self._is_nucleotide(tok_str):
                        seqs[i] += tok_str
                        if len(seqs[i]) >= max_length:
                            finished[i] = True

                # Append new token
                current_ids = torch.cat([current_ids, next_tokens], dim=1)

        return seqs
