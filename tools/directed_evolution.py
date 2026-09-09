#!/usr/bin/env python3
"""
Directed Evolution Script for RNA Design

This script performs directed evolution on RNA sequences using:
- Point mutations (not codon replacement)
- LLM-based scoring (compute_batch_likelihood)
- Dynamic beam search with simulated annealing
- MFE calculations via ViennaRNA

Supports all RNA types through tools.utils.conditions module.
"""

import argparse
import random
import time
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from tools.utils.io import read_fasta, write_fasta
from tools.utils.model import ModelLoader
from tools.utils.scorers.score_worker import compute_batch_likelihood
from tools.utils.conditions import GenerationCondition, LineageDatabase, get_rna_token, list_rna_types


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Directed Evolution for RNA Design",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input FASTA file with single RNA sequence")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output FASTA file for results")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file (overrides CLI args)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint directory")

    # RNA Type and Species Conditions
    parser.add_argument("--rna_type", type=str, default=None,
                        help="RNA type (mRNA, tRNA, circRNA, etc.)")
    parser.add_argument("--taxid", type=str, default=None,
                        help="NCBI Taxonomy ID (e.g., 9606)")
    parser.add_argument("--species", type=str, default=None,
                        help="Species name (e.g., homo_sapiens)")
    parser.add_argument("--lineage", type=str, default=None,
                        help="Full Greengenes lineage string")

    # Evolution Parameters
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of evolution iterations (default: 10)")
    parser.add_argument("--mutations", type=int, default=2,
                        help="Total mutations per iteration (default: 2)")
    parser.add_argument("--mutations_per_iter", type=int, default=1,
                        help="Mutations applied in each iteration step (default: 1)")
    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam search width (default: 10)")
    parser.add_argument("--output_count", type=int, default=5,
                        help="Number of output sequences (default: 5)")

    # Simulated Annealing Parameters
    parser.add_argument("--T_init", type=float, default=1.0,
                        help="Initial temperature (default: 1.0)")
    parser.add_argument("--T_min", type=float, default=0.01,
                        help="Minimum temperature (default: 0.01)")
    parser.add_argument("--cooling_rate", type=float, default=0.95,
                        help="Cooling rate (default: 0.95)")

    # Mutation Position Parameters
    parser.add_argument("--mutate_positions", type=str, default=None,
                        help="Comma-separated positions to mutate (0-based)")
    parser.add_argument("--mutate_range", type=str, default=None,
                        help="Range of positions to mutate (e.g., '0-100')")

    # Model Parameters
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type (bfloat16, float32, etc.)")

    # Additional Options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Get parser defaults for config merging
    parser_defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            parser_defaults[action.dest] = action.default

    return args, parser_defaults


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args, parser_defaults: dict) -> None:
    """Merge YAML config with command line arguments.

    Priority: CLI args > Config file > Parser defaults
    """
    for key, value in config.items():
        if value is not None:
            # Check if this argument was explicitly provided on command line
            current_value = getattr(args, key, None)
            default_value = parser_defaults.get(key)

            # Only override if the current value is the default (not explicitly set by user)
            if current_value == default_value:
                setattr(args, key, value)


def parse_mutate_positions(positions_str: str, seq_length: int) -> List[int]:
    """Parse comma-separated positions string to list of integers."""
    positions = []
    for part in positions_str.split(','):
        part = part.strip()
        if part.isdigit():
            pos = int(part)
            if 0 <= pos < seq_length:
                positions.append(pos)
    return positions


def parse_mutate_range(range_str: str, seq_length: int) -> List[int]:
    """Parse range string (e.g., '0-100') to list of integers."""
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'start-end'")
    start = int(parts[0].strip())
    end = int(parts[1].strip())
    start = max(0, start)
    end = min(seq_length, end)
    return list(range(start, end))


def generate_mutations_single_position(sequence: str, position: int) -> List[str]:
    """Generate mutations for a single position.

    Args:
        sequence: Input RNA sequence
        position: Position to mutate

    Returns:
        List of 4 mutated sequences (including original base)
    """
    bases = ['A', 'U', 'C', 'G']
    candidates = []

    for new_base in bases:
        mut_seq = sequence[:position] + new_base + sequence[position+1:]
        candidates.append(mut_seq)

    return candidates


def generate_mutations_with_beam_search(
    initial_sequence: str,
    positions: List[int],
    beam_width: int,
    score_fn
) -> List[str]:
    """Generate mutations with incremental beam search to avoid exponential explosion.

    Args:
        initial_sequence: Starting RNA sequence
        positions: Positions to mutate
        beam_width: Maximum number of candidates to keep after each position
        score_fn: Function to score candidates, takes List[str] and returns List[float]

    Returns:
        List of final candidate sequences (up to beam_width)
    """
    if not positions:
        return [initial_sequence]

    # Start with the initial sequence
    current_candidates = [initial_sequence]

    # Process each mutation position incrementally
    for pos_idx, position in enumerate(positions):
        print(f"  Processing mutation position {pos_idx + 1}/{len(positions)}: {position}")

        # Generate mutations for all current candidates at this position
        next_candidates = []
        for candidate in current_candidates:
            mutations = generate_mutations_single_position(candidate, position)
            next_candidates.extend(mutations)

        print(f"    Generated {len(next_candidates)} candidates")

        # Apply beam search if we exceed beam_width
        if len(next_candidates) > beam_width:
            # Score all candidates
            scores = score_fn(next_candidates)

            # Sort by score and keep top beam_width
            scored_pairs = list(zip(next_candidates, scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            current_candidates = [seq for seq, _ in scored_pairs[:beam_width]]

            print(f"    Beam search: kept top {len(current_candidates)} candidates")
        else:
            current_candidates = next_candidates
            print(f"    Kept all {len(current_candidates)} candidates (< beam_width)")

    return current_candidates


def dynamic_beam_search(candidates: List[str], scores: List[float], beam_width: int,
                        temperature: float, T_init: float, output_count: int) -> List[Tuple[str, float]]:
    """Dynamic beam search - expands beam width as temperature decreases.

    Args:
        candidates: Candidate sequences
        scores: LLM scores (log-likelihoods)
        beam_width: Base beam width
        temperature: Current temperature
        T_init: Initial temperature
        output_count: Maximum output count

    Returns:
        List of (sequence, score) tuples, sorted by score descending
    """
    # Dynamic beam width: as temperature decreases, we keep more candidates
    current_beam_width = min(int(beam_width * T_init / max(temperature, 0.001)), output_count)
    current_beam_width = max(current_beam_width, 1)

    # Sort by score and keep top-k
    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    return scored_candidates[:current_beam_width]


def format_sequence(sequence: str, condition: Optional[GenerationCondition] = None,
                   lineage_db: Optional[LineageDatabase] = None) -> str:
    """Format sequence with condition prefix for scoring.

    Args:
        sequence: RNA sequence
        condition: Generation condition (RNA type + species)
        lineage_db: Lineage database for resolving species

    Returns:
        Formatted sequence string for model input
    """
    if condition is None:
        return sequence

    # Build the conditional prompt
    prompt = condition.build_clm_prompt(lineage_db)
    # The scoring worker adds BOS and direction markers itself.
    if not prompt.startswith('<bos>') or not prompt.endswith('5'):
        raise ValueError('Unexpected CLM prompt format')
    return prompt[len('<bos>'):-1] + sequence


def calculate_mfe(sequence: str) -> float:
    """Calculate Minimum Free Energy using ViennaRNA.

    Args:
        sequence: RNA sequence

    Returns:
        MFE value in kcal/mol
    """
    try:
        import RNA
    except ImportError as exc:
        raise RuntimeError(
            "ViennaRNA is required for the MFE optimization objective. "
            "Install it with pip install '.[design]'."
        ) from exc
    structure, mfe = RNA.fold_compound(sequence).mfe()
    mfe = float(mfe)
    if not np.isfinite(mfe):
        raise ValueError("ViennaRNA returned a non-finite MFE")
    return mfe


def calculate_mfe_batch(sequences: List[str]) -> List[float]:
    """Calculate MFE for a batch of sequences.

    Args:
        sequences: List of RNA sequences

    Returns:
        List of MFE values
    """
    mfe_values = []
    for seq in sequences:
        mfe = calculate_mfe(seq)
        mfe_values.append(mfe)
    return mfe_values


def simulated_annealing_accept(current_score: float, new_score: float,
                               temperature: float) -> bool:
    """Simulated annealing acceptance decision.

    Args:
        current_score: Current sequence score
        new_score: New candidate sequence score
        temperature: Current temperature

    Returns:
        True if should accept new sequence, False otherwise
    """
    if new_score >= current_score:
        return True

    if temperature <= 0:
        return False

    # Calculate acceptance probability
    delta = new_score - current_score
    probability = np.exp(delta / temperature)

    return random.random() < probability


class DirectedEvolution:
    """Main directed evolution class."""

    def __init__(self, args):
        """Initialize directed evolution with configuration.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.model = None
        self.tokenizer = None
        self.condition = None
        self.lineage_db = None
        self.input_sequence = None
        self.input_header = None
        self.current_sequences = []
        self.current_scores = []
        self.log_entries = []

        # Set random seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.manual_seed(args.seed)

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.args.checkpoint}...")
        dtype = getattr(torch, self.args.dtype, torch.bfloat16)
        loader = ModelLoader(self.args.checkpoint)
        self.model, self.tokenizer = loader.load(self.args.device, dtype)
        self.model.eval()
        print("Model loaded successfully.")

    def load_sequence(self):
        """Load input sequence from FASTA file."""
        print(f"Loading sequence from {self.args.input}...")
        sequences = read_fasta(self.args.input)

        if len(sequences) == 0:
            raise ValueError(f"No sequences found in {self.args.input}")

        if len(sequences) > 1:
            print(f"Warning: Multiple sequences found in {self.args.input}. Using first one.")

        self.input_header, self.input_sequence = sequences[0]

        # Validate sequence (only A, U, C, G)
        seq_upper = self.input_sequence.upper()
        valid_bases = set('AUCG')
        if not all(c in valid_bases for c in seq_upper):
            raise ValueError(f"Invalid sequence: contains non-RNA characters. Only A, U, C, G allowed.")

        self.input_sequence = seq_upper
        print(f"Loaded sequence: {len(self.input_sequence)} bases")

    def setup_conditions(self):
        """Setup RNA type and species conditions."""
        if any([self.args.rna_type, self.args.taxid, self.args.species, self.args.lineage]):
            self.condition = GenerationCondition(
                rna_type=self.args.rna_type,
                taxid=self.args.taxid,
                species=self.args.species,
                lineage=self.args.lineage
            )
            self.condition.validate()
            self.lineage_db = LineageDatabase()
            print(f"Condition: {self.condition.build_clm_prompt(self.lineage_db)[:80]}...")

    def select_mutate_positions(self) -> List[int]:
        """Select positions to mutate.

        Returns:
            List of positions to mutate
        """
        seq_length = len(self.input_sequence)

        # Parse user-specified positions
        if self.args.mutate_positions:
            positions = parse_mutate_positions(self.args.mutate_positions, seq_length)
            if positions:
                return positions

        # Parse range
        if self.args.mutate_range:
            return parse_mutate_range(self.args.mutate_range, seq_length)

        # Default: randomly select positions from entire sequence
        positions = random.sample(range(seq_length), min(self.args.mutations_per_iter, seq_length))
        return sorted(positions)

    def generate_candidate_sequences(self, positions: List[int]) -> List[str]:
        """Generate candidate mutated sequences with incremental beam search.

        Args:
            positions: Positions to mutate

        Returns:
            List of candidate sequences
        """
        # Use the new beam search approach
        candidates = generate_mutations_with_beam_search(
            initial_sequence=self.input_sequence,
            positions=positions,
            beam_width=self.args.beam_width,
            score_fn=self.score_candidates
        )
        return candidates

    def score_candidates(self, candidates: List[str]) -> List[float]:
        """Score candidates using LLM.

        Args:
            candidates: Candidate sequences

        Returns:
            List of log-likelihood scores
        """
        # Format sequences with condition if available
        if self.condition is not None:
            formatted_seqs = [format_sequence(seq, self.condition, self.lineage_db) for seq in candidates]
        else:
            formatted_seqs = candidates

        # Compute batch likelihood
        scores = compute_batch_likelihood(
            model=self.model,
            tokenizer=self.tokenizer,
            sequences=formatted_seqs,
            device=self.args.device,
            reduce_method='mean',
            exclude_special_tokens=True
        )

        return scores

    def run(self):
        """Run the directed evolution process."""
        print(f"\n{'='*60}")
        print("Starting Directed Evolution")
        print(f"{'='*60}")
        print(f"Iterations: {self.args.iterations}")
        print(f"Mutations per iteration: {self.args.mutations_per_iter}")
        print(f"Beam width: {self.args.beam_width}")
        print(f"Output count: {self.args.output_count}")
        print(f"Temperature: {self.args.T_init} -> {self.args.T_min}")
        print(f"{'='*60}\n")

        # Initialize with input sequence
        self.current_sequences = [self.input_sequence]
        self.current_scores = [0.0]  # Initial score is 0 (not evaluated)

        # Candidate pool for collecting diverse sequences
        candidate_pool = []  # List of (sequence, score) tuples

        temperature = self.args.T_init
        best_sequence = self.input_sequence
        best_score = float('-inf')

        for iteration in range(self.args.iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.args.iterations} ---")
            print(f"Temperature: {temperature:.4f}")

            # Step 1: Select mutation positions
            positions = self.select_mutate_positions()
            print(f"Mutating positions: {positions}")

            # Step 2: Generate candidate sequences with incremental beam search
            candidates = self.generate_candidate_sequences(positions)
            print(f"Generated {len(candidates)} candidates (after beam search)")

            # Step 3: Score candidates with LLM
            scores = self.score_candidates(candidates)
            print(f"Scored candidates (LLM)")

            # Step 4: Dynamic beam search
            beam_candidates = dynamic_beam_search(
                candidates, scores,
                self.args.beam_width,
                temperature,
                self.args.T_init,
                self.args.output_count
            )

            # Step 5: Calculate MFE for beam candidates
            mfe_values = calculate_mfe_batch([seq for seq, _ in beam_candidates])
            print(f"MFE range: {min(mfe_values):.2f} to {max(mfe_values):.2f} kcal/mol")

            # Step 6: Simulated annealing - accept/reject
            accepted_sequences = []
            accepted_scores = []

            for (seq, score), mfe in zip(beam_candidates, mfe_values):
                # Combine LLM score and MFE (lower MFE is better)
                # Use negative MFE so higher is better
                combined_score = score - 0.1 * mfe  # Weight MFE contribution

                # Check if we should accept
                if self.current_scores:
                    current_best = max(self.current_scores)
                else:
                    current_best = float('-inf')

                if simulated_annealing_accept(current_best, combined_score, temperature):
                    accepted_sequences.append(seq)
                    accepted_scores.append(combined_score)

                    # Add to candidate pool
                    candidate_pool.append((seq, combined_score))

            # If no sequences accepted, keep the best beam candidate
            if not accepted_sequences:
                best_seq, best_s = beam_candidates[0]
                best_mfe = mfe_values[0]
                combined = best_s - 0.1 * best_mfe
                accepted_sequences = [best_seq]
                accepted_scores = [combined]
                candidate_pool.append((best_seq, combined))
                print("  (SA rejected all, keeping best beam candidate)")

            # Update current pool for next iteration
            self.current_sequences = accepted_sequences[:self.args.beam_width]
            self.current_scores = accepted_scores[:self.args.beam_width]

            # Track best
            if self.current_scores:
                current_best_idx = np.argmax(self.current_scores)
                current_best_seq = self.current_sequences[current_best_idx]
                current_best_score = self.current_scores[current_best_idx]

                if current_best_score > best_score:
                    best_score = current_best_score
                    best_sequence = current_best_seq
                    print(f"  New best: score={best_score:.4f}")

            print(f"  Candidate pool size: {len(candidate_pool)}")

            # Log iteration
            self.log_entries.append({
                'iteration': iteration + 1,
                'temperature': temperature,
                'positions': positions,
                'best_score': best_score,
                'mfe_values': mfe_values
            })

            # Cool down
            temperature = max(self.args.T_min, temperature * self.args.cooling_rate)

        # Final: Select top output_count sequences from candidate pool
        print(f"\n{'='*60}")
        print("Evolution Complete")
        print(f"{'='*60}")
        print(f"Best sequence score: {best_score:.4f}")
        print(f"Total candidates in pool: {len(candidate_pool)}")

        # Sort by score and select top sequences
        candidate_pool.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for seq, score in candidate_pool:
            if seq not in seen:
                seen.add(seq)
                unique_candidates.append((seq, score))

        # Select top output_count
        final_candidates = unique_candidates[:self.args.output_count]

        self.current_sequences = [seq for seq, _ in final_candidates]
        self.current_scores = [score for _, score in final_candidates]

        print(f"Selected {len(self.current_sequences)} unique sequences for output")

        return self.current_sequences, self.current_scores

        print(f"\n{'='*60}")
        print("Evolution Complete")
        print(f"{'='*60}")
        print(f"Best sequence score: {best_score:.4f}")

        return self.current_sequences, self.current_scores

    def save_results(self):
        """Save evolution results to output file."""
        print(f"\nSaving results to {self.args.output}...")

        # Prepare output sequences
        output_sequences = []
        for i, (seq, score) in enumerate(zip(self.current_sequences, self.current_scores)):
            header = f"candidate_{i+1}_score={score:.4f}"
            if i == 0:
                header += "_best"
            output_sequences.append((header, seq))

        # Write to FASTA
        write_fasta(self.args.output, output_sequences)
        print(f"Saved {len(output_sequences)} sequences to {self.args.output}")

        # Save log if requested
        if self.args.log_file:
            self.save_log()

    def save_log(self):
        """Save evolution log."""
        import json
        with open(self.args.log_file, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        print(f"Log saved to {self.args.log_file}")


def main():
    """Main entry point."""
    args, parser_defaults = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        merge_config_with_args(config, args, parser_defaults)

    # Validate required arguments
    if not args.input:
        raise ValueError("--input is required (either via CLI or config file)")
    if not args.output:
        raise ValueError("--output is required (either via CLI or config file)")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required (either via CLI or config file)")

    # Validate arguments
    if args.rna_type:
        supported_types = list_rna_types()
        if args.rna_type not in supported_types:
            raise ValueError(f"Unsupported RNA type: {args.rna_type}. Supported: {supported_types}")

    # Create and run directed evolution
    de = DirectedEvolution(args)

    try:
        calculate_mfe("GCGCUUCGCG")  # Preflight before allocating GPU memory.
        de.load_model()
        de.load_sequence()
        de.setup_conditions()
        sequences, scores = de.run()
        de.save_results()

        print("\nDone!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
