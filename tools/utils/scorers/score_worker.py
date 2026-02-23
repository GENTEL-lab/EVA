#!/usr/bin/env python3
"""
Sequence Scoring Worker Script

This script calculates the log-likelihood of RNA sequences.
Automatically adds 5'/3' direction markers.

Usage:
    python score_worker.py <sequences_json> <checkpoint_path> <device> <output_json> [reduce_method]

Arguments:
    sequences_json: JSON file path containing RNA sequence list
    checkpoint_path: Model path
    device: Compute device (cuda:0, cuda:1, etc.)
    output_json: Output JSON file path
    reduce_method: Reduction method ('mean' or 'sum'), default 'mean'

Output:
    JSON format results to output_json, containing log_likelihoods list
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List

# Add project path (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent.absolute()
TOOLS_DIR = SCRIPT_DIR.parent.parent  # tools/
sys.path.insert(0, str(TOOLS_DIR))

# Fix nan/memory errors caused by CUDA asynchronous execution
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn.functional as F

# Import model loader
from utils.model import ModelLoader


def prepare_input_with_direction_tokens(
    tokenizer,
    sequence: str,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Prepare model input - supports sequences with/without lineage prefix

    Input format:
    - No prefix: Pure RNA sequence (AUGCUAGC...)
      -> Output: <bos>5{sequence}3<eos>
    - With prefix: |d__...;s__species;<rna_mRNA>|{rna_sequence}
      -> Output: <bos>|d__...;s__species;<rna_mRNA>|5{rna_sequence}3<eos>

    Args:
        tokenizer: Tokenizer instance
        sequence: Input sequence (may contain lineage prefix)
        device: Compute device

    Returns:
        Dictionary containing input_ids, position_ids, sequence_ids
    """
    # Detect if lineage prefix is present (starts with |)
    if sequence.startswith('|'):
        # Find prefix end position (second |)
        second_pipe_idx = sequence.find('|', 1)
        if second_pipe_idx != -1:
            # Extract prefix and pure sequence
            prefix = sequence[:second_pipe_idx + 1]  # Include ending |
            rna_seq = sequence[second_pipe_idx + 1:]  # Pure RNA sequence
            # Add 5' and 3' direction markers to pure sequence
            sequence_with_direction = f"{prefix}5{rna_seq}3"
        else:
            # Abnormal case: only one |, treat as no prefix
            sequence_with_direction = f"5{sequence}3"
    else:
        # No prefix: add 5' and 3' direction markers to sequence
        sequence_with_direction = f"5{sequence}3"

    # Add BOS and EOS markers
    full_sequence = f"<bos>{sequence_with_direction}<eos>"

    # Encode
    token_ids = tokenizer.encode(full_sequence)

    # Create tensors
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    sequence_ids = torch.zeros((1, len(token_ids)), dtype=torch.long, device=device)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'sequence_ids': sequence_ids
    }


def compute_sequence_likelihood(
    model,
    tokenizer,
    sequence: str,
    device: str = 'cpu',
    reduce_method: str = 'mean'
) -> float:
    """
    Calculate log-likelihood for a single sequence

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        sequence: Input RNA sequence
        device: Device
        reduce_method: Reduction method ('mean' or 'sum')

    Returns:
        log-likelihood score
    """
    # Prepare input (add 5'/3' tokens)
    inputs = prepare_input_with_direction_tokens(tokenizer, sequence, device)

    # Forward pass
    with torch.no_grad():
        # Get model data type
        model_dtype = next(model.parameters()).dtype

        # Select autocast based on model data type
        if model_dtype == torch.bfloat16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    sequence_ids=inputs['sequence_ids']
                )
        else:
            outputs = model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                sequence_ids=inputs['sequence_ids']
            )

    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Check if logits contain nan/inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"[WARNING] logits contain nan or inf values", file=sys.stderr)
        return float('nan')

    # Calculate log-likelihood
    log_probs = F.log_softmax(logits, dim=-1)

    # Check if log_probs contain nan/inf
    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
        print(f"[WARNING] log_probs contain nan or inf values", file=sys.stderr)
        return float('nan')

    # Get log probabilities of actual tokens
    # Note: token at prediction position i is input_ids[i+1]
    input_ids = inputs['input_ids'][0]  # [seq_len]
    token_log_probs = []

    for i in range(len(input_ids) - 1):
        # Token at prediction position i is input_ids[i+1]
        predicted_token = input_ids[i + 1]
        log_prob = log_probs[0, i, predicted_token].item()
        token_log_probs.append(log_prob)

    # Reduce
    if reduce_method == 'mean':
        result = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0
    else:  # sum
        result = sum(token_log_probs) if token_log_probs else 0.0

    return result


def compute_batch_likelihood(
    model,
    tokenizer,
    sequences: List[str],
    device: str = 'cpu',
    reduce_method: str = 'mean',
    exclude_special_tokens: bool = False
) -> List[float]:
    """
    Batch calculate log-likelihood of sequences (performance optimized version)

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        sequences: List of input RNA sequences
        device: Device
        reduce_method: Reduction method ('mean' or 'sum')
        exclude_special_tokens: Whether to exclude 5/3/<eos> token log-likelihood

    Returns:
        List of log-likelihood scores
    """
    if not sequences:
        return []

    # Prepare inputs for all sequences
    all_input_ids = []
    all_position_ids = []
    all_sequence_ids = []
    seq_lengths = []

    for seq in sequences:
        # Handle sequences with/without lineage prefix
        if seq.startswith('|'):
            second_pipe_idx = seq.find('|', 1)
            if second_pipe_idx != -1:
                prefix = seq[:second_pipe_idx + 1]
                rna_seq = seq[second_pipe_idx + 1:]
                sequence_with_direction = f"{prefix}5{rna_seq}3"
            else:
                sequence_with_direction = f"5{seq}3"
        else:
            sequence_with_direction = f"5{seq}3"

        full_sequence = f"<bos>{sequence_with_direction}<eos>"

        # Encode
        token_ids = tokenizer.encode(full_sequence)
        seq_len = len(token_ids)
        seq_lengths.append(seq_len)

        # Create tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        sequence_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

        all_input_ids.append(input_ids)
        all_position_ids.append(position_ids)
        all_sequence_ids.append(sequence_ids)

    # Padding to maximum length
    max_len = max(seq_lengths)
    padded_input_ids = []
    padded_position_ids = []
    padded_sequence_ids = []

    pad_token_id = tokenizer.encode('<pad>')[0] if hasattr(tokenizer, 'pad_token') else 0

    for i in range(len(sequences)):
        seq_len = seq_lengths[i]
        pad_len = max_len - seq_len

        if pad_len > 0:
            padded_input = torch.cat([
                all_input_ids[i],
                torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device)
            ])
            padded_pos = torch.cat([
                all_position_ids[i],
                torch.arange(seq_len, max_len, dtype=torch.long, device=device)
            ])
            padded_seq = torch.cat([
                all_sequence_ids[i],
                torch.zeros(pad_len, dtype=torch.long, device=device)
            ])
        else:
            padded_input = all_input_ids[i]
            padded_pos = all_position_ids[i]
            padded_seq = all_sequence_ids[i]

        padded_input_ids.append(padded_input)
        padded_position_ids.append(padded_pos)
        padded_sequence_ids.append(padded_seq)

    # Stack into batch
    batch_input_ids = torch.stack(padded_input_ids)
    batch_position_ids = torch.stack(padded_position_ids)
    batch_sequence_ids = torch.stack(padded_sequence_ids)

    # Forward pass
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype

        if model_dtype == torch.bfloat16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch_input_ids,
                    position_ids=batch_position_ids,
                    sequence_ids=batch_sequence_ids
                )
        else:
            outputs = model(
                input_ids=batch_input_ids,
                position_ids=batch_position_ids,
                sequence_ids=batch_sequence_ids
            )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Get special token IDs (for exclusion)
    special_token_ids = set()
    if exclude_special_tokens:
        for token_str in ['5', '3', '<eos>']:
            ids = tokenizer.encode(token_str)
            if ids:
                special_token_ids.add(ids[0])

    # Calculate log-likelihood for each sequence
    results = []
    for i in range(len(sequences)):
        seq_len = seq_lengths[i]

        # Check for nan/inf
        if torch.isnan(logits[i]).any() or torch.isinf(logits[i]).any():
            print(f"[WARNING] Sequence {i} logits contain nan or inf values", file=sys.stderr)
            results.append(float('nan'))
            continue

        # Get log probabilities of actual tokens
        token_log_probs = []
        for j in range(seq_len - 1):
            predicted_token = batch_input_ids[i, j + 1]
            # Exclude special tokens
            if exclude_special_tokens and predicted_token.item() in special_token_ids:
                continue
            log_prob = log_probs[i, j, predicted_token].item()
            token_log_probs.append(log_prob)

        # Reduce
        if reduce_method == 'mean':
            result = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0
        else:
            result = sum(token_log_probs) if token_log_probs else 0.0

        results.append(result)

    return results


def main():
    """Main function"""
    if len(sys.argv) < 5:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python score_worker.py <sequences_json> <checkpoint_path> <device> <output_json> [reduce_method] [batch_size]'
        }))
        sys.exit(1)

    sequences_json = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3]
    output_json = sys.argv[4]
    reduce_method = sys.argv[5] if len(sys.argv) > 5 else 'mean'
    batch_size_arg = sys.argv[6] if len(sys.argv) > 6 else '128'

    try:
        # 1. Read sequences
        with open(sequences_json, 'r') as f:
            data = json.load(f)
            sequences = data['sequences']

        num_sequences = len(sequences)
        print(f"[INFO] Loaded {num_sequences} RNA sequences", file=sys.stderr)

        # 2. Load model (using ModelLoader)
        print(f"[INFO] Loading model: {checkpoint_path}", file=sys.stderr)
        print(f"[INFO] Device: {device}", file=sys.stderr)

        loader = ModelLoader(checkpoint_path)
        model, tokenizer = loader.load(device=device)

        print(f"[INFO] Model loaded successfully", file=sys.stderr)
        print(f"[INFO] Parameter count: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

        # 3. Calculate log-likelihood
        print(f"[INFO] Starting log-likelihood calculation (reduce_method={reduce_method})...", file=sys.stderr)
        log_likelihoods = []

        # Batch processing setup
        batch_size = int(batch_size_arg)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        print(f"[INFO] Using batch processing: batch_size={batch_size}, num_batches={num_batches}", file=sys.stderr)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_sequences = sequences[start_idx:end_idx]

            print(f"[PROGRESS] Processing sequences {start_idx+1}-{end_idx}/{num_sequences}", file=sys.stderr, flush=True)

            try:
                # Batch calculation
                batch_lls = compute_batch_likelihood(model, tokenizer, batch_sequences, device, reduce_method)
                log_likelihoods.extend(batch_lls)
            except Exception as e:
                import traceback
                print(f"[WARNING] Batch {batch_idx} calculation failed, falling back to sequential processing: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

                # Fall back to sequential processing
                for i, seq in enumerate(batch_sequences):
                    try:
                        ll = compute_sequence_likelihood(model, tokenizer, seq, device, reduce_method)
                        log_likelihoods.append(ll)
                    except Exception as e2:
                        seq_idx = start_idx + i
                        print(f"[WARNING] Sequence {seq_idx} calculation failed: {e2}", file=sys.stderr)
                        log_likelihoods.append(float('nan'))

        print(f"[INFO] Calculation completed", file=sys.stderr)

        # 4. Output results
        result = {
            'success': True,
            'log_likelihoods': log_likelihoods,
            'num_sequences': num_sequences,
            'reduce_method': reduce_method
        }

        with open(output_json, 'w') as f:
            json.dump(result, f)

        print(f"[INFO] Results written to: {output_json}", file=sys.stderr)

    except Exception as e:
        import traceback
        # Output error information
        error_result = {
            'success': False,
            'error': str(e)
        }
        with open(output_json, 'w') as f:
            json.dump(error_result, f)
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
