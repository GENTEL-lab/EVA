#!/usr/bin/env python3
"""
AIDO.RNA Log-Likelihood Calculation Script

Calculates log-likelihood (pseudo-likelihood) for RNA sequences using AIDO.RNA model.

Usage:
    python compute_aido_ll.py --fasta <fasta_file> --device <device>

Output:
    JSON to stdout with log_likelihoods list
"""

import sys
import os
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from safetensors.torch import load_file

# Add ModelGenerator to path
MODELGENERATOR_PATH = "/data4/huangyanjie/rna_benchmark/interpretability/AIdo.RNA/ModelGenerator-main"
sys.path.insert(0, MODELGENERATOR_PATH)

from modelgenerator.huggingface_models.rnabert import RNABertModel, RNABertConfig, RNABertTokenizer

# Default model path
DEFAULT_MODEL_PATH = "/data4/huangyanjie/rna_benchmark/interpretability/AIdo.RNA/model-weight"


def load_aido_model(model_path, device):
    """
    Load AIDO.RNA model (RNABert architecture)

    Args:
        model_path: Path to model directory
        device: Device to load model on

    Returns:
        model, tokenizer, config
    """
    print(f"Loading AIDO.RNA model from {model_path}...", file=sys.stderr)

    # Load config
    config = RNABertConfig.from_pretrained(model_path)
    print(f"  Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden", file=sys.stderr)

    # Load tokenizer
    tokenizer = RNABertTokenizer.from_pretrained(model_path)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}", file=sys.stderr)

    # Create model
    model = RNABertModel(config)

    # Load safetensors shards
    index_file = Path(model_path) / "pytorch_model.bin.index.json"
    with open(index_file, 'r') as f:
        index = json.load(f)

    weight_map = index['weight_map']
    shard_files = set(weight_map.values())

    state_dict = {}
    for shard_file in shard_files:
        shard_file_safe = shard_file.replace('.bin', '.safetensors')
        shard_path = Path(model_path) / shard_file_safe
        print(f"  Loading {shard_file_safe}...", file=sys.stderr)
        shard_state = load_file(str(shard_path))
        state_dict.update(shard_state)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters", file=sys.stderr)

    return model, tokenizer, config


def calculate_pseudo_likelihood(sequence, model, tokenizer, device, normalize=False, batch_size=1):
    """
    Calculate pseudo-likelihood for a single sequence using batched masking approach.

    For BERT-style masked LM, we mask each position one at a time and sum the
    log probabilities of the true tokens. With batch_size > 1, multiple masked
    positions are processed in a single forward pass for speedup.

    Args:
        sequence: RNA sequence string
        model: AIDO model
        tokenizer: AIDO tokenizer
        device: Computation device
        normalize: Whether to normalize by number of tokens (PTLL)
        batch_size: Number of masked positions to process per forward pass

    Returns:
        log_likelihood: Float value (sum or average of log probabilities)
    """
    # AIDO tokenizer expects space-separated nucleotides
    spaced_seq = ' '.join(list(sequence))

    # Tokenize
    inputs = tokenizer(
        spaced_seq,
        return_tensors="pt",
        padding=False,
        truncation=False
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get mask token id
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must have mask_token_id for pseudo-likelihood")

    # Get special token ids to skip
    special_token_ids = set()
    if tokenizer.cls_token_id is not None:
        special_token_ids.add(tokenizer.cls_token_id)
    if tokenizer.sep_token_id is not None:
        special_token_ids.add(tokenizer.sep_token_id)
    if tokenizer.pad_token_id is not None:
        special_token_ids.add(tokenizer.pad_token_id)
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_token_ids.add(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_token_ids.add(tokenizer.eos_token_id)

    seq_len = input_ids.shape[1]

    # Collect positions to mask (skip special tokens)
    mask_positions = []
    true_token_ids = []
    for i in range(seq_len):
        token_id = input_ids[0, i].item()
        if token_id not in special_token_ids:
            mask_positions.append(i)
            true_token_ids.append(token_id)

    num_effective_tokens = len(mask_positions)
    if num_effective_tokens == 0:
        return 0.0

    log_likelihood = 0.0
    embeddings = model.embeddings.word_embeddings.weight  # (vocab_size, hidden_size)

    # Process mask positions in batches
    for batch_start in range(0, num_effective_tokens, batch_size):
        batch_end = min(batch_start + batch_size, num_effective_tokens)
        batch_positions = mask_positions[batch_start:batch_end]
        batch_true_ids = true_token_ids[batch_start:batch_end]
        cur_batch_size = len(batch_positions)

        # Create batch of masked inputs: each item masks a different position
        batch_input_ids = input_ids.expand(cur_batch_size, -1).clone()  # (cur_batch_size, seq_len)
        batch_attention_mask = attention_mask.expand(cur_batch_size, -1)  # (cur_batch_size, seq_len)

        for idx, pos in enumerate(batch_positions):
            batch_input_ids[idx, pos] = mask_token_id

        # Single forward pass for the entire batch
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=False,
                return_dict=True
            )

            hidden_state = outputs.last_hidden_state  # (cur_batch_size, seq_len, hidden_size)
            logits = torch.matmul(hidden_state, embeddings.T)  # (cur_batch_size, seq_len, vocab_size)

        # Extract log probabilities for each masked position
        for idx, (pos, true_id) in enumerate(zip(batch_positions, batch_true_ids)):
            position_logits = logits[idx, pos, :]  # (vocab_size,)
            log_probs = F.log_softmax(position_logits, dim=-1)
            log_likelihood += log_probs[true_id].item()

    # 根据normalize参数决定返回总LL还是PTLL
    if normalize and num_effective_tokens > 0:
        return log_likelihood / num_effective_tokens  # PTLL
    return log_likelihood  # 总LL（默认）


def read_sequences_from_fasta(fasta_file):
    """Read sequences from FASTA file"""
    sequences = []
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq).upper()
            # Convert T to U for RNA
            seq = seq.replace('T', 'U')
            sequences.append(seq)
    return sequences


def compute_log_likelihoods(sequences, model, tokenizer, device, normalize=False, fasta_path=None, batch_size=1):
    """
    Compute log-likelihoods for multiple sequences

    Args:
        sequences: List of RNA sequences
        model: AIDO model
        tokenizer: AIDO tokenizer
        device: Computation device
        normalize: Whether to normalize by number of tokens (PTLL)
        fasta_path: Optional path to FASTA file for progress display
        batch_size: Number of masked positions to process per forward pass

    Returns:
        List of log-likelihood values (or PTLL if normalize=True)
    """
    log_likelihoods = []

    mode_str = "PTLL" if normalize else "log-likelihoods"
    print(f"Computing {mode_str} for {len(sequences)} sequences (mask batch_size={batch_size})...", file=sys.stderr)

    # Extract dataset name from fasta path for progress display
    if fasta_path:
        dataset_name = os.path.basename(fasta_path).replace('.fasta', '')
        desc = f"AIDO: {dataset_name}"
    else:
        desc = "AIDO"

    for seq in tqdm(sequences, desc=desc, file=sys.stderr):
        ll = calculate_pseudo_likelihood(seq, model, tokenizer, device, normalize=normalize, batch_size=batch_size)
        log_likelihoods.append(ll)

    return log_likelihoods


def main():
    parser = argparse.ArgumentParser(
        description="Calculate log-likelihood using AIDO.RNA model"
    )
    parser.add_argument('--fasta', type=str, required=True,
                        help='Input FASTA file path')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to AIDO model directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computation device (cuda:X or cpu)')
    parser.add_argument('--normalize-by-length', action='store_true',
                        help='Use PTLL (divide by token count) instead of total log-likelihood')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Number of masked positions per forward pass (default: 8)')

    args = parser.parse_args()

    try:
        # Check device
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU", file=sys.stderr)
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)

        # Read sequences
        sequences = read_sequences_from_fasta(args.fasta)
        print(f"Read {len(sequences)} sequences from {args.fasta}", file=sys.stderr)

        # Load model
        model, tokenizer, config = load_aido_model(args.model_path, device)

        # Compute log-likelihoods
        normalize = getattr(args, 'normalize_by_length', False)
        log_likelihoods = compute_log_likelihoods(sequences, model, tokenizer, device, normalize=normalize, fasta_path=args.fasta, batch_size=args.batch_size)

        # Output result as JSON
        result = {
            'success': True,
            'log_likelihoods': log_likelihoods,
            'num_sequences': len(log_likelihoods),
            'model': 'aido'
        }
        print(json.dumps(result))

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'log_likelihoods': [],
            'num_sequences': 0
        }
        print(json.dumps(result))
        sys.exit(1)


if __name__ == '__main__':
    main()
