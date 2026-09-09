"""HuggingFace model adapter for unified benchmark system"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from tqdm import tqdm

# Import multimolecule lazily in setup_environment to avoid requiring it for other adapters

from .base_adapter import BaseAdapter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta, convert_rna_to_dna


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for HuggingFace models (RNAFM, RNABERT, RNAMSM, RiNALMo, GENA, GROVER)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.max_length = None

    def setup_environment(self):
        """Load model and tokenizer"""
        # Import multimolecule here to register models to transformers
        import multimolecule

        # Import here to avoid issues in multiprocessing
        from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer
        import json

        print(f"Loading model: {self.config['model_name']}")

        # Get local path if specified
        local_path = self.config.get('local_path', None)
        model_path = local_path if local_path else self.config['model_name']
        tokenizer_path = local_path if local_path else self.config['tokenizer_name']

        # Try to load tokenizer
        try:
            # First try with trust_remote_code
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
        except (ValueError, ImportError) as e:
            # If it fails (e.g., RnaTokenizer not found), use BertTokenizer as fallback
            print(f"Warning: Failed to load custom tokenizer ({e}), using BertTokenizer as fallback")

            # Load tokenizer config to get vocab and special tokens
            import os
            if local_path:
                vocab_file = os.path.join(local_path, 'vocab.txt')
                tokenizer_config_file = os.path.join(local_path, 'tokenizer_config.json')
            else:
                # This shouldn't happen since we have local_path, but just in case
                vocab_file = None
                tokenizer_config_file = None

            if vocab_file and os.path.exists(vocab_file):
                # Load with BertTokenizer
                self.tokenizer = BertTokenizer(
                    vocab_file=vocab_file,
                    do_lower_case=False
                )

                # Load and apply tokenizer config
                if tokenizer_config_file and os.path.exists(tokenizer_config_file):
                    with open(tokenizer_config_file, 'r') as f:
                        config = json.load(f)

                    # Set special tokens
                    if 'pad_token' in config:
                        self.tokenizer.pad_token = config['pad_token']
                    if 'cls_token' in config:
                        self.tokenizer.cls_token = config['cls_token']
                    if 'sep_token' in config:
                        self.tokenizer.sep_token = config['sep_token']
                    if 'mask_token' in config:
                        self.tokenizer.mask_token = config['mask_token']
                    if 'unk_token' in config:
                        self.tokenizer.unk_token = config['unk_token']
            else:
                raise RuntimeError(f"Cannot load tokenizer from {tokenizer_path}")

        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        configured_max_length = self.config.get('max_seq_length') or self.config.get('max_length')
        model_max_length = getattr(self.model.config, 'max_position_embeddings', None)
        tokenizer_max_length = getattr(self.tokenizer, 'model_max_length', None)
        candidates = []
        for value in [configured_max_length, model_max_length, tokenizer_max_length]:
            if isinstance(value, int) and value > 0 and value < 10**9:
                candidates.append(value)
        self.max_length = min(candidates) if candidates else None

        print(f"Model loaded successfully")

    def _get_special_token_ids(self) -> set:
        """Get set of special token IDs from tokenizer"""
        special_token_ids = set()
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'eos_token_id', 'bos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special_token_ids.add(token_id)
        return special_token_ids

    def _find_effective_positions(self, input_ids: torch.Tensor, special_token_ids: set) -> tuple:
        """
        Find positions of non-special tokens using vectorized operations

        Args:
            input_ids: Token IDs tensor [seq_len] or [1, seq_len]
            special_token_ids: Set of special token IDs to skip

        Returns:
            Tuple of (effective_positions list, true_token_ids list)
        """
        # Flatten to 1D if needed
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)

        # Convert to CPU once for all comparisons
        token_ids_cpu = input_ids.cpu().tolist()

        effective_positions = []
        true_token_ids = []
        for i, token_id in enumerate(token_ids_cpu):
            if token_id not in special_token_ids:
                effective_positions.append(i)
                true_token_ids.append(token_id)

        return effective_positions, true_token_ids

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute pseudo-likelihood scores for sequences in FASTA file

        Args:
            fasta_path: Path to FASTA file
            device: GPU device (e.g., 'cuda:0')
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of log-likelihood scores
        """
        import os
        # Extract dataset name from fasta path for progress display
        dataset_name = os.path.basename(fasta_path).replace('.fasta', '')

        # Read sequences
        sequences = read_fasta(fasta_path)
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")
        print(f"Using batch_size: {batch_size}")

        # Convert RNA to DNA if needed
        if self.config['sequence_type'] == 'dna':
            sequences = [convert_rna_to_dna(seq) for seq in sequences]

        # Load model to device
        self.model.to(device)
        self.model.eval()

        # Compute scores
        scores = []
        model_display_name = self.config['model_name'].split('/')[-1]  # Short name
        progress_desc = f"{model_display_name} → {dataset_name}"

        if batch_size > 1:
            # Batch processing: process multiple sequences together
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(sequences), batch_size),
                         total=num_batches,
                         desc=progress_desc,
                         unit="batch"):
                batch = sequences[i:i+batch_size]
                batch_scores = self._calculate_batch_pseudo_likelihood(batch, device)
                scores.extend(batch_scores)
        else:
            # Single sequence processing (original behavior)
            for seq in tqdm(sequences, desc=progress_desc):
                score = self._calculate_pseudo_likelihood(seq, device)
                scores.append(score)

        return scores

    def _calculate_batch_pseudo_likelihood(self, sequences: List[str], device: str) -> List[float]:
        """
        Calculate pseudo-likelihood for multiple sequences in batch
        Uses sequence-level iteration with chunked masked positions.

        Args:
            sequences: List of input sequences
            device: GPU device

        Returns:
            List of log-likelihood scores
        """
        # The previous implementation expanded every effective token in every
        # sequence into one large masked batch. That is fast for short inputs but
        # can allocate >100 GiB for long coding sequences. Reuse the single
        # sequence chunked scorer to keep peak memory bounded.
        return [self._calculate_pseudo_likelihood(seq, device) for seq in sequences]

    def _calculate_pseudo_likelihood(self, sequence: str, device: str) -> float:
        """
        Calculate pseudo-likelihood for a single sequence
        Adapted from benchmark_rnagen/scripts/compute_model_scores.py (lines 48-116)

        Args:
            sequence: Input sequence
            device: GPU device

        Returns:
            Log-likelihood score
        """
        # Tokenize sequence. Keep truncation behavior aligned with the original
        # batch path so long coding sequences do not exceed the model context.
        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        ).to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        mask_token_id = self.tokenizer.mask_token_id

        log_likelihood = 0.0
        num_effective_tokens = 0

        # Get special token IDs (using helper method)
        special_token_ids = self._get_special_token_ids()

        # Find effective token positions (using helper method)
        effective_positions, true_token_ids = self._find_effective_positions(
            input_ids, special_token_ids
        )

        if len(effective_positions) == 0:
            return 0.0

        configured_chunk_size = int(self.config.get('mask_chunk_size', 64))
        full_mask_max_length = int(self.config.get('full_mask_max_length', 0))
        seq_len = input_ids.shape[1]
        if full_mask_max_length and seq_len <= full_mask_max_length:
            mask_chunk_size = len(effective_positions)
        else:
            mask_chunk_size = configured_chunk_size

        for start in range(0, len(effective_positions), mask_chunk_size):
            chunk_positions = effective_positions[start:start + mask_chunk_size]
            chunk_token_ids = true_token_ids[start:start + mask_chunk_size]

            batch_masked_input_ids = input_ids.repeat(len(chunk_positions), 1)
            batch_attention_mask = (
                attention_mask.repeat(len(chunk_positions), 1)
                if attention_mask is not None else None
            )
            for j, pos in enumerate(chunk_positions):
                batch_masked_input_ids[j, pos] = mask_token_id

            with torch.no_grad():
                if batch_attention_mask is not None:
                    outputs = self.model(
                        input_ids=batch_masked_input_ids,
                        attention_mask=batch_attention_mask
                    )
                else:
                    outputs = self.model(input_ids=batch_masked_input_ids)
                logits = outputs.logits

            for j, (pos, true_token_id) in enumerate(zip(chunk_positions, chunk_token_ids)):
                position_logits = logits[j, pos, :]
                log_probs = F.log_softmax(position_logits, dim=-1)
                token_log_prob = log_probs[true_token_id].item()
                log_likelihood += token_log_prob
                num_effective_tokens += 1

        return log_likelihood / num_effective_tokens
