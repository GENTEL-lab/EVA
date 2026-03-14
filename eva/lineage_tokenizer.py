"""
Lineage-based RNA specialized tokenizer
Supports Greengenes lineage string encoding, streamlined vocabulary without species tokens
"""

import json
import os
import re
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE

# RNA special tokens
RNA_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<bos_glm>",
    "<eos_span>",
    "<unk>",
]

# GLM span tokens (keep all 50, support future multi-span training)
GLM_SPAN_TOKENS = [f"<span_{i}>" for i in range(50)]

# RNA type tokens (15 RNA types with clear functions)
RNA_TYPE_TOKENS = [
    "<rna_mRNA>",        # messenger RNA
    "<rna_rRNA>",        # ribosomal RNA
    "<rna_tRNA>",        # transfer RNA
    "<rna_sRNA>",        # small RNA
    "<rna_lncRNA>",      # long non-coding RNA
    "<rna_circRNA>",     # circular RNA
    "<rna_viral_RNA>",   # viral RNA
    "<rna_miRNA>",       # microRNA
    "<rna_snoRNA>",      # small nucleolar RNA
    "<rna_snRNA>",       # small nuclear RNA
    "<rna_piRNA>",       # PIWI-interacting RNA
    "<rna_ribozyme>",    # ribozyme
    "<rna_scaRNA>",      # small Cajal body RNA
    "<rna_Y_RNA>",       # Y RNA
    "<rna_vault_RNA>",   # Vault RNA
]

# Greengenes lineage level prefix tokens (lowercase to avoid confusion with uppercase RNA bases AUCG)
LINEAGE_LEVEL_TOKENS = [
    "d__",  # Domain
    "p__",  # Phylum
    "c__",  # Class
    "o__",  # Order
    "f__",  # Family
    "g__",  # Genus
    "s__",  # Species
]

# Lineage special characters (characters needed after data preprocessing)
# Note: Lineage strings are preprocessed by clean_lineage(), removing parenthetical annotations, brackets, single quotes, slashes, dots, and replacing spaces with underscores
# Therefore, only core separators and hyphens need to be retained
LINEAGE_SPECIAL_CHARS = [
    ";",  # Level separator (required)
    "|",  # Lineage prefix boundary marker (required)
    "_",  # Underscore (required, used for D__ and other level prefixes, and species names after space replacement)
    "-",  # Hyphen (common in species names)
]

# RNA bases (as separate tokens, defined with priority, only used for RNA sequences)
RNA_BASES = ["A", "U", "G", "C"]

# Sequence direction markers (used to identify 5' and 3' ends of RNA sequences)
DIRECTION_TOKENS = ["5", "3"]

# Alphabetic characters - lineage is all lowercase, only lowercase letters needed (uppercase AUCG already in RNA_BASES)
ALPHANUMERIC_CHARS = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # a-z (26 characters)

# Complete vocabulary
def build_lineage_rna_vocab():
    """Build Lineage RNA vocabulary (no species tokens, no task tokens)"""
    vocab_list = (
        RNA_SPECIAL_TOKENS +
        GLM_SPAN_TOKENS +
        RNA_TYPE_TOKENS +
        LINEAGE_LEVEL_TOKENS +
        LINEAGE_SPECIAL_CHARS +
        RNA_BASES +
        DIRECTION_TOKENS +      # Sequence direction markers (5' and 3' ends)
        ALPHANUMERIC_CHARS      # Alphabetic characters placed last
    )
    return vocab_list

LINEAGE_RNA_VOCAB = build_lineage_rna_vocab()

END_OF_SPAN_TOKEN = "<eos_span>"
PAD_TOKEN_ID = 0


class LineageRNATokenizer:
    """Lineage-based RNA sequence specialized tokenizer"""

    def __init__(self):
        self.tokenizer = self._create_tokenizer()
        # Add token ID properties for compatibility
        self.pad_token_id = self.token_to_id("<pad>")
        self.bos_token_id = self.token_to_id("<bos>")
        self.eos_token_id = self.token_to_id("<eos>")

    def _create_tokenizer(self) -> Tokenizer:
        """Create Lineage RNA tokenizer

        Note: Uses BPE model but without merge rules to implement character-level tokenization.
        Since we use a custom encode() method to match tokens character by character (rather than BPE's standard tokenize flow),
        we need to declare all tokens as special, so that:
        1. BPE model won't attempt to merge characters (because special tokens don't participate in merging)
        2. Avoid "vocab contains holes" warning when saving
        3. Consistent with our character-by-character encoding logic: each character should be "specially handled" (kept independent)
        """
        # Build vocabulary
        vocab = {token: idx for idx, token in enumerate(LINEAGE_RNA_VOCAB)}
        merges = []  # Don't provide merge rules, force character-level tokenization

        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))

        # Declare all tokens as special to ensure they won't be merged by BPE
        # This is reasonable in our character-level tokenization scenario
        all_special_tokens = LINEAGE_RNA_VOCAB.copy()
        tokenizer.add_special_tokens(all_special_tokens)

        # Configure padding
        tokenizer.enable_padding(
            direction="right",
            pad_id=PAD_TOKEN_ID,
            pad_type_id=0,
            pad_token="<pad>"
        )

        print(f"Lineage RNA tokenizer created, vocabulary size: {len(LINEAGE_RNA_VOCAB)}")
        return tokenizer

    def encode(self, sequence: str) -> List[int]:
        """
        Encode RNA sequence (character-by-character encoding)

        Args:
            sequence: String containing lineage information and RNA sequence

        Returns:
            List of token IDs
        """
        # Character-by-character encoding
        token_ids = []
        i = 0
        while i < len(sequence):
            # Prioritize matching multi-character tokens (special markers, level prefixes, etc.)
            matched = False

            # Check RNA type tokens (longest first)
            for rna_token in sorted(RNA_TYPE_TOKENS, key=len, reverse=True):
                if sequence[i:i+len(rna_token)] == rna_token:
                    token_id = self.token_to_id(rna_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(rna_token)
                        matched = True
                        break

            if matched:
                continue

            # Check GLM span markers
            for span_token in GLM_SPAN_TOKENS:
                if sequence[i:i+len(span_token)] == span_token:
                    token_id = self.token_to_id(span_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(span_token)
                        matched = True
                        break

            if matched:
                continue

            # Check other special markers
            for special_token in RNA_SPECIAL_TOKENS:
                if sequence[i:i+len(special_token)] == special_token:
                    token_id = self.token_to_id(special_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(special_token)
                        matched = True
                        break

            if matched:
                continue

            # Check lineage level prefixes (D__, P__, C__, etc.)
            for level_token in LINEAGE_LEVEL_TOKENS:
                if sequence[i:i+len(level_token)] == level_token:
                    token_id = self.token_to_id(level_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(level_token)
                        matched = True
                        break

            if matched:
                continue

            # Single character encoding
            char = sequence[i]
            token_id = self.token_to_id(char)
            if token_id is not None:
                token_ids.append(token_id)
            else:
                # Unknown characters use <unk>
                unk_id = self.token_to_id("<unk>")
                if unk_id is not None:
                    token_ids.append(unk_id)
            i += 1

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token sequence (concatenate without spaces)"""
        tokens = [self.id_to_token(tid) for tid in token_ids if self.id_to_token(tid) is not None]
        return "".join(tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Get ID corresponding to token"""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Get token corresponding to ID"""
        return self.tokenizer.id_to_token(token_id)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (returns actual dictionary size, deduplicated)"""
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        """Return vocabulary size (compatible with Hugging Face interface)"""
        return self.vocab_size

    def get_output_token_ids(self) -> List[int]:
        """
        Return list of token IDs that the model actually needs to predict (general version, includes all end markers)

        In conditional generation tasks, the model only needs to predict the RNA sequence itself and end markers,
        not condition tokens (lineage information, RNA type, etc.).

        Returns:
            List of IDs for output tokens
        """
        output_tokens = [
            "A", "U", "G", "C",           # RNA bases (4 bases)
            "<eos>",                       # Stage 1 sequence generation end marker
            "<eos_span>",                  # Stage 2 sequence completion end marker
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def get_stage1_output_token_ids(self) -> List[int]:
        """
        Return output token ID list for Stage 1 sequence generation task

        Stage 1 needs to predict RNA bases, direction markers and <eos>, not <eos_span>

        Returns:
            List of IDs for output tokens (A, U, G, C, 5, 3, <eos>)
        """
        output_tokens = [
            "A", "U", "G", "C",    # RNA bases (4 bases)
            "5", "3",              # Sequence direction markers (5' and 3' ends)
            "<eos>",               # Stage 1 sequence generation end marker
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def get_stage2_output_token_ids(self) -> List[int]:
        """
        Return output token ID list for Stage 2 sequence completion task

        Stage 2 only needs to predict RNA bases and <eos_span>, not <eos>

        Returns:
            List of IDs for output tokens (A, U, G, C, <eos_span>)
        """
        output_tokens = [
            "A", "U", "G", "C",    # RNA bases (4 bases)
            "<eos_span>",          # Stage 2 sequence completion end marker
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def save(self, filepath: str):
        """Save tokenizer (backward compatible)"""
        self.tokenizer.save(filepath)

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to HuggingFace format"""
        import json

        # Ensure directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save tokenizer file
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        # Create vocabulary file - use actual tokenizer vocabulary, not hardcoded reconstruction
        vocab = self.tokenizer.get_vocab()
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # Create special_tokens_map.json
        special_tokens_map = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>"
        }
        special_tokens_path = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

        # Create tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": "LineageRNATokenizer",
            "auto_map": {
                "AutoTokenizer": ["lineage_tokenizer.py", "LineageRNATokenizer"]
            },
            "vocab_size": self.vocab_size,
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "rna_bases": RNA_BASES,
            "special_tokens": RNA_SPECIAL_TOKENS,
            "glm_span_tokens": GLM_SPAN_TOKENS,
            "rna_type_tokens": RNA_TYPE_TOKENS,
            "lineage_level_tokens": LINEAGE_LEVEL_TOKENS,
            "lineage_special_chars": LINEAGE_SPECIAL_CHARS,
            "mode": "lineage",
            "description": "Lineage-based tokenizer without species tokens or task tokens"
        }
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        print(f"LineageRNATokenizer saved to: {save_directory}")

    @classmethod
    def from_file(cls, filepath: str) -> 'LineageRNATokenizer':
        """Load tokenizer from file (backward compatible)"""
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(filepath)
        return instance

    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'LineageRNATokenizer':
        """Load tokenizer from HuggingFace format"""
        import json

        # Check for required files
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file does not exist: {tokenizer_path}")

        # Create instance
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Add token ID properties for compatibility
        instance.pad_token_id = instance.token_to_id("<pad>")
        instance.bos_token_id = instance.token_to_id("<bos>")
        instance.eos_token_id = instance.token_to_id("<eos>")

        return instance


def create_lineage_rna_tokenizer_json(output_path: str):
    """Create and save Lineage RNA tokenizer JSON file"""
    tokenizer = LineageRNATokenizer()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save tokenizer
    tokenizer.save(output_path)

    print(f"Lineage RNA tokenizer saved to: {output_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {RNA_SPECIAL_TOKENS[:8]}...")
    print(f"RNA bases: {RNA_BASES}")
    print(f"Lineage level tokens: {LINEAGE_LEVEL_TOKENS}")
    print(f"Lineage special characters: {LINEAGE_SPECIAL_CHARS}")

    return tokenizer


def get_lineage_rna_tokenizer(use_direction_tokens: bool = True) -> LineageRNATokenizer:
    """Get Lineage RNA tokenizer instance

    Args:
        use_direction_tokens: Whether to use 5/3 direction tokens
            - True: Load tokenizer with 5/3 direction tokens (vocab_size=114)
            - False: Load legacy tokenizer without direction tokens (vocab_size=112)

    Returns:
        LineageRNATokenizer instance
    """
    if use_direction_tokens:
        # Use tokenizer (with 5/3 direction tokens, vocab_size=114)
        tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    else:
        # Use legacy tokenizer (without direction tokens, vocab_size=112)
        tokenizer_path = os.path.join(os.path.dirname(__file__), "lineage_tokenizer_old.json")

    if os.path.exists(tokenizer_path):
        tokenizer = LineageRNATokenizer.from_file(tokenizer_path)
        print(f"Loaded Lineage RNA tokenizer: {os.path.basename(tokenizer_path)} (vocab_size={tokenizer.vocab_size})")
        return tokenizer
    else:
        if use_direction_tokens:
            # If new tokenizer file doesn't exist, create it
            print(f"Creating new Lineage RNA tokenizer: {tokenizer_path}")
            return create_lineage_rna_tokenizer_json(tokenizer_path)
        else:
            # Missing legacy tokenizer file is a critical error
            raise FileNotFoundError(
                f"Legacy tokenizer file does not exist: {tokenizer_path}\n"
                "Please ensure the legacy tokenizer has been backed up as lineage_tokenizer_vocab134.json"
            )


if __name__ == "__main__":
    # Test Lineage RNA tokenizer
    tokenizer = LineageRNATokenizer()

    # Test sequences (lineage in lowercase, RNA sequences in uppercase)
    test_sequences = [
        "|d__eukaryota;p__chordata;c__mammalia;<rna_mRNA>|AUGCUAGCUAGC<eos>",
        "|d__bacteria;p__;c__;o__;f__;g__escherichia;s__escherichia_coli;<rna_rRNA>|AUCGAUCG<eos>",
        "AUGCUAGC",  # Pure RNA sequence
    ]

    print("=== Lineage RNA Tokenizer Test ===")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()

    for seq in test_sequences:
        encoded = tokenizer.encode(seq)
        decoded = tokenizer.decode(encoded)
        is_correct = seq == decoded

        print(f"Original sequence: {seq}")
        print(f"Encoded length: {len(encoded)}")
        print(f"Decoded result: {decoded}")
        print(f"Encoding correct: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print(f"  Difference: original length={len(seq)}, decoded length={len(decoded)}")
        print("-" * 80)

    # Save tokenizer
    output_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    create_lineage_rna_tokenizer_json(output_path)
