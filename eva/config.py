# Copyright 2023 Mixtral AI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eva model configuration - Adapted from Mixtral architecture for RNA sequence tasks"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class EvaConfig(PretrainedConfig):
    model_type = "rnagen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Tokenizer (supports custom tokenizer)
        tokenizer=None,
        # Model architecture/initialization
        vocab_size=None,
        hidden_size=4096,
        intermediate_size=16384,
        gated_mlp=False,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        torch_dtype="bfloat16",
        use_cache=True,
        gradient_checkpointing=False,
        no_ffn_gradient_checkpointing=False,
        # Tokenization
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        # Attention implementation & rotary positional embeddings
        fused_attention_norm=False,
        msa_style_attention=True,
        max_num_sequences=512,
        max_position_embeddings=1024 * 64,
        rope_theta=100000.0,
        attention_dropout=0.0,
        clip_qkv=None,
        # Mixture of experts implementation
        moe_implementation="megablocks",
        moe_expert_selection="switch",
        moe_grouped_gemm=True,
        moe_memory_optimized=None,
        num_experts=8,
        num_experts_per_tok=2,
        moe_world_size=1,
        output_router_weights=False,
        # Additional activation quantization fn
        quantize_inputs_num_bits=None,
        quantize_rematerialize_num_bits=None,
        quantize_scatter_num_bits=None,
        # Loss function details
        mlm_loss_coef=1.0,
        # From DBRX, https://github.com/databricks/dbrx/blob/main/model/config.json
        router_aux_loss_coef=0.05,
        **kwargs,
    ) -> None:
        # If tokenizer is provided, get token IDs from it
        # Otherwise use the passed parameter values (these values will be provided when loading from JSON)
        if tokenizer is not None:
            if pad_token_id is None:
                pad_token_id = tokenizer.token_to_id("<pad>")
            if bos_token_id is None:
                bos_token_id = tokenizer.token_to_id("<bos>")
            if eos_token_id is None:
                eos_token_id = tokenizer.token_to_id("<eos>")
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = 3 * hidden_size if gated_mlp else 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.gated_mlp = gated_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.fused_attention_norm = fused_attention_norm
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.msa_style_attention = msa_style_attention
        self.max_num_sequences = max_num_sequences
        assert clip_qkv is None or clip_qkv > 0
        self.clip_qkv = clip_qkv

        num_experts_per_tok = min(num_experts_per_tok, num_experts)
        assert num_experts > 0 and num_experts_per_tok > 0
        assert (
            num_experts % moe_world_size == 0
        ), f"Expected {moe_world_size=} to perfectly divide {num_experts=}"  # noqa: E225
        if num_experts == 1:
            moe_implementation = "eager"
            moe_expert_selection = "switch"
        if num_experts == 1 or moe_expert_selection == "sinkhorn":
            router_aux_loss_coef = 0.0
        if moe_implementation != "megablocks":
            moe_world_size = 1
            output_router_weights = output_router_weights or router_aux_loss_coef > 0
        if moe_memory_optimized is None:
            moe_memory_optimized = moe_grouped_gemm

        self.quantize_inputs_num_bits = quantize_inputs_num_bits
        self.quantize_rematerialize_num_bits = quantize_rematerialize_num_bits
        self.quantize_scatter_num_bits = quantize_scatter_num_bits
        assert quantize_inputs_num_bits is None or quantize_inputs_num_bits == 8, "Only 8-bit quantization is supported"
        assert (
            self.quantize_inputs_num_bits == self.quantize_rematerialize_num_bits == self.quantize_scatter_num_bits
        ), "Different quantization bitwidths for inputs, rematerialize, and scatter are not supported"

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_weights = output_router_weights
        self.mlm_loss_coef = mlm_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

        self.moe_implementation = moe_implementation
        self.moe_expert_selection = moe_expert_selection
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_memory_optimized = moe_memory_optimized
        self.moe_world_size = max(1, moe_world_size)

        # Set vocab_size: prioritize tokenizer, then use passed parameter
        if tokenizer is not None:
            tokenizer_vocab_size = tokenizer.vocab_size
            # If vocab_size parameter is also provided, compare and warn
            if vocab_size is not None:
                if vocab_size < tokenizer_vocab_size:
                    logger.warning(f"Ignoring vocab_size {vocab_size}. Using larger {tokenizer_vocab_size} from tokenizer.")
                    self.vocab_size = tokenizer_vocab_size
                elif vocab_size > tokenizer_vocab_size:
                    logger.warning(f"Using vocab_size {vocab_size} instead of smaller {tokenizer_vocab_size} from tokenizer.")
                    self.vocab_size = vocab_size
                else:
                    self.vocab_size = tokenizer_vocab_size
            else:
                self.vocab_size = tokenizer_vocab_size
        elif vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            # When loading from JSON, vocab_size should be present; if neither exists, set to None (will error on subsequent use)
            self.vocab_size = None

        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing

        # Only check for token ID conflicts with passed parameters when tokenizer is provided
        # (When loading from JSON, there will be no tokenizer, these values will be loaded directly from config file)