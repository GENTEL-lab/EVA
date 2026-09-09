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
"""RNAGen模型配置 - 基于Mixtral架构适配RNA序列任务"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .tokenizer import get_rna_tokenizer

logger = logging.get_logger(__name__)


class RNAGenConfig(PretrainedConfig):
    model_type = "rnagen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Tokenizer (支持自定义tokenizer)
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
        # ============================================
        # Dropout regularization (Stage 1-3)
        # Reference: Srivastava et al. 2014, Vaswani et al. 2017
        # Biological sequence models: ProtGPT2, ESM-2, RNA-FM, DNABERT
        # ============================================
        # Stage 1: Core dropout
        resid_dropout=0.0,  # Residual dropout (applied to attention output)
        hidden_dropout=0.0,  # Hidden/MLP dropout (applied to MoE/FFN output)
        # Dropout warmup scheduling
        dropout_warmup_steps=3000,  # First 3k steps: dropout=0
        dropout_ramp_steps=6000,  # 3k-9k steps: linear ramp to target
        dropout_schedule="linear",  # "linear" | "cosine"
        # Stage 2: Optional optimizations
        gradient_clip_norm=0.0,  # 0 = disabled, typical value: 1.0
        label_smoothing=0.0,  # 0 = disabled, typical value: 0.05
        # Stage 3: Attention dropout (already defined above, kept for reference)
        # attention_dropout=0.0
        # Non-implemented dropout types (preserved for reference)
        embed_dropout=0.0,  # Not implemented (low ROI)
        layerdrop=0.0,  # Not implemented (conflicts with MoE)
        head_drop_p=0.0,  # Not implemented (uncertain benefits)
        rdrop_alpha=0.0,  # Not implemented (excessive cost)
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
        # 如果没有传入tokenizer，使用默认的RNATokenizer
        if tokenizer is None:
            tokenizer = get_rna_tokenizer()
        super().__init__(
            pad_token_id=tokenizer.token_to_id("<pad>"),
            bos_token_id=tokenizer.token_to_id("<bos>"),
            eos_token_id=tokenizer.token_to_id("<eos>"),
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

        # ============================================
        # Dropout regularization configuration
        # ============================================
        # Stage 1: Core dropout
        assert 0.0 <= resid_dropout <= 0.3, "resid_dropout should be in [0, 0.3]"
        assert 0.0 <= hidden_dropout <= 0.3, "hidden_dropout should be in [0, 0.3]"
        assert dropout_schedule in ["linear", "cosine"], "Only linear/cosine schedule supported"
        self.resid_dropout = resid_dropout
        self.hidden_dropout = hidden_dropout
        self.dropout_warmup_steps = dropout_warmup_steps
        self.dropout_ramp_steps = dropout_ramp_steps
        self.dropout_schedule = dropout_schedule

        # Stage 2: Optional optimizations
        assert gradient_clip_norm >= 0.0, "gradient_clip_norm must be non-negative"
        assert 0.0 <= label_smoothing <= 0.2, "label_smoothing should be in [0, 0.2]"
        self.gradient_clip_norm = gradient_clip_norm
        self.label_smoothing = label_smoothing

        # Non-implemented dropout types (preserved for reference)
        self.embed_dropout = embed_dropout
        self.layerdrop = layerdrop
        self.head_drop_p = head_drop_p
        self.rdrop_alpha = rdrop_alpha

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

        self.vocab_size = tokenizer.vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing

        # 确保vocab_size = tokenizer的设置，假如不同，大的覆盖小的
        if vocab_size is not None:
            if vocab_size < self.vocab_size:
                logger.warning(f"Ignoring vocab_size {vocab_size}. Using larger {self.vocab_size} from tokenizer.")
            elif vocab_size > self.vocab_size:
                logger.warning(f"Using vocab_size {vocab_size} instead of smaller {self.vocab_size} from tokenizer.")
                self.vocab_size = vocab_size
        if pad_token_id is not None and pad_token_id != self.pad_token_id:
            logger.warning(f"Ignoring pad_token_id. Using {self.pad_token_id} from tokenizer")
        if bos_token_id is not None and bos_token_id != self.bos_token_id:
            logger.warning(f"Ignoring bos_token_id. Using {self.bos_token_id} from tokenizer")
        if eos_token_id is not None and eos_token_id != self.eos_token_id:
            logger.warning(f"Ignoring eos_token_id. Using {self.eos_token_id} from tokenizer")