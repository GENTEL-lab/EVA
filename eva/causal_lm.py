"""
Eva model implementation
RNA-specific generative model based on MoE architecture, supporting expert parallelism and weight parallelism

Technical architecture: Based on MoE (Mixture of Experts) architecture
Application domain: RNA sequence generation and understanding
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.distributed.tensor import DeviceMesh
except ImportError:
    # Compatible with older PyTorch versions
    DeviceMesh = None

from transformers.cache_utils import DynamicCache

from .config import EvaConfig
from .modeling import EvaPreTrainedModel, MoeModelOutputWithPast, MoeCausalOutputWithPast
from .attention import Attention
from .mb_wrapper import mb_setup_args, mb_build_dmoe
from .moe import MOE_CLASSES

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """RMS normalization layer"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Ensure eps is numeric type - automatically handle string conversion
        self.variance_epsilon = float(eps)
        # Validate parameter type
        if not isinstance(self.variance_epsilon, (int, float)):
            raise TypeError(f"variance_epsilon must be numeric type, but received {type(self.variance_epsilon)}")

    def forward(self, hidden_states: torch.Tensor):
        # Standard RMSNorm implementation, avoiding Flash Attention issues
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class EvaLayer(nn.Module):
    """Eva's single transformer layer"""

    def __init__(
        self,
        config: EvaConfig,
        layer_idx: int,
        **moe_kwargs
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention layer
        self.attention = Attention(
            config=config,
            layer_idx=layer_idx,
        )

        # MoE layer
        self.moe_layer = MOE_CLASSES[config.moe_implementation](config, **moe_kwargs)

        # RMS layer normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing

        logger.info(f"EvaLayer {layer_idx} initialization completed")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_router_weights: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward propagation"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE layer processing
        if self.config.moe_implementation == "megablocks":
            hidden_states = self.moe_layer(hidden_states)
            router_weights = None
        else:
            hidden_states, router_weights = self.moe_layer(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_weights:
            outputs += (router_weights,)
        return outputs


class EvaBaseModel(EvaPreTrainedModel):
    """Eva base model"""

    def __init__(self, config: EvaConfig, meta_init: bool = False):
        super().__init__(config)
        self.config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)

        # Fix: Use correct device when MegaBlocks is set
        if config.moe_implementation == "megablocks":
            # Use current CUDA device, not from uninitialized parameters
            if torch.cuda.is_available():
                current_device = f"cuda:{torch.cuda.current_device()}"
            else:
                current_device = "cpu"

            # Determine data type based on configuration
            if hasattr(config, 'bf16') and config.bf16:
                dtype = torch.bfloat16
            elif hasattr(config, 'fp16') and config.fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32

            logger.info(f"MegaBlocks initialization: device={current_device}, dtype={dtype}")
            mb_args, device_mesh = mb_setup_args(config, dtype=dtype, device=current_device)
            kwargs = dict(args=mb_args, device_mesh=device_mesh)
            self.mb_args = mb_args
            self.expert_parallel_device_mesh = device_mesh
        else:
            kwargs = dict()

        # Transformer layers
        self.layers = nn.ModuleList([
            EvaLayer(config, i, **kwargs)
            for i in range(config.num_hidden_layers)
        ])

        # Final RMS layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing setting
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.post_init()

        logger.info(f"EvaBaseModel initialization completed:")
        logger.info(f"  - Vocabulary size: {config.vocab_size}")
        logger.info(f"  - Hidden layer size: {config.hidden_size}")
        logger.info(f"  - Number of layers: {config.num_hidden_layers}")
        logger.info(f"  - Number of experts: {config.num_experts}")
        logger.info(f"  - MoE world size: {config.moe_world_size}")
        logger.info(f"  - MoE implementation: {config.moe_implementation}")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,  # RNA model requires sequence ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_router_weights: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs  # Capture other possible parameters
    ) -> MoeModelOutputWithPast:
        """Forward propagation"""
        batch_size, seq_length = input_ids.shape

        # Parameter processing
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MegaBlocks load balancing loss cleanup
        if self.config.moe_implementation == "megablocks":
            import megablocks.layers.moe
            megablocks.layers.moe.clear_load_balancing_loss()

        # Embedding
        position_ids = position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids)

        # Type conversion
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.layers[0].attention.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)

        # KV cache initialization: create DynamicCache on first call
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_weights = () if output_router_weights else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Gradient checkpointing
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_ids,
                    past_key_values,
                    attention_mask,
                    output_attentions,
                    output_router_weights,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_router_weights=output_router_weights,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_weights:
                all_router_weights += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # DynamicCache is updated in-place, directly return past_key_values
        next_cache = past_key_values if use_cache else None
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_weights,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_weights=all_router_weights,
        )


class EvaForCausalLM(EvaPreTrainedModel):
    """Eva model for causal language modeling"""

    def __init__(self, config: EvaConfig):
        super().__init__(config)
        self.config = config

        # Base model
        self.model = EvaBaseModel(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Output token mask (for conditional generation tasks, restricts output vocabulary)
        # Set in training script: model.output_token_mask = mask
        self.output_token_mask = None

        # Initialize weights
        self.post_init()

        logger.info(f"EvaForCausalLM initialization completed")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,  # RNA model requires sequence ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_weights: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalOutputWithPast]:
        """Forward propagation"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Model forward propagation
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_weights=output_router_weights,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        loss = None

        # Calculate autoregressive language modeling loss
        logits = self.lm_head(hidden_states).float()
        if labels is not None:
            # Shift inputs and labels so that token < n predicts n, and flatten them
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)

            # Apply output token mask (conditional generation task: restrict output vocabulary)
            # Apply during both training and inference, compute softmax only on allowed tokens
            # Prerequisite: data layer must ensure tokens in labels are all in allowed set (guaranteed by extra protection in lineage_dataset.py)
            if self.output_token_mask is not None:
                # output_token_mask: [vocab_size], True indicates outputtable token
                mask_device = shift_logits.device
                output_mask = self.output_token_mask.to(mask_device)
                shift_logits = shift_logits.masked_fill(~output_mask, float('-inf'))

            # RNADataCollator uses -100 to pad labels (rna_collator.py:148)
            # Use CrossEntropyLoss's default ignore_index=-100
            ar_loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="mean"
            )
            loss = ar_loss.clone()  # or loss = ar_loss.detach().clone()
        else:
            ar_loss = None

        aux_loss = None
        if self.config.moe_implementation == "megablocks" and self.training:
            # Use MegaBlocks' batched load balancing loss
            # Note: batched_load_balancing_loss internally already applies router_aux_loss_coef (via moe_loss_weight parameter)
            # So no need to multiply by weight coefficient again here
            import megablocks.layers.moe
            aux_loss = megablocks.layers.moe.batched_load_balancing_loss(self.model.mb_args)

            # Add weighted auxiliary loss to total loss
            if loss is not None:
                loss += aux_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_weights:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output
        
        return MoeCausalOutputWithPast(
            loss=loss,
            ar_loss=ar_loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else outputs[1] if len(outputs) > 1 else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            router_weights=outputs.router_weights if hasattr(outputs, 'router_weights') else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        output_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """
        Autoregressive sequence generation

        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            sequence_ids: Sequence IDs [batch_size, seq_len]
            max_length: Maximum generation length (optional, choose one with max_new_tokens)
            max_new_tokens: Maximum number of new tokens to generate (optional, choose one with max_length)
            min_new_tokens: Minimum number of new tokens to generate (prevent premature EOS output)
            temperature: Temperature parameter (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty coefficient
            num_return_sequences: Number of sequences to generate per input
            do_sample: Whether to sample (True=sampling generation, False=greedy decoding)
            eos_token_id: End token ID
            pad_token_id: Padding token ID
            output_token_mask: Mask for outputtable tokens [vocab_size]

        Returns:
            Generated sequences [batch_size * num_return_sequences, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process max_length and max_new_tokens parameters
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = input_ids.shape[1] + 100  # Default generate 100 new tokens

        # Ensure output_token_mask is on correct device
        if output_token_mask is not None:
            output_token_mask = output_token_mask.to(device)

        # Duplicate input to support num_return_sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            sequence_ids = sequence_ids.repeat_interleave(num_return_sequences, dim=0)

        # Initialize generated sequences
        generated = input_ids.clone()
        current_length = input_ids.shape[1]

        # Track whether generation is complete
        unfinished_sequences = torch.ones(batch_size * num_return_sequences, dtype=torch.long, device=device)

        # Track number of generated tokens
        num_generated_tokens = 0

        for _ in range(max_length - current_length):
            # Forward propagation
            outputs = self.forward(
                input_ids=generated,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
            )

            # Get logits for last token
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply output_token_mask (restrict to only output specific tokens)
            if output_token_mask is not None:
                next_token_logits = next_token_logits.masked_fill(~output_token_mask, float('-inf'))

            # Prevent premature EOS output (before reaching min_new_tokens)
            if eos_token_id is not None and num_generated_tokens < min_new_tokens:
                next_token_logits[:, eos_token_id] = float('-inf')

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(generated.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability exceeding top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            # Sampling or greedy decoding
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding: select token with highest probability
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Use pad_token for completed sequences
            if pad_token_id is not None and eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Concatenate new token
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            # Update generated token count
            num_generated_tokens += 1

            # Update position_ids and sequence_ids
            current_length += 1
            new_position_ids = torch.full((position_ids.shape[0], 1), current_length - 1, dtype=torch.long, device=device)
            position_ids = torch.cat([position_ids, new_position_ids], dim=1)

            new_sequence_ids = sequence_ids[:, -1:].clone()
            sequence_ids = torch.cat([sequence_ids, new_sequence_ids], dim=1)

            # Check if EOS token encountered
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # Stop generation if all sequences are complete
            if unfinished_sequences.max() == 0:
                break

        return generated

    def _generate_one_chunk(
        self,
        sequence: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        chunk_size: int,
        temperature: float,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        output_token_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> dict:
        """
        Generate one chunk (internal helper function)

        Args:
            sequence: Current sequence [seq_len]
            position_ids: Position IDs [seq_len]
            sequence_ids: Sequence IDs [seq_len]
            chunk_size: Chunk size
            temperature: Temperature parameter
            eos_token_id: EOS token ID
            pad_token_id: PAD token ID
            output_token_mask: Mask for outputtable tokens
            device: Device

        Returns:
            dict: {
                'sequence': Complete sequence after generation,
                'position_ids': Updated position_ids,
                'sequence_ids': Updated sequence_ids,
                'log_prob': Cumulative log probability for this chunk,
                'has_eos': Whether EOS token was encountered
            }
        """
        current_seq = sequence.unsqueeze(0)  # [1, seq_len]
        current_pos = position_ids.unsqueeze(0)
        current_seq_ids = sequence_ids.unsqueeze(0)

        chunk_log_prob = 0.0
        has_eos = False

        for step in range(chunk_size):
            # Forward propagation
            outputs = self.forward(
                input_ids=current_seq,
                position_ids=current_pos,
                sequence_ids=current_seq_ids,
            )

            # Get logits for last token
            next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]

            # Apply output_token_mask
            if output_token_mask is not None:
                next_token_logits = next_token_logits.masked_fill(
                    ~output_token_mask, float('-inf')
                )

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Calculate probability
            probs = torch.softmax(next_token_logits, dim=-1)
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Sampling (use sampling instead of greedy to increase diversity)
            next_token = torch.multinomial(probs, num_samples=1)

            # Accumulate log probability
            chunk_log_prob += log_probs[next_token].item()

            # Check if it's EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                has_eos = True
                # Add EOS token
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                break

            # Add new token
            current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)

            # Update position_ids
            new_pos = torch.tensor([[current_pos.size(1)]], device=device)
            current_pos = torch.cat([current_pos, new_pos], dim=1)

            # Update sequence_ids
            new_seq_id = current_seq_ids[:, -1:].clone()
            current_seq_ids = torch.cat([current_seq_ids, new_seq_id], dim=1)

        return {
            'sequence': current_seq[0],
            'position_ids': current_pos[0],
            'sequence_ids': current_seq_ids[0],
            'log_prob': chunk_log_prob,
            'has_eos': has_eos,
        }

    def chunk_beam_search_generate(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        num_beams: int = 5,
        chunk_size: int = 10,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_return_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        output_token_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        early_stopping: bool = True,
        verbose: bool = False,
    ) -> torch.LongTensor:
        """
        Chunk-based Beam Search generation

        Generate chunk_size tokens each time, then perform beam selection,
        more efficient than traditional beam search

        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            sequence_ids: Sequence IDs [batch_size, seq_len]
            num_beams: Number of beams
            chunk_size: Number of tokens per chunk (adjustable)
            max_length: Maximum generation length
            max_new_tokens: Maximum number of new tokens to generate
            num_return_sequences: Number of sequences to return (must be <= num_beams)
            eos_token_id: End token ID
            pad_token_id: Padding token ID
            output_token_mask: Mask for outputtable tokens [vocab_size]
            temperature: Generation temperature (for internal chunk generation)
            early_stopping: Whether to early stop
            verbose: Whether to print detailed information

        Returns:
            Generated sequences [num_return_sequences, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Parameter validation
        assert num_return_sequences <= num_beams, \
            f"num_return_sequences ({num_return_sequences}) must be <= num_beams ({num_beams})"
        assert batch_size == 1, "Chunk beam search currently only supports batch_size=1"

        # Process max_length
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = input_ids.shape[1] + 100

        # Calculate how many chunks need to be generated
        current_length = input_ids.shape[1]
        total_new_tokens = max_length - current_length
        num_chunks = (total_new_tokens + chunk_size - 1) // chunk_size  # Round up

        if verbose:
            print(f"\n=== Chunk-based Beam Search ===")
            print(f"  Number of beams: {num_beams}")
            print(f"  Chunk size: {chunk_size}")
            print(f"  Total chunks: {num_chunks}")
            print(f"  Target length: {max_length}")
            print(f"  Temperature: {temperature}")

        # Ensure output_token_mask is on correct device
        if output_token_mask is not None:
            output_token_mask = output_token_mask.to(device)

        # Initialize beams
        # Each beam contains: sequence, position_ids, sequence_ids, cumulative log probability
        beams = [{
            'sequence': input_ids[0].clone(),
            'position_ids': position_ids[0].clone(),
            'sequence_ids': sequence_ids[0].clone(),
            'log_prob': 0.0,
            'finished': False,
        }]

        finished_beams = []

        # Generate chunk by chunk
        for chunk_idx in range(num_chunks):
            if verbose:
                print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ---")

            new_beams = []

            # Generate one chunk for each beam
            for beam_idx, beam in enumerate(beams):
                if beam['finished']:
                    # Keep finished beams as-is
                    new_beams.append(beam)
                    continue

                # Generate one chunk
                chunk_result = self._generate_one_chunk(
                    sequence=beam['sequence'],
                    position_ids=beam['position_ids'],
                    sequence_ids=beam['sequence_ids'],
                    chunk_size=chunk_size,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    output_token_mask=output_token_mask,
                    device=device,
                )

                # Check if EOS was encountered
                if chunk_result['has_eos']:
                    # This beam is finished
                    finished_beams.append({
                        'sequence': chunk_result['sequence'],
                        'log_prob': beam['log_prob'] + chunk_result['log_prob'],
                        'length': chunk_result['sequence'].size(0),
                    })
                    if verbose:
                        print(f"  Beam {beam_idx} finished (encountered <eos>)")
                else:
                    # Continue generating
                    new_beams.append({
                        'sequence': chunk_result['sequence'],
                        'position_ids': chunk_result['position_ids'],
                        'sequence_ids': chunk_result['sequence_ids'],
                        'log_prob': beam['log_prob'] + chunk_result['log_prob'],
                        'finished': False,
                    })

            # Stop if no unfinished beams remain
            if len(new_beams) == 0:
                if verbose:
                    print("  All beams finished")
                break

            # Beam selection: select top-k beams
            if len(new_beams) > num_beams:
                # Sort by log probability
                new_beams = sorted(new_beams, key=lambda x: x['log_prob'], reverse=True)
                new_beams = new_beams[:num_beams]
                if verbose:
                    print(f"  Keeping top-{num_beams} beams")

            beams = new_beams

            # Early stopping check
            if early_stopping and len(finished_beams) >= num_beams:
                if verbose:
                    print(f"  Early stopping: {len(finished_beams)} beams already finished")
                break

        # Add unfinished beams to finished_beams as well
        for beam in beams:
            if not beam['finished']:
                finished_beams.append({
                    'sequence': beam['sequence'],
                    'log_prob': beam['log_prob'],
                    'length': beam['sequence'].size(0),
                })

        # Sort by log probability
        finished_beams = sorted(finished_beams, key=lambda x: x['log_prob'], reverse=True)

        if verbose:
            print(f"\n=== Generation Complete ===")
            print(f"  Number of finished beams: {len(finished_beams)}")
            print(f"  Top-{num_return_sequences} beam scores:")
            for i, beam in enumerate(finished_beams[:num_return_sequences]):
                print(f"    Beam {i+1}: log_prob={beam['log_prob']:.4f}, length={beam['length']}")

        # Return top num_return_sequences sequences
        result_sequences = [beam['sequence'] for beam in finished_beams[:num_return_sequences]]

        # Pad to the same length
        max_len = max(seq.size(0) for seq in result_sequences)
        padded_sequences = []
        for seq in result_sequences:
            if seq.size(0) < max_len:
                padding = torch.full(
                    (max_len - seq.size(0),),
                    pad_token_id if pad_token_id is not None else 0,
                    device=device,
                    dtype=torch.long
                )
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)

        return torch.stack(padded_sequences)


    @property
    def device_mesh(self):
        """Return the expert parallel device mesh"""
        return getattr(self.model, 'expert_parallel_device_mesh', None)


def create_eva_model(
    config: EvaConfig,
) -> EvaForCausalLM:
    """
    Factory function for creating Eva model - simplified version

    Args:
        config: Model configuration

    Returns:
        EvaForCausalLM instance
    """
    # Create model directly, let MegaBlocks auto-detect current device
    return EvaForCausalLM(config)