"""
RNAGen模型实现
基于MoE架构的RNA专用生成模型，支持专家并行和权重并行

技术架构：基于MoE（Mixture of Experts）架构
应用领域：RNA序列生成和理解
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
    # 兼容旧版本PyTorch
    DeviceMesh = None

from .config import RNAGenConfig
from .modeling import RNAGenPreTrainedModel, MoeModelOutputWithPast, MoeCausalOutputWithPast
from .attention import Attention
from .mb_wrapper import mb_setup_args, mb_build_dmoe
from .moe import MOE_CLASSES

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """RMS归一化层 - 参考 src/progen3/modeling.py"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 确保 eps 是数值类型 - 自动处理字符串转换
        self.variance_epsilon = float(eps)
        # 验证参数类型
        if not isinstance(self.variance_epsilon, (int, float)):
            raise TypeError(f"variance_epsilon 必须是数值类型，但收到 {type(self.variance_epsilon)}")

    def forward(self, hidden_states: torch.Tensor):
        # 标准 RMSNorm 实现，避免 Flash Attention 的问题
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RNAGenLayer(nn.Module):
    """RNAGen的单个transformer层 - 基于ProGen3架构设计"""
    
    def __init__(
        self,
        config: RNAGenConfig,
        layer_idx: int,
        **moe_kwargs
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 注意力层
        self.attention = Attention(
            config=config,
            layer_idx=layer_idx,
        )
        
        # MoE层 - 使用原始ProGen3的MOE_CLASSES
        self.moe_layer = MOE_CLASSES[config.moe_implementation](config, **moe_kwargs)
        
        # RMS层归一化 - 参考 src/progen3/modeling.py
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 梯度检查点
        self.gradient_checkpointing = config.gradient_checkpointing
        
        logger.info(f"RNAProGen3Layer {layer_idx} 初始化完成")
    
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
        """前向传播 - 参考原始ProGen3的DecoderLayer实现"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 使用原始ProGen3的MoE层处理逻辑
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


class RNAProGen3Model(RNAGenPreTrainedModel):
    """RNA-ProGen3基础模型 - 遵循原始ProGen3架构"""
    
    def __init__(self, config: RNAGenConfig, meta_init: bool = False):
        super().__init__(config)
        self.config = config
        
        # 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        
        # 修复：MegaBlocks设置时使用正确的设备
        if config.moe_implementation == "megablocks":
            # 使用当前CUDA设备，而不是从未初始化的参数获取
            if torch.cuda.is_available():
                current_device = f"cuda:{torch.cuda.current_device()}"
            else:
                current_device = "cpu"
            
            # 根据配置确定数据类型
            if hasattr(config, 'bf16') and config.bf16:
                dtype = torch.bfloat16
            elif hasattr(config, 'fp16') and config.fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            logger.info(f"MegaBlocks初始化: device={current_device}, dtype={dtype}")
            mb_args, device_mesh = mb_setup_args(config, dtype=dtype, device=current_device)
            kwargs = dict(args=mb_args, device_mesh=device_mesh)
            self.mb_args = mb_args
            self.expert_parallel_device_mesh = device_mesh
        else:
            kwargs = dict()
            
        # transformer层
        self.layers = nn.ModuleList([
            RNAGenLayer(config, i, **kwargs)
            for i in range(config.num_hidden_layers)
        ])
        
        # 最终RMS层归一化 - 参考 src/progen3/modeling.py
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 梯度检查点设置 - 参考 src/progen3/modeling.py 第378行
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # 初始化权重
        self.post_init()
        
        logger.info(f"RNAProGen3Model初始化完成:")
        logger.info(f"  - 词汇表大小: {config.vocab_size}")
        logger.info(f"  - 隐藏层大小: {config.hidden_size}")
        logger.info(f"  - 层数: {config.num_hidden_layers}")
        logger.info(f"  - 专家数量: {config.num_experts}")
        logger.info(f"  - MoE世界大小: {config.moe_world_size}")
        logger.info(f"  - MoE实现: {config.moe_implementation}")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,  # RNA模型需要序列ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_router_weights: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs  # 捕获其他可能的参数
    ) -> MoeModelOutputWithPast:
        """前向传播 - 参考原始ProGen3实现"""
        batch_size, seq_length = input_ids.shape
        
        # 参数处理 - 参考原始ProGen3
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # MegaBlocks负载均衡损失清理
        if self.config.moe_implementation == "megablocks":
            import megablocks.layers.moe
            megablocks.layers.moe.clear_load_balancing_loss()
        
        # 嵌入 - 参考原始ProGen3
        position_ids = position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids)
        
        # 类型转换 - 参考原始ProGen3
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.layers[0].attention.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)
        
        # transformer层 - 参考原始ProGen3实现
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_weights = () if output_router_weights else None
        next_decoder_cache = None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # 梯度检查点 - 参考原始ProGen3
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
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_router_weights:
                all_router_weights += (layer_outputs[-1],)
        
        hidden_states = self.norm(hidden_states)
        
        # 添加最终的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
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


class RNAGenForCausalLM(RNAGenPreTrainedModel):
    """RNA-ProGen3用于因果语言建模的模型"""
    
    def __init__(self, config: RNAGenConfig):
        super().__init__(config)
        self.config = config

        # 基础模型
        self.model = RNAProGen3Model(config)

        # LM头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 输出token mask（用于条件生成任务，限制输出词汇表）
        # 在训练脚本中设置：model.output_token_mask = mask
        self.output_token_mask = None

        # 初始化权重
        self.post_init()

        logger.info(f"RNAGenForCausalLM初始化完成")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,  # RNA模型需要序列ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_weights: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalOutputWithPast]:
        """前向传播 - 参考原始ProGen3ForCausalLM"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 模型前向传播
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

        # 计算自回归语言建模损失 - 参考原始ProGen3
        logits = self.lm_head(hidden_states).float()
        if labels is not None:
            # 移位输入和标签使得token < n预测n，并展平它们
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)

            # 应用输出token mask（条件生成任务：限制输出词汇表）
            # 训练和推理时都应用，只在允许的token上计算softmax
            # 前提：数据层必须确保labels中的token都在允许集合中（通过lineage_dataset.py的额外保护保证）
            if self.output_token_mask is not None:
                # output_token_mask: [vocab_size], True表示可输出的token
                mask_device = shift_logits.device
                output_mask = self.output_token_mask.to(mask_device)
                shift_logits = shift_logits.masked_fill(~output_mask, float('-inf'))

            # RNADataCollator 使用 -100 来 padding labels (rna_collator.py:148)
            # 使用 CrossEntropyLoss 的默认 ignore_index=-100
            ar_loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="mean"
            )
            loss = ar_loss.clone()  # 或者 loss = ar_loss.detach().clone()
        else:
            ar_loss = None
        
        aux_loss = None
        if self.config.moe_implementation == "megablocks" and self.training:
            # 使用MegaBlocks的批量负载均衡损失
            # 注意：batched_load_balancing_loss 内部已经应用了 router_aux_loss_coef (通过 moe_loss_weight 参数)
            # 所以这里不需要再次乘以权重系数
            import megablocks.layers.moe
            aux_loss = megablocks.layers.moe.batched_load_balancing_loss(self.model.mb_args)

            # 将加权后的辅助损失加到总损失中
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
        自回归生成序列

        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            sequence_ids: 序列ID [batch_size, seq_len]
            max_length: 最大生成长度（可选，与max_new_tokens二选一）
            max_new_tokens: 最大新生成token数（可选，与max_length二选一）
            min_new_tokens: 最小新生成token数（防止过早输出EOS）
            temperature: 温度参数（越大越随机）
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚系数
            num_return_sequences: 每个输入生成的序列数
            do_sample: 是否采样（True=采样生成，False=贪婪解码）
            eos_token_id: 结束token ID
            pad_token_id: padding token ID
            output_token_mask: 可输出token的mask [vocab_size]

        Returns:
            生成的序列 [batch_size * num_return_sequences, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 处理max_length和max_new_tokens参数
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = input_ids.shape[1] + 100  # 默认生成100个新token

        # 确保output_token_mask在正确的设备上
        if output_token_mask is not None:
            output_token_mask = output_token_mask.to(device)

        # 复制输入以支持num_return_sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            sequence_ids = sequence_ids.repeat_interleave(num_return_sequences, dim=0)

        # 初始化生成的序列
        generated = input_ids.clone()
        current_length = input_ids.shape[1]

        # 跟踪是否已完成生成
        unfinished_sequences = torch.ones(batch_size * num_return_sequences, dtype=torch.long, device=device)

        # 跟踪已生成的token数
        num_generated_tokens = 0

        for _ in range(max_length - current_length):
            # 前向传播
            outputs = self.forward(
                input_ids=generated,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
            )

            # 获取最后一个token的logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # 应用output_token_mask（限制只能输出特定token）
            if output_token_mask is not None:
                next_token_logits = next_token_logits.masked_fill(~output_token_mask, float('-inf'))

            # 防止过早输出EOS（在达到min_new_tokens之前）
            if eos_token_id is not None and num_generated_tokens < min_new_tokens:
                next_token_logits[:, eos_token_id] = float('-inf')

            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for i in range(generated.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            # 采样或贪婪解码
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码：选择概率最高的token
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # 对已完成的序列使用pad_token
            if pad_token_id is not None and eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # 拼接新token
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            # 更新已生成token数
            num_generated_tokens += 1

            # 更新position_ids和sequence_ids
            current_length += 1
            new_position_ids = torch.full((position_ids.shape[0], 1), current_length - 1, dtype=torch.long, device=device)
            position_ids = torch.cat([position_ids, new_position_ids], dim=1)

            new_sequence_ids = sequence_ids[:, -1:].clone()
            sequence_ids = torch.cat([sequence_ids, new_sequence_ids], dim=1)

            # 检查是否遇到EOS token
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # 如果所有序列都完成了，停止生成
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
        生成一个chunk（内部辅助函数）

        Args:
            sequence: 当前序列 [seq_len]
            position_ids: 位置ID [seq_len]
            sequence_ids: 序列ID [seq_len]
            chunk_size: chunk大小
            temperature: 温度参数
            eos_token_id: EOS token ID
            pad_token_id: PAD token ID
            output_token_mask: 可输出token的mask
            device: 设备

        Returns:
            dict: {
                'sequence': 生成后的完整序列,
                'position_ids': 更新后的position_ids,
                'sequence_ids': 更新后的sequence_ids,
                'log_prob': 这个chunk的累积log概率,
                'has_eos': 是否遇到了EOS token
            }
        """
        current_seq = sequence.unsqueeze(0)  # [1, seq_len]
        current_pos = position_ids.unsqueeze(0)
        current_seq_ids = sequence_ids.unsqueeze(0)

        chunk_log_prob = 0.0
        has_eos = False

        for step in range(chunk_size):
            # 前向传播
            outputs = self.forward(
                input_ids=current_seq,
                position_ids=current_pos,
                sequence_ids=current_seq_ids,
            )

            # 获取最后一个token的logits
            next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]

            # 应用output_token_mask
            if output_token_mask is not None:
                next_token_logits = next_token_logits.masked_fill(
                    ~output_token_mask, float('-inf')
                )

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 计算概率
            probs = torch.softmax(next_token_logits, dim=-1)
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # 采样（使用采样而不是贪婪，增加多样性）
            next_token = torch.multinomial(probs, num_samples=1)

            # 累积log概率
            chunk_log_prob += log_probs[next_token].item()

            # 检查是否是EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                has_eos = True
                # 添加EOS token
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                break

            # 添加新token
            current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)

            # 更新position_ids
            new_pos = torch.tensor([[current_pos.size(1)]], device=device)
            current_pos = torch.cat([current_pos, new_pos], dim=1)

            # 更新sequence_ids
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
        基于Chunk的Beam Search生成

        每次生成chunk_size个token，然后进行beam selection，
        比传统beam search更高效

        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            sequence_ids: 序列ID [batch_size, seq_len]
            num_beams: beam数量
            chunk_size: 每个chunk的token数（可调整）
            max_length: 最大生成长度
            max_new_tokens: 最大新生成token数
            num_return_sequences: 返回序列数（必须 <= num_beams）
            eos_token_id: 结束token ID
            pad_token_id: padding token ID
            output_token_mask: 可输出token的mask [vocab_size]
            temperature: 生成温度（用于chunk内部生成）
            early_stopping: 是否早停
            verbose: 是否打印详细信息

        Returns:
            生成的序列 [num_return_sequences, generated_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 参数验证
        assert num_return_sequences <= num_beams, \
            f"num_return_sequences ({num_return_sequences}) must be <= num_beams ({num_beams})"
        assert batch_size == 1, "Chunk beam search currently only supports batch_size=1"

        # 处理max_length
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = input_ids.shape[1] + 100

        # 计算需要生成多少个chunk
        current_length = input_ids.shape[1]
        total_new_tokens = max_length - current_length
        num_chunks = (total_new_tokens + chunk_size - 1) // chunk_size  # 向上取整

        if verbose:
            print(f"\n=== Chunk-based Beam Search ===")
            print(f"  Beam数量: {num_beams}")
            print(f"  Chunk大小: {chunk_size}")
            print(f"  总chunk数: {num_chunks}")
            print(f"  目标长度: {max_length}")
            print(f"  温度: {temperature}")

        # 确保output_token_mask在正确的设备上
        if output_token_mask is not None:
            output_token_mask = output_token_mask.to(device)

        # 初始化beams
        # 每个beam包含: 序列、position_ids、sequence_ids、累积log概率
        beams = [{
            'sequence': input_ids[0].clone(),
            'position_ids': position_ids[0].clone(),
            'sequence_ids': sequence_ids[0].clone(),
            'log_prob': 0.0,
            'finished': False,
        }]

        finished_beams = []

        # 逐chunk生成
        for chunk_idx in range(num_chunks):
            if verbose:
                print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ---")

            new_beams = []

            # 对每个beam生成一个chunk
            for beam_idx, beam in enumerate(beams):
                if beam['finished']:
                    # 已完成的beam直接保留
                    new_beams.append(beam)
                    continue

                # 生成一个chunk
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

                # 检查是否遇到EOS
                if chunk_result['has_eos']:
                    # 这个beam已完成
                    finished_beams.append({
                        'sequence': chunk_result['sequence'],
                        'log_prob': beam['log_prob'] + chunk_result['log_prob'],
                        'length': chunk_result['sequence'].size(0),
                    })
                    if verbose:
                        print(f"  Beam {beam_idx} 完成 (遇到<eos>)")
                else:
                    # 继续生成
                    new_beams.append({
                        'sequence': chunk_result['sequence'],
                        'position_ids': chunk_result['position_ids'],
                        'sequence_ids': chunk_result['sequence_ids'],
                        'log_prob': beam['log_prob'] + chunk_result['log_prob'],
                        'finished': False,
                    })

            # 如果没有未完成的beam，停止
            if len(new_beams) == 0:
                if verbose:
                    print("  所有beam已完成")
                break

            # Beam selection: 选择top-k个beam
            if len(new_beams) > num_beams:
                # 按log概率排序
                new_beams = sorted(new_beams, key=lambda x: x['log_prob'], reverse=True)
                new_beams = new_beams[:num_beams]
                if verbose:
                    print(f"  保留top-{num_beams}个beam")

            beams = new_beams

            # 早停检查
            if early_stopping and len(finished_beams) >= num_beams:
                if verbose:
                    print(f"  早停: 已有{len(finished_beams)}个完成的beam")
                break

        # 将未完成的beam也加入finished_beams
        for beam in beams:
            if not beam['finished']:
                finished_beams.append({
                    'sequence': beam['sequence'],
                    'log_prob': beam['log_prob'],
                    'length': beam['sequence'].size(0),
                })

        # 按log概率排序
        finished_beams = sorted(finished_beams, key=lambda x: x['log_prob'], reverse=True)

        if verbose:
            print(f"\n=== 生成完成 ===")
            print(f"  完成的beam数: {len(finished_beams)}")
            print(f"  Top-{num_return_sequences}个beam的分数:")
            for i, beam in enumerate(finished_beams[:num_return_sequences]):
                print(f"    Beam {i+1}: log_prob={beam['log_prob']:.4f}, length={beam['length']}")

        # 返回top num_return_sequences个序列
        result_sequences = [beam['sequence'] for beam in finished_beams[:num_return_sequences]]

        # Pad到相同长度
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
        """Return the expert parallel device mesh - 参考原始ProGen3"""
        return getattr(self.model, 'expert_parallel_device_mesh', None)


def create_rnagen_model(
    config: RNAGenConfig,
) -> RNAGenForCausalLM:
    """
    创建RNA-ProGen3模型的工厂函数 - 简化版本
    
    Args:
        config: 模型配置
    
    Returns:
        RNAGenForCausalLM实例
    """
    # 直接创建模型，让MegaBlocks自动检测当前设备
    return RNAGenForCausalLM(config)