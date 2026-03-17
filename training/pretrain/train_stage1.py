#!/usr/bin/env python3
"""
Lineage-based Stage 1: 条件序列生成
基于Greengenes谱系字符串和RNA类型生成完整RNA序列
支持双机16卡分布式训练，使用MegaBlocks架构
"""

import os
import sys

# 抑制NCCL冗余日志输出（必须在import torch之前设置）
os.environ.setdefault('NCCL_DEBUG', 'WARN')

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.common.base_trainer import BaseTrainer

# 配置基础日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class LineageStage1Trainer(BaseTrainer):
    """Lineage Stage 1 条件序列生成训练器"""

    @property
    def stage_name(self) -> str:
        return 'lineage_stage1'

    @property
    def default_log_dir(self) -> str:
        return '/rna-multiverse/results/lineage_stage1_output/logs'

    @property
    def default_wandb_project(self) -> str:
        return 'rna-lineage-stage1'

    @property
    def uses_moe(self) -> bool:
        return True

    def _setup_model(self):
        """设置MoE模型"""
        from model.causal_lm import create_rnagen_model
        from model.lineage_tokenizer import get_lineage_rna_tokenizer
        from utils.device import create_device_manager, set_device_manager

        model_config_dict = self.config.get('model_config', {})
        distributed_config = self.config.get('distributed_config', {})
        data_config = self.config.get('data_config', {})

        use_direction_tokens = data_config.get('use_direction_tokens', True)
        self.tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)

        model_config_dict['vocab_size'] = self.tokenizer.vocab_size
        logger.info(f"自动设置vocab_size={self.tokenizer.vocab_size} (use_direction_tokens={use_direction_tokens})")

        if model_config_dict.get('moe_implementation') == "megablocks":
            moe_world_size = model_config_dict.get('moe_world_size', 1)
            if moe_world_size > 1:
                device_manager = create_device_manager(
                    world_size=self.world_size,
                    moe_world_size=moe_world_size,
                    weight_parallel_size=distributed_config.get('weight_parallel_size', 1)
                )
                set_device_manager(device_manager)
                self.device_manager = device_manager

        from model.config import RNAGenConfig
        model_config = RNAGenConfig(tokenizer=self.tokenizer, **model_config_dict)
        self.model = create_rnagen_model(model_config)
        logger.info(f"模型创建完成，设备: cuda:{self.local_rank}")

        training_config = self.config.get('training_config', {})
        if training_config.get('bf16', False):
            target_dtype = torch.bfloat16
        elif training_config.get('fp16', False):
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        self.model = self.model.to(f'cuda:{self.local_rank}')
        if target_dtype != torch.float32:
            self.model = self.model.to(target_dtype)

        if self.world_size > 1 and hasattr(self, 'device_manager'):
            if self.global_rank == 0:
                logger.info("MegaBlocks分布式配置:")
                logger.info(f"   - Expert Parallel组: {model_config_dict.get('moe_world_size', 1)} GPUs")
                logger.info(f"   - Data Parallel组: {self.world_size // model_config_dict.get('moe_world_size', 1)} GPUs")

        # 设置输出token mask
        model_to_set = self.model
        if model_config_dict.get('moe_implementation') == "megablocks" and model_config_dict.get('moe_world_size', 1) > 1:
            model_to_set.output_token_mask = None
            logger.info("MegaBlocks模式：禁用output_token_mask（避免DTensor混合错误）")
        else:
            output_token_ids = self.tokenizer.get_stage1_output_token_ids()
            output_token_mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool, device=f'cuda:{self.local_rank}')
            output_token_mask[output_token_ids] = True
            model_to_set.output_token_mask = output_token_mask
            logger.info(f"Stage 1输出token mask已设置: {len(output_token_ids)}/{self.tokenizer.vocab_size} 个可输出token")

    def _sync_gradients(self):
        """手动同步非 MoE 层的梯度"""
        if self.world_size <= 1 or not hasattr(self, 'device_manager'):
            return
        dp_group = self.device_manager.get_data_parallel_group()
        for name, param in self.model.named_parameters():
            if 'moe' in name.lower() or 'expert' in name.lower():
                continue
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, group=dp_group)

    def _extra_metrics(self, base_metrics):
        """添加 label smoothing 监控"""
        extra = {}
        model_cfg = self.config.get('model_config', {})
        label_smoothing = model_cfg.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            extra['regularization/label_smoothing'] = label_smoothing
        return extra


def main():
    LineageStage1Trainer.main(
        description='Lineage-based Stage 1: 序列生成',
        default_config='configs/lineage_training/lineage_stage1_16gpu.yaml',
        supports_resume=True,
    )


if __name__ == '__main__':
    main()
