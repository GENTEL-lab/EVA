#!/usr/bin/env python3
"""
Dense模型训练脚本
支持多GPU分布式训练，使用PyTorch DDP架构
"""

import os
import sys

# 抑制NCCL冗余日志输出
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class DenseTrainer(BaseTrainer):
    """Dense模型训练器 - 手动梯度同步模式"""

    @property
    def stage_name(self) -> str:
        return 'dense_training'

    @property
    def default_log_dir(self) -> str:
        return '/rna-multiverse/results/dense_training/logs'

    @property
    def default_wandb_project(self) -> str:
        return 'rna-dense-training'

    @property
    def uses_moe(self) -> bool:
        return False

    def __init__(self, config_path: str, resume_from_checkpoint: Optional[str] = None):
        super().__init__(config_path, resume_from_checkpoint)

    def _setup_model(self):
        """设置Dense模型"""
        from model_dense.causal_lm import create_dense_model
        from model_dense.config import RNAGenDenseConfig
        from model.lineage_tokenizer import get_lineage_rna_tokenizer

        model_config_dict = self.config.get('model_config', {})
        data_config = self.config.get('data_config', {})

        use_direction_tokens = data_config.get('use_direction_tokens', True)
        self.tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)

        model_config_dict['vocab_size'] = self.tokenizer.vocab_size
        logger.info(f"自动设置vocab_size={self.tokenizer.vocab_size}")

        # 创建Dense模型配置（移除MoE相关参数）
        dense_config_params = {
            k: v for k, v in model_config_dict.items()
            if k not in ['num_experts', 'num_experts_per_tok', 'expert_capacity_factor',
                        'router_aux_loss_coef', 'moe_implementation', 'moe_world_size']
        }

        model_config = RNAGenDenseConfig(**dense_config_params)
        self.model = create_dense_model(model_config)
        logger.info(f"Dense模型创建完成，设备: cuda:{self.local_rank}")

        # 设置数据类型
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

        if self.world_size > 1 and self.global_rank == 0:
            logger.info(f"手动梯度同步模式（无DDP）: {self.world_size} GPUs")

        # 禁用输出token mask（训练时不需要）
        self.model.output_token_mask = None
        logger.info("训练模式：禁用output_token_mask（避免loss=inf）")

    def _sync_gradients(self):
        """手动同步所有参数的梯度（替代DDP的自动同步）"""
        if self.world_size <= 1:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def main():
    DenseTrainer.main(
        description='Dense模型训练',
        default_config='configs/dense_training/base_dense.yaml',
    )


if __name__ == '__main__':
    main()
