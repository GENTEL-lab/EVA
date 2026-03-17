#!/usr/bin/env python3
"""
Mid-Training: 从预训练checkpoint继续训练

与预训练的主要区别：
1. 开启物种层次分类学前缀 (use_lineage_prefix: true)
2. EOS token增加loss权重，帮助模型学会正确断句
3. 从预训练checkpoint继续训练（只加载模型权重，不加载optimizer/scheduler）

支持双机16卡分布式训练，使用MegaBlocks + FSDP2架构
"""

import os
import sys
import re
import time

# 抑制NCCL冗余日志输出（必须在import torch之前设置）
os.environ.setdefault('NCCL_DEBUG', 'WARN')

import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

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


class MidTrainingTrainer(BaseTrainer):
    """Mid-Training 训练器 - 从预训练checkpoint继续训练"""

    @property
    def stage_name(self) -> str:
        return 'mid_training'

    @property
    def default_log_dir(self) -> str:
        return '/rna-multiverse/results/mid_training/logs'

    @property
    def default_wandb_project(self) -> str:
        return 'rna-mid-training'

    @property
    def uses_moe(self) -> bool:
        return True

    def setup(self):
        """覆写 setup：调用 super().setup() 后执行 _load_pretrain_checkpoint()"""
        # 注意：不传 resume_from_checkpoint 给 super().__init__，
        # 因为 mid-training 的 checkpoint 加载逻辑不同（只加载模型权重）
        self._setup_distributed()
        self._setup_logging()
        self._set_seed()
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_dropout_schedule()
        self._setup_memory_manager()
        self._calculate_model_flops()

        # Mid-Training核心：从预训练checkpoint加载模型权重
        self._load_pretrain_checkpoint()

        self.setup_complete = True

        if self.logger_manager and self.global_rank == 0:
            self.logger_manager.log_training_start(self.config)
            self._log_checkpoint_offset_to_wandb()

        logger.info("Mid-Training训练环境设置完成")

    def _load_pretrain_checkpoint(self):
        """
        从预训练checkpoint加载模型权重

        与普通resume的区别：
        - 只加载模型权重，不加载optimizer/scheduler
        - 重置global_step为0（重新开始训练）
        """
        training_config = self.config.get('training_config', {})
        pretrain_checkpoint = training_config.get('resume_from_pretrain', None)

        if not pretrain_checkpoint:
            if self.global_rank == 0:
                logger.warning("未指定预训练checkpoint (resume_from_pretrain)")
                logger.warning("   将从随机初始化开始训练")
            return

        checkpoint_dir = Path(pretrain_checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"预训练checkpoint不存在: {checkpoint_dir}")

        logger.info(f"[Rank {self.global_rank}] 从预训练checkpoint加载模型权重: {checkpoint_dir}")

        if dist.is_initialized():
            dist.barrier()

        try:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": {},
                "lr_scheduler": {},
                "metadata": {}
            }

            load_start_time = time.time()
            dcp.load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_dir),
            )
            load_time = time.time() - load_start_time
            logger.info(f"[Rank {self.global_rank}] 预训练checkpoint加载成功，耗时: {load_time:.2f}秒")

            self.model.load_state_dict(state_dict["model"])

            metadata = state_dict.get("metadata", {})
            pretrain_step = metadata.get('global_step', 0)

            match = re.search(r'checkpoint-(\d+)', checkpoint_dir.name)
            if match:
                self.checkpoint_step_offset = int(match.group(1))
            elif pretrain_step > 0:
                self.checkpoint_step_offset = pretrain_step
            else:
                self.checkpoint_step_offset = 0

            if self.global_rank == 0:
                logger.info(f"预训练模型权重加载完成:")
                logger.info(f"   - 预训练checkpoint: {checkpoint_dir}")
                logger.info(f"   - 预训练步数: {pretrain_step}")
                logger.info(f"   - Mid-Training起始step: 0 (offset={self.checkpoint_step_offset})")
                logger.info(f"   - Optimizer/Scheduler已重置")

        except Exception as e:
            logger.error(f"[Rank {self.global_rank}] 预训练checkpoint加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if dist.is_initialized():
                dist.barrier()

    def _log_checkpoint_offset_to_wandb(self):
        """记录checkpoint step offset到wandb"""
        if not self.logger_manager:
            return
        wandb_logger = getattr(self.logger_manager, 'wandb_logger', None)
        if wandb_logger and wandb_logger.enable:
            try:
                training_config = self.config.get('training_config', {})
                mid_training_info = {
                    'mid_training/checkpoint_step_offset': self.checkpoint_step_offset,
                    'mid_training/resume_from_pretrain': training_config.get('resume_from_pretrain', 'N/A'),
                    'mid_training/starting_step': self.checkpoint_step_offset,
                }
                wandb_logger.wandb.config.update(mid_training_info, allow_val_change=True)
            except Exception as e:
                logger.warning(f"记录checkpoint offset到WandB失败: {e}")

    def _setup_model(self):
        """设置MoE模型（与Stage1相同，但增加EOS loss权重日志）"""
        from model.causal_lm import create_rnagen_model
        from model.lineage_tokenizer import get_lineage_rna_tokenizer
        from utils.device import create_device_manager, set_device_manager

        model_config_dict = self.config.get('model_config', {})
        distributed_config = self.config.get('distributed_config', {})
        data_config = self.config.get('data_config', {})

        use_direction_tokens = data_config.get('use_direction_tokens', True)
        self.tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)

        model_config_dict['vocab_size'] = self.tokenizer.vocab_size
        logger.info(f"自动设置vocab_size={self.tokenizer.vocab_size}")

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

        # Mid-Training特性：显示EOS loss权重配置
        eos_loss_weight = model_config_dict.get('eos_loss_weight', 1.0)
        if self.global_rank == 0 and eos_loss_weight != 1.0:
            logger.info(f"Mid-Training EOS Loss权重: {eos_loss_weight}")

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

        model_to_set = self.model
        if model_config_dict.get('moe_implementation') == "megablocks" and model_config_dict.get('moe_world_size', 1) > 1:
            model_to_set.output_token_mask = None
            logger.info("MegaBlocks模式：禁用output_token_mask")
        else:
            output_token_ids = self.tokenizer.get_stage1_output_token_ids()
            output_token_mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool, device=f'cuda:{self.local_rank}')
            output_token_mask[output_token_ids] = True
            model_to_set.output_token_mask = output_token_mask

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

    def _extra_dataset_kwargs(self) -> Dict[str, Any]:
        """Mid-Training 额外的 dataset 参数"""
        data_config = self.config.get('data_config', {})
        return {
            'pretrain_ratio': data_config.get('pretrain_ratio', 0.0),
        }

    def _extra_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Mid-Training 额外指标"""
        extra = {
            'train/mid_training_step': self.global_step,
        }
        model_cfg = self.config.get('model_config', {})
        eos_loss_weight = model_cfg.get('eos_loss_weight', 1.0)
        if eos_loss_weight != 1.0:
            extra['mid_training/eos_loss_weight'] = eos_loss_weight
        return extra


def main():
    MidTrainingTrainer.main(
        description='Mid-Training: 从预训练checkpoint继续训练',
        default_config='configs/mid_training/base_mid_training.yaml',
    )


if __name__ == '__main__':
    main()
