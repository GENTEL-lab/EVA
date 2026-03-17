"""
BaseTrainer 抽象基类 - 模板方法模式

所有三个 Trainer（Stage1、MidTrain、Dense）共享的逻辑在此实现。
子类只需实现 _setup_model() 和 _sync_gradients() 等抽象方法。
"""

import os
import sys
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .constants import (
    DEFAULT_NCCL_TIMEOUT_MINUTES, DEFAULT_SEED,
    DEFAULT_DROPOUT_WARMUP_STEPS, DEFAULT_DROPOUT_RAMP_STEPS,
    DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY,
    DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2, DEFAULT_ADAM_EPSILON,
    DEFAULT_CLEANUP_FREQUENCY, DEFAULT_GC_FREQUENCY,
    MEMORY_MONITOR_BATCH_INTERVAL, MODEL_INPUT_KEYS,
    DEFAULT_MAX_EPOCHS, DEFAULT_WARMUP_RATIO, DEFAULT_MIN_LR_RATIO,
    DEFAULT_LOGGING_STEPS, DEFAULT_NUM_CHECKPOINTS,
    DEFAULT_BATCH_SIZE, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_CACHE_SIZE,
    DEFAULT_DATALOADER_NUM_WORKERS,
)
from .checkpoint_utils import save_dcp_checkpoint, save_auxiliary_files
from .flops_calculator import calculate_model_flops

logger = logging.getLogger(__name__)

# PLACEHOLDER_BASE_TRAINER_BODY


class BaseTrainer(ABC):
    """
    训练器抽象基类

    子类必须实现:
        - stage_name (property): 阶段名称，如 'lineage_stage1'
        - default_log_dir (property): 默认日志目录
        - default_wandb_project (property): 默认 wandb 项目名
        - uses_moe (property): 是否使用 MoE 架构
        - _setup_model(): 创建模型
        - _sync_gradients(): 梯度同步策略
    """

    # ---- 子类必须实现的抽象属性/方法 ----

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """阶段名称，用于 checkpoint metadata 和日志"""
        ...

    @property
    @abstractmethod
    def default_log_dir(self) -> str:
        ...

    @property
    @abstractmethod
    def default_wandb_project(self) -> str:
        ...

    @property
    @abstractmethod
    def uses_moe(self) -> bool:
        """是否使用 MoE 架构（影响优化器 foreach 和 FLOPs 计算）"""
        ...

    @abstractmethod
    def _setup_model(self):
        """创建并初始化模型，设置 self.model 和 self.tokenizer"""
        ...

    @abstractmethod
    def _sync_gradients(self):
        """梯度同步策略（MoE 用 data_parallel_group，Dense 用全局 all_reduce）"""
        ...

    # ---- 初始化 ----

    def __init__(self, config_path: str, resume_from_checkpoint: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.resume_from_checkpoint = resume_from_checkpoint

        # 分布式配置
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0
        self.node_rank = 0
        self.local_world_size = 1

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.setup_complete = False
        self.current_epoch = 0
        self.checkpoint_step_offset = 0

# PLACEHOLDER_INIT_CONTINUED

        # FLOPs统计
        self.total_flops = 0
        self.model_flops_per_step = None
        self.train_start_time = None

        # 模型组件
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tokenizer = None
        self.train_dataloader = None
        self.train_sampler = None

        # 内存管理
        self.memory_manager = None

        # Dropout调度
        self.current_dropout_values = {}
        self.dropout_warmup_steps = 0
        self.dropout_ramp_steps = 0
        self.dropout_schedule = "linear"
        self.target_resid_dropout = 0.0
        self.target_hidden_dropout = 0.0

        training_cfg = self.config.get('training_config', {})
        try:
            self.gradient_accumulation_steps = int(training_cfg.get('gradient_accumulation_steps', 1))
        except (TypeError, ValueError):
            self.gradient_accumulation_steps = 1
        self.gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)

    # ---- 共享方法 ----

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup(self):
        """设置训练环境（子类可覆写以添加额外步骤）"""
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

        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)

        self.setup_complete = True

        if self.logger_manager and self.global_rank == 0:
            self.logger_manager.log_training_start(self.config)

        logger.info(f"{self.stage_name}训练环境设置完成")

# PLACEHOLDER_DISTRIBUTED

    def _setup_distributed(self):
        """设置分布式训练环境"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.global_rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.node_rank = int(os.environ.get('NODE_RANK', 0))
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE',
                                                       torch.cuda.device_count()))

        if self.world_size > 1:
            nccl_timeout_minutes = int(os.environ.get(
                'NCCL_TIMEOUT_MINUTES', DEFAULT_NCCL_TIMEOUT_MINUTES))
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.global_rank,
                timeout=timedelta(minutes=nccl_timeout_minutes)
            )
            torch.cuda.set_device(self.local_rank)

            test_tensor = torch.ones(1, device=f'cuda:{self.local_rank}')
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            if test_tensor.item() == self.world_size and self.global_rank == 0:
                logger.info(f"分布式环境初始化成功: {self.world_size}个进程")

        self.device = torch.device(f'cuda:{self.local_rank}')

    def _setup_logging(self):
        from utils.logging import create_logger
        logging_config = self.config.get('logging_config', {})
        experiment_name = f"{self.stage_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir = logging_config.get('log_dir', self.default_log_dir)

        self.logger_manager = create_logger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            config=self.config,
            local_rank=self.local_rank,
            enable_wandb=logging_config.get('enable_wandb', False) and self.global_rank == 0,
            wandb_project=logging_config.get('wandb_project', self.default_wandb_project),
            wandb_run_name=logging_config.get('wandb_run_name', experiment_name)
        )

    def _set_seed(self):
        seed = self.config.get('training_config', {}).get('seed', DEFAULT_SEED)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# PLACEHOLDER_DATASETS

    def _setup_datasets(self):
        """设置数据集"""
        from model.lineage_tokenizer import get_lineage_rna_tokenizer
        from data.lineage_dataset import create_lineage_dataset
        from data.rna_collator import create_rna_data_collator

        data_config = self.config.get('data_config', {})
        training_config = self.config.get('training_config', {})

        use_lineage_prefix = data_config.get('use_lineage_prefix', True)
        use_rna_type_prefix = data_config.get('use_rna_type_prefix', True)
        use_chunked = data_config.get('use_chunked', True)
        cache_size = data_config.get('cache_size', DEFAULT_CACHE_SIZE)
        force_rebuild_index = data_config.get('force_rebuild_index', False)
        use_direction_tokens = data_config.get('use_direction_tokens', True)
        enable_reverse_augmentation = data_config.get('enable_reverse_augmentation', True)

        mode = data_config.get('mode', 'generation')
        glm_probability = data_config.get('glm_probability', 0.333)

        span_config = None
        if mode in ['mixed', 'completion']:
            from data.lineage_dataset import SpanConfig
            span_config_dict = data_config.get('span_config', {})
            span_config = SpanConfig(
                max_coverage_ratios=span_config_dict.get('max_coverage_ratios', [0.15, 0.25, 0.5, 0.8]),
                coverage_probs=span_config_dict.get('coverage_probs', [0.28, 0.30, 0.28, 0.14]),
                span_distributions=span_config_dict.get('span_distributions', [(10, 5), (20, 10), (50, 20)]),
                max_num_spans=span_config_dict.get('max_num_spans', 10),
                allow_overlap=span_config_dict.get('allow_overlap', False)
            )

        # 构建 dataset kwargs（子类可通过覆写 _extra_dataset_kwargs() 添加额外参数）
        dataset_kwargs = dict(
            data_file=data_config['train_file'],
            tokenizer=self.tokenizer,
            lineage_file=data_config['lineage_file'],
            mode=mode,
            span_config=span_config,
            glm_probability=glm_probability,
            max_seq_length=data_config.get('max_seq_length', DEFAULT_MAX_SEQ_LENGTH),
            max_samples=data_config.get('max_samples'),
            use_chunked=use_chunked,
            cache_size=cache_size,
            force_rebuild_index=force_rebuild_index,
            use_direction_tokens=use_direction_tokens,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
        )
        dataset_kwargs.update(self._extra_dataset_kwargs())

        self.train_dataset = create_lineage_dataset(**dataset_kwargs)

        data_collator = create_rna_data_collator(
            tokenizer=self.tokenizer,
            max_length=data_config.get('max_seq_length', DEFAULT_MAX_SEQ_LENGTH),
            device="cpu"
        )

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                seed=DEFAULT_SEED
            )
        else:
            self.train_sampler = None

        num_workers = training_config.get('dataloader_num_workers', DEFAULT_DATALOADER_NUM_WORKERS)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=training_config.get('per_device_train_batch_size', DEFAULT_BATCH_SIZE),
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=training_config.get('dataloader_pin_memory', True),
            drop_last=training_config.get('dataloader_drop_last', True),
        )

        if self.global_rank == 0:
            logger.info(f"{self.stage_name}数据集准备完成")
            logger.info(f"  训练样本数: {len(self.train_dataset)}")
            logger.info(f"  批次数: {len(self.train_dataloader)}")
            logger.info(f"  训练模式: {mode}")

    def _extra_dataset_kwargs(self) -> Dict[str, Any]:
        """子类可覆写以添加额外的 dataset 参数（如 pretrain_ratio）"""
        return {}

# PLACEHOLDER_OPTIMIZER

    def _setup_optimizer(self):
        training_config = self.config.get('training_config', {})
        learning_rate = float(training_config.get('learning_rate', DEFAULT_LEARNING_RATE))
        weight_decay = float(training_config.get('weight_decay', DEFAULT_WEIGHT_DECAY))
        adam_beta1 = float(training_config.get('adam_beta1', DEFAULT_ADAM_BETA1))
        adam_beta2 = float(training_config.get('adam_beta2', DEFAULT_ADAM_BETA2))
        adam_epsilon = float(training_config.get('adam_epsilon', DEFAULT_ADAM_EPSILON))

        optimizer_kwargs = dict(
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )

        # MoE 模式需要禁用 foreach（DTensor 不兼容）
        if self.uses_moe:
            model_config = self.config.get('model_config', {})
            if model_config.get('moe_implementation') == "megablocks" and model_config.get('moe_world_size', 1) > 1:
                optimizer_kwargs['foreach'] = False
                logger.info("MegaBlocks模式：禁用AdamW的foreach操作")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        if self.global_rank == 0:
            logger.info(f"{self.stage_name}优化器设置完成 - 学习率: {learning_rate}")

    def _setup_scheduler(self):
        training_config = self.config.get('training_config', {})
        max_epochs = training_config.get('max_epochs', DEFAULT_MAX_EPOCHS)
        warmup_ratio = training_config.get('warmup_ratio', DEFAULT_WARMUP_RATIO)
        min_lr_ratio = training_config.get('min_lr_ratio', DEFAULT_MIN_LR_RATIO)

        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        num_training_steps = (len(self.train_dataloader) // gradient_accumulation_steps) * max_epochs

        warmup_steps_config = training_config.get('warmup_steps', None)
        if warmup_steps_config is not None and int(warmup_steps_config) > 0:
            num_warmup_steps = int(warmup_steps_config)
        else:
            num_warmup_steps = int(num_training_steps * warmup_ratio)

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        if self.global_rank == 0:
            logger.info(f"学习率调度器设置完成")
            logger.info(f"  总步数: {num_training_steps}")
            logger.info(f"  Warmup步数: {num_warmup_steps}")

# PLACEHOLDER_DROPOUT

    def _setup_dropout_schedule(self):
        model_cfg = self.config.get('model_config', {})
        self.dropout_warmup_steps = model_cfg.get('dropout_warmup_steps', DEFAULT_DROPOUT_WARMUP_STEPS)
        self.dropout_ramp_steps = model_cfg.get('dropout_ramp_steps', DEFAULT_DROPOUT_RAMP_STEPS)
        self.dropout_schedule = model_cfg.get('dropout_schedule', 'linear')
        self.target_resid_dropout = model_cfg.get('resid_dropout', 0.0)
        self.target_hidden_dropout = model_cfg.get('hidden_dropout', 0.0)

    def _ramp_dropout_value(self, step: int, target: float) -> float:
        if step < self.dropout_warmup_steps:
            return 0.0
        if step >= self.dropout_warmup_steps + self.dropout_ramp_steps:
            return target
        progress = (step - self.dropout_warmup_steps) / max(1, self.dropout_ramp_steps)
        if self.dropout_schedule == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))
        return target * progress

    def _update_dropout_rates(self, step: int):
        p_resid = self._ramp_dropout_value(step, self.target_resid_dropout)
        p_hidden = self._ramp_dropout_value(step, self.target_hidden_dropout)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        for layer in model.model.layers:
            if hasattr(layer, 'drop_resid'):
                layer.drop_resid.p = p_resid
            if hasattr(layer, 'drop_mlp'):
                layer.drop_mlp.p = p_hidden

        self.current_dropout_values = {'resid': p_resid, 'hidden': p_hidden}

    def _setup_memory_manager(self):
        from utils.memory import create_memory_manager
        memory_config = self.config.get('memory_config', {})
        if memory_config.get('enable_monitoring', True):
            self.memory_manager = create_memory_manager(
                device=torch.device(f'cuda:{self.local_rank}'),
                cleanup_frequency=memory_config.get('cleanup_frequency', DEFAULT_CLEANUP_FREQUENCY),
                gc_frequency=memory_config.get('gc_frequency', DEFAULT_GC_FREQUENCY),
                enable_monitoring=memory_config.get('enable_monitoring', True)
            )

    def _calculate_model_flops(self):
        try:
            self.model_flops_per_step = calculate_model_flops(
                model_config=self.config.get('model_config', {}),
                training_config=self.config.get('training_config', {}),
                data_config=self.config.get('data_config', {}),
                world_size=self.world_size,
                distributed_config=self.config.get('distributed_config', {}),
                is_moe=self.uses_moe,
            )
            if self.global_rank == 0:
                logger.info(f"模型FLOPs: {self.model_flops_per_step/1e12:.2f} TFLOPs/optimizer_step")
        except Exception as e:
            logger.warning(f"FLOPs计算失败: {e}")
            self.model_flops_per_step = 0

# PLACEHOLDER_CHECKPOINT

    def save_checkpoint(self, epoch: int, is_final: bool = False, step: int = None):
        """使用DCP保存checkpoint"""
        if self.memory_manager:
            self.memory_manager.step(f"pre_checkpoint_step_{step}")
        torch.cuda.empty_cache()

        training_config = self.config.get('training_config', {})
        output_dir = Path(training_config['output_dir'])

        if is_final:
            checkpoint_dir = output_dir / 'final'
        elif step is not None:
            actual_step = step + self.checkpoint_step_offset
            checkpoint_dir = output_dir / f'checkpoint-{actual_step}'
        else:
            checkpoint_dir = output_dir / f'epoch-{epoch}'

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            torch.cuda.empty_cache()
            metadata = {
                'global_step': self.global_step,
                'checkpoint_step_offset': self.checkpoint_step_offset,
                'epoch': epoch,
                'stage': self.stage_name,
                'model_config': self.config.get('model_config', {}),
                'best_eval_loss': self.best_eval_loss,
                'world_size': self.world_size,
                'current_loss': getattr(self, 'current_loss', 0.0),
                'total_flops': self.total_flops,
            }

            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "metadata": metadata
            }
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[Rank {self.global_rank}] state_dict构建失败: {e}")
            torch.cuda.empty_cache()
            raise

        save_dcp_checkpoint(
            state_dict=state_dict,
            checkpoint_dir=checkpoint_dir,
            global_rank=self.global_rank,
            world_size=self.world_size,
            device=self.device,
        )

        save_auxiliary_files(
            checkpoint_dir=checkpoint_dir,
            tokenizer=self.tokenizer,
            model_config=self.config.get('model_config', {}),
            full_config=self.config,
            global_rank=self.global_rank,
        )

        logger.info(f"[Rank {self.global_rank}] Checkpoint保存流程完成: {checkpoint_dir}")

        if self.memory_manager:
            self.memory_manager.step(f"post_checkpoint_step_{step}")
        torch.cuda.empty_cache()

    def load_checkpoint(self, checkpoint_path: str):
        """从DCP格式的checkpoint恢复训练"""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

        logger.info(f"[Rank {self.global_rank}] 开始从checkpoint恢复: {checkpoint_dir}")

        if dist.is_initialized():
            dist.barrier()

# PLACEHOLDER_LOAD_CHECKPOINT_BODY

        try:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "metadata": {}
            }

            load_start_time = time.time()
            dcp.load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_dir),
            )
            load_time = time.time() - load_start_time
            logger.info(f"[Rank {self.global_rank}] DCP checkpoint加载成功，耗时: {load_time:.2f}秒")

            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

            metadata = state_dict["metadata"]
            self.global_step = metadata.get('global_step', 0)
            self.current_epoch = metadata.get('epoch', 0)
            self.total_flops = metadata.get('total_flops', 0)
            self.best_eval_loss = metadata.get('best_eval_loss', float('inf'))

            saved_offset = metadata.get('checkpoint_step_offset', 0)
            import re
            match = re.search(r'checkpoint-(\d+)', checkpoint_dir.name)

            if saved_offset > 0:
                self.checkpoint_step_offset = saved_offset
            elif match:
                checkpoint_step = int(match.group(1))
                if self.global_step == 0 and checkpoint_step > 0:
                    self.checkpoint_step_offset = checkpoint_step
                else:
                    self.checkpoint_step_offset = 0
            else:
                self.checkpoint_step_offset = 0

            if self.global_rank == 0:
                logger.info(f"Checkpoint恢复完成:")
                logger.info(f"   - 恢复步数: {self.global_step}")
                logger.info(f"   - Step偏移量: {self.checkpoint_step_offset}")
                logger.info(f"   - 恢复epoch: {self.current_epoch}")
                logger.info(f"   - 恢复FLOPs: {self.total_flops:.2e}")

        except Exception as e:
            logger.error(f"[Rank {self.global_rank}] Checkpoint加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if dist.is_initialized():
                dist.barrier()

# PLACEHOLDER_METRICS

    def _collect_metrics(self, loss: torch.Tensor, ar_loss: torch.Tensor,
                         lr: float, aux_loss: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """收集训练指标"""
        # MoE 模式：跨 data_parallel 组平均 loss
        if dist.is_initialized() and hasattr(self, 'device_manager'):
            loss_tensor = loss.clone().detach()
            ar_loss_tensor = ar_loss.clone().detach()
            dp_group = self.device_manager.get_data_parallel_group()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=dp_group)
            dist.all_reduce(ar_loss_tensor, op=dist.ReduceOp.AVG, group=dp_group)
            loss_for_log = loss_tensor.item()
            ar_loss_for_log = ar_loss_tensor.item()
        else:
            loss_for_log = loss.item()
            ar_loss_for_log = ar_loss.item()

        perplexity = torch.exp(torch.tensor(ar_loss_for_log)).item()

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
        else:
            memory_allocated = memory_reserved = 0

        display_step = self.global_step + self.checkpoint_step_offset

        metrics = {
            'train/loss': loss_for_log,
            'train/ar_loss': ar_loss_for_log,
            'train/perplexity': perplexity,
            'train/learning_rate': lr,
            'train/epoch': self.current_epoch,
            'train/global_step': display_step,
            'system/memory_allocated_gb': memory_allocated,
            'system/memory_reserved_gb': memory_reserved,
            'flops/step': self.model_flops_per_step or 0,
            'flops/cumulative': self.total_flops,
            'flops/cumulative_tera': self.total_flops / 1e12,
        }

        if aux_loss is not None:
            aux_loss_val = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            metrics['train/aux_loss'] = aux_loss_val
            if loss_for_log > 0:
                metrics['train/aux_loss_ratio'] = aux_loss_val / loss_for_log

        if self.current_dropout_values:
            metrics['dropout/resid'] = self.current_dropout_values.get('resid', 0.0)
            metrics['dropout/hidden'] = self.current_dropout_values.get('hidden', 0.0)

        # 子类可通过覆写 _extra_metrics() 添加额外指标
        metrics.update(self._extra_metrics(metrics))

        self.current_loss = loss_for_log
        return metrics

    def _extra_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """子类可覆写以添加额外指标"""
        return {}

    def _log_step(self, metrics: Dict[str, Any]):
        display_step = self.global_step + self.checkpoint_step_offset

        if self.logger_manager and self.global_rank == 0:
            self.logger_manager.log_step(display_step, metrics)

        if self.global_rank == 0:
            log_msg = (
                f"Step {display_step}: "
                f"Loss={metrics['train/loss']:.4f}, "
                f"AR_Loss={metrics['train/ar_loss']:.4f}, "
                f"PPL={metrics['train/perplexity']:.2f}, "
                f"LR={metrics['train/learning_rate']:.6f}"
            )
            if 'train/aux_loss' in metrics:
                log_msg += f", Aux_Loss={metrics['train/aux_loss']:.6f}"
            if 'dropout/resid' in metrics and (metrics['dropout/resid'] > 0 or metrics['dropout/hidden'] > 0):
                log_msg += f", Dropout(r={metrics['dropout/resid']:.4f}, h={metrics['dropout/hidden']:.4f})"
            logger.info(log_msg)

# PLACEHOLDER_TRAIN

    def train(self):
        """执行训练 - 支持基于FLOPs和运行时长的停止条件"""
        if not self.setup_complete:
            self.setup()

        self.train_start_time = time.time()
        training_config = self.config.get('training_config', {})

        target_flops = training_config.get('target_flops', None)
        if target_flops is not None:
            target_flops = float(target_flops)

        max_wall_time_hours = training_config.get('max_wall_time_hours', None)
        if max_wall_time_hours is not None:
            max_wall_time_hours = float(max_wall_time_hours)

        max_steps = training_config.get('max_steps', None)
        if max_steps is not None:
            max_steps = int(max_steps)

        max_epochs = training_config.get('max_epochs', None)
        if max_epochs is not None:
            max_epochs = int(max_epochs)

        save_steps = training_config.get('save_steps', None)
        if save_steps is not None:
            save_steps = int(save_steps)

        num_checkpoints = int(training_config.get('num_checkpoints', DEFAULT_NUM_CHECKPOINTS))

        self.save_steps = save_steps
        self.checkpoint_flops_milestones = []

        if save_steps and save_steps > 0:
            pass
        elif target_flops and num_checkpoints > 0:
            checkpoint_interval_flops = target_flops / num_checkpoints
            self.checkpoint_flops_milestones = [
                checkpoint_interval_flops * (i + 1) for i in range(num_checkpoints)
            ]

        if self.global_rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始{self.stage_name}训练")
            logger.info(f"批次大小: {training_config.get('per_device_train_batch_size', DEFAULT_BATCH_SIZE)}")
            logger.info(f"梯度累积步数: {self.gradient_accumulation_steps}")
            logger.info(f"World Size: {self.world_size}")
            logger.info(f"\n停止条件:")
            if target_flops:
                logger.info(f"  - 目标FLOPs: {target_flops:.2e}")
            if max_wall_time_hours:
                logger.info(f"  - 最大运行时间: {max_wall_time_hours} 小时")
            if max_steps:
                logger.info(f"  - 最大步数: {max_steps}")
            if max_epochs:
                logger.info(f"  - 最大epochs: {max_epochs}")
            logger.info(f"{'='*60}\n")

        epoch = 0
        while True:
            self.current_epoch = epoch
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_loss, should_stop = self.train_epoch(
                epoch,
                target_flops=target_flops,
                max_wall_time_hours=max_wall_time_hours,
                max_steps=max_steps
            )

            if self.global_rank == 0:
                elapsed_hours = (time.time() - self.train_start_time) / 3600
                logger.info(
                    f"Epoch {epoch + 1} 完成 | "
                    f"损失: {epoch_loss:.4f} | "
                    f"累计FLOPs: {self.total_flops:.2e} | "
                    f"已运行: {elapsed_hours:.2f}小时"
                )

            if should_stop:
                break

            if max_epochs and epoch + 1 >= max_epochs:
                if self.global_rank == 0:
                    logger.info(f"达到max_epochs={max_epochs}，停止训练")
                break

            epoch += 1

        self.save_checkpoint(self.current_epoch + 1, is_final=True)
        self._cleanup_and_exit(training_config)

# PLACEHOLDER_TRAIN_EPOCH

    def train_epoch(self, epoch: int, target_flops: Optional[float] = None,
                    max_wall_time_hours: Optional[float] = None,
                    max_steps: Optional[int] = None):
        self.model.train()
        total_loss = 0
        num_batches = 0
        should_stop = False
        grad_accum_steps = self.gradient_accumulation_steps

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"{self.stage_name} Epoch {epoch + 1}",
            disable=self.global_rank != 0
        )

        for batch_idx, batch in enumerate(progress_bar):
            self._update_dropout_rates(self.global_step)

            batch = {k: v.to(f'cuda:{self.local_rank}') if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            model_inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}

            outputs = self.model(**model_inputs)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            ar_loss = outputs.get('ar_loss', loss) if isinstance(outputs, dict) else loss
            aux_loss = outputs.get('aux_loss', None) if isinstance(outputs, dict) else None

            loss_to_backprop = loss / grad_accum_steps
            loss_to_backprop.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                self._sync_gradients()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.model_flops_per_step:
                    self.total_flops += self.model_flops_per_step

                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = self._collect_metrics(loss, ar_loss, current_lr, aux_loss)

                training_config = self.config.get('training_config', {})
                logging_steps = training_config.get('logging_steps', DEFAULT_LOGGING_STEPS)
                if self.global_step % logging_steps == 0:
                    self._log_step(metrics)

                # Checkpoint保存策略
                checkpoint_saved = False
                if hasattr(self, 'save_steps') and self.save_steps and self.save_steps > 0:
                    if self.global_step > 0 and self.global_step % self.save_steps == 0:
                        if self.global_rank == 0:
                            logger.info(f"达到步数间隔 {self.save_steps}，保存checkpoint")
                        self.save_checkpoint(self.current_epoch + 1, step=self.global_step)
                        checkpoint_saved = True

                if not checkpoint_saved and hasattr(self, 'checkpoint_flops_milestones') and self.checkpoint_flops_milestones:
                    for milestone_flops in self.checkpoint_flops_milestones:
                        prev_flops = self.total_flops - self.model_flops_per_step
                        if prev_flops < milestone_flops <= self.total_flops:
                            if self.global_rank == 0:
                                logger.info(f"达到FLOPs里程碑 {milestone_flops:.2e}，保存checkpoint")
                            self.save_checkpoint(self.current_epoch + 1, step=self.global_step)
                            break

                # 停止条件检查
                if target_flops and self.total_flops >= target_flops:
                    if self.global_rank == 0:
                        logger.info(f"达到target_flops={target_flops:.2e}，停止训练")
                    should_stop = True
                    break

                if max_wall_time_hours and self.train_start_time:
                    elapsed_hours = (time.time() - self.train_start_time) / 3600
                    if elapsed_hours >= max_wall_time_hours:
                        if self.global_rank == 0:
                            logger.info(f"达到max_wall_time={max_wall_time_hours}小时，停止训练")
                        should_stop = True
                        break

                if max_steps and self.global_step >= max_steps:
                    if self.global_rank == 0:
                        logger.info(f"达到max_steps={max_steps}，停止训练")
                    should_stop = True
                    break

            total_loss += loss.item()
            num_batches += 1

            if self.global_rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                display_step = self.global_step + self.checkpoint_step_offset
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}',
                    'FLOPs': f'{self.total_flops:.2e}',
                    'step': display_step
                })

            if self.memory_manager and batch_idx % MEMORY_MONITOR_BATCH_INTERVAL == 0:
                self.memory_manager.step(f"batch_{batch_idx}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss, should_stop

    def _cleanup_and_exit(self, training_config):
        try:
            if self.global_rank == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"{self.stage_name}训练完成！")
                logger.info(f"最终模型: {training_config['output_dir']}/final/")
                logger.info(f"总步数: {self.global_step:,}")
                logger.info(f"总FLOPs: {self.total_flops/1e12:.2f} TFLOPs")
                logger.info(f"{'='*60}\n")

            if self.world_size > 1:
                dist.barrier()

            if self.logger_manager and self.global_rank == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                except Exception as e:
                    logger.warning(f"关闭WandB时出错: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if self.world_size > 1:
                dist.barrier()

            if self.world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()

        except Exception as e:
            logger.error(f"清理过程出错: {e}")
        finally:
            sys.exit(0)

    @classmethod
    def main(cls, description: str, default_config: str, supports_resume: bool = False):
        """通用的 CLI 入口"""
        import argparse
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--config', type=str, default=default_config, help='配置文件路径')
        if supports_resume:
            parser.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练的路径')
        args = parser.parse_args()

        trainer = None
        try:
            kwargs = {'config_path': args.config}
            if supports_resume:
                kwargs['resume_from_checkpoint'] = args.resume
            trainer = cls(**kwargs)
            trainer.train()
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            if trainer:
                trainer._cleanup_and_exit(trainer.config.get('training_config', {}))
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            if trainer:
                trainer._cleanup_and_exit(trainer.config.get('training_config', {}))
            raise
