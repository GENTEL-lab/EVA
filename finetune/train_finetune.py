#!/usr/bin/env python3
"""
Finetune Training: Full-parameter finetuning from a pretrained checkpoint (single-GPU).

Finetuning for specific RNA types (e.g. aptamer, gRNA).
Key features:
1. Custom RNA type prefix (using existing tokens such as <rna_sRNA>)
2. Optional species lineage prefix
3. Increased EOS token loss weight for proper sequence termination
4. Loads only model weights from pretrained checkpoint (optimizer/scheduler re-initialized)
5. Supports .pt and DCP checkpoint formats

Single-GPU finetuning, no distributed training.
"""

import os
import sys
import re
import traceback

# Single-GPU mode, NCCL not needed

import yaml
import argparse
import logging
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.causal_lm import create_rnagen_model
from model.config import RNAGenConfig
from model.lineage_tokenizer import get_lineage_rna_tokenizer
from data.lineage_dataset import create_lineage_dataset, SpanConfig
from data.rna_collator import create_rna_data_collator
from utils.memory import MemoryManager, create_memory_manager
from utils.logging import create_logger

# Configure basic logging to ensure logger.info outputs to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfigure even if already configured
)
logger = logging.getLogger(__name__)


class FinetuneTrainer:
    """Finetune Trainer - Full-parameter finetuning from pretrained checkpoint"""

    def __init__(self, config_path: str):
        """Initialize trainer"""
        # --- Config ---
        self.config_path = config_path
        self.config = self._load_config()
        self.training_config = self.config.get('training_config', {})
        self.model_config = self.config.get('model_config', {})
        self.data_config = self.config.get('data_config', {})
        self.logging_config = self.config.get('logging_config', {})
        self.memory_config = self.config.get('memory_config', {})

        # --- Distributed config ---
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0
        self.node_rank = 0
        self.local_world_size = 1

        # --- Training state ---
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.setup_complete = False
        self.current_epoch = 0
        self.checkpoint_step_offset = 0  # Step offset when resuming from checkpoint

        # --- FLOPs tracking ---
        self.total_flops = 0
        self.model_flops_per_step = None
        self.train_start_time = None  # Training start time (for max_wall_time)

        # --- Model components ---
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tokenizer = None
        self.train_dataloader = None
        self.eval_dataloader = None

        # --- Memory management ---
        self.memory_manager = None

        # --- Dropout schedule ---
        self.current_dropout_values = {}  # Current dropout values
        self.dropout_warmup_steps = 0
        self.dropout_ramp_steps = 0
        self.dropout_schedule = "linear"
        self.target_resid_dropout = 0.0
        self.target_hidden_dropout = 0.0

        # --- Gradient accumulation ---
        try:
            self.gradient_accumulation_steps = int(self.training_config.get('gradient_accumulation_steps', 1))
        except (TypeError, ValueError):
            self.gradient_accumulation_steps = 1
        self.gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)

    def _load_config(self) -> Dict[str, Any]:
        """Load config file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup(self):
        """Set up training environment"""
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

        # Core finetune step: load model weights from pretrained checkpoint
        self._load_pretrain_checkpoint()

        self.setup_complete = True

        if self.logger_manager:
            self.logger_manager.log_training_start(self.config)
            # Log checkpoint step offset to wandb for tracking
            self._log_checkpoint_offset_to_wandb()

        logger.info("Finetune training environment setup complete")

    def _load_pretrain_checkpoint(self):
        """
        Load model weights from pretrained checkpoint (single-GPU, .pt format only).

        Unlike regular resume:
        - Only loads model weights, not optimizer/scheduler
        - Resets global_step to 0 (fresh training start)
        - Used for finetuning (with lineage prefix + EOS weight)
        """
        pretrain_checkpoint = self.training_config.get('resume_from_pretrain', None)

        if not pretrain_checkpoint:
            logger.warning("⚠️  未指定预训练checkpoint (resume_from_pretrain)")
            logger.warning("   将从随机初始化开始训练")
            return

        checkpoint_path = Path(pretrain_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

        # If directory, auto-find .pt file
        if checkpoint_path.is_dir():
            pt_files = list(checkpoint_path.glob('*.pt'))
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in directory: {checkpoint_path}")
            # Prefer model_weights.pt, otherwise use first .pt file found
            model_weights_pt = checkpoint_path / 'model_weights.pt'
            if model_weights_pt.exists():
                checkpoint_path = model_weights_pt
            else:
                checkpoint_path = pt_files[0]
            logger.info(f"Found checkpoint file in directory: {checkpoint_path}")

        logger.info(f"Loading model weights from pretrained checkpoint: {checkpoint_path}")

        try:
            load_start_time = time.time()

            # Load .pt checkpoint
            logger.info(f"Loading checkpoint with torch.load...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Support multiple checkpoint formats
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                metadata = checkpoint.get('metadata', {})
            elif 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                metadata = checkpoint.get('metadata', {})
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
                metadata = checkpoint.get('metadata', {})
            else:
                # Direct state_dict
                model_state_dict = checkpoint
                metadata = {}

            # Handle possible 'module.' prefix
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict

            # Load model weights
            load_result = self.model.load_state_dict(model_state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys[:5]}...")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:5]}...")

            pretrain_step = metadata.get('global_step', 0)

            load_time = time.time() - load_start_time
            logger.info(f"Pretrained checkpoint loaded successfully, took: {load_time:.2f}s")

            # Extract step number from checkpoint path as offset
            checkpoint_name = checkpoint_path.stem
            match = re.search(r'checkpoint[-_]?(\d+)', checkpoint_name)
            if match:
                self.checkpoint_step_offset = int(match.group(1))
            elif pretrain_step > 0:
                self.checkpoint_step_offset = pretrain_step
            else:
                self.checkpoint_step_offset = 0

            logger.info(f"Pretrained model weights loaded:")
            logger.info(f"   - Pretrained checkpoint: {checkpoint_path}")
            logger.info(f"   - Pretrained steps: {pretrain_step}")
            logger.info(f"   - Finetune starting step: 0 (checkpoint naming offset={self.checkpoint_step_offset})")
            logger.info(f"   - Note: Optimizer/Scheduler reset, starting with fresh learning rate")

        except Exception as e:
            logger.error(f"Failed to load pretrained checkpoint: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        logger.info(f"Pretrained checkpoint loading complete")

    def _log_checkpoint_offset_to_wandb(self):
        """Log checkpoint step offset to wandb"""
        if not self.logger_manager:
            return

        # Get wandb logger
        wandb_logger = getattr(self.logger_manager, 'wandb_logger', None)
        if wandb_logger and wandb_logger.enable:
            try:
                finetune_info = {
                    'finetune/checkpoint_step_offset': self.checkpoint_step_offset,
                    'finetune/resume_from_pretrain': self.training_config.get('resume_from_pretrain', 'N/A'),
                    'finetune/starting_step': self.checkpoint_step_offset,
                }
                wandb_logger.wandb.config.update(finetune_info, allow_val_change=True)
                logger.info(f"Finetune info logged to WandB:")
                logger.info(f"   - Starting step: {self.checkpoint_step_offset}")
                logger.info(f"   - WandB curves will start from step {self.checkpoint_step_offset}")
            except Exception as e:
                logger.warning(f"Failed to log checkpoint offset to WandB: {e}")

    def _setup_distributed(self):
        """设置单卡训练环境"""
        # 单卡模式，不使用分布式
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0

        # 设置设备
        if torch.cuda.is_available():
            # 可以通过环境变量 CUDA_VISIBLE_DEVICES 指定使用哪张卡
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            logger.info(f"✅ 单卡训练模式: 使用 GPU {self.local_rank}")
        else:
            self.device = torch.device('cpu')
            logger.warning("⚠️ CUDA不可用，使用CPU训练")

    def _setup_logging(self):
        """设置日志系统"""
        logging_config = self.logging_config

        experiment_name = f"finetune_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir = logging_config.get('log_dir', './results/finetune/logs')

        self.logger_manager = create_logger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            config=self.config,
            local_rank=self.local_rank,
            enable_wandb=logging_config.get('enable_wandb', False),  # 单卡模式直接使用配置
            wandb_project=logging_config.get('wandb_project', 'rna-finetune'),
            wandb_run_name=logging_config.get('wandb_run_name', experiment_name)
        )

    def _set_seed(self):
        """设置随机种子"""
        seed = self.training_config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_model(self):
        """设置模型（单卡版本）"""
        model_config_dict = self.model_config
        data_config = self.data_config

        # 读取use_direction_tokens配置（默认True）
        use_direction_tokens = data_config.get('use_direction_tokens', True)

        # 加载tokenizer（根据配置选择新/旧版本）
        self.tokenizer = get_lineage_rna_tokenizer(use_direction_tokens=use_direction_tokens)

        # 自动从tokenizer设置vocab_size，确保一致性
        model_config_dict['vocab_size'] = self.tokenizer.vocab_size
        logger.info(f"✅ 自动设置vocab_size={self.tokenizer.vocab_size} (use_direction_tokens={use_direction_tokens})")

        # 单卡模式：禁用 MegaBlocks 分布式特性
        if model_config_dict.get('moe_implementation') == "megablocks":
            model_config_dict['moe_world_size'] = 1  # 强制设为1，单卡不支持专家并行
            logger.info("⚠️ 单卡模式：MegaBlocks moe_world_size 强制设为 1")

        model_config = RNAGenConfig(tokenizer=self.tokenizer, **model_config_dict)
        self.model = create_rnagen_model(model_config)
        logger.info(f"模型创建完成，设备: cuda:{self.local_rank}")

        # Finetune特性：显示EOS loss权重配置
        eos_loss_weight = model_config_dict.get('eos_loss_weight', 1.0)
        if eos_loss_weight != 1.0:
            logger.info(f"📊 Finetune EOS Loss权重: {eos_loss_weight}")
            logger.info(f"   用于解决EOS预测不足的问题")

        training_config = self.training_config
        if training_config.get('bf16', False):
            target_dtype = torch.bfloat16
        elif training_config.get('fp16', False):
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        self.model = self.model.to(f'cuda:{self.local_rank}')
        if target_dtype != torch.float32:
            self.model = self.model.to(target_dtype)

        # 禁用输出token mask（避免GLM模式下<eos_span>导致loss=inf）
        self.model.output_token_mask = None
        logger.info("⚠️ Finetune模式：禁用output_token_mask（支持GLM模式）")

    def _setup_datasets(self):
        """设置数据集 - Finetune序列生成"""
        data_config = self.data_config
        training_config = self.training_config

        # Finetune核心配置
        use_lineage_prefix = data_config.get('use_lineage_prefix', True)
        use_rna_type_prefix = data_config.get('use_rna_type_prefix', True)  # 是否使用RNA类型前缀
        pretrain_ratio = data_config.get('pretrain_ratio', 0.0)  # pretraining任务比例
        fixed_lineage = data_config.get('fixed_lineage', None)  # 固定谱系字符串（用于微调特定物种）

        # 创建基于谱系的数据集
        use_chunked = data_config.get('use_chunked', True)
        cache_size = data_config.get('cache_size', 10000)
        force_rebuild_index = data_config.get('force_rebuild_index', False)
        use_direction_tokens = data_config.get('use_direction_tokens', True)
        enable_reverse_augmentation = data_config.get('enable_reverse_augmentation', True)

        # Mixed GLM模式配置
        mode = data_config.get('mode', 'generation')
        glm_probability = data_config.get('glm_probability', 0.333)

        # 如果是completion或mixed模式，需要构建SpanConfig
        span_config = self._build_span_config(mode, data_config)

        # GLM fuzzy factor配置
        fuzzy_factor = data_config.get('fuzzy_factor', 0.2)

        self.train_dataset = create_lineage_dataset(
            data_file=data_config['train_file'],
            tokenizer=self.tokenizer,
            lineage_file=data_config.get('lineage_file', ''),  # 使用fixed_lineage时可以为空
            mode=mode,
            span_config=span_config,
            glm_probability=glm_probability,
            max_seq_length=data_config.get('max_seq_length', 1024),
            max_samples=data_config.get('max_samples'),
            use_chunked=use_chunked,
            cache_size=cache_size,
            force_rebuild_index=force_rebuild_index,
            use_direction_tokens=use_direction_tokens,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
            pretrain_ratio=pretrain_ratio,
            fixed_lineage=fixed_lineage,
            fuzzy_factor=fuzzy_factor,
        )

        data_collator = create_rna_data_collator(
            tokenizer=self.tokenizer,
            max_length=data_config.get('max_seq_length', 1024),
            device="cpu"
        )

        # 单卡模式：不使用 DistributedSampler，直接使用 shuffle
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=training_config.get('per_device_train_batch_size', 2),
            shuffle=True,  # 单卡模式直接 shuffle
            collate_fn=data_collator,
            num_workers=training_config.get('dataloader_num_workers', 2),
            pin_memory=training_config.get('dataloader_pin_memory', True),
            drop_last=training_config.get('dataloader_drop_last', True)
        )

        self._log_dataset_info(mode, fuzzy_factor, use_lineage_prefix,
                               use_rna_type_prefix, use_direction_tokens,
                               pretrain_ratio, fixed_lineage)

    def _build_span_config(self, mode: str, data_config: Dict[str, Any]) -> Optional['SpanConfig']:
        """构建SpanConfig（用于completion/mixed模式）"""
        if mode not in ['mixed', 'completion']:
            return None

        span_config_dict = data_config.get('span_config', {})

        # 读取固定区域GLM配置
        fixed_region_config = data_config.get('fixed_region_glm', {})
        fixed_regions = None
        fixed_region_ratio = 0.0
        if fixed_region_config.get('enabled', False):
            fixed_regions = fixed_region_config.get('regions', [])
            fixed_region_ratio = fixed_region_config.get('ratio', 0.0)
            logger.info(f"启用固定区域GLM: ratio={fixed_region_ratio}, regions={fixed_regions}")

        return SpanConfig(
            max_coverage_ratios=span_config_dict.get('max_coverage_ratios', [0.15, 0.25, 0.5, 0.8]),
            coverage_probs=span_config_dict.get('coverage_probs', [0.28, 0.30, 0.28, 0.14]),
            span_distributions=span_config_dict.get('span_distributions', [(10, 5), (20, 10), (50, 20)]),
            max_num_spans=span_config_dict.get('max_num_spans', 10),
            allow_overlap=span_config_dict.get('allow_overlap', False),
            fixed_regions=fixed_regions,
            fixed_region_ratio=fixed_region_ratio,
            fixed_span_id=span_config_dict.get('fixed_span_id', None),
        )

    def _log_dataset_info(self, mode: str, fuzzy_factor: float,
                          use_lineage_prefix: bool, use_rna_type_prefix: bool,
                          use_direction_tokens: bool, pretrain_ratio: float,
                          fixed_lineage: Optional[str]):
        """打印数据集和输入格式信息"""
        logger.info(f"Finetune数据集准备完成")
        logger.info(f"  训练样本数: {len(self.train_dataset)}")
        logger.info(f"  批次数: {len(self.train_dataloader)}")
        logger.info(f"  训练模式: {mode}")
        if mode in ['glm', 'mixed']:
            logger.info(f"  GLM fuzzy_factor: {fuzzy_factor}")

        # 输入格式配置
        logger.info(f"输入格式配置:")
        logger.info(f"  use_lineage_prefix: {use_lineage_prefix} ({'开启物种谱系' if use_lineage_prefix else '关闭物种谱系'})")
        logger.info(f"  use_rna_type_prefix: {use_rna_type_prefix} ({'开启RNA类型' if use_rna_type_prefix else '关闭RNA类型'})")
        logger.info(f"  use_direction_tokens: {use_direction_tokens} ({'开启方向标记' if use_direction_tokens else '关闭方向标记'})")
        logger.info(f"  pretrain_ratio: {pretrain_ratio} ({int(pretrain_ratio * 100)}% 无前缀纯序列任务)")
        if fixed_lineage:
            logger.info(f"  fixed_lineage: {fixed_lineage} (使用固定谱系，不从文件读取)")

        # 输出一个实际的训练样本示例
        try:
            sample = self.train_dataset[0]
            input_ids = sample.get('input_ids', None)
            if input_ids is not None:
                # 解码前50个token作为示例
                example_tokens = input_ids[:min(50, len(input_ids))].tolist()
                example_text = self.tokenizer.decode(example_tokens)
                logger.info(f"训练输入格式示例 (前50 tokens):")
                logger.info(f"  {example_text}")
        except Exception as e:
            logger.warning(f"无法获取训练样本示例: {e}")

    def _setup_optimizer(self):
        """设置优化器（单卡版本）"""
        training_config = self.training_config

        learning_rate = float(training_config.get('learning_rate', 2e-3))
        weight_decay = float(training_config.get('weight_decay', 1e-5))
        adam_beta1 = float(training_config.get('adam_beta1', 0.9))
        adam_beta2 = float(training_config.get('adam_beta2', 0.95))
        adam_epsilon = float(training_config.get('adam_epsilon', 1e-8))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon
        )

        logger.info(f"Finetune优化器设置完成 - 学习率: {learning_rate}")

    def _setup_scheduler(self):
        """设置学习率调度器"""
        training_config = self.training_config

        max_epochs = training_config.get('max_epochs', 2)
        warmup_ratio = training_config.get('warmup_ratio', 0.1)
        min_lr_ratio = training_config.get('min_lr_ratio', 0.1)

        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        num_training_steps = (len(self.train_dataloader) // gradient_accumulation_steps) * max_epochs

        warmup_steps_config = training_config.get('warmup_steps', None)
        if warmup_steps_config is not None:
            warmup_steps_config = int(warmup_steps_config)
            if warmup_steps_config > 0:
                num_warmup_steps = warmup_steps_config
            else:
                num_warmup_steps = int(num_training_steps * warmup_ratio)
        else:
            num_warmup_steps = int(num_training_steps * warmup_ratio)


        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        logger.info(f"学习率调度器设置完成")
        logger.info(f"  总步数: {num_training_steps}")
        used_fixed_warmup = (training_config.get('warmup_steps', None) is not None and
                            int(training_config.get('warmup_steps', 0)) > 0)
        if used_fixed_warmup:
            logger.info(f"  Warmup步数: {num_warmup_steps} (手动设定)")
        else:
            logger.info(f"  Warmup步数: {num_warmup_steps} ({warmup_ratio*100:.1f}%)")
        logger.info(f"  峰值学习率: {training_config.get('learning_rate', 'N/A')}")
        logger.info(f"  最终学习率: {float(training_config.get('learning_rate', 1.0)) * min_lr_ratio:.2e}")

    def _setup_dropout_schedule(self):
        """设置Dropout调度器"""
        model_cfg = self.model_config

        self.dropout_warmup_steps = model_cfg.get('dropout_warmup_steps', 3000)
        self.dropout_ramp_steps = model_cfg.get('dropout_ramp_steps', 6000)
        self.dropout_schedule = model_cfg.get('dropout_schedule', 'linear')

        self.target_resid_dropout = model_cfg.get('resid_dropout', 0.0)
        self.target_hidden_dropout = model_cfg.get('hidden_dropout', 0.0)

        if self.target_resid_dropout > 0 or self.target_hidden_dropout > 0:
            logger.info("📊 Dropout调度器设置完成")
            logger.info(f"  目标resid_dropout: {self.target_resid_dropout}")
            logger.info(f"  目标hidden_dropout: {self.target_hidden_dropout}")

    def _ramp_dropout_value(self, step: int, target: float) -> float:
        """计算当前步的dropout值"""
        if step < self.dropout_warmup_steps:
            return 0.0
        if step >= self.dropout_warmup_steps + self.dropout_ramp_steps:
            return target
        progress = (step - self.dropout_warmup_steps) / max(1, self.dropout_ramp_steps)
        if self.dropout_schedule == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))
        return target * progress

    def _update_dropout_rates(self, step: int):
        """更新模型中的dropout率"""
        p_resid = self._ramp_dropout_value(step, self.target_resid_dropout)
        p_hidden = self._ramp_dropout_value(step, self.target_hidden_dropout)

        for layer in self.model.model.layers:
            if hasattr(layer, 'drop_resid'):
                layer.drop_resid.p = p_resid
            if hasattr(layer, 'drop_mlp'):
                layer.drop_mlp.p = p_hidden

        self.current_dropout_values = {
            'resid': p_resid,
            'hidden': p_hidden,
        }

    def _setup_memory_manager(self):
        """设置内存管理器"""
        memory_config = self.memory_config
        if memory_config.get('enable_monitoring', True):
            self.memory_manager = create_memory_manager(
                device=torch.device(f'cuda:{self.local_rank}'),
                cleanup_frequency=memory_config.get('cleanup_frequency', 100),
                gc_frequency=memory_config.get('gc_frequency', 50),
                enable_monitoring=memory_config.get('enable_monitoring', True)
            )

    def save_checkpoint(self, epoch: int, is_final: bool = False, step: int = None):
        """保存checkpoint为 .pt 格式（单卡版本）"""
        logger.info(f"开始checkpoint保存流程 - epoch={epoch}, step={step}, is_final={is_final}")

        if self.memory_manager:
            self.memory_manager.step(f"pre_checkpoint_step_{step}")
        torch.cuda.empty_cache()

        training_config = self.training_config
        output_dir = Path(training_config['output_dir'])

        if is_final:
            checkpoint_dir = output_dir / 'final'
            checkpoint_name = 'model_final.pt'
        elif step is not None:
            actual_step = step + self.checkpoint_step_offset
            checkpoint_dir = output_dir / f'checkpoint-{actual_step}'
            checkpoint_name = f'model_checkpoint_{actual_step}.pt'
        else:
            checkpoint_dir = output_dir / f'epoch-{epoch}'
            checkpoint_name = f'model_epoch_{epoch}.pt'

        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # 设置目录权限为777，允许非root用户删除
            os.chmod(checkpoint_dir, 0o777)
            # 同时设置父目录权限
            if output_dir.exists():
                os.chmod(output_dir, 0o777)
        except Exception as e:
            logger.error(f"目录创建失败: {e}")
            raise

        try:
            torch.cuda.empty_cache()

            metadata = {
                'global_step': self.global_step,
                'checkpoint_step_offset': self.checkpoint_step_offset,
                'epoch': epoch,
                'stage': 'finetune',  # Finetune标识
                'model_config': self.model_config,
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
            logger.error(f"state_dict构建失败: {e}")
            torch.cuda.empty_cache()
            raise

        try:
            save_start_time = time.time()
            checkpoint_path = checkpoint_dir / checkpoint_name
            torch.save(state_dict, checkpoint_path)
            # 设置文件权限为666，允许非root用户删除
            os.chmod(checkpoint_path, 0o666)
            save_time = time.time() - save_start_time
            logger.info(f"✅ Checkpoint保存成功: {checkpoint_path}, 耗时: {save_time:.2f}秒")

        except Exception as e:
            logger.error(f"❌ Checkpoint保存失败: {e}")
            raise
        finally:
            del state_dict
            torch.cuda.empty_cache()

        # 保存额外文件
        try:
            self.tokenizer.save_pretrained(checkpoint_dir)
            # 设置tokenizer文件权限
            for f in checkpoint_dir.glob('*.json'):
                os.chmod(f, 0o666)
        except Exception as e:
            logger.warning(f"Tokenizer保存失败: {e}")

        try:
            config_path = checkpoint_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)
            os.chmod(config_path, 0o666)
        except Exception as e:
            logger.warning(f"Config文件保存失败: {e}")

        try:
            config_path = checkpoint_dir / 'training_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            os.chmod(config_path, 0o666)
        except Exception as e:
            logger.warning(f"训练配置保存失败: {e}")

        logger.info(f"✅ Checkpoint保存流程完成: {checkpoint_dir}")

        if self.memory_manager:
            self.memory_manager.step(f"post_checkpoint_step_{step}")
        torch.cuda.empty_cache()

    def _calculate_model_flops(self):
        """计算模型FLOPs（单卡版本）"""
        try:
            model_config = self.model_config
            training_config = self.training_config
            data_config = self.data_config

            batch_size = max(1, int(training_config.get('per_device_train_batch_size', 2)))
            seq_length = int(data_config.get('max_seq_length', model_config.get('max_position_embeddings', 1024)))
            hidden_size = int(model_config.get('hidden_size', 512))
            num_layers = int(model_config.get('num_hidden_layers', 8))
            intermediate_size = int(model_config.get('intermediate_size', 2048))
            num_heads = max(1, int(model_config.get('num_attention_heads', 8)))
            num_kv_heads = max(1, int(model_config.get('num_key_value_heads', num_heads)))
            experts_per_tok = max(1, int(model_config.get('num_experts_per_tok', 1)))
            num_experts = int(model_config.get('num_experts', 1))
            gated_mlp = bool(model_config.get('gated_mlp', False))
            grad_accum_steps = max(1, int(training_config.get('gradient_accumulation_steps', 1)))

            tokens_per_device = batch_size * seq_length
            head_dim = hidden_size // num_heads
            kv_ratio = num_kv_heads / num_heads

            attn_proj_flops = tokens_per_device * 4 * hidden_size * hidden_size * (1 + kv_ratio)
            attn_scores = 2 * batch_size * num_heads * seq_length * seq_length * head_dim
            attn_values = attn_scores
            attn_flops = attn_proj_flops + attn_scores + attn_values

            mlp_per_token = 4 * hidden_size * intermediate_size
            if gated_mlp:
                mlp_per_token += 2 * hidden_size * intermediate_size
                mlp_per_token += intermediate_size

            mlp_flops = tokens_per_device * mlp_per_token
            router_flops = 0
            if num_experts > 1:
                mlp_flops *= experts_per_tok
                router_flops = tokens_per_device * 2 * hidden_size * num_experts
            mlp_flops += router_flops

            flops_per_layer = attn_flops + mlp_flops
            flops_per_forward_replica = flops_per_layer * num_layers
            # 单卡模式：data_parallel_size = 1
            total_flops = flops_per_forward_replica * grad_accum_steps * 6
            self.model_flops_per_step = total_flops

            logger.info(f"📊 模型FLOPs: {self.model_flops_per_step/1e12:.2f} TFLOPs/optimizer_step")

        except Exception as e:
            logger.warning(f"FLOPs计算失败: {e}")
            self.model_flops_per_step = 0

    def _update_flops(self):
        """更新FLOPs"""
        if self.model_flops_per_step:
            self.total_flops += self.model_flops_per_step

    def _collect_metrics(self, loss: torch.Tensor, ar_loss: torch.Tensor, lr: float,
                         aux_loss: Optional[torch.Tensor] = None,
                         clm_stats: Optional[Dict] = None,
                         glm_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """收集训练指标（单卡版本）"""
        # 单卡模式：直接使用 loss 值，不需要 all_reduce
        loss_for_log = loss.item()
        ar_loss_for_log = ar_loss.item()

        perplexity = torch.exp(torch.tensor(ar_loss_for_log)).item()

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
        else:
            memory_allocated = memory_reserved = 0

        # 计算显示步数（从checkpoint的step开始）
        display_step = self.global_step + self.checkpoint_step_offset

        metrics = {
            'train/loss': loss_for_log,
            'train/ar_loss': ar_loss_for_log,
            'train/perplexity': perplexity,
            'train/learning_rate': lr,
            'train/epoch': self.current_epoch,
            'train/global_step': display_step,  # 使用display_step，从checkpoint继续
            'train/finetune_step': self.global_step,  # Finetune内部步数（从0开始）
            'system/memory_allocated_gb': memory_allocated,
            'system/memory_reserved_gb': memory_reserved,
            'flops/step': self.model_flops_per_step or 0,
            'flops/cumulative': self.total_flops,
            'flops/cumulative_tera': self.total_flops / 1e12,
        }

        # CLM/GLM分别的perplexity
        if clm_stats and clm_stats['count'] > 0:
            clm_avg_loss = clm_stats['total_loss'] / clm_stats['count']
            clm_ppl = torch.exp(torch.tensor(clm_avg_loss)).item()
            metrics['train/clm_loss'] = clm_avg_loss
            metrics['train/clm_perplexity'] = clm_ppl
            metrics['train/clm_count'] = clm_stats['count']

        if glm_stats and glm_stats['count'] > 0:
            glm_avg_loss = glm_stats['total_loss'] / glm_stats['count']
            glm_ppl = torch.exp(torch.tensor(glm_avg_loss)).item()
            metrics['train/glm_loss'] = glm_avg_loss
            metrics['train/glm_perplexity'] = glm_ppl
            metrics['train/glm_count'] = glm_stats['count']

        if aux_loss is not None:
            aux_loss_val = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            metrics['train/aux_loss'] = aux_loss_val
            if loss_for_log > 0:
                metrics['train/aux_loss_ratio'] = aux_loss_val / loss_for_log

        if self.current_dropout_values:
            metrics['dropout/resid'] = self.current_dropout_values.get('resid', 0.0)
            metrics['dropout/hidden'] = self.current_dropout_values.get('hidden', 0.0)

        # Finetune特性：记录EOS loss权重
        model_cfg = self.model_config
        eos_loss_weight = model_cfg.get('eos_loss_weight', 1.0)
        if eos_loss_weight != 1.0:
            metrics['finetune/eos_loss_weight'] = eos_loss_weight

        self.current_loss = loss_for_log
        return metrics

    def _log_step(self, metrics: Dict[str, Any]):
        """记录训练步骤（单卡版本）"""
        display_step = self.global_step + self.checkpoint_step_offset

        if self.logger_manager:
            self.logger_manager.log_step(display_step, metrics)

        log_msg = (
            f"Step {display_step}: "
            f"Loss={metrics['train/loss']:.4f}, "
            f"AR_Loss={metrics['train/ar_loss']:.4f}, "
            f"PPL={metrics['train/perplexity']:.2f}, "
            f"LR={metrics['train/learning_rate']:.6f}"
        )
        # CLM/GLM分别的perplexity
        if 'train/clm_perplexity' in metrics:
            log_msg += f", CLM_PPL={metrics['train/clm_perplexity']:.2f}({int(metrics.get('train/clm_count', 0))})"
        if 'train/glm_perplexity' in metrics:
            log_msg += f", GLM_PPL={metrics['train/glm_perplexity']:.2f}({int(metrics.get('train/glm_count', 0))})"
        if 'train/aux_loss' in metrics:
            log_msg += f", Aux_Loss={metrics['train/aux_loss']:.6f}"
        if 'finetune/eos_loss_weight' in metrics:
            log_msg += f", EOS_W={metrics['finetune/eos_loss_weight']}"
        logger.info(log_msg)

    def train(self):
        """执行训练"""
        if not self.setup_complete:
            self.setup()

        self.train_start_time = time.time()
        training_config = self.training_config

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

        num_checkpoints = int(training_config.get('num_checkpoints', 24))

        self.save_steps = save_steps
        self.checkpoint_flops_milestones = []

        if save_steps and save_steps > 0:
            pass
        elif target_flops and num_checkpoints > 0:
            checkpoint_interval_flops = target_flops / num_checkpoints
            self.checkpoint_flops_milestones = [
                checkpoint_interval_flops * (i + 1) for i in range(num_checkpoints)
            ]

        logger.info(f"\n{'='*60}")
        logger.info(f"开始Finetune - 从预训练checkpoint继续训练")
        logger.info(f"批次大小: {training_config.get('per_device_train_batch_size', 2)}")

        # 显示Finetune特性
        model_cfg = self.model_config
        eos_loss_weight = model_cfg.get('eos_loss_weight', 1.0)
        data_cfg = self.data_config
        use_lineage_prefix = data_cfg.get('use_lineage_prefix', True)

        logger.info(f"\nFinetune特性:")
        logger.info(f"  - 谱系前缀: {'开启' if use_lineage_prefix else '关闭'}")
        logger.info(f"  - EOS Loss权重: {eos_loss_weight}")

        logger.info(f"\n停止条件:")
        if target_flops:
            logger.info(f"  - 目标FLOPs: {target_flops:.2e}")
        if max_wall_time_hours:
            logger.info(f"  - 最大运行时间: {max_wall_time_hours} 小时")
        if max_steps:
            logger.info(f"  - 最大步数: {max_steps}")
        if max_epochs:
            logger.info(f"  - 最大epochs: {max_epochs}")

        logger.info(f"\nCheckpoint保存策略:")
        if self.save_steps and self.save_steps > 0:
            logger.info(f"  - 基于步数间隔: 每{self.save_steps}步保存一次")
        elif self.checkpoint_flops_milestones:
            logger.info(f"  - 基于FLOPs均匀分布{num_checkpoints}个点")
        else:
            logger.info(f"  - 仅保存最终checkpoint")
        logger.info(f"{'='*60}\n")

        epoch = 0
        while True:
            self.current_epoch = epoch
            # 单卡模式：不需要 set_epoch

            epoch_loss, should_stop = self.train_epoch(
                epoch,
                target_flops=target_flops,
                max_wall_time_hours=max_wall_time_hours,
                max_steps=max_steps
            )

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
                logger.info(f"达到max_epochs={max_epochs}，停止训练")
                break

            epoch += 1

        # 不保存final checkpoint，只依赖中间checkpoint
        self._cleanup_and_exit(training_config)

    def train_epoch(self, epoch: int, target_flops: Optional[float] = None,
                    max_wall_time_hours: Optional[float] = None,
                    max_steps: Optional[int] = None):
        """训练一个epoch（单卡版本）"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        should_stop = False
        grad_accum_steps = self.gradient_accumulation_steps

        # CLM/GLM分别的loss统计（用于计算分别的perplexity）
        clm_stats = {'total_loss': 0.0, 'count': 0}
        glm_stats = {'total_loss': 0.0, 'count': 0}

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Finetune Epoch {epoch + 1}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            self._update_dropout_rates(self.global_step)

            # 提取task_types用于统计（在移动到GPU之前）
            task_types = batch.get('task_types', None)

            batch = {k: v.to(f'cuda:{self.local_rank}') if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            loss, ar_loss, aux_loss = self._process_batch(batch)

            self._update_task_stats(task_types, ar_loss, clm_stats, glm_stats)

            loss_to_backprop = loss / grad_accum_steps
            loss_to_backprop.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                # 梯度裁剪，防止梯度爆炸导致loss=inf
                max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                self._update_flops()

                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = self._collect_metrics(loss, ar_loss, current_lr, aux_loss,
                                               clm_stats=clm_stats, glm_stats=glm_stats)

                logging_steps = self.training_config.get('logging_steps', 20)
                if self.global_step % logging_steps == 0:
                    self._log_step(metrics)
                    # 重置统计（每次log后重置，统计的是logging_steps期间的平均值）
                    clm_stats = {'total_loss': 0.0, 'count': 0}
                    glm_stats = {'total_loss': 0.0, 'count': 0}

                self._maybe_save_checkpoint()

                should_stop = self._check_stop_conditions(
                    target_flops, max_wall_time_hours, max_steps)
                if should_stop:
                    break

            total_loss += loss.item()
            num_batches += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            display_step = self.global_step + self.checkpoint_step_offset
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}',
                'FLOPs': f'{self.total_flops:.2e}',
                'step': display_step
            })

            if self.memory_manager and batch_idx % 100 == 0:
                self.memory_manager.step(f"batch_{batch_idx}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss, should_stop

    def _process_batch(self, batch: Dict[str, Any]):
        """前向传播，返回 (loss, ar_loss, aux_loss)"""
        model_inputs = {
            k: v for k, v in batch.items()
            if k in ['input_ids', 'position_ids', 'sequence_ids', 'attention_mask', 'labels',
                    'past_key_values', 'use_cache', 'output_attentions', 'output_hidden_states']
        }

        outputs = self.model(**model_inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        ar_loss = outputs.get('ar_loss', loss) if isinstance(outputs, dict) else loss
        aux_loss = outputs.get('aux_loss', None) if isinstance(outputs, dict) else None
        return loss, ar_loss, aux_loss

    def _update_task_stats(self, task_types, ar_loss: torch.Tensor,
                           clm_stats: Dict, glm_stats: Dict):
        """根据task_types统计CLM/GLM的loss"""
        # 注意：当batch_size=1时，ar_loss就是该样本的loss
        # 当batch_size>1时，ar_loss是batch内所有样本的平均loss，此时统计不精确
        # 但由于当前配置batch_size=1，所以统计是准确的
        if task_types is None:
            return

        ar_loss_val = ar_loss.item()
        batch_size = len(task_types)
        if batch_size == 1:
            # batch_size=1时，精确统计
            task_type = task_types[0]
            if task_type == 'lineage_generation':
                clm_stats['total_loss'] += ar_loss_val
                clm_stats['count'] += 1
            elif task_type == 'lineage_completion_multi_span':
                glm_stats['total_loss'] += ar_loss_val
                glm_stats['count'] += 1
        else:
            # batch_size>1时，按类型数量加权分配（近似统计）
            clm_count = sum(1 for t in task_types if t == 'lineage_generation')
            glm_count = sum(1 for t in task_types if t == 'lineage_completion_multi_span')
            if clm_count > 0:
                clm_stats['total_loss'] += ar_loss_val * clm_count
                clm_stats['count'] += clm_count
            if glm_count > 0:
                glm_stats['total_loss'] += ar_loss_val * glm_count
                glm_stats['count'] += glm_count

    def _maybe_save_checkpoint(self):
        """检查是否需要保存checkpoint"""
        checkpoint_saved = False

        if hasattr(self, 'save_steps') and self.save_steps and self.save_steps > 0:
            if self.global_step > 0 and self.global_step % self.save_steps == 0:
                logger.info(f"💾 达到步数间隔 {self.save_steps}，保存checkpoint")
                self.save_checkpoint(self.current_epoch + 1, step=self.global_step)
                checkpoint_saved = True

        if not checkpoint_saved and hasattr(self, 'checkpoint_flops_milestones') and self.checkpoint_flops_milestones:
            for milestone_flops in self.checkpoint_flops_milestones:
                prev_flops = self.total_flops - self.model_flops_per_step
                if prev_flops < milestone_flops <= self.total_flops:
                    logger.info(f"💾 达到FLOPs里程碑 {milestone_flops:.2e}，保存checkpoint")
                    self.save_checkpoint(self.current_epoch + 1, step=self.global_step)
                    break

    def _check_stop_conditions(self, target_flops: Optional[float],
                               max_wall_time_hours: Optional[float],
                               max_steps: Optional[int]) -> bool:
        """检查是否满足停止条件"""
        if target_flops and self.total_flops >= target_flops:
            logger.info(f"✅ 达到target_flops={target_flops:.2e}，停止训练")
            return True

        if max_wall_time_hours and self.train_start_time:
            elapsed_hours = (time.time() - self.train_start_time) / 3600
            if elapsed_hours >= max_wall_time_hours:
                logger.info(f"⏰ 达到max_wall_time={max_wall_time_hours}小时，停止训练")
                return True

        if max_steps and self.global_step >= max_steps:
            logger.info(f"达到max_steps={max_steps}，停止训练")
            return True

        return False

    def _cleanup_and_exit(self, training_config):
        """清理资源（单卡版本）"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🏁 Finetune训练完成！")
            logger.info(f"最终模型: {training_config.get('output_dir', 'N/A')}/final/")
            logger.info(f"总步数: {self.global_step:,}")
            logger.info(f"总FLOPs: {self.total_flops/1e12:.2f} TFLOPs")
            logger.info(f"{'='*60}\n")

            if self.logger_manager:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                except Exception as e:
                    logger.warning(f"关闭WandB时出错: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"清理过程出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='Finetune: 从预训练checkpoint继续训练')
    parser.add_argument('--config', type=str,
                       default='configs/finetune/base_finetune.yaml',
                       help='配置文件路径')
    args = parser.parse_args()

    trainer = None
    try:
        trainer = FinetuneTrainer(args.config)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        if trainer:
            trainer._cleanup_and_exit(trainer.training_config)
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        if trainer:
            trainer._cleanup_and_exit(trainer.training_config)
        raise


if __name__ == '__main__':
    main()
