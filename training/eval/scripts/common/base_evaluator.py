"""
BaseEvaluator 基类 + EvalMetrics dataclass
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """评估指标"""
    loss: float = 0.0
    perplexity: float = 0.0
    total_tokens: int = 0
    total_samples: int = 0


class BaseEvaluator(ABC):
    """
    评估器基类

    子类只需实现:
        - model_type (property): 'lineage' 或 'dense'
        - _load_model(): 调用各自的 model loader
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        ...

    @abstractmethod
    def _load_model(self):
        """加载模型，返回 (model, tokenizer, config)"""
        ...

    def __init__(
        self,
        checkpoint_path: str,
        eval_config: Dict[str, Any],
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.eval_config = eval_config
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.stage = self._detect_stage()

        stage_config = self.eval_config.get(self.stage, {})
        self.eval_mode = stage_config.get('mode', 'generation')
        logger.info(f"{'='*60}")
        logger.info(f"评估模式: {self.eval_mode}")
        logger.info(f"  - generation = CLM (因果语言建模)")
        logger.info(f"  - completion = GLM (span infilling)")
        logger.info(f"{'='*60}")

        self.model, self.tokenizer, self.config = self._load_model()
        self._resolve_training_config_params()
        self._setup_output_token_mask()
        self.eval_dataloader = self._prepare_dataloader()

    def _detect_stage(self) -> str:
        """检测checkpoint的训练阶段"""
        metadata_file = self.checkpoint_path / ".metadata"
        if metadata_file.exists():
            try:
                training_config = self.checkpoint_path / "training_config.yaml"
                if training_config.exists():
                    with open(training_config, 'r') as f:
                        config = yaml.safe_load(f)
                        run_name = config.get('training_config', {}).get('run_name', '')
                        if 'stage1' in run_name or 'generation' in run_name:
                            return 'stage1'
                        elif 'stage2' in run_name or 'completion' in run_name:
                            return 'stage2'
            except Exception as e:
                logger.warning(f"读取训练配置失败: {e}")
        logger.warning("无法检测训练阶段，默认使用stage1")
        return 'stage1'

    def _resolve_training_config_params(self):
        """从训练配置读取关键参数"""
        self.use_direction_tokens_from_training = False
        self.use_lineage_prefix_from_training = True
        self.use_rna_type_prefix_from_training = True
        self.enable_reverse_augmentation_from_training = False

        eval_override_direction = self.eval_config.get('use_direction_tokens')
        eval_override_lineage = self.eval_config.get('use_lineage_prefix')
        eval_override_rna_type = self.eval_config.get('use_rna_type_prefix')
        has_eval_override = any(x is not None for x in [eval_override_direction, eval_override_lineage, eval_override_rna_type])

        try:
            config_path = None
            for parent_level in range(3):
                check_path = self.checkpoint_path
                for _ in range(parent_level):
                    check_path = check_path.parent
                training_config_path = check_path / 'training_config.yaml'
                if training_config_path.exists():
                    config_path = training_config_path
                    break

            if config_path is None:
                for parent_level in range(3):
                    check_path = self.checkpoint_path
                    for _ in range(parent_level):
                        check_path = check_path.parent
                    experiment_config_path = check_path / 'experiment_config.yaml'
                    if experiment_config_path.exists():
                        config_path = experiment_config_path
                        break

            if config_path and config_path.exists():
                with open(config_path, 'r') as f:
                    exp_config = yaml.safe_load(f)
                data_config = exp_config.get('data_config', {})
                self.use_direction_tokens_from_training = data_config.get('use_direction_tokens', False)
                self.use_lineage_prefix_from_training = data_config.get('use_lineage_prefix', True)
                self.use_rna_type_prefix_from_training = data_config.get('use_rna_type_prefix', True)
                logger.info(f"从训练配置读取数据参数:")
                logger.info(f"   - use_direction_tokens: {self.use_direction_tokens_from_training}")
                logger.info(f"   - use_lineage_prefix: {self.use_lineage_prefix_from_training}")
                logger.info(f"   - use_rna_type_prefix: {self.use_rna_type_prefix_from_training}")
        except Exception as e:
            logger.warning(f"读取训练配置失败: {e}，使用默认值")

        if has_eval_override:
            if eval_override_direction is not None:
                self.use_direction_tokens_from_training = eval_override_direction
            if eval_override_lineage is not None:
                self.use_lineage_prefix_from_training = eval_override_lineage
            if eval_override_rna_type is not None:
                self.use_rna_type_prefix_from_training = eval_override_rna_type

# PLACEHOLDER_OUTPUT_MASK

    def _setup_output_token_mask(self):
        """根据评估模式设置output_token_mask"""
        if self.eval_mode == 'completion':
            output_tokens = ["A", "U", "G", "C", "<eos_span>"]
        else:
            if self.use_direction_tokens_from_training:
                output_tokens = ["A", "U", "G", "C", "5", "3", "<eos>"]
            else:
                output_tokens = ["A", "U", "G", "C", "<eos>"]

        output_token_ids = [self.tokenizer.token_to_id(t) for t in output_tokens if self.tokenizer.token_to_id(t) is not None]

        model_vocab_size = self.config.vocab_size
        output_token_mask = torch.zeros(model_vocab_size, dtype=torch.bool)
        output_token_mask[output_token_ids] = True
        self.model.output_token_mask = output_token_mask

        logger.info(f"Output token mask已更新: {len(output_token_ids)}/{model_vocab_size} 个可输出token")

    def _prepare_dataloader(self) -> DataLoader:
        """准备评估数据集"""
        import sys
        project_root = Path(__file__).parent.parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from data.lineage_dataset import create_lineage_dataset, SpanConfig
        from data.rna_collator import create_rna_data_collator

        stage_config = self.eval_config.get(self.stage, {})

        data_file = stage_config.get('data_file')
        if not data_file or not Path(data_file).exists():
            raise FileNotFoundError(f"验证数据文件不存在: {data_file}")

        lineage_file = stage_config.get('lineage_file')
        if not lineage_file or not Path(lineage_file).exists():
            raise FileNotFoundError(f"谱系文件不存在: {lineage_file}")

        mode = stage_config.get('mode', 'generation')
        max_seq_length = stage_config.get('max_seq_length', 8192)
        max_samples = stage_config.get('max_samples', None)
        legacy_mode = self.eval_config.get('legacy_tokenization', False)

        span_config = None
        if mode == "completion":
            span_cfg = stage_config.get('span_config', {})
            fixed_span_length = span_cfg.get('fixed_span_length', None)
            span_config = SpanConfig(
                fixed_span_length=fixed_span_length,
                max_num_spans=span_cfg.get('max_num_spans', 1),
                span_id_range=tuple(span_cfg.get('span_id_range', [0, 49]))
            )

        eval_dataset = create_lineage_dataset(
            data_file=data_file,
            tokenizer=self.tokenizer,
            lineage_file=lineage_file,
            mode=mode,
            max_seq_length=max_seq_length,
            span_config=span_config,
            max_samples=max_samples,
            use_direction_tokens=self.use_direction_tokens_from_training,
            use_lineage_prefix=self.use_lineage_prefix_from_training,
            use_rna_type_prefix=self.use_rna_type_prefix_from_training,
            add_bos_token=(not legacy_mode),
            enable_reverse_augmentation=self.enable_reverse_augmentation_from_training
        )

        logger.info(f"  - 样本数: {len(eval_dataset)}")

        data_collator = create_rna_data_collator(
            tokenizer=self.tokenizer,
            max_length=max_seq_length,
            device="cpu"
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

        logger.info(f"  - 批次数: {len(eval_dataloader)}")
        return eval_dataloader

# PLACEHOLDER_EVALUATE

    def evaluate(self) -> Dict[str, float]:
        mode_desc = "GLM (span infilling)" if self.eval_mode == "completion" else "CLM (generation)"
        logger.info(f"\n{'='*60}")
        logger.info(f"开始{self.model_type}评估 - {mode_desc}")
        logger.info(f"{'='*60}\n")

        self.model.eval()
        metrics = EvalMetrics()

        total_ar_loss_sum = 0.0
        total_tokens = 0
        num_batches = 0

        start_time = time.time()
        total_batches = len(self.eval_dataloader)
        model_dtype = next(self.model.parameters()).dtype

        with torch.no_grad():
            progress_bar = tqdm(
                self.eval_dataloader,
                desc=f"Evaluating {mode_desc}",
                total=total_batches
            )

            for batch_idx, batch in enumerate(progress_bar):
                processed_batch = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        v = v.to(self.device)
                        if v.dtype in [torch.float32, torch.float64]:
                            v = v.to(model_dtype)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                batch = processed_batch

                model_inputs = {
                    k: v for k, v in batch.items()
                    if k in ['input_ids', 'position_ids', 'sequence_ids',
                            'attention_mask', 'labels']
                }

                try:
                    outputs = self.model(**model_inputs)

                    if isinstance(outputs, dict):
                        loss = outputs.get('loss')
                        ar_loss = outputs.get('ar_loss', loss)
                    else:
                        loss = outputs[0] if len(outputs) > 0 else None
                        ar_loss = loss

                    batch_tokens = 0
                    if 'labels' in batch:
                        labels = batch['labels']
                        valid_mask = labels != -100
                        if valid_mask.any():
                            batch_tokens = valid_mask.sum().item()
                            total_tokens += batch_tokens

                    if ar_loss is not None and batch_tokens > 0:
                        total_ar_loss_sum += ar_loss.item() * batch_tokens
                        num_batches += 1

                    if total_tokens > 0:
                        avg_ar_loss = total_ar_loss_sum / total_tokens
                        avg_ppl = torch.exp(torch.tensor(avg_ar_loss)).item()
                        progress_bar.set_postfix({
                            'ar_loss': f'{avg_ar_loss:.4f}',
                            'ppl': f'{avg_ppl:.2f}'
                        })

                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                        logger.info(f"EVAL_PROGRESS: {batch_idx + 1}/{total_batches}")

                except Exception as e:
                    logger.error(f"Batch {batch_idx} 评估失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

        eval_time = time.time() - start_time

        if total_tokens > 0:
            metrics.loss = total_ar_loss_sum / total_tokens
            metrics.perplexity = torch.exp(torch.tensor(metrics.loss)).item()
            metrics.total_tokens = total_tokens
            metrics.total_samples = num_batches * self.batch_size

        logger.info(f"\n{'='*60}")
        logger.info(f"{self.model_type}评估完成 - {mode_desc}")
        logger.info(f"{'='*60}")
        logger.info(f"Loss: {metrics.loss:.4f}")
        logger.info(f"Perplexity: {metrics.perplexity:.2f}")
        logger.info(f"Total Tokens: {metrics.total_tokens:,}")
        logger.info(f"Evaluation Time: {eval_time:.2f}s")
        logger.info(f"{'='*60}\n")

        return {
            'loss': metrics.loss,
            'perplexity': metrics.perplexity,
            'total_tokens': metrics.total_tokens,
            'total_samples': metrics.total_samples,
            'eval_time': eval_time,
            'eval_mode': self.eval_mode,
            'stage': self.stage,
            'eval_dataset': str(self.eval_config.get(self.stage, {}).get('data_file', 'unknown')),
            'batch_size': self.batch_size,
            'num_batches': num_batches,
            'model_type': self.model_type,
        }

    def _get_result_filename(self) -> str:
        base_name = f"eval_results_{self.stage}"
        if self.eval_mode == 'completion':
            stage_config = self.eval_config.get(self.stage, {})
            span_config = stage_config.get('span_config', {})
            fixed_span_length = span_config.get('fixed_span_length')
            if fixed_span_length is not None:
                base_name = f"{base_name}_glm_span{fixed_span_length}"
            else:
                base_name = f"{base_name}_glm"
        return f"{base_name}.json"

    def save_results(self, results: Dict[str, float], output_file: Optional[str] = None):
        if output_file is None:
            result_filename = self._get_result_filename()
            output_file = self.checkpoint_path / result_filename

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"评估结果已保存: {output_file}")

        # 也保存到父目录（兼容嵌套 checkpoint 结构）
        if self.checkpoint_path.name == 'model_only':
            parent_output_file = self.checkpoint_path.parent / self._get_result_filename()
            with open(parent_output_file, 'w') as f:
                json.dump(results, f, indent=2)
        elif self.checkpoint_path.name.startswith('checkpoint-') and self.checkpoint_path.parent.name.startswith('checkpoint-'):
            parent_output_file = self.checkpoint_path.parent / self._get_result_filename()
            with open(parent_output_file, 'w') as f:
                json.dump(results, f, indent=2)
