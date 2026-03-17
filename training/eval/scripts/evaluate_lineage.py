#!/usr/bin/env python3
"""
Lineage模型评估脚本
支持Stage 1 (generation)和Stage 2 (completion)的评估
使用单GPU DCP加载，支持并行评估
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Triton/MegaBlocks 兼容性修复
os.environ.setdefault('TRITON_DISABLE_LINE_INFO', '1')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from training.eval.scripts.common.base_evaluator import BaseEvaluator, EvalMetrics
from training.eval.scripts.lineage_model_loader import (
    load_lineage_model,
    validate_lineage_checkpoint,
)

logger = logging.getLogger(__name__)


class LineageEvaluator(BaseEvaluator):
    """Lineage模型评估器"""

    @property
    def model_type(self) -> str:
        return 'lineage'

    def _load_model(self):
        logger.info(f"加载lineage模型: {self.checkpoint_path}")

        if not validate_lineage_checkpoint(str(self.checkpoint_path)):
            raise ValueError(f"无效的lineage checkpoint: {self.checkpoint_path}")

        model, tokenizer, config = load_lineage_model(
            str(self.checkpoint_path),
            device=self.device
        )

        logger.info(f"Lineage模型加载完成")
        logger.info(f"  - 配置: {config.num_hidden_layers}层, {config.hidden_size}维, {config.num_experts}专家")
        logger.info(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description='Lineage模型评估')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--config', type=str, default='/rna-multiverse/training/eval/configs/lineage_eval_config.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    try:
        import yaml
        with open(args.config, 'r') as f:
            eval_config = yaml.safe_load(f)

        evaluator = LineageEvaluator(
            checkpoint_path=args.checkpoint_path,
            eval_config=eval_config,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        results = evaluator.evaluate()
        evaluator.save_results(results, args.output)
        logger.info("评估完成")

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
