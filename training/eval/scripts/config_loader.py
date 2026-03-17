#!/usr/bin/env python3
"""
配置加载器 - 统一管理训练和评估配置
从checkpoint或训练配置文件中加载配置，支持灵活覆盖
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器 - 统一配置管理"""

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        从checkpoint加载训练配置

        Args:
            checkpoint_path: checkpoint目录或文件路径

        Returns:
            配置字典，如果找不到则返回None
        """
        checkpoint_path = Path(checkpoint_path)

        # 情况1: checkpoint目录（包含config.yaml或config.json）
        if checkpoint_path.is_dir():
            config_file = checkpoint_path / "config.yaml"
            if config_file.exists():
                logger.info(f"从checkpoint加载配置: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)

            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                logger.info(f"从checkpoint加载配置: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            # 尝试从pytorch_model.bin的同级目录查找
            model_file = checkpoint_path / "pytorch_model.bin"
            if model_file.exists():
                parent_config = checkpoint_path.parent / "training_config.yaml"
                if parent_config.exists():
                    logger.info(f"从父目录加载配置: {parent_config}")
                    with open(parent_config, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)

        # 情况2: checkpoint文件（.bin, .pt, .pth）
        elif checkpoint_path.is_file() and checkpoint_path.suffix in ['.bin', '.pt', '.pth']:
            # 尝试从同级目录查找config
            config_file = checkpoint_path.parent / "config.yaml"
            if config_file.exists():
                logger.info(f"从checkpoint同级目录加载配置: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)

        logger.warning(f"未找到配置文件: {checkpoint_path}")
        return None

    @staticmethod
    def load_from_training_config(config_path: str) -> Dict[str, Any]:
        """
        直接从训练配置文件加载

        Args:
            config_path: 训练配置文件路径

        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        logger.info(f"从训练配置加载: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def extract_eval_config(
        training_config: Dict[str, Any],
        override_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从训练配置中提取评估配置，并应用覆盖

        Args:
            training_config: 完整的训练配置
            override_config: 覆盖配置（可选）

        Returns:
            评估配置字典
        """
        # 提取eval_config部分
        eval_config = training_config.get('eval_config', {}).copy()

        # 如果没有eval_config，从training_config推断
        if not eval_config:
            logger.warning("训练配置中没有eval_config，将从training_config推断")
            eval_config = ConfigLoader._infer_eval_config(training_config)

        # 应用覆盖配置
        if override_config:
            logger.info(f"应用覆盖配置: {list(override_config.keys())}")
            eval_config.update(override_config)

        return eval_config

    @staticmethod
    def _infer_eval_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从训练配置推断评估配置（向后兼容）

        Args:
            training_config: 训练配置

        Returns:
            推断的评估配置
        """
        train_cfg = training_config.get('training_config', {})
        data_cfg = training_config.get('data_config', {})

        # 尝试检测stage
        stage = None
        if 'stage1' in str(train_cfg.get('run_name', '')).lower():
            stage = 1
        elif 'stage2' in str(train_cfg.get('run_name', '')).lower():
            stage = 2
        elif 'stage3' in str(train_cfg.get('run_name', '')).lower():
            stage = 3

        # 推断验证数据路径
        train_file = data_cfg.get('train_file', '')
        eval_data_file = train_file.replace('_train.fa', '_val.fa')

        eval_config = {
            'eval_data_file': eval_data_file,
            'eval_batch_size': 64,
            'max_seq_length': data_cfg.get('max_seq_length', 8096),
            'max_val_samples': None,
            'num_workers': 4,
            'stage': stage,
            'clm_ratio': train_cfg.get('clm_ratio', 0.0),
            'task_ratios': train_cfg.get('task_ratios', [0.0, 0.0, 0.0]),
            'use_multitask': train_cfg.get('use_multitask_dataset', True),
            'output_base_dir': '/rna-multiverse/training/eval',
            'save_predictions': False,
            'compute_metrics': {
                'perplexity': True,
                'accuracy': True,
                'f1_score': True
            }
        }

        logger.info(f"推断评估配置: stage={stage}, clm_ratio={eval_config['clm_ratio']}")
        return eval_config

    @staticmethod
    def auto_detect_stage(checkpoint_path: str) -> Optional[int]:
        """
        自动检测训练阶段

        Args:
            checkpoint_path: checkpoint路径

        Returns:
            阶段编号 (1, 2, 3) 或 None
        """
        checkpoint_path = Path(checkpoint_path)
        path_str = str(checkpoint_path).lower()

        if 'stage1' in path_str or 'classification' in path_str:
            return 1
        elif 'stage2' in path_str or 'clm' in path_str:
            return 2
        elif 'stage3' in path_str or 'generation' in path_str:
            return 3

        return None

    @staticmethod
    def load_override_config(override_path: str) -> Dict[str, Any]:
        """
        加载覆盖配置文件

        Args:
            override_path: 覆盖配置文件路径

        Returns:
            覆盖配置字典
        """
        if not override_path:
            return {}

        override_path = Path(override_path)
        if not override_path.exists():
            logger.warning(f"覆盖配置文件不存在: {override_path}")
            return {}

        logger.info(f"加载覆盖配置: {override_path}")
        with open(override_path, 'r', encoding='utf-8') as f:
            if override_path.suffix == '.json':
                return json.load(f)
            else:
                return yaml.safe_load(f)

    @staticmethod
    def get_eval_config(
        checkpoint_path: str,
        training_config_path: Optional[str] = None,
        override_config_path: Optional[str] = None,
        override_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取完整的评估配置（主入口）

        优先级（已优化）：
        1. override_dict (命令行参数)
        2. override_config_path (覆盖配置文件)
        3. training_config_path (训练配置文件，优先级提升)
        4. checkpoint中的config (降级为fallback)
        5. 自动推断

        Args:
            checkpoint_path: checkpoint路径
            training_config_path: 训练配置文件路径（可选）
            override_config_path: 覆盖配置文件路径（可选）
            override_dict: 直接覆盖的配置字典（可选）

        Returns:
            完整的评估配置
        """
        training_config = None

        # 1. 优先使用显式指定的训练配置文件（优先级提升）
        if training_config_path:
            training_config = ConfigLoader.load_from_training_config(training_config_path)
            logger.info(f"✅ 使用显式指定的训练配置: {training_config_path}")

        # 2. 如果没有指定训练配置，尝试从checkpoint加载
        if not training_config:
            training_config = ConfigLoader.load_from_checkpoint(checkpoint_path)
            if training_config:
                logger.info(f"✅ 从checkpoint加载配置")

        # 3. 如果仍然没有配置，尝试自动查找训练配置
        if not training_config:
            stage = ConfigLoader.auto_detect_stage(checkpoint_path)
            if stage:
                config_map = {
                    1: "/rna-multiverse/configs/stage1_classification_16gpu.yaml",
                    2: "/rna-multiverse/configs/stage2_dual_mode_clm_16gpu.yaml",
                    3: "/rna-multiverse/configs/stage3_dual_mode_generation_16gpu.yaml"
                }
                default_config_path = config_map.get(stage)
                if default_config_path and Path(default_config_path).exists():
                    logger.info(f"自动使用默认配置: {default_config_path}")
                    training_config = ConfigLoader.load_from_training_config(default_config_path)

        # 4. 如果还是没有，抛出错误
        if not training_config:
            raise ValueError(
                f"无法加载训练配置！\n"
                f"请检查:\n"
                f"1. checkpoint目录中是否有config.yaml\n"
                f"2. 是否提供了--training_config参数\n"
                f"3. checkpoint路径是否包含stage信息\n"
                f"当前checkpoint路径: {checkpoint_path}"
            )

        # 5. 提取eval_config
        override_config = {}

        # 合并文件覆盖配置
        if override_config_path:
            file_override = ConfigLoader.load_override_config(override_config_path)
            override_config.update(file_override)

        # 合并字典覆盖配置
        if override_dict:
            override_config.update(override_dict)

        eval_config = ConfigLoader.extract_eval_config(training_config, override_config)

        # 6. 添加完整的训练配置引用（用于模型加载等）
        eval_config['_full_training_config'] = training_config

        logger.info(f"✅ 评估配置加载完成: stage={eval_config.get('stage')}")
        return eval_config


def main():
    """测试配置加载"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python config_loader.py <checkpoint_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    checkpoint_path = sys.argv[1]
    eval_config = ConfigLoader.get_eval_config(checkpoint_path)

    print("\n评估配置:")
    print(yaml.dump(eval_config, default_flow_style=False, allow_unicode=True))


if __name__ == "__main__":
    main()