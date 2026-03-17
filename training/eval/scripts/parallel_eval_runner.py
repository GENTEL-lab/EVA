#!/usr/bin/env python3
"""
8GPU并行评估调度器
使用GPU池管理器并行评估多个checkpoint（单GPU模式）

用法:
    python parallel_eval_runner.py \\
        --checkpoint_dir /path/to/checkpoints \\
        --training_config /path/to/config.yaml \\
        --num_gpus 8

特点:
- 每个GPU独立运行评估进程（非分布式）
- 单GPU DCP加载（自动合并16个分片）
- 智能任务队列（8个GPU并行处理checkpoint）
- 实时进度监控
"""

import os
import sys
import json
import yaml
import argparse
import logging
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.path_utils import setup_project_paths
setup_project_paths()

from scripts.gpu_pool_manager import create_gpu_pool_manager, CheckpointTask
try:
    from scripts.multinode_gpu_pool import create_multinode_gpu_pool
    HAS_MULTINODE = True
except ImportError:
    HAS_MULTINODE = False
from scripts.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelEvalRunner:
    """并行评估运行器"""

    def __init__(self, checkpoint_dir: str, training_config: str = None,
                 override_config: str = None, eval_config_file: str = None, num_gpus: int = 8,
                 mode: str = "multistage", multinode: bool = False,
                 multinode_config_file: str = None, dry_run: bool = False,
                 force_reeval: bool = False):
        """
        初始化并行评估运行器

        Args:
            checkpoint_dir: checkpoint父目录
            training_config: 训练配置文件（可选）
            override_config: 覆盖配置文件（可选）
            eval_config_file: 评估配置文件路径（可选，优先级最高）
            num_gpus: 使用的GPU数量（多节点模式下会被覆盖为16）
            mode: 评估模式 ("multistage" 或 "lineage")
            multinode: 是否启用多节点模式
            multinode_config_file: 多节点配置文件路径
            dry_run: Dry-run模式（只打印命令，不实际执行）
            force_reeval: 强制重新评估（忽略已有结果）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.training_config = training_config
        self.override_config = override_config
        self.eval_config_file = eval_config_file  # 指定的评估配置文件（优先级最高）
        self.num_gpus = num_gpus
        self.mode = mode
        self.multinode = multinode
        self.dry_run = dry_run
        self.force_reeval = force_reeval

        # 预加载评估配置文件以获取span信息（用于结果文件名判断）
        self._preload_eval_config_for_span()

        # 验证checkpoint目录
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint目录不存在: {self.checkpoint_dir}")

        # 查找所有checkpoint
        self.checkpoints = self._find_checkpoints()

        if not self.checkpoints:
            raise ValueError(f"未找到任何checkpoint: {self.checkpoint_dir}")

        logger.info(f"找到 {len(self.checkpoints)} 个checkpoint")

        # 加载评估配置
        self.eval_config = self._load_eval_config()

        # 初始化输出目录（必须在GPU管理器之前，因为多节点模式需要使用）
        self.output_dir = Path(self.eval_config.get('output_base_dir', '/rna-multiverse/training/eval'))

        # 创建GPU池管理器
        if self.multinode:
            # 多节点模式
            import yaml

            # 加载多节点配置
            if multinode_config_file is None:
                multinode_config_file = str(Path(__file__).parent.parent / 'configs' / 'multinode_config.yaml')

            with open(multinode_config_file, 'r') as f:
                full_config = yaml.safe_load(f)

            multinode_cfg = full_config['multinode']

            if not multinode_cfg.get('enabled'):
                raise ValueError("多节点配置中未启用multinode模式")

            # 计算总GPU数
            num_local_gpus = len(multinode_cfg['node0']['gpu_ids'])
            num_remote_gpus = len(multinode_cfg['node1']['virtual_gpu_ids'])
            self.num_gpus = num_local_gpus + num_remote_gpus

            logger.info(f"多节点模式启用:")
            logger.info(f"  本地GPU: {num_local_gpus}")
            logger.info(f"  远程GPU: {num_remote_gpus}")
            logger.info(f"  总GPU数: {self.num_gpus}")
            logger.info(f"  Dry-run: {self.dry_run}")

            # 创建多节点GPU池
            gpu_pool_config = {
                'max_gpus': self.num_gpus,
                'memory_threshold': 0.7,
                'check_interval': 10.0
            }
            self.gpu_manager = create_multinode_gpu_pool(
                config=gpu_pool_config,
                multinode_config=multinode_cfg,
                dry_run=self.dry_run
            )

            # 设置日志同步目录（多节点模式）
            local_log_base = str(self.output_dir / "logs")
            self.gpu_manager.set_log_base_dir(local_log_base)
        else:
            # 单节点模式
            gpu_pool_config = {
                'max_gpus': num_gpus,
                'memory_threshold': 0.7,
                'check_interval': 10.0
            }
            self.gpu_manager = create_gpu_pool_manager(gpu_pool_config)

    def _preload_eval_config_for_span(self):
        """预加载评估配置以获取span信息（用于确定结果文件名）"""
        self._fixed_span_length = None

        if self.eval_config_file:
            try:
                import yaml
                with open(self.eval_config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # 检查stage1和stage2的span配置
                for stage in ['stage1', 'stage2']:
                    stage_config = config.get(stage, {})
                    span_config = stage_config.get('span_config', {})
                    fixed_span = span_config.get('fixed_span_length')
                    if fixed_span is not None:
                        self._fixed_span_length = fixed_span
                        logger.info(f"检测到GLM评估配置: fixed_span_length={fixed_span}")
                        break
            except Exception as e:
                logger.warning(f"预加载评估配置失败: {e}")

    def _get_expected_result_filename(self, stage: str = 'stage1') -> str:
        """根据评估配置获取期望的结果文件名"""
        base_name = f"eval_results_{stage}"
        if self._fixed_span_length is not None:
            base_name = f"{base_name}_glm_span{self._fixed_span_length}"
        return f"{base_name}.json"

    def _has_eval_results(self, checkpoint_path: str) -> bool:
        """
        检查checkpoint是否已有评估结果

        Args:
            checkpoint_path: checkpoint路径

        Returns:
            True如果已有结果文件，False否则
        """
        checkpoint_path = Path(checkpoint_path)

        # 根据mode确定结果文件名
        if self.mode == "lineage" or self.mode == "dense":
            # lineage/dense模式：使用配置感知的结果文件名
            result_filename_stage1 = self._get_expected_result_filename('stage1')
            result_filename_stage2 = self._get_expected_result_filename('stage2')
            result_files = [
                checkpoint_path / result_filename_stage1,
                checkpoint_path / result_filename_stage2,
            ]
            # 如果是model_only目录，也检查父目录
            if checkpoint_path.name == "model_only":
                result_files.extend([
                    checkpoint_path.parent / result_filename_stage1,
                    checkpoint_path.parent / result_filename_stage2,
                ])
        else:
            # multistage模式：检查checkpoint_metrics.json
            result_files = [
                checkpoint_path / "checkpoint_metrics.json",
            ]
            if checkpoint_path.name == "model_only":
                result_files.append(checkpoint_path.parent / "checkpoint_metrics.json")

        # 检查是否有任何结果文件存在
        for result_file in result_files:
            if result_file.exists():
                return True

        return False

    def _find_checkpoints(self) -> List[str]:
        """查找所有checkpoint"""
        checkpoints = []

        # 检查是否为单个checkpoint目录
        if (self.checkpoint_dir / '.metadata').exists():
            checkpoints.append(str(self.checkpoint_dir))
            logger.info(f"单个checkpoint: {self.checkpoint_dir.name}")
            return checkpoints

        # 查找所有checkpoint子目录
        for ckpt_dir in sorted(self.checkpoint_dir.glob('checkpoint-*')):
            if ckpt_dir.is_dir():
                found = False

                # 1. 优先使用model_only子目录（如果存在）
                model_only_dir = ckpt_dir / 'model_only'
                if model_only_dir.is_dir():
                    # 检查是否为DCP格式（有.metadata）或单文件格式（有model_weights.pt）
                    if (model_only_dir / '.metadata').exists():
                        checkpoints.append(str(model_only_dir))
                        logger.info(f"  使用model_only: {ckpt_dir.name}/model_only (DCP格式)")
                        found = True
                    elif (model_only_dir / 'model_weights.pt').exists():
                        checkpoints.append(str(model_only_dir))
                        logger.info(f"  使用model_only: {ckpt_dir.name}/model_only (单文件格式)")
                        found = True

                # 2. 查找checkpoint-*格式的合并权重目录（如checkpoint-28000）
                if not found:
                    for merged_dir in ckpt_dir.glob('checkpoint-*'):
                        if merged_dir.is_dir() and (merged_dir / 'model_weights.pt').exists():
                            checkpoints.append(str(merged_dir))
                            logger.info(f"  使用合并权重: {ckpt_dir.name}/{merged_dir.name} (单文件格式)")
                            found = True
                            break

                # 3. 回退到DCP格式
                if not found and (ckpt_dir / '.metadata').exists():
                    checkpoints.append(str(ckpt_dir))
                    logger.info(f"  使用checkpoint: {ckpt_dir.name} (DCP格式)")

        # 智能提示：如果没找到checkpoint，检查是否指定了错误的目录层级
        if not checkpoints:
            # 检查是否指定了experiments父目录（包含多个实验子目录）
            experiment_subdirs = [d for d in self.checkpoint_dir.iterdir()
                                 if d.is_dir() and not d.name.startswith('.')]

            if experiment_subdirs:
                logger.warning("="*60)
                logger.warning("⚠️  未找到checkpoint，但发现以下子目录:")
                for subdir in experiment_subdirs[:5]:  # 只显示前5个
                    has_checkpoints = any(subdir.glob('checkpoint-*'))
                    status = "✅ 包含checkpoint" if has_checkpoints else "❌ 无checkpoint"
                    logger.warning(f"  - {subdir.name}/ ({status})")

                if len(experiment_subdirs) > 5:
                    logger.warning(f"  ... 共{len(experiment_subdirs)}个子目录")

                logger.warning("")
                logger.warning("💡 提示：请指定具体的实验目录，而不是父目录")
                logger.warning("   正确示例: .../experiments/660M_3M_20251008_142613")
                logger.warning("   错误示例: .../experiments")
                logger.warning("="*60)

        # 按checkpoint编号排序（支持数字和非数字名称）
        def checkpoint_sort_key(path):
            # 从路径中提取checkpoint目录名
            # 可能的格式：checkpoint-xxx, checkpoint-xxx/model_only, checkpoint-xxx/checkpoint-xxx
            path_obj = Path(path)
            name = path_obj.name

            # 如果是model_only或嵌套的checkpoint-xxx，使用父目录名
            if name == 'model_only' or (name.startswith('checkpoint-') and path_obj.parent.name.startswith('checkpoint-')):
                name = path_obj.parent.name

            if '-' in name:
                try:
                    # 尝试提取数字编号
                    return (0, int(name.split('-')[1]))
                except ValueError:
                    # 非数字名称（如checkpoint-random），排在最后
                    return (1, name.split('-')[1])
            return (2, name)  # 无'-'的名称
        
        checkpoints.sort(key=checkpoint_sort_key)

        # 过滤已有结果的checkpoint（除非force_reeval=True）
        if not self.force_reeval:
            original_count = len(checkpoints)
            skipped_checkpoints = []
            filtered_checkpoints = []

            for ckpt_path in checkpoints:
                if self._has_eval_results(ckpt_path):
                    skipped_checkpoints.append(ckpt_path)
                else:
                    filtered_checkpoints.append(ckpt_path)

            if skipped_checkpoints:
                logger.info("\n" + "="*60)
                logger.info(f"📋 跳过已有评估结果的checkpoint: {len(skipped_checkpoints)}/{original_count}")
                logger.info("="*60)
                for ckpt_path in skipped_checkpoints:
                    ckpt_name = Path(ckpt_path).parent.name if Path(ckpt_path).name == 'model_only' else Path(ckpt_path).name
                    logger.info(f"  ⏭️  {ckpt_name}")
                logger.info("")
                logger.info(f"✅ 需要评估的checkpoint: {len(filtered_checkpoints)}/{original_count}")
                if filtered_checkpoints:
                    logger.info("待评估列表:")
                    for ckpt_path in filtered_checkpoints:
                        ckpt_name = Path(ckpt_path).parent.name if Path(ckpt_path).name == 'model_only' else Path(ckpt_path).name
                        logger.info(f"  🔄 {ckpt_name}")
                logger.info("="*60)
                logger.info("💡 提示: 使用 --force-reeval 参数可强制重新评估所有checkpoint")
                logger.info("="*60 + "\n")

            checkpoints = filtered_checkpoints

        return checkpoints

    def _load_eval_config(self) -> Dict[str, Any]:
        """加载评估配置"""
        try:
            # 使用配置加载器
            config = ConfigLoader.get_eval_config(
                checkpoint_path=self.checkpoints[0],  # 使用第一个checkpoint的配置
                training_config_path=self.training_config,
                override_config_path=self.override_config,
                override_dict={
                    'distributed_eval': False,  # 强制单GPU模式
                }
            )

            logger.info(f"评估配置加载成功:")
            logger.info(f"  - Stage: {config.get('stage')}")
            logger.info(f"  - CLM比例: {config.get('clm_ratio')}")
            logger.info(f"  - 任务比例: {config.get('task_ratios')}")

            return config

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    def _generate_lineage_eval_script(self, checkpoint_path: str, gpu_id: int, checkpoint_name: str) -> str:
        """生成lineage评估脚本"""
        # 如果指定了评估配置文件，直接使用；否则自动推断
        if self.eval_config_file:
            config_file_logic = f'''
# 使用指定的评估配置文件（优先级最高）
config_file = '{self.eval_config_file}'
logger.info(f"使用指定的评估配置: {{config_file}}")
'''
        else:
            config_file_logic = '''
# 自动推断评估配置（根据checkpoint的训练配置）
logger.info("自动推断评估配置...")
from training.eval.scripts.auto_config_helper import update_lineage_eval_config
try:
    config_file = update_lineage_eval_config('{checkpoint_path}')
    logger.info(f"使用评估配置: {{config_file}}")
except Exception as e:
    logger.warning(f"自动推断失败: {{e}}")
    logger.info("使用默认配置文件")
    config_file = '/rna-multiverse/training/eval/configs/lineage_eval_config.yaml'
'''

        return f'''
import os
# 设置CUDA环境变量以避免内存访问问题
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA执行，便于调试
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用设备端断言
# Triton/MegaBlocks 兼容性修复
os.environ['TRITON_DISABLE_LINE_INFO'] = '1'  # 禁用line info减少内存访问
os.environ['TRITON_KERNEL_CACHE_DIR'] = '/tmp/triton_cache'  # 使用显式缓存目录
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'  # 支持多代GPU架构
# CUDA内存分配优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 减少内存碎片

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path('/rna-multiverse')
sys.path.insert(0, str(project_root))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GPU{gpu_id} - %(message)s')
logger = logging.getLogger(__name__)

from training.eval.scripts.evaluate_lineage import LineageEvaluator
import yaml

{config_file_logic}

# 加载评估配置
logger.info("加载评估配置...")
with open(config_file, 'r') as f:
    eval_config = yaml.safe_load(f)

# 创建评估器
logger.info("创建lineage评估器...")
try:
    evaluator = LineageEvaluator(
        checkpoint_path='{checkpoint_path}',
        eval_config=eval_config,
        device='cuda',
        batch_size=eval_config.get('evaluation', {{}}).get('batch_size', 4),
        num_workers=eval_config.get('evaluation', {{}}).get('num_workers', 2)
    )

    # 执行评估
    logger.info(f"开始评估: {checkpoint_name}")
    results = evaluator.evaluate()

    # 保存结果
    evaluator.save_results(results)

    logger.info(f"✅ Lineage评估完成: {checkpoint_name}")
    logger.info(f"  Loss: {{results['loss']:.4f}}")
    logger.info(f"  Perplexity: {{results['perplexity']:.2f}}")
    sys.exit(0)

except Exception as e:
    logger.error(f"❌ Lineage评估失败: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    def _generate_dense_eval_script(self, checkpoint_path: str, gpu_id: int, checkpoint_name: str) -> str:
        """生成dense模型评估脚本"""
        # 如果指定了评估配置文件，直接使用；否则使用默认配置
        if self.eval_config_file:
            config_file_logic = f'''
# 使用指定的评估配置文件（优先级最高）
config_file = '{self.eval_config_file}'
logger.info(f"使用指定的评估配置: {{config_file}}")
'''
        else:
            config_file_logic = '''
# 使用默认的dense评估配置
config_file = '/rna-multiverse/training/eval/configs/dense_eval_config.yaml'
logger.info(f"使用默认评估配置: {config_file}")
'''

        return f'''
import os
# 设置CUDA环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA执行，便于调试
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用设备端断言
# CUDA内存分配优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 减少内存碎片

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path('/rna-multiverse')
sys.path.insert(0, str(project_root))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GPU{gpu_id} - %(message)s')
logger = logging.getLogger(__name__)

from training.eval.scripts.evaluate_dense import DenseEvaluator
import yaml

{config_file_logic}

# 加载评估配置
logger.info("加载评估配置...")
with open(config_file, 'r') as f:
    eval_config = yaml.safe_load(f)

# 创建评估器
logger.info("创建Dense评估器...")
try:
    evaluator = DenseEvaluator(
        checkpoint_path='{checkpoint_path}',
        eval_config=eval_config,
        device='cuda',
        batch_size=eval_config.get('evaluation', {{}}).get('batch_size', 4),
        num_workers=eval_config.get('evaluation', {{}}).get('num_workers', 2)
    )

    # 执行评估
    logger.info(f"开始评估: {checkpoint_name}")
    results = evaluator.evaluate()

    # 保存结果
    evaluator.save_results(results)

    logger.info(f"✅ Dense评估完成: {checkpoint_name}")
    logger.info(f"  Loss: {{results['loss']:.4f}}")
    logger.info(f"  Perplexity: {{results['perplexity']:.2f}}")
    sys.exit(0)

except Exception as e:
    logger.error(f"❌ Dense评估失败: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    def _generate_multistage_eval_script(self, checkpoint_path: str, gpu_id: int, checkpoint_name: str) -> str:
        """生成multistage评估脚本"""
        return f'''
import os
# 设置CUDA环境变量以避免内存访问问题
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA执行，便于调试
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用设备端断言
# Triton/MegaBlocks 兼容性修复
os.environ['TRITON_DISABLE_LINE_INFO'] = '1'  # 禁用line info减少内存访问
os.environ['TRITON_KERNEL_CACHE_DIR'] = '/tmp/triton_cache'  # 使用显式缓存目录
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'  # 支持多代GPU架构
# CUDA内存分配优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 减少内存碎片

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path('/rna-multiverse')
sys.path.insert(0, str(project_root))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GPU{gpu_id} - %(message)s')
logger = logging.getLogger(__name__)

# 导入评估器
from training.eval.scripts.evaluate_multistage import MultistageEvaluator
from training.eval.scripts.config_loader import ConfigLoader

# 加载评估配置
logger.info("加载评估配置...")
config = ConfigLoader.get_eval_config(
    checkpoint_path='{checkpoint_path}',
    training_config_path='/rna-multiverse/configs/stage1_classification_16gpu.yaml',
    override_dict={{'distributed_eval': False}}
)

# 创建评估器
logger.info("创建评估器...")
evaluator = MultistageEvaluator(config, stage={self.eval_config.get('stage', 1)})

# 加载验证数据
logger.info("加载验证数据...")
val_dataloader = evaluator.load_validation_data()

# 评估checkpoint
try:
    logger.info(f"开始评估: {checkpoint_name}")
    result = evaluator.evaluate_checkpoint('{checkpoint_path}', val_dataloader)
    logger.info(f"✅ 评估完成: {checkpoint_name}")
    logger.info(f"  Loss: {{result['metrics']['val_loss']:.4f}}")
    logger.info(f"  Perplexity: {{result['metrics']['val_perplexity']:.2f}}")
    sys.exit(0)
except Exception as e:
    logger.error(f"❌ 评估失败: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    def run_single_checkpoint_eval(self, checkpoint_path: str, gpu_id: int):
        """
        在指定GPU上运行单个checkpoint评估（使用独立Python进程）
        支持本地GPU和远程GPU

        Args:
            checkpoint_path: checkpoint路径
            gpu_id: GPU ID (0-7为本地，8-15为远程)

        Returns:
            本地GPU: subprocess.Popen对象
            远程GPU: RemoteProcessInfo对象（或None）
        """
        # 提取checkpoint名称（如果路径以model_only结尾，取父目录名）
        path_obj = Path(checkpoint_path)
        if path_obj.name == 'model_only':
            checkpoint_name = path_obj.parent.name
        else:
            checkpoint_name = path_obj.name

        # 判断是本地还是远程GPU，确定实际CUDA设备ID
        is_remote = self.multinode and hasattr(self.gpu_manager, 'is_remote_gpu') and self.gpu_manager.is_remote_gpu(gpu_id)
        if is_remote:
            # 远程GPU：将虚拟ID（8-15）转换为实际ID（0-7）
            actual_gpu_id = self.gpu_manager.node1_config['actual_gpu_ids'][
                self.gpu_manager.remote_virtual_gpu_ids.index(gpu_id)
            ]
        else:
            # 本地GPU：直接使用gpu_id
            actual_gpu_id = gpu_id

        # 根据mode选择评估脚本（使用实际GPU ID）
        if self.mode == "lineage":
            eval_script_content = self._generate_lineage_eval_script(checkpoint_path, actual_gpu_id, checkpoint_name)
        elif self.mode == "dense":
            eval_script_content = self._generate_dense_eval_script(checkpoint_path, actual_gpu_id, checkpoint_name)
        else:
            eval_script_content = self._generate_multistage_eval_script(checkpoint_path, actual_gpu_id, checkpoint_name)

        # 判断是本地还是远程GPU
        if is_remote:
            # 远程GPU
            logger.info(f"远程GPU {gpu_id} 开始评估: {checkpoint_name}")

            success, process_info = self.gpu_manager.start_remote_eval_process(
                virtual_gpu_id=gpu_id,
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                eval_script_content=eval_script_content
            )

            if success:
                logger.info(f"  - 远程PID: {process_info.pid if process_info else 'Unknown'}")
                return process_info
            else:
                logger.error(f"  - 远程进程启动失败")
                return None
        else:
            # 本地GPU
            # 创建临时脚本文件
            temp_script_dir = self.output_dir / "temp_scripts"
            temp_script_dir.mkdir(parents=True, exist_ok=True)
            temp_script = temp_script_dir / f"eval_gpu{gpu_id}_{checkpoint_name}.py"

            with open(temp_script, 'w') as f:
                f.write(eval_script_content)

            # 日志文件
            log_dir = self.output_dir / "logs" / f"gpu_{gpu_id}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{checkpoint_name}.log"

            # 启动评估进程
            logger.info(f"本地GPU {gpu_id} 开始评估: {checkpoint_name}")

            if self.dry_run:
                logger.info(f"[DRY-RUN] 本地进程: python3 {temp_script}")
                # Dry-run: 返回模拟进程
                class DummyProcess:
                    def __init__(self, gpu_id):
                        self.pid = 77777 + gpu_id
                        self.returncode = None
                        self._start_time = time.time()
                    def poll(self):
                        if time.time() - self._start_time > 5:
                            self.returncode = 0
                        return self.returncode
                return DummyProcess(gpu_id)

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ['python3', str(temp_script)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent.parent)
                )

            logger.info(f"  - PID: {process.pid}")
            logger.info(f"  - 日志: {log_file}")

            return process

    def run(self):
        """运行并行评估"""
        mode_label = f"{self.num_gpus}GPU" + (" (多节点)" if self.multinode else "")
        logger.info("="*60)
        logger.info(f"开始{mode_label}并行评估")
        logger.info("="*60)
        logger.info(f"Checkpoint目录: {self.checkpoint_dir}")
        logger.info(f"Checkpoint数量: {len(self.checkpoints)}")
        logger.info(f"使用GPU数: {self.num_gpus}")
        logger.info(f"输出目录: {self.output_dir}")
        if self.multinode:
            logger.info(f"多节点模式: 本地8GPU + 远程8GPU")
        if self.dry_run:
            logger.info(f"⚠️ DRY-RUN模式：仅模拟，不实际执行")
        logger.info("="*60)

        # 添加checkpoint到任务队列
        self.gpu_manager.add_checkpoints(self.checkpoints)

        # 启动监控线程
        self.gpu_manager.start_monitoring()

        # 进程跟踪（支持本地和远程进程）
        active_processes: Dict[int, Any] = {}  # gpu_id -> process (subprocess.Popen or RemoteProcessInfo)
        active_tasks: Dict[int, CheckpointTask] = {}  # gpu_id -> task

        start_time = time.time()
        last_status_print = 0

        try:
            # 主调度循环
            while not self.gpu_manager.is_all_tasks_completed():
                # 检查已完成的进程
                for gpu_id in list(active_processes.keys()):
                    process = active_processes[gpu_id]
                    task = active_tasks[gpu_id]

                    # 检查进程是否结束（区分本地/远程）
                    is_completed = False
                    return_code = None

                    if self.multinode and hasattr(self.gpu_manager, 'is_remote_gpu') and self.gpu_manager.is_remote_gpu(gpu_id):
                        # 远程进程
                        is_running = self.gpu_manager.check_remote_process_status(gpu_id)
                        if is_running is False:  # 明确为False表示已结束
                            is_completed = True
                            return_code = 0  # 假设成功（可以后续增强错误检测）
                    else:
                        # 本地进程
                        return_code = process.poll()
                        if return_code is not None:
                            is_completed = True

                    if is_completed:
                        # 进程已结束

                        # 如果是远程GPU，做最终日志同步
                        if self.multinode and hasattr(self.gpu_manager, 'is_remote_gpu') and self.gpu_manager.is_remote_gpu(gpu_id):
                            logger.info(f"同步远程GPU {gpu_id} 最终日志...")
                            success, log_path = self.gpu_manager.sync_single_remote_log(gpu_id)
                            if success:
                                logger.info(f"  ✅ 日志已同步: {log_path}")
                            else:
                                logger.warning(f"  ⚠️ 日志同步失败: {log_path}")

                        if return_code == 0:
                            # 成功
                            result_file = self.output_dir / "experiments" / f"exp_*_{task.checkpoint_name}" / "checkpoint_metrics.json"
                            self.gpu_manager.complete_task(gpu_id, task, str(result_file))
                        else:
                            # 失败
                            error_msg = f"评估进程返回码: {return_code}"
                            self.gpu_manager.fail_task(gpu_id, task, error_msg)

                        # 清理
                        del active_processes[gpu_id]
                        del active_tasks[gpu_id]

                # 为空闲GPU分配新任务
                available_gpu = self.gpu_manager.get_available_gpu()
                if available_gpu is not None and available_gpu not in active_processes:
                    task = self.gpu_manager.assign_task_to_gpu(available_gpu)
                    if task:
                        # 启动评估进程
                        process = self.run_single_checkpoint_eval(task.checkpoint_path, available_gpu)
                        if process is not None:  # 可能启动失败
                            active_processes[available_gpu] = process
                            active_tasks[available_gpu] = task

                            # 设置PID
                            if hasattr(process, 'pid'):
                                self.gpu_manager.gpu_pool[available_gpu].process_id = process.pid
                            else:
                                # 远程进程
                                self.gpu_manager.gpu_pool[available_gpu].process_id = getattr(process, 'pid', None)

                # 定期打印状态和同步日志
                current_time = time.time()
                if current_time - last_status_print >= 30:  # 每30秒打印一次
                    self._print_status()

                    # 多节点模式下同步远程日志
                    if self.multinode and hasattr(self.gpu_manager, 'sync_remote_logs'):
                        self.gpu_manager.sync_remote_logs()

                    last_status_print = current_time

                # 短暂休眠
                time.sleep(2)

            # 所有任务完成
            total_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("🎉 所有评估任务完成！")
            logger.info("="*60)

            # 多节点模式：最终日志同步
            if self.multinode and hasattr(self.gpu_manager, 'sync_remote_logs'):
                logger.info("执行最终日志同步...")
                sync_results = self.gpu_manager.sync_remote_logs(force=True)
                success_count = sum(1 for success, _ in sync_results.values() if success)
                logger.info(f"最终日志同步: {success_count}/{len(sync_results)} 成功")

            # 最终统计
            summary = self.gpu_manager.get_status_summary()
            logger.info(f"总任务数: {summary['total_tasks']}")
            logger.info(f"成功完成: {summary['completed']}")
            logger.info(f"失败任务: {summary['failed']}")
            logger.info(f"总耗时: {total_time/60:.1f} 分钟")
            logger.info(f"平均速度: {summary['completed']/(total_time/60):.2f} checkpoints/分钟")
            logger.info(f"结果目录: {self.output_dir / 'experiments'}")
            logger.info("="*60)

            # 显示失败任务详情
            if summary['failed'] > 0:
                logger.warning("\n失败的任务:")
                for task in self.gpu_manager.get_failed_results():
                    logger.warning(f"  - {task.checkpoint_name}: {task.error_msg}")

        except KeyboardInterrupt:
            logger.warning("\n收到中断信号，正在停止所有任务...")
            # 终止所有活动进程
            for gpu_id, process in active_processes.items():
                logger.info(f"终止GPU {gpu_id}的评估进程...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"GPU {gpu_id}进程未响应，强制杀死...")
                    process.kill()

        finally:
            # 清理
            self.gpu_manager.shutdown()

    def _print_status(self):
        """打印当前状态"""
        summary = self.gpu_manager.get_status_summary()

        logger.info("\n" + "─"*60)
        logger.info(f"进度: {summary['completed']}/{summary['total_tasks']} ({summary['progress_percent']:.1f}%)")

        if summary['eta_minutes'] is not None:
            logger.info(f"预计剩余时间: {summary['eta_minutes']:.1f} 分钟")

        logger.info("\nGPU状态:")
        for gpu_detail in summary['gpu_details']:
            status_emoji = {
                'idle': '💤',
                'busy': '🔥',
                'completed': '✅',
                'error': '❌',
                'disabled': '🚫'
            }
            emoji = status_emoji.get(gpu_detail['status'], '❓')

            status_line = f"  GPU {gpu_detail['gpu_id']}: {emoji} {gpu_detail['status']}"

            if gpu_detail['current_checkpoint']:
                status_line += f" | {gpu_detail['current_checkpoint']}"

            # 显示batch进度（如果有）
            if gpu_detail.get('current_batch') and gpu_detail.get('total_batches'):
                current_batch = gpu_detail['current_batch']
                total_batches = gpu_detail['total_batches']
                batch_percent = (current_batch / total_batches * 100) if total_batches > 0 else 0
                status_line += f" | {current_batch}/{total_batches} ({batch_percent:.1f}%)"
            elif gpu_detail['progress'] > 0:
                # 回退到百分比显示（如果没有batch信息）
                status_line += f" | {gpu_detail['progress']*100:.1f}%"

            if gpu_detail['eta_minutes'] is not None:
                status_line += f" | ETA: {gpu_detail['eta_minutes']:.0f}min"

            status_line += f" | 完成: {gpu_detail['completed_count']}"

            logger.info(status_line)

        logger.info("─"*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多GPU并行评估调度器（支持单节点8GPU和双节点16GPU）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  1. 单节点8GPU评估:
     python parallel_eval_runner.py \\
         --checkpoint_dir /path/to/results/stage1_output \\
         --num_gpus 8 --mode lineage

  2. 双节点16GPU评估:
     python parallel_eval_runner.py \\
         --checkpoint_dir /path/to/results/stage1_output \\
         --multinode --mode lineage

  3. Dry-run测试（不实际执行）:
     python parallel_eval_runner.py \\
         --checkpoint_dir /path/to/results/stage1_output \\
         --multinode --dry-run --mode lineage

优势:
  - 单节点: 8倍加速
  - 双节点: 16倍加速
  - 单GPU DCP加载（自动合并16分片）
  - 智能任务调度（GPU池管理）
  - 实时进度监控（本地+远程）
        """
    )

    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint父目录或单个checkpoint路径')
    parser.add_argument('--training_config', type=str,
                       help='训练配置文件（可选，优先从checkpoint读取）')
    parser.add_argument('--override_config', type=str,
                       help='覆盖配置文件（可选）')
    parser.add_argument('--eval-config', type=str,
                       help='评估配置文件路径（可选，优先级最高，覆盖自动推断）')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='使用的GPU数量（默认: 8，多节点模式下自动为16）')
    parser.add_argument('--mode', type=str, default='multistage',
                       choices=['multistage', 'lineage', 'dense'],
                       help='评估模式（默认: multistage，可选: lineage, dense）')
    parser.add_argument('--multinode', action='store_true',
                       help='启用多节点模式（16GPU: 本地8+远程8）')
    parser.add_argument('--multinode-config', type=str,
                       help='多节点配置文件路径（可选）')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry-run模式：只打印命令，不实际执行（用于测试）')
    parser.add_argument('--force-reeval', action='store_true',
                       help='强制重新评估：忽略已有结果，重新评估所有checkpoint')

    args = parser.parse_args()

    try:
        runner = ParallelEvalRunner(
            checkpoint_dir=args.checkpoint_dir,
            training_config=args.training_config,
            override_config=args.override_config,
            eval_config_file=args.eval_config,
            num_gpus=args.num_gpus,
            mode=args.mode,
            multinode=args.multinode,
            multinode_config_file=args.multinode_config,
            dry_run=args.dry_run,
            force_reeval=args.force_reeval
        )

        runner.run()

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
