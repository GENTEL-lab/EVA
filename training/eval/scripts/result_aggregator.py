#!/usr/bin/env python3
"""
结果聚合系统
负责将多GPU并行评估的结果聚合为与单GPU评估兼容的格式
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 安全导入matplotlib和numpy，处理依赖问题
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    print(f"警告：numpy不可用 ({e})，将使用内置数学函数")
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用无GUI后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True and NUMPY_AVAILABLE  # 需要numpy支持
except ImportError as e:
    print(f"警告：matplotlib不可用 ({e})，将跳过图表生成")
    MATPLOTLIB_AVAILABLE = False
    plt = None

logger = logging.getLogger(__name__)


class ResultAggregator:
    """结果聚合器"""

    def __init__(self, output_dir: Path, stage: int):
        """
        初始化结果聚合器
        Args:
            output_dir: 输出目录
            stage: 训练阶段
        """
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        logger.info(f"结果聚合器初始化: 输出目录 {output_dir}, 阶段 {stage}")

    def normalize_result_format(self, result: Dict) -> Dict:
        """标准化结果格式，确保有metrics字段"""
        # 如果已经有metrics字段，直接返回
        if 'metrics' in result:
            return result
        
        # 否则，从扁平结构创建嵌套结构
        metrics = {}
        metric_fields = ['val_loss', 'val_perplexity', 'total_tokens', 'num_batches', 
                       'tokens_per_second', 'eval_time', 'load_time']
        
        for field in metric_fields:
            if field in result:
                metrics[field] = result[field]
        
        # 创建标准化的结果
        normalized = result.copy()
        normalized['metrics'] = metrics
        return normalized

    def aggregate_results(self, completed_tasks: List, failed_tasks: List):
        """
        聚合评估结果
        Args:
            completed_tasks: 完成的任务列表
            failed_tasks: 失败的任务列表
        """
        logger.info(f"开始聚合结果: {len(completed_tasks)} 个成功, {len(failed_tasks)} 个失败")

        try:
            # 收集所有结果数据
            all_results = []
            collected_results = self.collect_individual_results(completed_tasks)

            for task, result_data in collected_results:
                if result_data:
                    # 标准化结果格式
                    normalized_result = self.normalize_result_format(result_data)
                    all_results.append(normalized_result)

            if not all_results:
                logger.warning("没有有效的评估结果可以聚合")
                return

            # 排序结果（按checkpoint编号）
            all_results.sort(key=lambda x: self.extract_checkpoint_number(x.get('checkpoint_name', '')))

            # 生成聚合的JSON文件
            self.generate_checkpoint_metrics(all_results)

            # 生成可视化图表
            self.generate_plots(all_results)

            # 生成评估摘要
            self.generate_evaluation_summary(all_results, failed_tasks)

            # 生成兼容的日志文件
            self.generate_evaluation_log(all_results, completed_tasks, failed_tasks)

            logger.info("结果聚合完成")

        except Exception as e:
            logger.error(f"结果聚合失败: {e}")
            raise

    def collect_individual_results(self, completed_tasks: List) -> List[Tuple[Any, Dict]]:
        """收集各个GPU的评估结果"""
        collected_results = []

        for task in completed_tasks:
            try:
                # 读取结果文件
                result_data = self.load_result_file(task.result_file)
                if result_data:
                    # 如果是列表，取第一个（单个checkpoint的结果）
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result_data = result_data[0]

                    collected_results.append((task, result_data))
                    logger.debug(f"收集到结果: {task.checkpoint_name}")
                else:
                    logger.warning(f"无法读取结果文件: {task.result_file}")

            except Exception as e:
                logger.error(f"收集任务 {task.checkpoint_name} 结果失败: {e}")

        logger.info(f"收集到 {len(collected_results)} 个有效结果")
        return collected_results

    def load_result_file(self, result_file_path: str) -> Optional[Dict]:
        """加载结果文件"""
        if not result_file_path or not os.path.exists(result_file_path):
            return None

        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取结果文件失败 {result_file_path}: {e}")
            return None

    def extract_checkpoint_number(self, checkpoint_name: str) -> int:
        """从checkpoint名称提取编号"""
        try:
            if 'checkpoint-' in checkpoint_name:
                return int(checkpoint_name.split('checkpoint-')[1])
            return 0
        except (ValueError, IndexError):
            return 0

    def generate_checkpoint_metrics(self, all_results: List[Dict]):
        """生成checkpoint_metrics.json文件（与单GPU格式兼容）"""
        output_file = self.output_dir / "checkpoint_metrics.json"

        # 确保每个结果都有必要的字段
        processed_results = []
        for result in all_results:
            # 兼容两种格式：嵌套的metrics字段 或 扁平结构
            if 'metrics' in result:
                # 原有格式：有metrics字段
                metrics = result['metrics']
            else:
                # 扁平格式：从result中提取metrics相关字段
                metrics = {}
                metric_fields = ['val_loss', 'val_perplexity', 'total_tokens', 'num_batches', 
                               'tokens_per_second', 'eval_time', 'load_time']
                
                has_metrics = False
                for field in metric_fields:
                    if field in result:
                        metrics[field] = result[field]
                        has_metrics = True
                
                if not has_metrics:
                    logger.warning(f"结果缺少metrics字段: {result.get('checkpoint_name', 'unknown')}")
                    continue

            # 补充可能缺失的字段
            processed_result = {
                'checkpoint_path': result.get('checkpoint_path', ''),
                'checkpoint_name': result.get('checkpoint_name', ''),
                'stage': result.get('stage', self.stage),
                'evaluation_time': result.get('evaluation_time', metrics.get('eval_time', 0.0)),
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
                'metrics': metrics,
                'stage_config': result.get('stage_config', {}),
                'model_config': result.get('model_config', {})
            }

            # 确保metrics有必要字段
            if 'stage' not in processed_result['metrics']:
                processed_result['metrics']['stage'] = self.stage

            processed_results.append(processed_result)

        # 保存聚合结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)

        # 设置文件权限
        os.chmod(str(output_file), 0o666)

        logger.info(f"已生成checkpoint_metrics.json: {len(processed_results)} 个结果")

    def generate_plots(self, all_results: List[Dict]):
        """生成可视化图表"""
        if not MATPLOTLIB_AVAILABLE:
            logger.info("matplotlib不可用，跳过图表生成")
            return

        if len(all_results) < 1:
            logger.info("结果数量不足，跳过图表生成")
            return

        try:
            # 设置matplotlib使用无中文字体，避免乱码
            matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False

            # 提取数据用于绘图
            checkpoint_names = []
            steps = []
            losses = []
            perplexities = []

            for result in all_results:
                # 兼容两种格式：嵌套结构(metrics字段)和扁平结构(直接顶层)
                metrics = result.get('metrics', {})
                checkpoint_name = result.get('checkpoint_name', '')

                checkpoint_names.append(checkpoint_name)
                steps.append(self.extract_checkpoint_number(checkpoint_name))

                # 优先从metrics中取，如果没有则从顶层取
                val_loss = metrics.get('val_loss') or result.get('val_loss', 0.0)
                val_perplexity = metrics.get('val_perplexity') or result.get('val_perplexity', 0.0)

                losses.append(val_loss)
                perplexities.append(val_perplexity)

            # 生成Loss图表
            self._generate_loss_plot(steps, losses, checkpoint_names)

            # 生成Perplexity图表
            self._generate_perplexity_plot(steps, perplexities, checkpoint_names)

        except Exception as e:
            logger.error(f"生成图表失败: {e}")

    def _generate_loss_plot(self, steps, losses, checkpoint_names):
        """生成Loss图表"""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Stage {self.stage} Validation Loss Analysis', fontsize=16, fontweight='bold')

        # 左图: Loss趋势线
        ax1.plot(steps, losses, 'o-', linewidth=2, markersize=8, color='blue', markerfacecolor='lightblue', markeredgecolor='blue')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Loss Trend Over Training', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (step, loss) in enumerate(zip(steps, losses)):
            ax1.annotate(f'{loss:.3f}', (step, loss),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        # 右图: Checkpoint性能对比
        bars = ax2.bar(range(len(losses)), losses, alpha=0.7, color=['lightcoral' if i == losses.index(min(losses)) else 'lightsteelblue' for i in range(len(losses))])
        ax2.set_xlabel('Checkpoint', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Checkpoint Performance Comparison', fontsize=14)
        ax2.set_xticks(range(len(checkpoint_names)))
        ax2.set_xticklabels([name.split('-')[1] if '-' in name else name for name in checkpoint_names], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, loss) in enumerate(zip(bars, losses)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(losses) * 0.01, f'{loss:.3f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存Loss图表
        loss_plot_file = self.plots_dir / f'stage{self.stage}_validation_loss.png'
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 设置文件权限
        os.chmod(str(loss_plot_file), 0o666)
        logger.info(f"Loss图表已保存: {loss_plot_file}")

    def _generate_perplexity_plot(self, steps, perplexities, checkpoint_names):
        """生成Perplexity图表"""
        if not MATPLOTLIB_AVAILABLE:
            return

        # 过滤掉异常高的困惑度值
        valid_data = [(step, ppl, name, i) for i, (step, ppl, name) in enumerate(zip(steps, perplexities, checkpoint_names)) if ppl < 10000]

        if not valid_data:
            logger.warning("没有有效的困惑度数据可以绘制")
            return

        valid_steps, valid_ppls, valid_names, valid_indices = zip(*valid_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Stage {self.stage} Validation Perplexity Analysis', fontsize=16, fontweight='bold')

        # 左图: Perplexity趋势线
        ax1.plot(valid_steps, valid_ppls, 'o-', linewidth=2, markersize=8, color='orange', markerfacecolor='lightgoldenrodyellow', markeredgecolor='orange')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Validation Perplexity', fontsize=12)
        ax1.set_title('Perplexity Trend Over Training', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (step, ppl) in enumerate(zip(valid_steps, valid_ppls)):
            ax1.annotate(f'{ppl:.1f}', (step, ppl),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        # 右图: Checkpoint困惑度对比
        bars = ax2.bar(range(len(valid_ppls)), valid_ppls, alpha=0.7,
                      color=['lightgreen' if i == valid_indices[list(valid_ppls).index(min(valid_ppls))] else 'lightsalmon' for i in valid_indices])
        ax2.set_xlabel('Checkpoint', fontsize=12)
        ax2.set_ylabel('Validation Perplexity', fontsize=12)
        ax2.set_title('Checkpoint Perplexity Comparison', fontsize=14)
        ax2.set_xticks(range(len(valid_names)))
        ax2.set_xticklabels([name.split('-')[1] if '-' in name else name for name in valid_names], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, ppl) in enumerate(zip(bars, valid_ppls)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(valid_ppls) * 0.01, f'{ppl:.1f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存Perplexity图表
        ppl_plot_file = self.plots_dir / f'stage{self.stage}_validation_perplexity.png'
        plt.savefig(ppl_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 设置文件权限
        os.chmod(str(ppl_plot_file), 0o666)
        logger.info(f"Perplexity图表已保存: {ppl_plot_file}")

    def generate_evaluation_summary(self, all_results: List[Dict], failed_tasks: List):
        """生成评估摘要markdown文件"""
        if not all_results:
            logger.warning("没有结果可以生成摘要")
            return

        try:
            # 计算统计信息
            losses = [result['metrics']['val_loss'] for result in all_results]
            perplexities = [result['metrics']['val_perplexity'] for result in all_results]

            if NUMPY_AVAILABLE:
                best_idx = np.argmin(losses)
            else:
                best_idx = losses.index(min(losses))
            best_checkpoint = all_results[best_idx]

            # 获取阶段名称
            stage_names = {1: "CLM基础训练", 2: "分类训练", 3: "生成训练"}
            stage_name = stage_names.get(self.stage, f"阶段{self.stage}")

            # 生成摘要文本
            summary_text = f"""# 并行评估结果摘要 - 阶段{self.stage}

## 评估概览
- **评估模式**: 多GPU并行评估
- **阶段名称**: {stage_name}
- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总checkpoint数**: {len(all_results)}
- **失败checkpoint数**: {len(failed_tasks)}

## 最佳性能
- **最佳Checkpoint**: {best_checkpoint['checkpoint_name']}
- **验证损失**: {best_checkpoint['metrics']['val_loss']:.6f}
- **困惑度**: {best_checkpoint['metrics']['val_perplexity']:.3f}
- **处理token数**: {best_checkpoint['metrics'].get('total_tokens', 'N/A'):,}

## 训练进展
- **损失范围**: {min(losses):.6f} ~ {max(losses):.6f}
- **平均困惑度**: {(sum(perplexities)/len(perplexities) if perplexities else 0):.3f}
- **损失标准差**: {(sum((x - sum(losses)/len(losses))**2 for x in losses) / len(losses))**0.5 if len(losses) > 1 else 0:.6f}
- **总体改善**: {((losses[0] - losses[-1]) / losses[0] * 100 if len(losses) > 1 else 0):.2f}%

## 详细结果

| Checkpoint | Step | 验证损失 | 困惑度 | Token数 | 评估时间(s) |
|------------|------|----------|--------|---------|-------------|
"""

            # 添加详细结果表格
            for result in all_results:
                metrics = result['metrics']
                step = self.extract_checkpoint_number(result['checkpoint_name'])
                summary_text += f"| {result['checkpoint_name']} | {step} | {metrics['val_loss']:.6f} | {metrics['val_perplexity']:.3f} | {metrics.get('total_tokens', 'N/A'):,} | {result.get('evaluation_time', 0):.1f} |\n"

            # 添加失败信息
            if failed_tasks:
                summary_text += f"""
## 失败的Checkpoint

| Checkpoint | 失败原因 | 重试次数 |
|------------|----------|----------|
"""
                for task in failed_tasks:
                    summary_text += f"| {task.checkpoint_name} | {task.error_msg or 'Unknown'} | {task.retry_count} |\n"

            summary_text += f"""
## 生成的文件
- `checkpoint_metrics.json`: 完整的评估指标数据
- `plots/stage{self.stage}_evaluation_summary.png`: 可视化图表
- `evaluation.log`: 详细的评估日志

## 使用说明
这些结果与单GPU评估格式完全兼容，可以直接用于后续分析和比较。
"""

            # 保存摘要文件
            summary_file = self.output_dir / f'stage{self.stage}_evaluation_summary.md'
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_text)

            os.chmod(str(summary_file), 0o666)
            logger.info(f"评估摘要已保存: {summary_file}")

        except Exception as e:
            logger.error(f"生成评估摘要失败: {e}")

    def generate_evaluation_log(self, all_results: List[Dict], completed_tasks: List, failed_tasks: List):
        """生成兼容的evaluation.log文件"""
        try:
            log_file = self.output_dir / "evaluation.log"

            with open(log_file, 'w', encoding='utf-8') as f:
                # 写入日志头
                f.write(f"RNA-ProGen3 阶段{self.stage} 并行评估日志\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总checkpoint数: {len(all_results)}\n")
                f.write("=" * 80 + "\n\n")

                # 写入每个checkpoint的结果
                for i, result in enumerate(all_results):
                    metrics = result['metrics']
                    f.write(f"📊 评估Checkpoint {i+1}/{len(all_results)}: {result['checkpoint_name']}\n")
                    f.write(f"  验证损失: {metrics['val_loss']:.6f}\n")
                    f.write(f"  困惑度: {metrics['val_perplexity']:.3f}\n")
                    f.write(f"  Token数: {metrics.get('total_tokens', 'N/A'):,}\n")
                    f.write(f"  评估时间: {result.get('evaluation_time', 0):.1f}秒\n")
                    f.write("-" * 40 + "\n")

                # 写入失败信息
                if failed_tasks:
                    f.write(f"\n❌ 失败的checkpoint ({len(failed_tasks)} 个):\n")
                    for task in failed_tasks:
                        f.write(f"  - {task.checkpoint_name}: {task.error_msg or 'Unknown error'}\n")

                # 写入摘要
                if all_results:
                    losses = [result['metrics']['val_loss'] for result in all_results]
                    if NUMPY_AVAILABLE:
                        best_idx = np.argmin(losses)
                    else:
                        best_idx = losses.index(min(losses))
                    best_result = all_results[best_idx]

                    f.write(f"\n✅ 评估完成摘要:\n")
                    f.write(f"  最佳checkpoint: {best_result['checkpoint_name']}\n")
                    f.write(f"  最佳验证损失: {best_result['metrics']['val_loss']:.6f}\n")
                    f.write(f"  最佳困惑度: {best_result['metrics']['val_perplexity']:.3f}\n")
                    f.write(f"  损失改善: {((losses[0] - losses[-1]) / losses[0] * 100 if len(losses) > 1 else 0):.2f}%\n")

            os.chmod(str(log_file), 0o666)
            logger.info(f"评估日志已生成: {log_file}")

        except Exception as e:
            logger.error(f"生成评估日志失败: {e}")

    def copy_best_model_artifacts(self, all_results: List[Dict]):
        """复制最佳模型的相关文件（可选功能）"""
        if not all_results:
            return

        try:
            # 找到最佳checkpoint
            losses = [result['metrics']['val_loss'] for result in all_results]
            if NUMPY_AVAILABLE:
                best_idx = np.argmin(losses)
            else:
                best_idx = losses.index(min(losses))
            best_result = all_results[best_idx]

            best_checkpoint_path = Path(best_result['checkpoint_path'])
            if not best_checkpoint_path.exists():
                logger.warning(f"最佳checkpoint路径不存在: {best_checkpoint_path}")
                return

            # 创建best_model目录
            best_model_dir = self.output_dir / "best_model"
            best_model_dir.mkdir(exist_ok=True)

            # 复制重要文件
            files_to_copy = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            for filename in files_to_copy:
                src_file = best_checkpoint_path / filename
                if src_file.exists():
                    dst_file = best_model_dir / filename
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"复制最佳模型文件: {filename}")

            # 创建模型信息文件
            model_info = {
                'checkpoint_name': best_result['checkpoint_name'],
                'checkpoint_path': str(best_checkpoint_path),
                'val_loss': best_result['metrics']['val_loss'],
                'val_perplexity': best_result['metrics']['val_perplexity'],
                'total_tokens': best_result['metrics'].get('total_tokens'),
                'stage': self.stage,
                'selected_time': datetime.now().isoformat()
            }

            with open(best_model_dir / "model_info.json", 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            logger.info(f"最佳模型文件已复制到: {best_model_dir}")

        except Exception as e:
            logger.error(f"复制最佳模型文件失败: {e}")


def main():
    """测试结果聚合器"""
    import tempfile

    # 创建临时测试环境
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_output"
        output_dir.mkdir()

        # 创建聚合器
        aggregator = ResultAggregator(output_dir, stage=1)

        # 模拟一些测试数据
        class MockTask:
            def __init__(self, name, result_file):
                self.checkpoint_name = name
                self.result_file = result_file
                self.retry_count = 0
                self.error_msg = None

        # 创建模拟结果文件
        test_results = []
        for i in range(3):
            result_data = {
                'checkpoint_name': f'checkpoint-{(i+1)*1000}',
                'checkpoint_path': f'/path/to/checkpoint-{(i+1)*1000}',
                'stage': 1,
                'evaluation_time': 900.0 + i * 50,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'val_loss': 8.0 - i * 0.5,
                    'val_perplexity': 3000 - i * 500,
                    'total_tokens': 296900000 + i * 1000,
                    'num_batches': 235,
                    'stage': 1
                },
                'stage_config': {
                    'clm_ratio': 0.0,
                    'task_ratios': [0.0, 0.5, 0.5],
                    'use_multitask': True
                },
                'model_config': {
                    'model_type': 'rnagen',
                    'hidden_size': 768,
                    'num_hidden_layers': 12
                }
            }

            result_file = output_dir / f"result_{i}.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f)

            test_results.append(MockTask(f'checkpoint-{(i+1)*1000}', str(result_file)))

        # 运行聚合
        aggregator.aggregate_results(test_results, [])

        print(f"测试完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()