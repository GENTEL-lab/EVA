#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集并可视化Lineage评估结果
从各个checkpoint目录收集eval_results_*.json，汇总并生成可视化图表
"""

import os
import json
import argparse
import logging
import traceback
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import numpy as np
from scipy.interpolate import make_interp_spline, UnivariateSpline

# ============================================================================
# 可配置参数：训练曲线提前显示的步数
# ============================================================================
# 在绘制带提前步数的图表时，训练曲线会比验证曲线提前显示TRAINING_OFFSET步
# 例如：TRAINING_OFFSET=300 表示如果第一个验证点在step 1000，训练曲线从step 700开始显示
TRAINING_OFFSET = 300  # 默认提前300步，可根据需要修改
# ============================================================================

def parse_training_log(log_files: List[str]) -> Dict[str, List[float]]:
    """解析训练日志，提取训练步数、loss和perplexity
    
    支持多个日志文件（断点续训场景），会自动合并并按步数排序
    
    Args:
        log_files: 日志文件路径列表（单个文件也需要放在列表中）
    
    Returns:
        包含steps, losses, ar_losses, perplexities的字典
    """
    # 如果传入的是单个字符串，转换为列表
    if isinstance(log_files, str):
        log_files = [log_files]
    
    all_steps = []
    all_losses = []
    all_ar_losses = []
    all_perplexities = []

    # 正则表达式匹配日志行
    # 示例: Step 30: Loss=0.0000, LR=0.000000, Metrics={train/loss=4.3241, train/ar_loss=4.3086, train/perplexity=74.3394, ...}
    pattern = r'Step (\d+):.*train/loss=([\d.]+).*train/ar_loss=([\d.]+).*train/perplexity=([\d.]+)'

    # 逐个解析日志文件
    for idx, log_file in enumerate(log_files, 1):
        logging.info(f"📖 解析训练日志 [{idx}/{len(log_files)}]: {log_file}")
        steps_in_file = 0
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        step = int(match.group(1))
                        loss = float(match.group(2))
                        ar_loss = float(match.group(3))
                        ppl = float(match.group(4))

                        all_steps.append(step)
                        all_losses.append(loss)
                        all_ar_losses.append(ar_loss)
                        all_perplexities.append(ppl)
                        steps_in_file += 1

            logging.info(f"   ✅ 从该日志解析了 {steps_in_file} 个训练步骤")
        except Exception as e:
            logging.error(f"   ❌ 解析训练日志失败: {e}")
            continue
    
    # 按步数排序（处理可能的乱序情况）
    if all_steps:
        sorted_indices = sorted(range(len(all_steps)), key=lambda i: all_steps[i])
        all_steps = [all_steps[i] for i in sorted_indices]
        all_losses = [all_losses[i] for i in sorted_indices]
        all_ar_losses = [all_ar_losses[i] for i in sorted_indices]
        all_perplexities = [all_perplexities[i] for i in sorted_indices]
        
        logging.info(f"✅ 总共从 {len(log_files)} 个训练日志中解析了 {len(all_steps)} 个训练步骤")
        logging.info(f"   步数范围: {all_steps[0]} - {all_steps[-1]}")
    else:
        logging.warning("⚠️  未能从任何训练日志中解析到有效数据")
    
    return {
        'steps': all_steps,
        'losses': all_losses,
        'ar_losses': all_ar_losses,
        'perplexities': all_perplexities
    }


def collect_results(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """收集所有checkpoint的评估结果（仅CLM结果）"""
    results = []
    checkpoint_parent_dir = Path(checkpoint_dir)

    checkpoint_dirs = sorted(
        [d for d in checkpoint_parent_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[1])
    )
    logging.info(f"在 {checkpoint_dir} 中找到 {len(checkpoint_dirs)} 个checkpoint")

    for ckpt_dir in checkpoint_dirs:
        # 只收集CLM结果（eval_results_stage1.json 或 eval_results_stage2.json）
        eval_files = [f for f in ckpt_dir.glob('eval_results_stage*.json') if 'glm' not in f.name]
        if not eval_files:
            logging.warning(f"未找到CLM评估结果: {ckpt_dir.name}")
            continue

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    result = json.load(f)

                step = int(ckpt_dir.name.split('-')[1])
                results.append({
                    'step': step,
                    'loss': result.get('loss'),
                    'ppl': result.get('perplexity')
                })
                logging.info(f"✅ {ckpt_dir.name}: loss={result.get('loss'):.4f}, ppl={result.get('perplexity'):.2f}")
            except Exception as e:
                logging.error(f"读取或解析失败 {eval_file}: {e}")

    results.sort(key=lambda x: x['step'])
    return results


def collect_all_eval_results(checkpoint_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    收集所有类型的评估结果（CLM和GLM）

    返回格式:
    {
        'clm': [{'step': ..., 'loss': ..., 'ppl': ...}, ...],
        'glm_span10': [...],
        'glm_span50': [...],
        'glm_span200': [...],
    }
    """
    all_results = {
        'clm': [],
        'glm_span10': [],
        'glm_span50': [],
        'glm_span200': [],
    }

    checkpoint_parent_dir = Path(checkpoint_dir)

    checkpoint_dirs = sorted(
        [d for d in checkpoint_parent_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[1])
    )
    logging.info(f"在 {checkpoint_dir} 中找到 {len(checkpoint_dirs)} 个checkpoint")

    for ckpt_dir in checkpoint_dirs:
        step = int(ckpt_dir.name.split('-')[1])

        # 收集所有eval_results_*.json文件
        eval_files = list(ckpt_dir.glob('eval_results_*.json'))

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    result = json.load(f)

                result_data = {
                    'step': step,
                    'loss': result.get('loss'),
                    'ppl': result.get('perplexity')
                }

                filename = eval_file.name

                # 判断结果类型
                if 'glm_span10' in filename:
                    all_results['glm_span10'].append(result_data)
                    logging.debug(f"✅ {ckpt_dir.name} GLM span10: loss={result.get('loss'):.4f}")
                elif 'glm_span50' in filename:
                    all_results['glm_span50'].append(result_data)
                    logging.debug(f"✅ {ckpt_dir.name} GLM span50: loss={result.get('loss'):.4f}")
                elif 'glm_span200' in filename:
                    all_results['glm_span200'].append(result_data)
                    logging.debug(f"✅ {ckpt_dir.name} GLM span200: loss={result.get('loss'):.4f}")
                elif 'glm' not in filename:
                    # CLM结果
                    all_results['clm'].append(result_data)
                    logging.debug(f"✅ {ckpt_dir.name} CLM: loss={result.get('loss'):.4f}")

            except Exception as e:
                logging.error(f"读取或解析失败 {eval_file}: {e}")

    # 排序
    for key in all_results:
        all_results[key].sort(key=lambda x: x['step'])

    # 统计
    logging.info("=" * 60)
    logging.info("评估结果收集统计:")
    for key, results in all_results.items():
        if results:
            logging.info(f"  {key}: {len(results)} 个checkpoint")
    logging.info("=" * 60)

    return all_results

def save_summary_json(results: List[Dict[str, Any]], output_dir: str, all_results: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """保存汇总的JSON结果（包括CLM和GLM）"""
    summary_file = os.path.join(output_dir, 'eval_results_summary.json')

    # 构建包含所有结果的数据结构
    if all_results:
        summary_data = {
            'clm': results,  # CLM结果
            'glm_span10': all_results.get('glm_span10', []),
            'glm_span50': all_results.get('glm_span50', []),
            'glm_span200': all_results.get('glm_span200', []),
        }
    else:
        summary_data = {
            'clm': results,
            'glm_span10': [],
            'glm_span50': [],
            'glm_span200': [],
        }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logging.info(f"\n✅ 汇总结果已保存: {summary_file}")
    logging.info(f"   CLM: {len(results)} 个checkpoint")
    if all_results:
        for key in ['glm_span10', 'glm_span50', 'glm_span200']:
            if all_results.get(key):
                logging.info(f"   {key}: {len(all_results[key])} 个checkpoint")

def save_training_data_json(train_data: Dict[str, List[float]], output_dir: str):
    """保存训练数据到JSON文件"""
    if not train_data or not train_data['steps']:
        logging.warning("训练数据为空，跳过保存")
        return

    train_summary_file = os.path.join(output_dir, 'training_data_summary.json')

    # 构建保存的数据结构
    train_summary = {
        'num_steps': len(train_data['steps']),
        'step_range': {
            'min': int(min(train_data['steps'])),
            'max': int(max(train_data['steps']))
        },
        'loss_stats': {
            'min': float(np.min(train_data['losses'])),
            'max': float(np.max(train_data['losses'])),
            'mean': float(np.mean(train_data['losses'])),
            'std': float(np.std(train_data['losses']))
        },
        'ar_loss_stats': {
            'min': float(np.min(train_data['ar_losses'])),
            'max': float(np.max(train_data['ar_losses'])),
            'mean': float(np.mean(train_data['ar_losses'])),
            'std': float(np.std(train_data['ar_losses']))
        },
        'perplexity_stats': {
            'min': float(np.min(train_data['perplexities'])),
            'max': float(np.max(train_data['perplexities'])),
            'mean': float(np.mean(train_data['perplexities'])),
            'std': float(np.std(train_data['perplexities']))
        },
        'data_points': [
            {
                'step': int(step),
                'loss': float(loss),
                'ar_loss': float(ar_loss),
                'perplexity': float(ppl)
            }
            for step, loss, ar_loss, ppl in zip(
                train_data['steps'],
                train_data['losses'],
                train_data['ar_losses'],
                train_data['perplexities']
            )
        ]
    }

    with open(train_summary_file, 'w') as f:
        json.dump(train_summary, f, indent=2)

    logging.info(f"✅ 训练数据已保存: {train_summary_file}")
    logging.info(f"   共 {len(train_data['steps'])} 个训练步骤")
    logging.info(f"   Loss范围: {train_summary['loss_stats']['min']:.4f} - {train_summary['loss_stats']['max']:.4f}")
    logging.info(f"   Perplexity范围: {train_summary['perplexity_stats']['min']:.2f} - {train_summary['perplexity_stats']['max']:.2f}")


def visualize_results_line(results: List[Dict[str, Any]], output_dir: str, stage: str):
    """可视化评估结果（折线图）"""
    steps = [r['step'] for r in results]
    losses = [r['loss'] for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, losses, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(f'Lineage Evaluation - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 固定纵轴范围
    ax.set_ylim(1.20, 1.35)

    # 标注最低点
    min_loss_val = min(losses)
    best_step = steps[losses.index(min_loss_val)]
    ax.scatter(best_step, min_loss_val, color='red', zorder=5, label=f'Best: {min_loss_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.2f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ 折线图可视化结果已保存: {output_path}")

def visualize_results_fitted(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, glm_results: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """可视化评估结果（仅散点图，不含拟合曲线）

    Args:
        results: CLM评估结果
        output_dir: 输出目录
        stage: 阶段名称
        train_data: 训练数据（可选，不再使用）
        glm_results: GLM评估结果字典，格式 {'glm_span10': [...], 'glm_span50': [...], ...}
    """
    steps = np.array([r['step'] for r in results])
    losses = np.array([r['loss'] for r in results])

    if len(steps) < 1:
        logging.warning("无数据点，将跳过。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制CLM验证集散点
    ax.scatter(steps, losses, c='red', s=50, label='Val Loss (CLM)', zorder=5, edgecolors='darkred', linewidth=1)

    # 绘制GLM评估结果（如果有）- 只显示散点
    glm_plot_configs = {
        'glm_span10': {'color': 'green', 'label': 'GLM span=10', 'marker': 's'},
        'glm_span50': {'color': 'orange', 'label': 'GLM span=50', 'marker': '^'},
        'glm_span200': {'color': 'purple', 'label': 'GLM span=200', 'marker': 'd'},
    }
    has_glm = False
    if glm_results:
        for glm_type, glm_data in glm_results.items():
            if not glm_data or glm_type not in glm_plot_configs:
                continue
            has_glm = True
            config = glm_plot_configs[glm_type]
            glm_steps = np.array([r['step'] for r in glm_data])
            glm_losses = np.array([r['loss'] for r in glm_data])

            # 只绘制散点，不绘制曲线
            ax.scatter(glm_steps, glm_losses, c=config['color'], s=50, marker=config['marker'],
                       label=config['label'], alpha=0.8, edgecolors='black', linewidth=0.5, zorder=4)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if has_glm:
        ax.set_title(f'CLM & GLM Validation Loss - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Loss - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注CLM最低点
    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    ax.scatter(best_step, min_loss_val, color='gold', s=100, zorder=10, marker='*', label=f'Best CLM: {min_loss_val:.4f} at step {best_step}')

    # 为CLM数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.4f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}_fitted.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ 散点图可视化结果已保存: {output_path}")

def visualize_perplexity(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, glm_results: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """可视化困惑度（Perplexity）仅散点图

    Args:
        results: CLM评估结果
        output_dir: 输出目录
        stage: 阶段名称
        train_data: 训练数据（可选，不再使用）
        glm_results: GLM评估结果字典
    """
    steps = np.array([r['step'] for r in results])
    ppls = np.array([r['ppl'] for r in results])

    if len(steps) < 1:
        logging.warning("无数据点，将跳过Perplexity图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 只绘制CLM验证集散点
    ax.scatter(steps, ppls, marker='o', s=50, color='darkred', zorder=5, edgecolors='black', linewidth=1, label='Val PPL (CLM)')

    # 绘制GLM评估结果（如果有）- 只显示散点
    glm_plot_configs = {
        'glm_span10': {'color': 'green', 'label': 'GLM span=10', 'marker': 's'},
        'glm_span50': {'color': 'orange', 'label': 'GLM span=50', 'marker': '^'},
        'glm_span200': {'color': 'purple', 'label': 'GLM span=200', 'marker': 'd'},
    }
    has_glm = False
    if glm_results:
        for glm_type, glm_data in glm_results.items():
            if not glm_data or glm_type not in glm_plot_configs:
                continue
            has_glm = True
            config = glm_plot_configs[glm_type]
            glm_steps = np.array([r['step'] for r in glm_data])
            glm_ppls = np.array([r['ppl'] for r in glm_data])

            # 只绘制散点，不绘制曲线
            ax.scatter(glm_steps, glm_ppls, c=config['color'], s=50, marker=config['marker'],
                       label=config['label'], alpha=0.8, edgecolors='black', linewidth=0.5, zorder=4)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    if has_glm:
        ax.set_title(f'CLM & GLM Validation Perplexity - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Perplexity - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注CLM最低困惑度点
    min_ppl_val = np.min(ppls)
    min_ppl_idx = np.argmin(ppls)
    best_step = steps[min_ppl_idx]
    ax.scatter(best_step, min_ppl_val, color='gold', s=150, zorder=10, marker='*',
               label=f'Best CLM PPL: {min_ppl_val:.4f} at step {best_step}')

    # 为CLM数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{ppls[i]:.4f}',
                   xy=(steps[i], ppls[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"perplexity_{stage}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ Perplexity散点图已保存: {output_path}")


def generate_summary_text(results: List[Dict[str, Any]], output_dir: str, stage: str):
    """生成评估结果的文本摘要"""
    steps = np.array([res['step'] for res in results])
    losses = np.array([res['loss'] for res in results])
    ppls = np.array([res['ppl'] for res in results])

    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    summary = []
    summary.append("========================================")
    summary.append(f"Lineage评估摘要 - {stage}")
    summary.append("========================================")
    summary.append(f"共 {len(steps)} 个checkpoint的评估结果")
    summary.append("")
    summary.append("📈 性能趋势:")
    summary.append(f"  - 最低Loss: {min_loss_val:.4f}")
    summary.append(f"  - 对应困惑度(PPL): {ppls[min_loss_idx]:.2f}")
    summary.append(f"  - 最佳Checkpoint: step {best_step}")
    summary.append("")
    summary.append(f"  - 起始Loss: {losses[0]:.4f} (step {steps[0]})")
    summary.append(f"  - 最终Loss: {losses[-1]:.4f} (step {steps[-1]})")
    summary.append("")
    summary.append("💡 结论:")
    if min_loss_idx == 0:
        summary.append("  - ⚠️ 模型性能从一开始就在下降，请检查训练设置。")
    elif min_loss_idx == len(losses) - 1:
        summary.append("  - ✅ 模型仍在持续改进，可以考虑继续训练。")
    else:
        summary.append(f"  - ⚠️ 模型可能在 step {best_step} 之后开始过拟合。")
    summary.append("========================================")

    output_path = os.path.join(output_dir, f"eval_summary_{stage}.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    logging.info(f"✅ 摘要文件已保存: {output_path}")

def visualize_results_fitted_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化评估结果（曲线拟合，训练曲线提前offset步开始）"""
    steps = np.array([r['step'] for r in results])
    losses = np.array([r['loss'] for r in results])

    if len(steps) < 4:
        logging.warning("数据点少于4个，无法进行平滑曲线拟合，将跳过带offset的Loss图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，先绘制训练集曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ar_losses = np.array(train_data['ar_losses'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ar_losses_filtered = train_ar_losses[mask]

        logging.info(f"[带offset] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            loss_std = np.std(train_ar_losses_filtered)

            # 第一条曲线：浅蓝色，适度平滑
            smoothing_factor_light = n * (loss_std * 0.5) ** 2

            # 第二条曲线：深蓝色，更平滑
            smoothing_factor_smooth = n * (loss_std * 1.0) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ar_losses_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                losses_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, losses_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train AR Loss')

                # 深蓝色更平滑曲线
                spl_smooth = UnivariateSpline(train_steps_filtered, train_ar_losses_filtered, k=3, s=smoothing_factor_smooth)
                losses_smooth_deep = spl_smooth(steps_smooth)
                ax.plot(steps_smooth, losses_smooth_deep, color='darkblue', linewidth=2.5, alpha=0.8, label='Train AR Loss (smooth)')

                logging.info(f"✅ [带offset] 训练集Loss曲线已添加 (提前{offset}步)")
            except Exception as e:
                logging.warning(f"[带offset] 训练集Loss平滑失败: {e}")
        else:
            logging.warning(f"[带offset] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    n = len(steps)
    loss_std = np.std(losses)
    smoothing_factor = n * (loss_std * 0.5) ** 2

    logging.info(f"[带offset] 验证集平滑参数: n={n}, loss_std={loss_std:.6f}, smoothing_factor={smoothing_factor:.6f}")

    try:
        spl = UnivariateSpline(steps, losses, k=3, s=smoothing_factor)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = spl(steps_smooth)

        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)
        logging.info("✅ [带offset] 使用 UnivariateSpline 生成验证集平滑趋势曲线")
    except Exception as e:
        logging.warning(f"[带offset] UnivariateSpline拟合失败: {e}，使用多项式拟合")
        degree = min(5, n - 1)
        poly_coeffs = np.polyfit(steps, losses, degree)
        poly_func = np.poly1d(poly_coeffs)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = poly_func(steps_smooth)
        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)

    ax.scatter(steps, losses, c='red', s=50, label='Val Data Points', zorder=5, edgecolors='darkred', linewidth=1)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training & Validation Loss (Train Offset: -{offset} steps) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低点
    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    ax.scatter(best_step, min_loss_val, color='orange', s=100, zorder=10, marker='*', label=f'Best: {min_loss_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.2f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}_fitted_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [带offset] 曲线拟合可视化结果已保存: {output_path}")

def visualize_perplexity_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化困惑度（Perplexity）折线图（训练曲线提前offset步开始）"""
    steps = np.array([r['step'] for r in results])
    ppls = np.array([r['ppl'] for r in results])

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，先绘制训练集perplexity曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ppls = np.array(train_data['perplexities'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ppls_filtered = train_ppls[mask]

        logging.info(f"[带offset] 训练Perplexity数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            ppl_std = np.std(train_ppls_filtered)

            # 第一条曲线：浅蓝色，适度平滑
            smoothing_factor_light = n * (ppl_std * 0.5) ** 2

            # 第二条曲线：深蓝色，更平滑
            smoothing_factor_smooth = n * (ppl_std * 1.0) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ppls_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                ppls_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, ppls_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train Perplexity')

                # 深蓝色更平滑曲线
                spl_smooth = UnivariateSpline(train_steps_filtered, train_ppls_filtered, k=3, s=smoothing_factor_smooth)
                ppls_smooth_deep = spl_smooth(steps_smooth)
                ax.plot(steps_smooth, ppls_smooth_deep, color='darkblue', linewidth=2.5, alpha=0.8, label='Train Perplexity (smooth)')

                logging.info(f"✅ [带offset] 训练集Perplexity曲线已添加 (提前{offset}步)")
            except Exception as e:
                logging.warning(f"[带offset] 训练集Perplexity平滑失败: {e}")
        else:
            logging.warning(f"[带offset] 裁剪后训练Perplexity数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    if len(steps) >= 4:
        n = len(steps)
        ppl_std = np.std(ppls)
        smoothing_factor = n * (ppl_std * 0.5) ** 2

        try:
            spl = UnivariateSpline(steps, ppls, k=3, s=smoothing_factor)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = spl(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
            logging.info(f"✅ [带offset] Perplexity平滑曲线: n={n}, ppl_std={ppl_std:.6f}, s={smoothing_factor:.6f}")
        except Exception as e:
            logging.warning(f"[带offset] Perplexity曲线平滑失败: {e}，使用多项式拟合")
            degree = min(5, n - 1)
            poly_coeffs = np.polyfit(steps, ppls, degree)
            poly_func = np.poly1d(poly_coeffs)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = poly_func(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
    else:
        ax.plot(steps, ppls, color='red', linewidth=2, alpha=0.7, label='Val Perplexity')

    # 绘制原始数据点
    ax.scatter(steps, ppls, marker='o', s=50, color='darkred', zorder=5, edgecolors='black', linewidth=1, label='Val Data Points')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title(f'Training & Validation Perplexity (Train Offset: -{offset} steps) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低困惑度点
    min_ppl_val = np.min(ppls)
    min_ppl_idx = np.argmin(ppls)
    best_step = steps[min_ppl_idx]
    ax.scatter(best_step, min_ppl_val, color='red', s=150, zorder=10, marker='*',
               label=f'Best PPL: {min_ppl_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{ppls[i]:.2f}',
                   xy=(steps[i], ppls[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"perplexity_{stage}_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [带offset] 困惑度折线图已保存: {output_path}")

def visualize_results_fitted_val_only(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None):
    """可视化评估结果（曲线拟合，保留浅蓝色训练曲线，不显示深蓝色训练曲线）"""
    steps = np.array([r['step'] for r in results])
    losses = np.array([r['loss'] for r in results])

    if len(steps) < 4:
        logging.warning("数据点少于4个，无法进行平滑曲线拟合，将跳过仅验证集的Loss图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，绘制浅蓝色训练集曲线（但不绘制深蓝色的更平滑曲线）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ar_losses = np.array(train_data['ar_losses'])

        # 裁剪训练数据：从第一个验证点的step开始
        first_val_step = steps.min()
        mask = train_steps >= first_val_step
        train_steps_filtered = train_steps[mask]
        train_ar_losses_filtered = train_ar_losses[mask]

        logging.info(f"[仅验证集] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {first_val_step})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            loss_std = np.std(train_ar_losses_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (loss_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ar_losses_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                losses_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, losses_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train AR Loss')

                logging.info(f"✅ [仅验证集] 训练集Loss曲线已添加 (浅蓝色 s={smoothing_factor_light:.6f})")
            except Exception as e:
                logging.warning(f"[仅验证集] 训练集Loss平滑失败: {e}")
        else:
            logging.warning(f"[仅验证集] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    n = len(steps)
    loss_std = np.std(losses)
    smoothing_factor = n * (loss_std * 0.5) ** 2

    logging.info(f"[仅验证集] 验证集平滑参数: n={n}, loss_std={loss_std:.6f}, smoothing_factor={smoothing_factor:.6f}")

    try:
        spl = UnivariateSpline(steps, losses, k=3, s=smoothing_factor)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = spl(steps_smooth)

        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)
        logging.info("✅ [仅验证集] 使用 UnivariateSpline 生成验证集平滑趋势曲线")
    except Exception as e:
        logging.warning(f"[仅验证集] UnivariateSpline拟合失败: {e}，使用多项式拟合")
        degree = min(5, n - 1)
        poly_coeffs = np.polyfit(steps, losses, degree)
        poly_func = np.poly1d(poly_coeffs)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = poly_func(steps_smooth)
        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)

    ax.scatter(steps, losses, c='red', s=50, label='Val Data Points', zorder=5, edgecolors='darkred', linewidth=1)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if train_data and train_data['steps']:
        ax.set_title(f'Training & Validation Loss (Light Train Curve) - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Loss - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低点
    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    ax.scatter(best_step, min_loss_val, color='orange', s=100, zorder=10, marker='*', label=f'Best: {min_loss_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.2f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}_fitted_val_only.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [仅验证集] 曲线拟合可视化结果已保存: {output_path}")

def visualize_results_fitted_val_only_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化评估结果（曲线拟合，训练曲线提前offset步，保留浅蓝色训练曲线，不显示深蓝色训练曲线）"""
    steps = np.array([r['step'] for r in results])
    losses = np.array([r['loss'] for r in results])

    if len(steps) < 4:
        logging.warning("数据点少于4个，无法进行平滑曲线拟合，将跳过仅验证集带offset的Loss图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，先绘制浅蓝色训练集曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ar_losses = np.array(train_data['ar_losses'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ar_losses_filtered = train_ar_losses[mask]

        logging.info(f"[仅验证集-offset] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            loss_std = np.std(train_ar_losses_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (loss_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ar_losses_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                losses_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, losses_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train AR Loss')

                logging.info(f"✅ [仅验证集-offset] 训练集Loss曲线已添加 (提前{offset}步)")
            except Exception as e:
                logging.warning(f"[仅验证集-offset] 训练集Loss平滑失败: {e}")
        else:
            logging.warning(f"[仅验证集-offset] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    n = len(steps)
    loss_std = np.std(losses)
    smoothing_factor = n * (loss_std * 0.5) ** 2

    logging.info(f"[仅验证集-offset] 验证集平滑参数: n={n}, loss_std={loss_std:.6f}, smoothing_factor={smoothing_factor:.6f}")

    try:
        spl = UnivariateSpline(steps, losses, k=3, s=smoothing_factor)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = spl(steps_smooth)

        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)
        logging.info("✅ [仅验证集-offset] 使用 UnivariateSpline 生成验证集平滑趋势曲线")
    except Exception as e:
        logging.warning(f"[仅验证集-offset] UnivariateSpline拟合失败: {e}，使用多项式拟合")
        degree = min(5, n - 1)
        poly_coeffs = np.polyfit(steps, losses, degree)
        poly_func = np.poly1d(poly_coeffs)
        steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        losses_smooth = poly_func(steps_smooth)
        ax.plot(steps_smooth, losses_smooth, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)

    ax.scatter(steps, losses, c='red', s=50, label='Val Data Points', zorder=5, edgecolors='darkred', linewidth=1)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training & Validation Loss (Light Train, Offset: -{offset} steps) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低点
    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    ax.scatter(best_step, min_loss_val, color='orange', s=100, zorder=10, marker='*', label=f'Best: {min_loss_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.2f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}_fitted_val_only_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [仅验证集-offset] 曲线拟合可视化结果已保存: {output_path}")

def visualize_perplexity_val_only(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None):
    """可视化困惑度（Perplexity）折线图（仅验证集，保留浅蓝色训练曲线）"""
    steps = np.array([r['step'] for r in results])
    ppls = np.array([r['ppl'] for r in results])

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，绘制浅蓝色训练集困惑度曲线（但不绘制深蓝色的更平滑曲线）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ppls = np.array(train_data['perplexities'])

        # 裁剪训练数据：从第一个验证点的step开始
        first_val_step = steps.min()
        mask = train_steps >= first_val_step
        train_steps_filtered = train_steps[mask]
        train_ppls_filtered = train_ppls[mask]

        logging.info(f"[仅验证集] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {first_val_step})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            ppl_std = np.std(train_ppls_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (ppl_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ppls_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                ppls_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, ppls_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train Perplexity')

                logging.info(f"✅ [仅验证集] 训练集Perplexity曲线已添加 (浅蓝色 s={smoothing_factor_light:.6f})")
            except Exception as e:
                logging.warning(f"[仅验证集] 训练集Perplexity平滑失败: {e}")
        else:
            logging.warning(f"[仅验证集] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    if len(steps) >= 4:
        n = len(steps)
        ppl_std = np.std(ppls)
        smoothing_factor = n * (ppl_std * 0.5) ** 2

        try:
            spl = UnivariateSpline(steps, ppls, k=3, s=smoothing_factor)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = spl(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
            logging.info(f"✅ [仅验证集] Perplexity平滑曲线: n={n}, ppl_std={ppl_std:.6f}, s={smoothing_factor:.6f}")
        except Exception as e:
            logging.warning(f"[仅验证集] Perplexity曲线平滑失败: {e}，使用多项式拟合")
            degree = min(5, n - 1)
            poly_coeffs = np.polyfit(steps, ppls, degree)
            poly_func = np.poly1d(poly_coeffs)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = poly_func(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
    else:
        ax.plot(steps, ppls, color='red', linewidth=2, alpha=0.7, label='Val Perplexity')

    # 绘制原始数据点
    ax.scatter(steps, ppls, marker='o', s=50, color='darkred', zorder=5, edgecolors='black', linewidth=1, label='Val Data Points')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    if train_data and train_data['steps']:
        ax.set_title(f'Training & Validation Perplexity (Light Train Curve) - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Perplexity (Val Only) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 自动调整纵轴范围
    ax.set_ylim(3.3, 3.8)

    # 标注最低困惑度点
    min_ppl_val = np.min(ppls)
    min_ppl_idx = np.argmin(ppls)
    best_step = steps[min_ppl_idx]
    ax.scatter(best_step, min_ppl_val, color='red', s=150, zorder=10, marker='*',
               label=f'Best PPL: {min_ppl_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{ppls[i]:.2f}',
                   xy=(steps[i], ppls[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"perplexity_{stage}_val_only.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [仅验证集] 困惑度折线图已保存: {output_path}")

def visualize_perplexity_val_only_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化困惑度（Perplexity）折线图（仅验证集，带offset，保留浅蓝色训练曲线）"""
    steps = np.array([r['step'] for r in results])
    ppls = np.array([r['ppl'] for r in results])

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，先绘制浅蓝色训练集困惑度曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ppls = np.array(train_data['perplexities'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ppls_filtered = train_ppls[mask]

        logging.info(f"[仅验证集-offset] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            ppl_std = np.std(train_ppls_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (ppl_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ppls_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                ppls_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, ppls_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train Perplexity')

                logging.info(f"✅ [仅验证集-offset] 训练集Perplexity曲线已添加 (浅蓝色 s={smoothing_factor_light:.6f}, offset={offset})")
            except Exception as e:
                logging.warning(f"[仅验证集-offset] 训练集Perplexity平滑失败: {e}")
        else:
            logging.warning(f"[仅验证集-offset] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 验证集平滑曲线
    if len(steps) >= 4:
        n = len(steps)
        ppl_std = np.std(ppls)
        smoothing_factor = n * (ppl_std * 0.5) ** 2

        try:
            spl = UnivariateSpline(steps, ppls, k=3, s=smoothing_factor)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = spl(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
            logging.info(f"✅ [仅验证集-offset] Perplexity平滑曲线: n={n}, ppl_std={ppl_std:.6f}, s={smoothing_factor:.6f}")
        except Exception as e:
            logging.warning(f"[仅验证集-offset] Perplexity曲线平滑失败: {e}，使用多项式拟合")
            degree = min(5, n - 1)
            poly_coeffs = np.polyfit(steps, ppls, degree)
            poly_func = np.poly1d(poly_coeffs)
            steps_smooth = np.linspace(steps.min(), steps.max(), 500)
            ppls_smooth = poly_func(steps_smooth)
            ax.plot(steps_smooth, ppls_smooth, color='red', linewidth=2.5, alpha=0.7, label='Val Perplexity')
    else:
        ax.plot(steps, ppls, color='red', linewidth=2, alpha=0.7, label='Val Perplexity')

    # 绘制原始数据点
    ax.scatter(steps, ppls, marker='o', s=50, color='darkred', zorder=5, edgecolors='black', linewidth=1, label='Val Data Points')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    if train_data and train_data['steps']:
        ax.set_title(f'Training & Validation Perplexity (Light Train Curve, Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Perplexity (Val Only, Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低困惑度点
    min_ppl_val = np.min(ppls)
    min_ppl_idx = np.argmin(ppls)
    best_step = steps[min_ppl_idx]
    ax.scatter(best_step, min_ppl_val, color='red', s=150, zorder=10, marker='*',
               label=f'Best PPL: {min_ppl_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{ppls[i]:.2f}',
                   xy=(steps[i], ppls[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"perplexity_{stage}_val_only_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [仅验证集-offset] 困惑度折线图已保存: {output_path}")

def visualize_results_fitted_train_only_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化评估结果（只显示训练曲线+验证散点，不显示验证拟合曲线，带offset）"""
    steps = np.array([r['step'] for r in results])
    losses = np.array([r['loss'] for r in results])

    if len(steps) < 4:
        logging.warning("数据点少于4个，无法绘制，将跳过训练曲线+验证散点图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，绘制浅蓝色训练集曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ar_losses = np.array(train_data['ar_losses'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ar_losses_filtered = train_ar_losses[mask]

        logging.info(f"[训练曲线+验证散点-offset] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            loss_std = np.std(train_ar_losses_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (loss_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ar_losses_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                losses_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, losses_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train AR Loss')

                logging.info(f"✅ [训练曲线+验证散点-offset] 训练集Loss曲线已添加 (浅蓝色 s={smoothing_factor_light:.6f}, offset={offset})")
            except Exception as e:
                logging.warning(f"[训练曲线+验证散点-offset] 训练集Loss平滑失败: {e}")
        else:
            logging.warning(f"[训练曲线+验证散点-offset] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 只绘制验证集散点，不绘制拟合曲线
    ax.scatter(steps, losses, c='red', s=50, label='Val Data Points', zorder=5, edgecolors='darkred', linewidth=1)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if train_data and train_data['steps']:
        ax.set_title(f'Training Curve & Validation Points (Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Points Only (Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低点
    min_loss_val = np.min(losses)
    min_loss_idx = np.argmin(losses)
    best_step = steps[min_loss_idx]

    ax.scatter(best_step, min_loss_val, color='orange', s=100, zorder=10, marker='*', label=f'Best: {min_loss_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{losses[i]:.2f}',
                   xy=(steps[i], losses[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"eval_results_{stage}_train_curve_val_points_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [训练曲线+验证散点-offset] 可视化结果已保存: {output_path}")

def visualize_perplexity_train_only_with_offset(results: List[Dict[str, Any]], output_dir: str, stage: str, train_data: Optional[Dict[str, List[float]]] = None, offset: int = TRAINING_OFFSET):
    """可视化困惑度（只显示训练曲线+验证散点，不显示验证拟合曲线，带offset）"""
    steps = np.array([r['step'] for r in results])
    ppls = np.array([r['ppl'] for r in results])

    if len(steps) < 4:
        logging.warning("数据点少于4个，无法绘制，将跳过训练曲线+验证散点图。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 如果提供了训练数据，先绘制浅蓝色训练集困惑度曲线（提前offset步）
    if train_data and train_data['steps']:
        train_steps = np.array(train_data['steps'])
        train_ppls = np.array(train_data['perplexities'])

        # 裁剪训练数据：从（第一个验证点 - offset）开始
        first_val_step = steps.min()
        train_start_step = max(0, first_val_step - offset)
        mask = train_steps >= train_start_step
        train_steps_filtered = train_steps[mask]
        train_ppls_filtered = train_ppls[mask]

        logging.info(f"[训练曲线+验证散点-offset] 训练数据裁剪: 原始{len(train_steps)}点 -> 裁剪后{len(train_steps_filtered)}点 (起始step: {train_start_step}, offset={offset})")

        if len(train_steps_filtered) >= 4:
            n = len(train_steps_filtered)
            ppl_std = np.std(train_ppls_filtered)

            # 只绘制浅蓝色适度平滑曲线
            smoothing_factor_light = n * (ppl_std * 0.5) ** 2

            try:
                # 浅蓝色适度平滑曲线
                spl_light = UnivariateSpline(train_steps_filtered, train_ppls_filtered, k=3, s=smoothing_factor_light)
                steps_smooth = np.linspace(train_steps_filtered.min(), train_steps_filtered.max(), 500)
                ppls_smooth_light = spl_light(steps_smooth)
                ax.plot(steps_smooth, ppls_smooth_light, color='lightblue', linewidth=1.5, alpha=0.5, label='Train Perplexity')

                logging.info(f"✅ [训练曲线+验证散点-offset] 训练集Perplexity曲线已添加 (浅蓝色 s={smoothing_factor_light:.6f}, offset={offset})")
            except Exception as e:
                logging.warning(f"[训练曲线+验证散点-offset] 训练集Perplexity平滑失败: {e}")
        else:
            logging.warning(f"[训练曲线+验证散点-offset] 裁剪后训练数据点不足({len(train_steps_filtered)}个)，无法绘制平滑曲线")

    # 只绘制验证集散点，不绘制拟合曲线
    ax.scatter(steps, ppls, marker='o', s=50, color='darkred', zorder=5, edgecolors='black', linewidth=1, label='Val Data Points')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    if train_data and train_data['steps']:
        ax.set_title(f'Training Curve & Validation Points (Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Validation Points Only (Offset {offset}) - {stage}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最低困惑度点
    min_ppl_val = np.min(ppls)
    min_ppl_idx = np.argmin(ppls)
    best_step = steps[min_ppl_idx]
    ax.scatter(best_step, min_ppl_val, color='red', s=150, zorder=10, marker='*',
               label=f'Best PPL: {min_ppl_val:.2f} at step {best_step}')

    # 为每个数据点添加数值标注
    for i in range(len(steps)):
        ax.annotate(f'{ppls[i]:.2f}',
                   xy=(steps[i], ppls[i]),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"perplexity_{stage}_train_curve_val_points_offset{offset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"✅ [训练曲线+验证散点-offset] 困惑度可视化结果已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="收集并可视化Lineage评估结果")
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint父目录')
    parser.add_argument('--output-dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--stage', type=str, required=True, help='训练阶段名称')
    parser.add_argument('--training-log', type=str, nargs='+', default=None, help='训练日志文件路径（可选，支持多个日志文件）')
    parser.add_argument('--training-offset', type=int, default=TRAINING_OFFSET, help=f'训练曲线提前显示的步数（默认: {TRAINING_OFFSET}）')
    args = parser.parse_args()

    # 使用命令行参数或默认值
    training_offset = args.training_offset

    # 设置日志
    log_file_path = os.path.join(args.output_dir, 'visualization.log')
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info("============================================================")
    logging.info("Lineage评估结果收集与可视化")
    logging.info("============================================================")
    logging.info(f"Checkpoint目录: {args.checkpoint_dir}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"训练阶段: {args.stage}")
    if args.training_log:
        logging.info(f"训练日志数量: {len(args.training_log)}")
        for idx, log in enumerate(args.training_log, 1):
            logging.info(f"  [{idx}] {log}")
        logging.info(f"训练曲线提前步数: {training_offset}")
    logging.info("============================================================")

    try:
        results = collect_results(args.checkpoint_dir)
        if not results:
            logging.error("未能收集到任何有效的评估结果，程序终止。")
            sys.exit(1)

        # 解析训练日志（如果提供）
        train_data = None
        if args.training_log:
            logging.info("============================================================")
            logging.info("解析训练日志以添加训练曲线")
            logging.info("============================================================")
            train_data = parse_training_log(args.training_log)

            if not train_data['steps']:
                logging.warning("⚠️  训练日志解析失败或无有效数据，将仅绘制验证集曲线。")
                train_data = None

        # 收集GLM评估结果（在保存之前先收集）
        all_results = collect_all_eval_results(args.checkpoint_dir)
        glm_results = {k: v for k, v in all_results.items() if 'glm' in k and v}

        # 保存汇总JSON（包括CLM和GLM）
        save_summary_json(results, args.output_dir, all_results)

        # 保存训练数据（如果有）
        if train_data and train_data['steps']:
            save_training_data_json(train_data, args.output_dir)
        if glm_results:
            logging.info(f"✅ 找到GLM评估结果: {list(glm_results.keys())}")

        visualize_results_line(results, args.output_dir, args.stage)
        visualize_results_fitted(results, args.output_dir, args.stage, train_data, glm_results)
        visualize_perplexity(results, args.output_dir, args.stage, train_data, glm_results)
        generate_summary_text(results, args.output_dir, args.stage)

        # 生成仅验证集的图表（无训练曲线）
        logging.info("============================================================")
        logging.info("生成仅验证集的图表（保留浅蓝色训练曲线，不显示深蓝色训练曲线）")
        logging.info("============================================================")
        visualize_results_fitted_val_only(results, args.output_dir, args.stage, train_data)
        visualize_results_fitted_val_only_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)
        visualize_perplexity_val_only(results, args.output_dir, args.stage, train_data)
        visualize_perplexity_val_only_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)

        # 如果提供了训练数据，生成带offset的额外图表
        if train_data and train_data['steps']:
            logging.info("============================================================")
            logging.info(f"生成带offset的额外图表 (训练曲线提前 {training_offset} 步)")
            logging.info("============================================================")
            visualize_results_fitted_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)
            visualize_perplexity_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)

            logging.info("============================================================")
            logging.info(f"生成训练曲线+验证散点图表 (不含验证拟合曲线，offset={training_offset})")
            logging.info("============================================================")
            visualize_results_fitted_train_only_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)
            visualize_perplexity_train_only_with_offset(results, args.output_dir, args.stage, train_data, offset=training_offset)

        logging.info("✅ 所有任务成功完成！")
    except Exception as e:
        logging.error(f"处理失败: {e}")
        logging.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
