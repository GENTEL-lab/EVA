#!/usr/bin/env python3
"""
位置消融实验：测试 premature stop codon 在不同位置的 AUROC

实验设计：
- 在 CDS 的 5 个不同位置插入 5 个串联终止密码子
- 位置：5%, 25%, 50%, 75%, 95%
- 原始 essentiality benchmark 日志显示插入点为第 12 个核苷酸之后，
  本实验改用相对位置做机制消融
- 计算每个位置的 AUROC
- 如果各位置 AUROC 相近，说明 EVA 捕捉的是全局基因功能约束，而非 NMD 信号

使用方法：
    python position_ablation.py --checkpoint /path/to/checkpoint --device cuda:1 --sample 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from usage_helpers import load_model, prepare_input_ids


# ============== 配置 ==============
MUTATION = "UAAUAAUAAUAGUGA"  # 与原始日志一致的5个串联终止密码子
POSITIONS = [0.05, 0.25, 0.50, 0.75, 0.95]  # 插入位置（相对位置）
POSITION_NAMES = ["5%", "25%", "50%", "75%", "95%"]


def get_label(item: dict) -> int:
    """兼容细菌(label)和真核(essential)数据集的标签字段。"""
    if 'label' in item:
        return int(item['label'])
    if 'essential' in item:
        return int(bool(item['essential']))
    raise KeyError("item must contain 'label' or 'essential'")


def insert_mutation_at_position(sequence: str, position_ratio: float, mutation: str) -> str:
    """在序列的指定相对位置插入突变序列"""
    seq_len = len(sequence)
    # 计算插入位置（确保是整数索引）
    insert_pos = max(1, min(int(seq_len * position_ratio), seq_len - 1))
    return sequence[:insert_pos] + mutation + sequence[insert_pos:]


def compute_log_likelihood(model, tokenizer, sequence: str, device: str) -> float:
    """计算序列的 log-likelihood（sum，不归一化）"""
    inputs = prepare_input_ids(tokenizer, sequence, device)

    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
        if model_dtype == torch.bfloat16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    sequence_ids=inputs['sequence_ids']
                )
        else:
            outputs = model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                sequence_ids=inputs['sequence_ids']
            )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    input_ids = inputs['input_ids'][0]
    token_log_probs = []

    for i in range(len(input_ids) - 1):
        predicted_token = input_ids[i + 1]
        log_prob = log_probs[0, i, predicted_token].item()
        token_log_probs.append(log_prob)

    total_ll = sum(token_log_probs) if token_log_probs else 0.0

    # 清理 GPU 内存
    del inputs, outputs, logits, log_probs
    torch.cuda.empty_cache()

    return total_ll


def compute_delta_ll_at_position(model, tokenizer, sequence: str, position_ratio: float,
                                  mutation: str, device: str) -> dict:
    """计算在指定位置的 ΔLL"""
    # 计算野生型 LL
    wt_ll = compute_log_likelihood(model, tokenizer, sequence, device)

    # 生成突变序列并计算 LL
    mutant_seq = insert_mutation_at_position(sequence, position_ratio, mutation)
    mut_ll = compute_log_likelihood(model, tokenizer, mutant_seq, device)

    delta_ll = wt_ll - mut_ll

    return {
        'wt_ll': wt_ll,
        'mut_ll': mut_ll,
        'delta_ll': delta_ll
    }


def calc_auroc(items: list) -> float:
    """计算 AUROC"""
    if len(items) < 20:
        return None
    labels = [x['label'] for x in items]
    scores = [x['delta_ll'] for x in items]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos == 0 or n_neg == 0:
        return None
    try:
        return roc_auc_score(labels, scores)
    except:
        return None


def load_and_sample_data(dataset_path: str, sample_per_species: int = 100,
                         random_seed: int = 42) -> list:
    """加载数据集并按物种采样"""
    print(f"加载数据集: {dataset_path}")
    with open(dataset_path, 'r') as f:
        full_data = json.load(f)
    print(f"总基因数: {len(full_data)}")

    # 按物种分组
    by_organism = defaultdict(list)
    for item in full_data:
        if not item.get('sequence'):
            continue
        item['label'] = get_label(item)
        by_organism[item['organism']].append(item)

    print(f"物种数: {len(by_organism)}")

    # 按物种采样
    random.seed(random_seed)
    sampled_data = []
    for org, items in by_organism.items():
        # 确保每个物种至少采样 10 个 essential 和 10 个 non-essential
        essential = [x for x in items if x['label'] == 1]
        non_essential = [x for x in items if x['label'] == 0]

        n_sample = min(sample_per_species // 2, len(essential), len(non_essential))
        if n_sample < 10:
            # 样本太少，跳过该物种
            continue

        sampled = random.sample(essential, n_sample) + random.sample(non_essential, n_sample)
        sampled_data.extend(sampled)

    print(f"采样后基因数: {len(sampled_data)}")

    # 统计
    n_essential = sum(1 for x in sampled_data if x['label'] == 1)
    n_non_essential = len(sampled_data) - n_essential
    print(f"  Essential: {n_essential}")
    print(f"  Non-essential: {n_non_essential}")

    return sampled_data


def run_position_ablation(model, tokenizer, data: list, device: str,
                          max_seqlen: int = 8192) -> dict:
    """运行位置消融实验"""
    results = {pos_name: [] for pos_name in POSITION_NAMES}

    total = len(data)
    start_time = time.time()

    for i, item in enumerate(data):
        sequence = item['sequence']

        # 截断过长的序列（与原始实验一致）
        if len(sequence) > max_seqlen:
            sequence = sequence[:max_seqlen]

        result_item = {
            'gene_name': item.get('gene_name') or item.get('gene') or '',
            'locus_tag': item.get('locus_tag', ''),
            'organism': item['organism'],
            'label': get_label(item),
            'length': len(sequence)
        }

        # 计算各位置的 ΔLL
        for pos_ratio, pos_name in zip(POSITIONS, POSITION_NAMES):
            pos_result = compute_delta_ll_at_position(
                model, tokenizer, sequence, pos_ratio, MUTATION, device
            )
            result_item[f'delta_ll_{pos_name}'] = pos_result['delta_ll']
            result_item['delta_ll'] = pos_result['delta_ll']
            results[pos_name].append(result_item.copy())

        # 打印进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"进度: {i+1}/{total} ({100*(i+1)/total:.1f}%) - "
                  f"预计剩余时间: {remaining:.0f}秒")

    return results


def analyze_results(results: dict) -> dict:
    """分析实验结果，计算各位置的 AUROC"""
    analysis = {}

    for pos_name in POSITION_NAMES:
        items = results[pos_name]

        # 按物种计算 AUROC
        by_organism = defaultdict(list)
        for item in items:
            by_organism[item['organism']].append(item)

        species_aurocs = []
        for org, org_items in by_organism.items():
            auroc = calc_auroc(org_items)
            if auroc is not None:
                species_aurocs.append({
                    'organism': org,
                    'auroc': auroc,
                    'n_genes': len(org_items)
                })

        # 计算整体 AUROC
        overall_auroc = calc_auroc(items)

        analysis[pos_name] = {
            'overall_auroc': overall_auroc,
            'mean_auroc': np.mean([x['auroc'] for x in species_aurocs]) if species_aurocs else None,
            'std_auroc': np.std([x['auroc'] for x in species_aurocs]) if species_aurocs else None,
            'n_species': len(species_aurocs),
            'per_species': species_aurocs
        }

    return analysis


def print_results(analysis: dict):
    """打印结果"""
    print("\n" + "=" * 70)
    print("位置消融实验结果")
    print("=" * 70)
    print(f"\n突变序列: {MUTATION}")
    print(f"插入位置: {POSITION_NAMES}")
    print()

    print(f"{'位置':<10} {'整体AUROC':<12} {'平均AUROC':<12} {'标准差':<12} {'物种数':<8}")
    print("-" * 54)
    for pos_name in POSITION_NAMES:
        a = analysis[pos_name]
        overall = f"{a['overall_auroc']:.4f}" if a['overall_auroc'] else "N/A"
        mean = f"{a['mean_auroc']:.4f}" if a['mean_auroc'] else "N/A"
        std = f"{a['std_auroc']:.4f}" if a['std_auroc'] else "N/A"
        print(f"{pos_name:<10} {overall:<12} {mean:<12} {std:<12} {a['n_species']:<8}")

    print("\n" + "=" * 70)

    # 结论分析
    aurocs = [analysis[pos]['overall_auroc'] for pos in POSITION_NAMES
              if analysis[pos]['overall_auroc'] is not None]

    if len(aurocs) >= 3:
        auroc_std = np.std(aurocs)
        auroc_range = max(aurocs) - min(aurocs)

        print("\n结论分析:")
        print(f"  AUROC 范围: {max(aurocs):.4f} - {min(aurocs):.4f} = {auroc_range:.4f}")
        print(f"  AUROC 标准差: {auroc_std:.4f}")

        if auroc_std < 0.02:
            print("\n  ✓ 各位置 AUROC 基本一致（std < 0.02）")
            print("  → EVA 捕捉的是全局基因功能约束，而非位置依赖的 NMD 信号")
        elif auroc_range < 0.05:
            print("\n  ✓ 各位置 AUROC 差异较小（range < 0.05）")
            print("  → 结果支持 EVA 捕捉的是全局基因功能约束")
        else:
            print("\n  ! 各位置 AUROC 存在差异")
            print("  → 可能存在位置依赖效应，需要进一步分析")


def main():
    parser = argparse.ArgumentParser(description='位置消融实验：premature stop codon 位置对 AUROC 的影响')

    parser.add_argument('--checkpoint', type=str,
                       default='/data/yanjie_huang/eva/EVA_checkpoint/30M_1124/checkpoint-56844',
                       help='模型 checkpoint 路径')
    parser.add_argument('--device', type=str, default='cuda:1', help='计算设备')
    parser.add_argument('--dataset', type=str,
                       default='/data/yanjie_huang/rna_benchmark/interpretability/zeroshot_essentiality/output/essentiality_dataset.json',
                       help='essentiality 数据集路径')
    parser.add_argument('--sample', type=int, default=100,
                       help='每个物种采样数量（默认100）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max-seqlen', type=int, default=8192, help='最大序列长度')
    parser.add_argument('--output', type=str, default=None,
                       help='结果输出路径（默认不保存）')

    args = parser.parse_args()

    print("=" * 70)
    print("位置消融实验：Premature Stop Codon 位置对 AUROC 的影响")
    print("=" * 70)
    print(f"模型: {args.checkpoint}")
    print(f"设备: {args.device}")
    print(f"数据集: {args.dataset}")
    print(f"每物种采样数: {args.sample}")
    print(f"随机种子: {args.seed}")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载数据
    data = load_and_sample_data(args.dataset, args.sample, args.seed)

    if not data:
        print("错误: 没有足够的样本数据")
        sys.exit(1)

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device
    print(f"模型加载完成，设备: {device}")

    # 运行实验
    print("\n开始位置消融实验...")
    results = run_position_ablation(model, tokenizer, data, args.device, args.max_seqlen)

    # 分析结果
    print("\n分析结果...")
    analysis = analyze_results(results)
    print_results(analysis)

    # 保存结果
    if args.output:
        output = {
            'config': {
                'mutation': MUTATION,
                'positions': POSITION_NAMES,
                'position_ratios': POSITIONS,
                'sample_per_species': args.sample,
                'checkpoint': args.checkpoint
            },
            'analysis': analysis,
            'raw_results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n结果已保存到: {args.output}")

    print("\n实验完成!")


if __name__ == '__main__':
    main()
