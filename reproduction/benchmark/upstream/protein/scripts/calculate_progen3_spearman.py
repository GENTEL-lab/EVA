#!/usr/bin/env python3
"""
计算ProGen3模型结果的Spearman相关系数

将ProGen3的log_likelihood与DMS_score进行匹配，计算每个数据集的Spearman相关系数。
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


def calculate_spearman_for_dataset(progen3_scores_file, dms_file):
    """
    计算单个数据集的Spearman相关系数

    Args:
        progen3_scores_file: ProGen3分数文件路径
        dms_file: DMS数据文件路径

    Returns:
        dict: 包含spearman_corr, p_value, n_samples等信息
    """
    # 读取ProGen3分数
    progen3_df = pd.read_csv(progen3_scores_file)

    # 读取DMS数据
    dms_df = pd.read_csv(dms_file)

    # 从sequence_id中提取索引: {dataset_name}+{index}
    progen3_df['idx'] = progen3_df['sequence_id'].str.split('+').str[-1].astype(int)

    # 按索引排序
    progen3_df = progen3_df.sort_values('idx').reset_index(drop=True)

    # 验证索引匹配
    expected_indices = set(range(len(dms_df)))
    actual_indices = set(progen3_df['idx'].tolist())

    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        if missing:
            print(f"  警告: 缺少 {len(missing)} 个索引", file=sys.stderr)
        if extra:
            print(f"  警告: 多余 {len(extra)} 个索引", file=sys.stderr)

    # 构建匹配的数据
    matched_dms_scores = []
    matched_model_scores = []

    for _, row in progen3_df.iterrows():
        idx = row['idx']
        if 0 <= idx < len(dms_df):
            matched_dms_scores.append(dms_df.iloc[idx]['DMS_score'])
            matched_model_scores.append(row['log_likelihood'])

    # 过滤NaN值
    valid_pairs = [(dms, model) for dms, model in zip(matched_dms_scores, matched_model_scores)
                   if not (np.isnan(dms) or np.isnan(model))]

    if len(valid_pairs) < 2:
        return {
            'spearman_corr': None,
            'p_value': None,
            'n_samples': len(valid_pairs),
            'n_total': len(dms_df),
            'status': 'insufficient_data'
        }

    valid_dms = [p[0] for p in valid_pairs]
    valid_model = [p[1] for p in valid_pairs]

    spearman_corr, p_value = spearmanr(valid_dms, valid_model)

    return {
        'spearman_corr': float(spearman_corr) if not np.isnan(spearman_corr) else None,
        'p_value': float(p_value) if not np.isnan(p_value) else None,
        'n_samples': len(valid_pairs),
        'n_total': len(dms_df),
        'status': 'success'
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='计算ProGen3 Spearman相关系数')
    parser.add_argument('--model', type=str, default='progen3-1b',
                        choices=['progen3-1b', 'progen3-3b'],
                        help='模型名称')
    parser.add_argument('--scores_dir', type=str, default=None,
                        help='ProGen3分数目录 (默认根据model自动设置)')
    parser.add_argument('--dms_dir', type=str,
                        default='/data4/huangyanjie/rna_benchmark/DMS_ProteinGym_substitutions',
                        help='DMS数据目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (默认自动生成)')
    args = parser.parse_args()

    # 设置默认路径
    if args.scores_dir is None:
        args.scores_dir = f'/data4/huangyanjie/rna_benchmark/progen3_models/results/scores/{args.model}'

    if args.output is None:
        args.output = f'/data4/huangyanjie/rna_benchmark/progen3_models/results/{args.model}_spearman_results.csv'

    print("=" * 80)
    print(f"ProGen3 Spearman相关系数计算")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"分数目录: {args.scores_dir}")
    print(f"DMS目录: {args.dms_dir}")
    print(f"输出文件: {args.output}")
    print("=" * 80)

    # 获取所有分数文件
    scores_dir = Path(args.scores_dir)
    dms_dir = Path(args.dms_dir)

    if not scores_dir.exists():
        print(f"错误: 分数目录不存在: {scores_dir}")
        sys.exit(1)

    score_files = sorted(scores_dir.glob('*.csv'))
    print(f"\n找到 {len(score_files)} 个分数文件")

    results = []

    for score_file in tqdm(score_files, desc="计算Spearman相关系数"):
        dataset_name = score_file.stem
        dms_file = dms_dir / f"{dataset_name}.csv"

        if not dms_file.exists():
            print(f"\n警告: DMS文件不存在: {dms_file}")
            results.append({
                'dataset': dataset_name,
                'spearman_corr': None,
                'p_value': None,
                'n_samples': 0,
                'n_total': 0,
                'status': 'dms_file_missing'
            })
            continue

        try:
            result = calculate_spearman_for_dataset(score_file, dms_file)
            result['dataset'] = dataset_name
            results.append(result)
        except Exception as e:
            print(f"\n错误处理 {dataset_name}: {e}")
            results.append({
                'dataset': dataset_name,
                'spearman_corr': None,
                'p_value': None,
                'n_samples': 0,
                'n_total': 0,
                'status': f'error: {str(e)}'
            })

    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)

    # 重新排列列
    cols = ['dataset', 'spearman_corr', 'p_value', 'n_samples', 'n_total', 'status']
    results_df = results_df[cols]

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)

    successful = results_df[results_df['status'] == 'success']
    print(f"成功计算: {len(successful)}/{len(results_df)} 个数据集")

    if len(successful) > 0:
        mean_corr = successful['spearman_corr'].mean()
        median_corr = successful['spearman_corr'].median()
        std_corr = successful['spearman_corr'].std()
        min_corr = successful['spearman_corr'].min()
        max_corr = successful['spearman_corr'].max()

        print(f"\nSpearman相关系数统计:")
        print(f"  平均值: {mean_corr:.4f}")
        print(f"  中位数: {median_corr:.4f}")
        print(f"  标准差: {std_corr:.4f}")
        print(f"  最小值: {min_corr:.4f}")
        print(f"  最大值: {max_corr:.4f}")

        # 统计正/负相关数量
        n_positive = (successful['spearman_corr'] > 0).sum()
        n_negative = (successful['spearman_corr'] < 0).sum()
        print(f"\n  正相关数据集: {n_positive}")
        print(f"  负相关数据集: {n_negative}")

        # 显著性统计 (p < 0.05)
        significant = successful[successful['p_value'] < 0.05]
        print(f"\n  显著相关 (p<0.05): {len(significant)}/{len(successful)}")

    print(f"\n结果已保存到: {args.output}")

    # 保存JSON统计摘要
    stats_file = args.output.replace('.csv', '_summary.json')
    if len(successful) > 0:
        summary = {
            'model': args.model,
            'total_datasets': len(results_df),
            'successful_datasets': len(successful),
            'mean_spearman': float(mean_corr),
            'median_spearman': float(median_corr),
            'std_spearman': float(std_corr),
            'min_spearman': float(min_corr),
            'max_spearman': float(max_corr),
            'n_positive_corr': int(n_positive),
            'n_negative_corr': int(n_negative),
            'n_significant': int(len(significant)),
        }
        with open(stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"统计摘要已保存到: {stats_file}")

    print("\n计算完成!")


if __name__ == '__main__':
    main()
