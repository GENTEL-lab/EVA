#!/usr/bin/env python3
"""
ESMC模型 Domain DMS评估脚本

使用ESMC模型计算Domain DMS数据集上的突变效果预测，并与实验DMS分数计算相关性。

用法:
    python evaluate_dms_esmc_domain.py --model esmc_300m --dms_file <path> --output <path> [options]

示例:
    python evaluate_dms_esmc_domain.py \
        --model esmc_300m \
        --dms_file /data4/huangyanjie/rna_benchmark/Domainone_benchmark/top10_datasets/ckpt25500_top10/O15151_PF00641_299.csv \
        --output results.csv \
        --device cuda:1
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


class ESMCDomainEvaluator:
    """ESMC模型Domain DMS评估器"""

    def __init__(self, model_name: str = "esmc_300m", device: str = None, batch_size: int = 1):
        """
        初始化评估器

        Args:
            model_name: ESMC模型名称
            device: 计算设备
            batch_size: 批处理大小（ESMC通常逐序列处理）
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"使用设备: {self.device}")
        print(f"正在加载模型 {model_name}...")
        start_time = time.time()

        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒\n")

    @torch.no_grad()
    def calculate_sequence_score(self, seq: str) -> float:
        """
        计算单个序列的得分

        使用平均logit置信度作为序列质量评分
        这比完整的PLL快得多，且在DMS任务上相关性较好

        Args:
            seq: 蛋白质序列

        Returns:
            序列得分（平均logit置信度）
        """
        try:
            # 编码序列
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.model.encode(protein)

            # 获取 logits
            logits_output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=False)
            )

            # 计算每个位置的最大 logit（置信度）
            logits = logits_output.logits.sequence[0]  # (seq_len, vocab_size)
            max_logits = torch.max(logits, dim=-1)[0]  # (seq_len,)

            # 返回平均置信度
            return max_logits.mean().item()

        except Exception as e:
            print(f"警告: 序列计算失败: {e}", file=sys.stderr)
            return float('nan')

    def evaluate_dms_dataset(
        self,
        dms_file: str,
        max_sequences: int = None,
        use_wild_type: bool = False
    ) -> Tuple[List[float], List[float], Dict]:
        """
        评估DMS数据集

        Args:
            dms_file: DMS CSV文件路径
            max_sequences: 最多处理的序列数（用于测试）
            use_wild_type: 是否计算相对于野生型的delta分数

        Returns:
            (dms_scores, model_scores, metadata)
        """
        print(f"加载DMS数据: {dms_file}")

        # 读取DMS数据
        df = pd.read_csv(dms_file)

        if max_sequences:
            df = df.head(max_sequences)

        print(f"总序列数: {len(df)}")

        dms_scores = df['DMS_score'].tolist()
        sequences = df['mutated_sequence'].tolist()

        # 如果需要计算delta分数，先计算野生型分数
        wild_type_score = None
        if use_wild_type and 'mutated_sequence' in df.columns:
            # 假设第一个序列是参考序列，或者从文件中提取
            # 这里简化处理，使用所有序列的平均作为基线
            print("注意: 未实现野生型delta计算，使用绝对分数")

        # 计算模型分数
        print("计算模型分数...")
        model_scores = []

        for seq in tqdm(sequences, desc="评估序列"):
            score = self.calculate_sequence_score(seq)
            model_scores.append(score)

        # 计算Spearman相关性
        valid_pairs = [(dms, model) for dms, model in zip(dms_scores, model_scores)
                       if not np.isnan(model)]

        if len(valid_pairs) < 2:
            print(f"警告: 有效预测数量不足 ({len(valid_pairs)}个)")
            spearman_corr = float('nan')
            p_value = float('nan')
        else:
            valid_dms = [p[0] for p in valid_pairs]
            valid_model = [p[1] for p in valid_pairs]
            spearman_corr, p_value = spearmanr(valid_dms, valid_model)

        metadata = {
            'dms_file': os.path.basename(dms_file),
            'total_sequences': len(sequences),
            'valid_predictions': len(valid_pairs),
            'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
        }

        return dms_scores, model_scores, metadata


def parse_args():
    parser = argparse.ArgumentParser(description='ESMC模型 Domain DMS评估')
    parser.add_argument('--model', type=str, default='esmc_300m',
                        help='ESMC模型名称 (default: esmc_300m)')
    parser.add_argument('--dms_file', type=str, required=True,
                        help='DMS CSV文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出CSV文件路径')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='计算设备 (default: cuda:1)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='最多处理的序列数（用于测试）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小 (default: 1)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ESMC模型 Domain DMS评估")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"DMS文件: {args.dms_file}")
    print(f"输出文件: {args.output}")
    print(f"设备: {args.device}")
    print("=" * 80)

    # 初始化评估器
    evaluator = ESMCDomainEvaluator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # 评估DMS数据集
    dms_scores, model_scores, metadata = evaluator.evaluate_dms_dataset(
        args.dms_file,
        max_sequences=args.max_sequences
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    print(f"DMS文件: {metadata['dms_file']}")
    print(f"总序列数: {metadata['total_sequences']}")
    print(f"有效预测数: {metadata['valid_predictions']}")
    if metadata['spearman_correlation'] is not None:
        print(f"Spearman相关系数: {metadata['spearman_correlation']:.4f}")
        print(f"P值: {metadata['p_value']:.6f}")
    else:
        print("Spearman相关系数: N/A (数据不足)")
    print("=" * 80)

    # 保存详细结果
    print(f"\n保存结果到 {args.output}...")

    # 读取原始DMS数据以获取mutant列
    df = pd.read_csv(args.dms_file)
    if args.max_sequences:
        df = df.head(args.max_sequences)

    output_df = pd.DataFrame({
        'mutant': df['mutant'].tolist() if 'mutant' in df.columns else [''] * len(dms_scores),
        'sequence': df['mutated_sequence'].tolist(),
        'DMS_score': dms_scores,
        'model_score': model_scores
    })

    output_df.to_csv(args.output, index=False)

    # 保存统计信息
    stats_file = args.output.replace('.csv', '_stats.json')
    metadata['model'] = args.model
    metadata['device'] = args.device

    with open(stats_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"结果已保存到: {args.output}")
    print(f"统计信息已保存到: {stats_file}")
    print("\n评估完成!")


if __name__ == '__main__':
    main()
