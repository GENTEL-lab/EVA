#!/usr/bin/env python3
"""
使用ESM模型评估ProteinGym DMS数据集 - Docker版本

在docker容器内运行ESM变异预测评估

使用方法 (在宿主机上):
    # 单个数据集
    docker exec rnagen4 python /data/benchmark_protein_dms/evaluate_dms_esm_docker.py \
        --model_path /data/esm_models/esm1v_t33_650M_UR90S_1.pt \
        --dms_file /data/DMS_ProteinGym_substitutions/A0A140D2T1_ZIKV_Sourisseau_2019.csv \
        --output /data/benchmark_protein_dms/test_results/A0A140D2T1_ZIKV_esm.csv \
        --device cuda:0
"""

import argparse
import csv
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import torch

# ESM is installed in the container
try:
    import esm
    from esm import pretrained
except ImportError as e:
    print(f"Error importing ESM: {e}")
    print("Please ensure fair-esm is installed: pip install fair-esm")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用ESM模型评估ProteinGym DMS数据集 (Zero-shot Variant Prediction)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='ESM模型路径(.pt文件)或预训练模型名称(如esm1v_t33_650M_UR90S_1)'
    )

    parser.add_argument(
        '--dms_file',
        type=str,
        required=True,
        help='DMS CSV文件路径(ProteinGym格式,包含mutant和mutated_sequence列)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出结果CSV文件路径'
    )

    parser.add_argument(
        '--reference_file',
        type=str,
        default='/data/ProteinGym/reference_files/DMS_substitutions.csv',
        help='ProteinGym参考文件路径'
    )

    parser.add_argument(
        '--scoring_strategy',
        type=str,
        default='wt-marginals',
        choices=['wt-marginals', 'masked-marginals'],
        help='评分策略: wt-marginals(快速) 或 masked-marginals(更准确但更慢,默认: wt-marginals)'
    )

    parser.add_argument(
        '--max_sequences',
        type=int,
        default=None,
        help='最多处理的序列数量(用于快速测试,默认处理全部)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备 (默认: cuda if available else cpu)'
    )

    parser.add_argument(
        '--nogpu',
        action='store_true',
        help='强制使用CPU'
    )

    return parser.parse_args()


def load_esm_model(model_path: str, device: str = 'cuda'):
    """
    加载ESM模型

    Args:
        model_path: 模型路径或预训练模型名称
        device: 计算设备

    Returns:
        (model, alphabet)
    """
    print(f"Loading ESM model from {model_path}...")

    # 检查是否是.pt文件路径
    if model_path.endswith('.pt') and os.path.exists(model_path):
        # 从本地文件加载
        print(f"Loading from local file: {model_path}")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        # 尝试作为预训练模型名称加载
        print(f"Loading pretrained model: {model_path}")
        try:
            model, alphabet = pretrained.load_model_and_alphabet(model_path)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            raise

    model.eval()
    model = model.to(device)
    print(f"Model loaded successfully on {device}")

    return model, alphabet


def get_wildtype_sequence(dms_file: str, reference_file: str = None) -> Tuple[str, int]:
    """
    从DMS数据中获取野生型序列

    Args:
        dms_file: DMS数据文件
        reference_file: 参考文件路径

    Returns:
        (wildtype_sequence, offset_idx)
    """
    # 首先尝试从参考文件获取
    if reference_file and os.path.exists(reference_file):
        dms_basename = os.path.basename(dms_file).replace('.csv', '')
        ref_df = pd.read_csv(reference_file)

        # 查找匹配的DMS数据集
        match = ref_df[ref_df['DMS_filename'].str.replace('.csv', '') == dms_basename]
        if len(match) > 0:
            wt_seq = match.iloc[0]['target_seq']
            offset_idx = 1  # ProteinGym默认1-indexed
            print(f"Found wildtype sequence from reference file (length: {len(wt_seq)})")
            return wt_seq, offset_idx

    # 否则从DMS数据反推野生型序列
    print("Reference not found, inferring wildtype from mutations...")
    df = pd.read_csv(dms_file)

    if 'mutated_sequence' not in df.columns or 'mutant' not in df.columns:
        raise ValueError("DMS file must contain 'mutated_sequence' and 'mutant' columns")

    # 取第一个突变信息来反推野生型
    first_row = df.iloc[0]
    mutant = first_row['mutant']
    mutated_seq = first_row['mutated_sequence']

    # 解析突变: 格式通常是 "W123A" 或 "W123A:G456T" (多突变)
    wt_seq = list(mutated_seq)

    for single_mut in mutant.split(':'):
        if len(single_mut) < 3:
            continue
        wt_aa = single_mut[0]
        mt_aa = single_mut[-1]
        pos = int(single_mut[1:-1])

        # 恢复野生型氨基酸 (位置是1-indexed的)
        wt_seq[pos - 1] = wt_aa

    wt_seq = ''.join(wt_seq)
    offset_idx = 1

    print(f"Inferred wildtype sequence from mutations (length: {len(wt_seq)})")
    return wt_seq, offset_idx


def label_row(row_mutant: str, sequence: str, token_probs, alphabet, offset_idx: int) -> float:
    """
    计算单个突变的ESM分数

    Args:
        row_mutant: 突变标识 (如 "W123A" 或 "W123A:G456T")
        sequence: 野生型序列
        token_probs: 模型输出的token概率 (log softmax)
        alphabet: ESM alphabet
        offset_idx: 位置偏移量

    Returns:
        突变效应分数 (mutant log-prob - wildtype log-prob)
    """
    score = 0.0

    for single_mut in row_mutant.split(':'):
        if len(single_mut) < 3:
            continue

        wt_aa = single_mut[0]
        mt_aa = single_mut[-1]
        idx = int(single_mut[1:-1]) - offset_idx

        # 验证位置范围
        if idx < 0 or idx >= len(sequence):
            # print(f"Warning: position {idx + offset_idx} out of range for sequence of length {len(sequence)}")
            continue

        wt_encoded = alphabet.get_idx(wt_aa)
        mt_encoded = alphabet.get_idx(mt_aa)

        # token_probs shape: (1, seq_len + 2, vocab_size)
        # +1 for BOS token
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()

    return score


def compute_esm_scores_wt_marginals(
    df: pd.DataFrame,
    wt_sequence: str,
    model,
    alphabet,
    offset_idx: int = 1,
    device: str = 'cuda'
) -> List[float]:
    """
    使用wt-marginals策略计算ESM分数

    这是最快的策略:只需要对野生型序列做一次前向传播
    """
    batch_converter = alphabet.get_batch_converter()

    # 准备野生型序列数据
    data = [("wildtype", wt_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 获取野生型序列的token概率
    print("Computing wildtype token probabilities...")
    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)

    # 计算每个突变的分数
    print("Scoring variants...")
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mutant = row['mutant']
        try:
            score = label_row(mutant, wt_sequence, token_probs, alphabet, offset_idx)
            scores.append(score)
        except Exception as e:
            # print(f"Warning: failed to score mutant {mutant}: {e}")
            scores.append(float('nan'))

    return scores


def compute_esm_scores_masked_marginals(
    df: pd.DataFrame,
    wt_sequence: str,
    model,
    alphabet,
    offset_idx: int = 1,
    device: str = 'cuda'
) -> List[float]:
    """
    使用masked-marginals策略计算ESM分数

    对每个位置进行mask然后预测,更准确但更慢
    """
    batch_converter = alphabet.get_batch_converter()

    # 准备野生型序列数据
    data = [("wildtype", wt_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 对每个位置进行masked预测
    print("Computing masked marginals for all positions...")
    all_token_probs = []

    for i in tqdm(range(batch_tokens.size(1)), desc="Masking positions"):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
        all_token_probs.append(token_probs[:, i])  # (1, vocab_size)

    # 合并所有位置的概率
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)  # (1, seq_len, vocab_size)

    # 计算每个突变的分数
    print("Scoring variants...")
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mutant = row['mutant']
        try:
            score = label_row(mutant, wt_sequence, token_probs, alphabet, offset_idx)
            scores.append(score)
        except Exception as e:
            scores.append(float('nan'))

    return scores


def main():
    args = parse_args()

    # 设备设置
    device = 'cpu' if args.nogpu else args.device

    print("=" * 80)
    print("ProteinGym DMS Evaluation - ESM Zero-shot Variant Prediction (Docker)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"DMS file: {args.dms_file}")
    print(f"Output file: {args.output}")
    print(f"Scoring strategy: {args.scoring_strategy}")
    print(f"Device: {device}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. 加载DMS数据
    print("\n[1/5] Loading DMS data...")
    df = pd.read_csv(args.dms_file)

    if args.max_sequences:
        df = df.head(args.max_sequences)

    print(f"Loaded {len(df)} variants")

    # 检查必要的列
    if 'mutant' not in df.columns:
        print("Error: DMS file must contain 'mutant' column")
        sys.exit(1)
    if 'DMS_score' not in df.columns:
        print("Error: DMS file must contain 'DMS_score' column")
        sys.exit(1)

    # 2. 获取野生型序列
    print("\n[2/5] Getting wildtype sequence...")
    wt_sequence, offset_idx = get_wildtype_sequence(args.dms_file, args.reference_file)
    print(f"Wildtype sequence length: {len(wt_sequence)}")
    print(f"Position offset: {offset_idx}")

    # 3. 加载ESM模型
    print("\n[3/5] Loading ESM model...")
    model, alphabet = load_esm_model(args.model_path, device)

    # 4. 计算ESM分数
    print(f"\n[4/5] Computing ESM scores ({args.scoring_strategy})...")

    if args.scoring_strategy == 'wt-marginals':
        esm_scores = compute_esm_scores_wt_marginals(
            df, wt_sequence, model, alphabet, offset_idx, device
        )
    else:  # masked-marginals
        esm_scores = compute_esm_scores_masked_marginals(
            df, wt_sequence, model, alphabet, offset_idx, device
        )

    # 5. 计算相关系数
    print("\n[5/5] Computing Spearman correlation...")

    dms_scores = df['DMS_score'].tolist()

    # 过滤NaN值
    valid_pairs = [
        (dms, esm) for dms, esm in zip(dms_scores, esm_scores)
        if not np.isnan(esm) and not np.isnan(dms)
    ]

    if len(valid_pairs) < 2:
        print("Error: Not enough valid predictions to compute correlation")
        spearman_corr, p_value = float('nan'), float('nan')
    else:
        valid_dms_scores = [p[0] for p in valid_pairs]
        valid_esm_scores = [p[1] for p in valid_pairs]
        spearman_corr, p_value = spearmanr(valid_dms_scores, valid_esm_scores)

    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Total variants: {len(df)}")
    print(f"Valid predictions: {len(valid_pairs)}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.6e}")
    print("=" * 80)

    # 6. 保存结果
    print(f"\nSaving results to {args.output}...")

    df['esm_score'] = esm_scores
    df.to_csv(args.output, index=False)

    # 保存统计信息
    stats_file = args.output.replace('.csv', '_stats.json')
    stats = {
        'dms_file': args.dms_file,
        'model_path': args.model_path,
        'scoring_strategy': args.scoring_strategy,
        'total_variants': len(df),
        'valid_predictions': len(valid_pairs),
        'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else None,
        'p_value': float(p_value) if not np.isnan(p_value) else None,
        'wildtype_sequence_length': len(wt_sequence),
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Statistics saved to {stats_file}")
    print("\nEvaluation complete!")

    # 返回相关系数用于批量评估
    return spearman_corr


if __name__ == '__main__':
    main()
