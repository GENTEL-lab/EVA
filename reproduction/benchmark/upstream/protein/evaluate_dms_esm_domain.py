#!/usr/bin/env python3
"""
使用ESM模型评估Domain DMS数据集

针对只包含domain片段的DMS数据集，从文件名获取offset

使用方法:
    docker exec rnagen4 /usr/bin/python3.11 /rna-multiverse/benchmark_protein_dms/evaluate_dms_esm_domain.py \
        --model_path /rna-multiverse/esm_models/esm1v_t33_650M_UR90S_1.pt \
        --dms_file /rna-multiverse/path/to/data.csv \
        --output /rna-multiverse/path/to/output.csv \
        --device cuda:0
"""

import argparse
import sys
import json
import os
import re
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import torch

try:
    import esm
    from esm import pretrained
except ImportError as e:
    print(f"Error importing ESM: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='ESM evaluation for domain DMS datasets'
    )
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dms_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--scoring_strategy', type=str, default='wt-marginals',
                        choices=['wt-marginals', 'masked-marginals'])
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--offset', type=int, default=None,
                        help='Manual offset. If not provided, extracted from filename')
    return parser.parse_args()


def load_esm_model(model_path: str, device: str = 'cuda'):
    """Load ESM model"""
    print(f"Loading ESM model from {model_path}...")
    if model_path.endswith('.pt') and os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, alphabet


def get_offset_from_filename(filename: str) -> int:
    """Extract offset from filename like O15151_PF00641_299.csv"""
    basename = os.path.basename(filename).replace('.csv', '')
    parts = basename.split('_')
    # 最后一部分应该是offset数字
    try:
        offset = int(parts[-1])
        return offset
    except ValueError:
        return 1  # 默认offset


def get_wildtype_sequence(df: pd.DataFrame, offset: int) -> str:
    """从突变数据反推野生型序列"""
    # 取第一行
    first_row = df.iloc[0]
    mutant = first_row['mutant']
    mutated_seq = list(first_row['mutated_sequence'])

    # 解析突变并恢复野生型
    for single_mut in mutant.split(':'):
        if len(single_mut) < 3:
            continue
        wt_aa = single_mut[0]
        mt_aa = single_mut[-1]
        pos = int(single_mut[1:-1])

        # 计算在序列中的实际位置
        seq_pos = pos - offset
        if 0 <= seq_pos < len(mutated_seq):
            mutated_seq[seq_pos] = wt_aa

    return ''.join(mutated_seq)


def label_row(row_mutant: str, sequence: str, token_probs, alphabet, offset: int) -> float:
    """计算突变分数"""
    score = 0.0
    for single_mut in row_mutant.split(':'):
        if len(single_mut) < 3:
            continue
        wt_aa = single_mut[0]
        mt_aa = single_mut[-1]
        pos = int(single_mut[1:-1])

        # 计算序列中的位置
        idx = pos - offset
        if idx < 0 or idx >= len(sequence):
            continue

        wt_encoded = alphabet.get_idx(wt_aa)
        mt_encoded = alphabet.get_idx(mt_aa)

        # +1 for BOS token
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()

    return score


def main():
    args = parse_args()

    print("=" * 70)
    print("ESM Domain DMS Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"DMS file: {args.dms_file}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 加载数据
    df = pd.read_csv(args.dms_file)
    print(f"Loaded {len(df)} variants")

    # 获取offset
    if args.offset is not None:
        offset = args.offset
    else:
        offset = get_offset_from_filename(args.dms_file)
    print(f"Using offset: {offset}")

    # 获取野生型序列
    wt_sequence = get_wildtype_sequence(df, offset)
    print(f"Wildtype sequence (length={len(wt_sequence)}): {wt_sequence[:50]}...")

    # 加载模型
    model, alphabet = load_esm_model(args.model_path, args.device)

    # 计算token probabilities
    batch_converter = alphabet.get_batch_converter()
    data = [("wildtype", wt_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(args.device)

    print(f"Computing scores ({args.scoring_strategy})...")

    if args.scoring_strategy == 'wt-marginals':
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
    else:  # masked-marginals
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1)), desc="Masking"):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                tp = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
            all_token_probs.append(tp[:, i])
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

    # 计算每个突变的分数
    esm_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        try:
            score = label_row(row['mutant'], wt_sequence, token_probs, alphabet, offset)
            esm_scores.append(score)
        except Exception as e:
            esm_scores.append(float('nan'))

    # 计算相关系数
    dms_scores = df['DMS_score'].tolist()
    valid_pairs = [(d, e) for d, e in zip(dms_scores, esm_scores)
                   if not np.isnan(e) and not np.isnan(d)]

    if len(valid_pairs) >= 2:
        valid_dms = [p[0] for p in valid_pairs]
        valid_esm = [p[1] for p in valid_pairs]
        corr, pval = spearmanr(valid_dms, valid_esm)
    else:
        corr, pval = float('nan'), float('nan')

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Total variants: {len(df)}")
    print(f"Valid predictions: {len(valid_pairs)}")
    print(f"Spearman correlation: {corr:.4f}")
    print(f"P-value: {pval:.6e}")
    print("=" * 70)

    # 保存结果
    df['esm_score'] = esm_scores
    df.to_csv(args.output, index=False)

    # 保存统计
    stats = {
        'dms_file': args.dms_file,
        'model_path': args.model_path,
        'offset': offset,
        'wildtype_length': len(wt_sequence),
        'total_variants': len(df),
        'valid_predictions': len(valid_pairs),
        'spearman_correlation': float(corr) if not np.isnan(corr) else None,
        'p_value': float(pval) if not np.isnan(pval) else None,
    }

    stats_file = args.output.replace('.csv', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to {args.output}")
    return corr


if __name__ == '__main__':
    main()
