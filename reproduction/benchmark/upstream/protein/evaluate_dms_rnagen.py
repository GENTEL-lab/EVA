#!/usr/bin/env python3
"""
使用RNAGen模型评估ProteinGym DMS数据集

专门适配RNAGen模型格式的评估脚本
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr

import torch

# 添加路径
sys.path.insert(0, "/rna-multiverse")
sys.path.insert(0, "/rna-multiverse/rnagen")
sys.path.insert(0, "/rna-multiverse/dna_rna_tools")

from model1018.config import RNAGenConfig
from model1018.causal_lm import RNAGenForCausalLM
from model1018.lineage_tokenizer import LineageRNATokenizer
from dna_rna_converter import reverse_translate_protein


def parse_args():
    parser = argparse.ArgumentParser(description='使用RNAGen模型评估ProteinGym DMS数据集')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dms_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_sequences', type=int, default=None)
    parser.add_argument('--codon_optimization', type=str, default='first')
    parser.add_argument('--reduce_method', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_rna_model(model_path: str, device: str = 'cuda'):
    """加载RNAGen模型"""
    print(f"从 {model_path} 加载模型...")

    # 加载配置
    config_file = Path(model_path) / "config.json"
    config = RNAGenConfig.from_json_file(str(config_file))

    # 适配单GPU推理
    config.moe_world_size = 1

    # 加载tokenizer
    tokenizer = LineageRNATokenizer.from_pretrained(str(model_path))

    # 创建模型
    model = RNAGenForCausalLM(config)

    # 加载权重
    weights_file = Path(model_path) / "model_weights.pt"
    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"✓ 模型已加载到 {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def load_dms_data(dms_file: str, max_sequences: int = None) -> List[Dict]:
    """加载DMS数据"""
    data = []
    with open(dms_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_sequences and i >= max_sequences:
                break
            data.append(row)
    print(f"从 {dms_file} 加载了 {len(data)} 个突变序列")
    return data


def protein_to_rna_batch(protein_sequences: List[str], optimization: str = 'first') -> List[str]:
    """批量将蛋白质序列反向翻译为RNA序列"""
    rna_sequences = []
    for protein_seq in protein_sequences:
        try:
            rna_seq = reverse_translate_protein(
                protein_seq,
                code_type='standard',
                optimization=optimization,
                output_format='rna',
                add_start_stop=False,
            )
            rna_sequences.append(rna_seq)
        except Exception as e:
            print(f"警告: 蛋白质序列反向翻译失败: {e}")
            rna_sequences.append("")
    return rna_sequences


def compute_sequence_loglikelihood(
    sequences: List[str],
    model,
    tokenizer,
    batch_size: int = 8,
    reduce_method: str = 'mean',
    device: str = 'cuda'
) -> List[float]:
    """计算RNA序列的对数似然"""
    scores = []
    model.eval()

    for i in tqdm(range(0, len(sequences), batch_size), desc="计算序列对数似然"):
        batch_seqs = sequences[i:i + batch_size]

        # 跳过空序列
        valid_indices = [j for j, seq in enumerate(batch_seqs) if seq]
        valid_seqs = [batch_seqs[j] for j in valid_indices]

        if not valid_seqs:
            scores.extend([float('nan')] * len(batch_seqs))
            continue

        # Tokenize - 手动批处理编码
        try:
            # 逐个编码序列
            batch_input_ids = []
            max_len = 0

            for seq in valid_seqs:
                tokens = tokenizer.encode(seq)
                batch_input_ids.append(tokens)
                max_len = max(max_len, len(tokens))

            # Padding到相同长度
            pad_id = tokenizer.token_to_id('<pad>')
            padded_input_ids = []
            attention_masks = []

            for tokens in batch_input_ids:
                # Padding
                padding_len = max_len - len(tokens)
                padded_tokens = tokens + [pad_id] * padding_len
                mask = [1] * len(tokens) + [0] * padding_len

                padded_input_ids.append(padded_tokens)
                attention_masks.append(mask)

            input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)

        except Exception as e:
            print(f"警告: tokenization失败: {e}")
            scores.extend([float('nan')] * len(batch_seqs))
            continue

        # 前向传播
        with torch.no_grad():
            try:
                # 生成position_ids和sequence_ids
                batch_size, seq_len = input_ids.shape
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
                sequence_ids = torch.zeros_like(input_ids)  # 所有序列ID设为0

                outputs = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits  # (batch, seq_len, vocab_size)
            except Exception as e:
                print(f"警告: 模型前向传播失败: {e}")
                scores.extend([float('nan')] * len(batch_seqs))
                continue

        # 计算对数概率
        batch_scores = []
        for b_idx in range(len(valid_seqs)):
            seq_logits = logits[b_idx, :-1, :]
            seq_input_ids = input_ids[b_idx, 1:]

            log_probs = torch.log_softmax(seq_logits, dim=-1)
            token_log_probs = log_probs[range(len(seq_input_ids)), seq_input_ids]

            mask = attention_mask[b_idx, 1:]
            valid_log_probs = token_log_probs[mask.bool()]

            if reduce_method == 'mean':
                score = valid_log_probs.mean().item()
            else:
                score = valid_log_probs.sum().item()

            batch_scores.append(score)

        # 插入到正确位置
        result_scores = []
        valid_idx = 0
        for j in range(len(batch_seqs)):
            if j in valid_indices:
                result_scores.append(batch_scores[valid_idx])
                valid_idx += 1
            else:
                result_scores.append(float('nan'))

        scores.extend(result_scores)

    return scores


def main():
    args = parse_args()

    print("=" * 80)
    print("ProteinGym DMS 评估 - RNAGen模型")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"DMS文件: {args.dms_file}")
    print(f"输出文件: {args.output}")
    print("=" * 80)

    # 1. 加载DMS数据
    print("\n[1/5] 加载DMS数据...")
    dms_data = load_dms_data(args.dms_file, args.max_sequences)

    # 2. 提取蛋白质序列
    print("\n[2/5] 提取蛋白质序列...")
    protein_sequences = [row['mutated_sequence'] for row in dms_data]
    dms_scores = [float(row['DMS_score']) for row in dms_data]

    # 3. 反向翻译
    print(f"\n[3/5] 反向翻译为RNA序列(策略: {args.codon_optimization})...")
    rna_sequences = protein_to_rna_batch(protein_sequences, args.codon_optimization)

    # 4. 加载模型
    print("\n[4/5] 加载RNAGen模型...")
    model, tokenizer = load_rna_model(args.model_path, args.device)

    # 5. 计算对数似然
    print(f"\n[5/5] 计算序列对数似然(归约: {args.reduce_method})...")
    model_scores = compute_sequence_loglikelihood(
        rna_sequences,
        model,
        tokenizer,
        batch_size=args.batch_size,
        reduce_method=args.reduce_method,
        device=args.device
    )

    # 6. 计算斯皮尔曼相关系数
    print("\n[6/6] 计算斯皮尔曼相关系数...")
    valid_pairs = [(dms, model) for dms, model in zip(dms_scores, model_scores) if not np.isnan(model)]

    if len(valid_pairs) < 2:
        print("✗ 错误: 有效预测数量不足")
        sys.exit(1)

    valid_dms_scores = [p[0] for p in valid_pairs]
    valid_model_scores = [p[1] for p in valid_pairs]

    spearman_corr, p_value = spearmanr(valid_dms_scores, valid_model_scores)

    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    print(f"总序列数: {len(dms_data)}")
    print(f"有效预测数: {len(valid_pairs)}")
    print(f"斯皮尔曼相关系数: {spearman_corr:.4f}")
    print(f"P值: {p_value:.6f}")
    print("=" * 80)

    # 7. 保存结果
    print(f"\n保存结果到 {args.output}...")

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mutant', 'protein_sequence', 'rna_sequence', 'DMS_score', 'model_score'])
        writer.writeheader()
        for i, row in enumerate(dms_data):
            writer.writerow({
                'mutant': row.get('mutant', f'variant_{i}'),
                'protein_sequence': protein_sequences[i],
                'rna_sequence': rna_sequences[i],
                'DMS_score': dms_scores[i],
                'model_score': model_scores[i],
            })

    # 保存统计信息
    stats_file = args.output.replace('.csv', '_stats.json')
    stats = {
        'dms_file': args.dms_file,
        'model_path': args.model_path,
        'total_sequences': len(dms_data),
        'valid_predictions': len(valid_pairs),
        'spearman_correlation': float(spearman_corr),
        'p_value': float(p_value),
        'codon_optimization': args.codon_optimization,
        'reduce_method': args.reduce_method,
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ 结果已保存到 {args.output}")
    print(f"✓ 统计信息已保存到 {stats_file}")
    print("\n评估完成!")


if __name__ == '__main__':
    main()
