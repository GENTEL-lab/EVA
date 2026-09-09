#!/usr/bin/env python3
"""
批处理DMS评估脚本 - 一次加载模型，处理多个DMS文件
用于减少重复的模型加载时间
"""

import sys
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from scipy.stats import spearmanr
import time

# 添加项目路径
sys.path.insert(0, '/rna-multiverse/rnagen')

from model.lineage_tokenizer import LineageRNATokenizer
from usage.usage_helpers import load_model
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_sequence_likelihood(model, tokenizer, sequence: str, device: str, reduce_method: str = 'mean'):
    """计算单条序列的log-likelihood"""
    input_ids = tokenizer.encode(sequence, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    target_ids = input_ids[:, 1:]
    log_probs_for_targets = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    if reduce_method == 'mean':
        return log_probs_for_targets.mean().item()
    elif reduce_method == 'sum':
        return log_probs_for_targets.sum().item()
    else:
        raise ValueError(f"Unknown reduce_method: {reduce_method}")


def compute_batch_likelihood(model, tokenizer, sequences: List[str], device: str, reduce_method: str = 'mean'):
    """批量计算log-likelihood"""
    batch_size = len(sequences)
    encoded = [tokenizer.encode(seq, return_tensors='pt') for seq in sequences]

    max_len = max(e.size(1) for e in encoded)
    padded_input_ids = []
    attention_masks = []

    for enc in encoded:
        seq_len = enc.size(1)
        if seq_len < max_len:
            pad_len = max_len - seq_len
            padded = torch.cat([enc, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
            mask = torch.cat([torch.ones((1, seq_len)), torch.zeros((1, pad_len))], dim=1)
        else:
            padded = enc
            mask = torch.ones_like(enc, dtype=torch.float)

        padded_input_ids.append(padded)
        attention_masks.append(mask)

    batch_input_ids = torch.stack(padded_input_ids).squeeze(1).to(device)
    batch_attention_masks = torch.stack(attention_masks).squeeze(1).to(device)

    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    batch_lls = []
    for i in range(batch_size):
        input_ids = batch_input_ids[i]
        attention_mask = batch_attention_masks[i]
        seq_len = int(attention_mask.sum().item())

        target_ids = input_ids[1:seq_len]
        log_probs_for_seq = log_probs[i, :seq_len-1, :]
        log_probs_for_targets = log_probs_for_seq.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

        if reduce_method == 'mean':
            ll = log_probs_for_targets.mean().item()
        elif reduce_method == 'sum':
            ll = log_probs_for_targets.sum().item()
        else:
            raise ValueError(f"Unknown reduce_method: {reduce_method}")

        batch_lls.append(ll)

    return batch_lls


def process_single_dms_file(model, tokenizer, dms_file: str, device: str, reduce_method: str, batch_size: int) -> Dict:
    """处理单个DMS文件"""
    print(f"\n[INFO] 开始处理: {os.path.basename(dms_file)}", file=sys.stderr, flush=True)
    start_time = time.time()

    # 1. 读取JSON格式的序列数据
    with open(dms_file, 'r') as f:
        data = json.load(f)

    sequences = data['sequences']
    num_sequences = len(sequences)

    print(f"[INFO] 序列数量: {num_sequences}", file=sys.stderr, flush=True)

    # 2. 批量计算log-likelihood
    log_likelihoods = []
    num_batches = (num_sequences + batch_size - 1) // batch_size

    print(f"[INFO] 批处理: batch_size={batch_size}, num_batches={num_batches}", file=sys.stderr, flush=True)

    for batch_idx in tqdm(range(num_batches), desc="批处理", file=sys.stderr):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_sequences = sequences[start_idx:end_idx]

        print(f"[PROGRESS] 处理序列 {start_idx+1}-{end_idx}/{num_sequences} (batch {batch_idx+1}/{num_batches})",
              file=sys.stderr, flush=True)

        try:
            batch_lls = compute_batch_likelihood(model, tokenizer, batch_sequences, device, reduce_method)
            log_likelihoods.extend(batch_lls)
        except Exception as e:
            print(f"[WARNING] 批次 {batch_idx} 失败，回退到逐条处理: {e}", file=sys.stderr)
            for i, seq in enumerate(batch_sequences):
                try:
                    ll = compute_sequence_likelihood(model, tokenizer, seq, device, reduce_method)
                    log_likelihoods.append(ll)
                except Exception as e2:
                    seq_idx = start_idx + i
                    print(f"[WARNING] 序列 {seq_idx} 失败: {e2}", file=sys.stderr)
                    log_likelihoods.append(float('nan'))

    elapsed = time.time() - start_time
    print(f"[INFO] 处理完成，耗时: {elapsed:.2f}秒", file=sys.stderr, flush=True)

    return {
        'success': True,
        'log_likelihoods': log_likelihoods,
        'num_sequences': num_sequences,
        'reduce_method': reduce_method,
        'elapsed_time': elapsed
    }


def main():
    if len(sys.argv) < 5:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python compute_batch_dms_in_docker.py <dms_files_list_json> <checkpoint_path> <device> <output_dir> [reduce_method] [batch_size]'
        }))
        sys.exit(1)

    dms_files_list_json = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3]
    output_dir = sys.argv[4]
    reduce_method = sys.argv[5] if len(sys.argv) > 5 else 'mean'
    batch_size = int(sys.argv[6]) if len(sys.argv) > 6 else 128

    try:
        # 读取要处理的DMS文件列表
        with open(dms_files_list_json, 'r') as f:
            dms_files = json.load(f)

        print(f"[INFO] 总共 {len(dms_files)} 个DMS文件待处理", file=sys.stderr, flush=True)
        print(f"[INFO] 加载模型: {checkpoint_path}", file=sys.stderr, flush=True)
        print(f"[INFO] 设备: {device}", file=sys.stderr, flush=True)

        # 1. 加载tokenizer
        tokenizer = LineageRNATokenizer()

        # 2. 加载模型（只加载一次！）
        model_load_start = time.time()
        model = load_model(checkpoint_path, device=device)
        model.eval()
        model_load_time = time.time() - model_load_start

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] 模型加载完成，耗时: {model_load_time:.2f}秒", file=sys.stderr, flush=True)
        print(f"[INFO] 参数量: {num_params:,}", file=sys.stderr, flush=True)
        print(f"[INFO] 固定 batch_size={batch_size}", file=sys.stderr, flush=True)

        # 3. 处理每个DMS文件
        total_start = time.time()
        results = {}

        for i, dms_file in enumerate(dms_files):
            print(f"\n{'='*60}", file=sys.stderr, flush=True)
            print(f"[{i+1}/{len(dms_files)}] {os.path.basename(dms_file)}", file=sys.stderr, flush=True)
            print(f"{'='*60}", file=sys.stderr, flush=True)

            result = process_single_dms_file(model, tokenizer, dms_file, device, reduce_method, batch_size)

            # 保存结果到输出目录
            output_file = os.path.join(output_dir, os.path.basename(dms_file).replace('.json', '_output.json'))
            with open(output_file, 'w') as f:
                json.dump(result, f)

            results[dms_file] = {
                'success': result['success'],
                'num_sequences': result['num_sequences'],
                'elapsed_time': result['elapsed_time'],
                'output_file': output_file
            }

        total_elapsed = time.time() - total_start

        # 4. 输出汇总
        summary = {
            'success': True,
            'total_files': len(dms_files),
            'model_load_time': model_load_time,
            'total_processing_time': total_elapsed,
            'average_time_per_file': total_elapsed / len(dms_files) if dms_files else 0,
            'results': results
        }

        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"[INFO] 全部完成！", file=sys.stderr, flush=True)
        print(f"[INFO] 模型加载时间: {model_load_time:.2f}秒", file=sys.stderr, flush=True)
        print(f"[INFO] 总处理时间: {total_elapsed:.2f}秒", file=sys.stderr, flush=True)
        print(f"[INFO] 平均每文件: {total_elapsed/len(dms_files):.2f}秒", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        print(json.dumps(summary))

    except Exception as e:
        import traceback
        print(json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
