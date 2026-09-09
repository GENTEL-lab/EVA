#!/usr/bin/env python3
"""
Docker容器内运行的Protein DMS log-likelihood计算脚本

此脚本在容器内运行，接收RNA序列列表，计算log-likelihood
使用新模型（自动添加5'/3'方向标记）

用法（在容器内）：
    python compute_dms_ll_in_docker.py <sequences_json> <checkpoint_path> <device>

参数：
    sequences_json: JSON文件路径，包含RNA序列列表
    checkpoint_path: 容器内模型路径
    device: 计算设备（cuda:0, cuda:1等）

输出：
    JSON格式结果到stdout，包含log_likelihoods列表
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 强制设置单GPU推理环境变量（必须在导入torch之前）
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# 🔧 修复CUDA异步执行导致的nan/内存错误
# 参考: benchmark_rnagen/scripts/new_model/CLAUDE.md 问题1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置项目路径（容器内路径）
project_root = '/rna-multiverse/rnagen'
os.chdir(project_root)
sys.path.insert(0, project_root)

# 导入必要的模块
from model.lineage_tokenizer import LineageRNATokenizer
from usage.usage_helpers import load_model
import torch
import torch.nn.functional as F


def prepare_input_ids_with_direction_tokens(
    tokenizer: LineageRNATokenizer,
    sequence: str,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    准备模型输入 - 支持带/不带 lineage 前缀的序列

    输入格式（根据 lineage_model_input_format.md）：
    - 无前缀: 纯RNA序列 (AUGCUAGC...)
      -> 输出: <bos>5{sequence}3<eos>
    - 有前缀: |d__...;s__species;<rna_mRNA>|{rna_sequence}
      -> 输出: <bos>|d__...;s__species;<rna_mRNA>|5{rna_sequence}3<eos>
    """
    # 检测是否包含 lineage 前缀（以 | 开头）
    if sequence.startswith('|'):
        # 找到前缀结束位置（第二个 |）
        second_pipe_idx = sequence.find('|', 1)
        if second_pipe_idx != -1:
            # 提取前缀和纯序列
            prefix = sequence[:second_pipe_idx + 1]  # 包含结尾的 |
            rna_seq = sequence[second_pipe_idx + 1:]  # 纯RNA序列
            # 在纯序列两端添加5'和3'方向标记
            sequence_with_direction = f"{prefix}5{rna_seq}3"
        else:
            # 异常情况：只有一个 |，按无前缀处理
            sequence_with_direction = f"5{sequence}3"
    else:
        # 无前缀：在序列两端添加5'和3'方向标记
        sequence_with_direction = f"5{sequence}3"

    # 添加BOS和EOS标记
    full_sequence = f"<bos>{sequence_with_direction}<eos>"

    # 编码
    token_ids = tokenizer.encode(full_sequence)

    # 创建tensors
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    sequence_ids = torch.zeros((1, len(token_ids)), dtype=torch.long, device=device)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'sequence_ids': sequence_ids
    }


def compute_sequence_likelihood(
    model,
    tokenizer,
    sequence: str,
    device: str = 'cpu',
    reduce_method: str = 'mean'
) -> float:
    """
    计算序列的log-likelihood

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        sequence: 输入RNA序列
        device: 设备
        reduce_method: 归约方式 ('mean' 或 'sum')

    Returns:
        log-likelihood分数
    """
    # 准备输入（使用新版本，添加5'/3' tokens）
    inputs = prepare_input_ids_with_direction_tokens(tokenizer, sequence, device)

    # 前向传播（使用autocast确保数据类型匹配）
    with torch.no_grad():
        # 获取模型数据类型
        model_dtype = next(model.parameters()).dtype

        # 根据模型数据类型选择autocast
        if model_dtype == torch.bfloat16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    sequence_ids=inputs['sequence_ids']
                )
        elif model_dtype == torch.float32:
            # float32不需要autocast
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

    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # 检查logits是否包含nan/inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"[WARNING] logits包含nan或inf值", file=sys.stderr)
        return float('nan')

    # 计算log-likelihood（参考compute_ll_for_new_models.py的实现）
    log_probs = F.log_softmax(logits, dim=-1)

    # 检查log_probs是否包含nan/inf
    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
        print(f"[WARNING] log_probs包含nan或inf值", file=sys.stderr)
        return float('nan')

    # 获取实际token的log概率
    # 注意：预测位置i的token是input_ids[i+1]
    input_ids = inputs['input_ids'][0]  # [seq_len]
    token_log_probs = []

    for i in range(len(input_ids) - 1):
        # 预测位置i的token是input_ids[i+1]
        predicted_token = input_ids[i + 1]
        log_prob = log_probs[0, i, predicted_token].item()
        token_log_probs.append(log_prob)

    # 归约
    if reduce_method == 'mean':
        result = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0
    else:  # sum
        result = sum(token_log_probs) if token_log_probs else 0.0

    # 最后检查结果
    if result != result:  # nan check
        print(f"[WARNING] 计算结果为nan", file=sys.stderr)

    return result


def compute_batch_likelihood(
    model,
    tokenizer,
    sequences: List[str],
    device: str = 'cpu',
    reduce_method: str = 'mean'
) -> List[float]:
    """
    批量计算序列的log-likelihood（性能优化版本）

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        sequences: 输入RNA序列列表
        device: 设备
        reduce_method: 归约方式 ('mean' 或 'sum')

    Returns:
        log-likelihood分数列表
    """
    if not sequences:
        return []

    # 准备所有序列的输入
    all_input_ids = []
    all_position_ids = []
    all_sequence_ids = []
    seq_lengths = []

    for seq in sequences:
        # 处理带/不带 lineage 前缀的序列
        if seq.startswith('|'):
            # 找到前缀结束位置（第二个 |）
            second_pipe_idx = seq.find('|', 1)
            if second_pipe_idx != -1:
                # 提取前缀和纯序列
                prefix = seq[:second_pipe_idx + 1]  # 包含结尾的 |
                rna_seq = seq[second_pipe_idx + 1:]  # 纯RNA序列
                # 在纯序列两端添加5'和3'方向标记
                sequence_with_direction = f"{prefix}5{rna_seq}3"
            else:
                # 异常情况：只有一个 |，按无前缀处理
                sequence_with_direction = f"5{seq}3"
        else:
            # 无前缀：添加5'/3'方向标记
            sequence_with_direction = f"5{seq}3"

        full_sequence = f"<bos>{sequence_with_direction}<eos>"

        # 编码
        token_ids = tokenizer.encode(full_sequence)
        seq_len = len(token_ids)
        seq_lengths.append(seq_len)

        # 创建tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        sequence_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

        all_input_ids.append(input_ids)
        all_position_ids.append(position_ids)
        all_sequence_ids.append(sequence_ids)

    # Padding到最大长度
    max_len = max(seq_lengths)
    padded_input_ids = []
    padded_position_ids = []
    padded_sequence_ids = []

    pad_token_id = tokenizer.encode('<pad>')[0] if hasattr(tokenizer, 'pad_token') else 0

    for i in range(len(sequences)):
        seq_len = seq_lengths[i]
        pad_len = max_len - seq_len

        # Padding
        if pad_len > 0:
            padded_input = torch.cat([
                all_input_ids[i],
                torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device)
            ])
            padded_pos = torch.cat([
                all_position_ids[i],
                torch.arange(seq_len, max_len, dtype=torch.long, device=device)
            ])
            padded_seq = torch.cat([
                all_sequence_ids[i],
                torch.zeros(pad_len, dtype=torch.long, device=device)
            ])
        else:
            padded_input = all_input_ids[i]
            padded_pos = all_position_ids[i]
            padded_seq = all_sequence_ids[i]

        padded_input_ids.append(padded_input)
        padded_position_ids.append(padded_pos)
        padded_sequence_ids.append(padded_seq)

    # 堆叠成batch
    batch_input_ids = torch.stack(padded_input_ids)  # [batch_size, max_len]
    batch_position_ids = torch.stack(padded_position_ids)
    batch_sequence_ids = torch.stack(padded_sequence_ids)

    # 前向传播
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype

        if model_dtype == torch.bfloat16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch_input_ids,
                    position_ids=batch_position_ids,
                    sequence_ids=batch_sequence_ids
                )
        elif model_dtype == torch.float32:
            outputs = model(
                input_ids=batch_input_ids,
                position_ids=batch_position_ids,
                sequence_ids=batch_sequence_ids
            )
        else:
            outputs = model(
                input_ids=batch_input_ids,
                position_ids=batch_position_ids,
                sequence_ids=batch_sequence_ids
            )

    logits = outputs.logits  # [batch_size, max_len, vocab_size]

    # 计算log-likelihood
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, max_len, vocab_size]

    # 为每个序列计算log-likelihood
    results = []
    for i in range(len(sequences)):
        seq_len = seq_lengths[i]

        # 检查nan/inf
        if torch.isnan(logits[i]).any() or torch.isinf(logits[i]).any():
            print(f"[WARNING] 序列 {i} 的logits包含nan或inf值", file=sys.stderr)
            results.append(float('nan'))
            continue

        # 获取实际token的log概率
        token_log_probs = []
        for j in range(seq_len - 1):
            predicted_token = batch_input_ids[i, j + 1]
            log_prob = log_probs[i, j, predicted_token].item()
            token_log_probs.append(log_prob)

        # 归约
        if reduce_method == 'mean':
            result = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0
        else:  # sum
            result = sum(token_log_probs) if token_log_probs else 0.0

        results.append(result)

    return results


def get_optimal_batch_size(device: str, model_size_hint: str = 'auto') -> int:
    """
    根据GPU可用内存动态计算最优批量大小

    Args:
        device: CUDA设备
        model_size_hint: 模型大小提示 ('30M', '244M', '687M', 'auto')

    Returns:
        最优批量大小
    """
    try:
        device_idx = int(device.replace('cuda:', ''))

        # 获取GPU内存信息
        total_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
        reserved_mem = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
        allocated_mem = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
        free_mem = total_mem - reserved_mem

        print(f"[INFO] GPU {device_idx}: 总内存={total_mem:.1f}GB, 已分配={allocated_mem:.1f}GB, 可用≈{free_mem:.1f}GB", file=sys.stderr)

        # 根据模型大小和可用内存估算batch_size
        # 经验值：每个序列(~1000 tokens)大约占用内存
        # 30M模型: ~0.5GB/batch
        # 244M模型: ~2GB/batch
        # 687M模型: ~4GB/batch

        if model_size_hint == '30M' or (model_size_hint == 'auto' and free_mem > 70):
            mem_per_sample = 0.3  # GB per sample in batch
        elif model_size_hint == '244M' or (model_size_hint == 'auto' and free_mem > 50):
            mem_per_sample = 1.0
        elif model_size_hint == '687M' or (model_size_hint == 'auto' and free_mem > 30):
            mem_per_sample = 2.0
        else:
            mem_per_sample = 0.5

        # 保留20GB给模型本身和其他开销
        available_for_batch = max(free_mem - 20, 10)
        optimal_batch = int(available_for_batch / mem_per_sample)

        # 限制范围 [4, 64]
        optimal_batch = max(4, min(64, optimal_batch))

        print(f"[INFO] 自动选择 batch_size={optimal_batch}", file=sys.stderr)
        return optimal_batch

    except Exception as e:
        print(f"[WARNING] 无法获取GPU信息，使用默认batch_size=16: {e}", file=sys.stderr)
        return 16


def main():
    """主函数"""
    if len(sys.argv) < 5:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python compute_dms_ll_in_docker.py <sequences_json> <checkpoint_path> <device> <output_json> [reduce_method] [batch_size]'
        }))
        sys.exit(1)

    sequences_json = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3]
    output_json = sys.argv[4]
    reduce_method = sys.argv[5] if len(sys.argv) > 5 else 'mean'
    batch_size_arg = sys.argv[6] if len(sys.argv) > 6 else 'auto'

    try:
        # 1. 读取序列
        with open(sequences_json, 'r') as f:
            data = json.load(f)
            sequences = data['sequences']

        num_sequences = len(sequences)
        print(f"[INFO] 加载了 {num_sequences} 条RNA序列", file=sys.stderr)

        # 2. 加载模型
        print(f"[INFO] 加载模型: {checkpoint_path}", file=sys.stderr)
        print(f"[INFO] 设备: {device}", file=sys.stderr)

        model, tokenizer, config = load_model(checkpoint_path, device=device)

        # 确保模型使用统一的数据类型（bfloat16）
        # 这个模型内部有些操作（如megablocks）必须使用bfloat16
        model = model.to(torch.bfloat16)

        model.eval()

        print(f"[INFO] 模型加载完成", file=sys.stderr)
        print(f"[INFO] 参数量: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

        # 3. 计算log-likelihood（使用批处理优化）
        print(f"[INFO] 开始计算log-likelihood (reduce_method={reduce_method})...", file=sys.stderr)
        log_likelihoods = []

        # 批处理设置 - 固定batch_size=128
        if batch_size_arg == 'auto':
            batch_size = 128  # 固定批量大小
            print(f"[INFO] 使用固定 batch_size={batch_size}", file=sys.stderr)
        else:
            batch_size = int(batch_size_arg)
            print(f"[INFO] 使用指定的 batch_size={batch_size}", file=sys.stderr)

        num_batches = (num_sequences + batch_size - 1) // batch_size

        print(f"[INFO] 使用批处理: batch_size={batch_size}, num_batches={num_batches}", file=sys.stderr)

        for batch_idx in tqdm(range(num_batches), desc="批处理", file=sys.stderr):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_sequences = sequences[start_idx:end_idx]

            # 实时显示处理进度
            print(f"[PROGRESS] 处理序列 {start_idx+1}-{end_idx}/{num_sequences} (batch {batch_idx+1}/{num_batches})", file=sys.stderr, flush=True)

            try:
                # 批量计算
                batch_lls = compute_batch_likelihood(model, tokenizer, batch_sequences, device, reduce_method)
                log_likelihoods.extend(batch_lls)
            except Exception as e:
                import traceback
                print(f"[WARNING] 批次 {batch_idx} 计算失败，回退到逐条处理: {e}", file=sys.stderr)
                print(f"[DEBUG] 完整错误信息:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

                # 回退到逐条处理
                for i, seq in enumerate(batch_sequences):
                    try:
                        ll = compute_sequence_likelihood(model, tokenizer, seq, device, reduce_method)
                        log_likelihoods.append(ll)
                    except Exception as e2:
                        seq_idx = start_idx + i
                        print(f"[WARNING] 序列 {seq_idx} 计算失败: {e2}", file=sys.stderr)
                        print(f"[DEBUG] 序列长度: {len(seq)}", file=sys.stderr)
                        print(f"[DEBUG] 序列前100bp: {seq[:100]}", file=sys.stderr)
                        log_likelihoods.append(float('nan'))

        print(f"[INFO] 计算完成", file=sys.stderr)

        # 4. 输出结果（JSON到文件，避免docker exec stdout大小限制）
        result = {
            'success': True,
            'log_likelihoods': log_likelihoods,
            'num_sequences': num_sequences,
            'reduce_method': reduce_method
        }

        # 写入到输出文件
        with open(output_json, 'w') as f:
            json.dump(result, f)

        print(f"[INFO] 结果已写入: {output_json}", file=sys.stderr)

    except Exception as e:
        # 输出错误信息（写入文件）
        error_result = {
            'success': False,
            'error': str(e)
        }
        with open(output_json, 'w') as f:
            json.dump(error_result, f)
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
