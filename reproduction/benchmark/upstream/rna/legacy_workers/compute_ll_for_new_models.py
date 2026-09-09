#!/usr/bin/env python3
"""
在Docker容器内运行的log-likelihood计算脚本（新模型专用）

此脚本专门为新的scaling checkpoints设计，在tokenization时自动添加5'/3'方向标记
与原有的compute_ll_in_docker.py相比的区别：
1. 在prepare_input_ids中添加"5"和"3" tokens
2. 返回总log-likelihood（与其他模型一致），而非平均值

用法（在容器内）：
    python compute_ll_for_new_models.py <fasta_path> <checkpoint_path> <device>

参数：
    fasta_path: 容器内FASTA文件路径
    checkpoint_path: 容器内模型路径
    device: 计算设备（cuda:0, cuda:1, cpu等）

输出：
    JSON格式结果到stdout，包含log_likelihoods列表
"""

import sys
import json
import os
from pathlib import Path

# 设置CUDA同步模式，避免异步执行导致的内存访问问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from typing import Dict

# 强制设置单GPU推理环境变量（必须在导入torch之前）
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# 设置项目路径（容器内路径）
project_root = '/rna-multiverse/rnagen'
os.chdir(project_root)
sys.path.insert(0, project_root)

# 导入必要的模块
from model1018.lineage_tokenizer import LineageRNATokenizer
from usage.usage_helpers import load_model
import torch
import torch.nn.functional as F


def prepare_input_ids_with_direction_tokens(
    tokenizer: LineageRNATokenizer,
    sequence: str,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    准备模型输入 - 新模型版本（自动添加5'/3'方向标记）

    与原始prepare_input_ids的唯一区别：
    - 原始格式: <bos>{sequence}<eos>
    - 新格式:   <bos>5{sequence}3<eos>

    Args:
        tokenizer: tokenizer实例
        sequence: 输入序列（纯RNA序列或包含条件前缀）
        device: 设备

    Returns:
        包含input_ids, position_ids, sequence_ids的字典
    """
    # 在序列两端添加5'和3'方向标记
    # 注意：tokenizer.encode()会逐字符编码"5"和"3"
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


def read_sequences_from_fasta(fasta_file):
    """从FASTA文件读取序列（简单解析，无需Bio库）"""
    sequences = []
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # 新序列开始
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        # 添加最后一条序列
        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences


def compute_sequence_likelihood(
    model,
    tokenizer,
    sequence: str,
    device: str = 'cpu',
    normalize: bool = False
) -> float:
    """
    计算序列的log-likelihood（总和或PTLL）

    使用修改版的prepare_input_ids，自动添加5'/3' tokens

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        sequence: 输入序列（纯RNA序列）
        device: 设备
        normalize: 是否归一化为PTLL（除以token数量）

    Returns:
        log-likelihood分数（总和或PTLL，取决于normalize参数）
    """
    # 准备输入（使用新版本，添加5'/3' tokens）
    inputs = prepare_input_ids_with_direction_tokens(tokenizer, sequence, device)

    try:
        # 前向传播（使用bfloat16以匹配megablocks）
        with torch.no_grad():
            # 强制使用bfloat16 autocast，因为megablocks需要它
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    sequence_ids=inputs['sequence_ids']
                )
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                # 在autocast内计算log_softmax以保持数值稳定
                log_probs = F.log_softmax(logits.float(), dim=-1)

        # 获取实际token的log概率
        # 注意：预测位置i的token是input_ids[i+1]
        input_ids = inputs['input_ids'][0]  # [seq_len]
        token_log_probs = []

        for i in range(len(input_ids) - 1):
            # 预测位置i的token是input_ids[i+1]
            predicted_token = input_ids[i + 1]
            log_prob = log_probs[0, i, predicted_token].item()
            token_log_probs.append(log_prob)

        # 计算总log-likelihood（与其他模型保持一致）
        total_log_likelihood = sum(token_log_probs) if token_log_probs else 0.0

        # 检查nan/inf，如果出现则返回nan（让调用方处理）
        import math
        if math.isnan(total_log_likelihood) or math.isinf(total_log_likelihood):
            return float('nan')

        # 根据normalize参数决定返回总LL还是PTLL
        if normalize and token_log_probs:
            return total_log_likelihood / len(token_log_probs)  # PTLL
        return total_log_likelihood  # 总LL（默认）

    finally:
        # 确保释放GPU内存
        del inputs
        if 'outputs' in dir():
            del outputs
        if 'logits' in dir():
            del logits
        if 'log_probs' in dir():
            del log_probs
        # 同步CUDA确保操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def main():
    if len(sys.argv) < 4:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python compute_ll_for_new_models.py <fasta_path> <checkpoint_path> <device> [normalize]'
        }))
        sys.exit(1)

    fasta_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3]
    # 可选参数: normalize (默认为False)
    normalize = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

    try:
        # 读取序列
        sequences = read_sequences_from_fasta(fasta_path)

        # 加载模型（使用新API）
        model, tokenizer, config = load_model(checkpoint_path, device)

        # 不进行dtype转换，保持模型原始数据类型
        # 推理时会在compute_sequence_likelihood中使用autocast

        # 验证tokenizer包含5'/3' tokens
        token_5_id = tokenizer.token_to_id("5")
        token_3_id = tokenizer.token_to_id("3")

        if token_5_id is None or token_3_id is None:
            raise ValueError(
                f"Tokenizer缺少方向标记: '5' (ID={token_5_id}), '3' (ID={token_3_id})\n"
                f"请确保使用包含DIRECTION_TOKENS的新版tokenizer（vocab_size=114）"
            )

        # 计算log-likelihood
        mode_str = "PTLL" if normalize else "总log-likelihood"
        print(f"开始计算{mode_str}...", file=sys.stderr)
        log_likelihoods = []
        for i, sequence in enumerate(sequences):
            score = compute_sequence_likelihood(model, tokenizer, sequence, device, normalize=normalize)
            log_likelihoods.append(score)

            # 清理GPU缓存，避免内存碎片和数值不稳定
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 每10条序列打印进度（stderr不影响JSON输出）
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{len(sequences)} 序列已处理", file=sys.stderr)

        # 输出JSON结果到stdout
        result = {
            'success': True,
            'log_likelihoods': log_likelihoods,
            'num_sequences': len(log_likelihoods),
            'vocab_size': tokenizer.vocab_size,
            'direction_tokens': {'5': token_5_id, '3': token_3_id}
        }
        print(json.dumps(result))

    except Exception as e:
        # 错误信息也输出JSON格式
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == '__main__':
    main()
