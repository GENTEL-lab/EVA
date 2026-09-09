#!/usr/bin/env python3
"""
GenerRNA模型Conda Wrapper - 从主环境调用generna conda环境计算log-likelihood

此脚本运行在主环境，通过subprocess调用generna conda环境中的GenerRNA模型，
避免在主环境中安装GenerRNA相关依赖

用法：
    python compute_generrna_scores_conda.py <fasta_file> <ckpt_path> <tokenizer_path> <device>

参数：
    fasta_file: FASTA文件路径
    ckpt_path: GenerRNA模型checkpoint路径
    tokenizer_path: Tokenizer目录路径
    device: 计算设备（cuda, cuda:0, cpu等）

返回：
    返回log-likelihood列表（JSON格式）
"""

import sys
import json
import subprocess
import os
from pathlib import Path


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
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences


def compute_log_likelihoods_via_conda(fasta_file, ckpt_path, tokenizer_path, device='cuda', normalize=False):
    """
    通过conda环境调用GenerRNA计算log-likelihood

    Args:
        fasta_file: FASTA文件路径
        ckpt_path: 模型checkpoint路径
        tokenizer_path: Tokenizer目录路径
        device: 计算设备
        normalize: 是否归一化为PTLL（除以token数量）

    Returns:
        log_likelihoods: 对数似然值列表（总LL或PTLL）

    Raises:
        RuntimeError: 计算失败时抛出异常
    """

    # 检查文件是否存在
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_file}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型checkpoint不存在: {ckpt_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer目录不存在: {tokenizer_path}")

    # GenerRNA calculate_likelihood.py的路径
    generrna_root = Path(ckpt_path).parent
    calculate_script = generrna_root / "calculate_loglikelihood_me" / "calculate_likelihood.py"

    if not calculate_script.exists():
        raise FileNotFoundError(f"GenerRNA计算脚本不存在: {calculate_script}")

    # 构建conda激活和python命令
    # 使用bash -c来确保conda命令可用
    cmd = [
        'bash', '-c',
        f'''
        source /home/huangyanjie/miniconda3/etc/profile.d/conda.sh && \
        conda activate hf_model && \
        python {calculate_script} \
            --input_file {fasta_file} \
            --ckpt_path {ckpt_path} \
            --tokenizer_path {tokenizer_path} \
            --device {device}
        '''
    ]

    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=7200  # 2小时超时（大数据集需要更长时间）
        )

        # 解析输出
        if result.returncode != 0:
            print(f"GenerRNA stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"GenerRNA计算失败，返回码: {result.returncode}")

        # GenerRNA的calculate_likelihood.py会打印很多信息
        # 我们需要从stdout中提取log-likelihood值
        log_likelihoods = []

        # 解析输出，查找Log-Likelihood行（排除平均Log-Likelihood行）
        for line in result.stdout.split('\n'):
            # 查找包含"Log-Likelihood:"的行，但排除"平均Log-Likelihood"
            if 'Log-Likelihood:' in line and '平均' not in line:
                # 格式: "  Log-Likelihood: -48.4366"
                try:
                    ll_str = line.split('Log-Likelihood:')[1].strip()
                    log_likelihood = float(ll_str)
                    log_likelihoods.append(log_likelihood)
                except (IndexError, ValueError) as e:
                    print(f"警告: 无法解析行: {line}", file=sys.stderr)
                    continue

        if len(log_likelihoods) == 0:
            print(f"GenerRNA stdout:\n{result.stdout}", file=sys.stderr)
            print(f"GenerRNA stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError("未能从GenerRNA输出中提取log-likelihood值")

        # 如果需要归一化，读取序列并计算PTLL
        if normalize:
            sequences = read_sequences_from_fasta(fasta_file)
            if len(sequences) == len(log_likelihoods):
                # GenerRNA使用单核苷酸tokenization，token数 = 序列长度
                normalized_lls = []
                for ll, seq in zip(log_likelihoods, sequences):
                    num_tokens = len(seq)  # 单核苷酸token
                    if num_tokens > 0:
                        normalized_lls.append(ll / num_tokens)
                    else:
                        normalized_lls.append(ll)
                return normalized_lls
            else:
                print(f"警告: 序列数({len(sequences)})与LL数({len(log_likelihoods)})不匹配，无法归一化", file=sys.stderr)

        return log_likelihoods

    except subprocess.TimeoutExpired:
        raise RuntimeError("GenerRNA计算超时（超过10分钟）")
    except Exception as e:
        raise RuntimeError(f"调用GenerRNA时发生错误: {e}")


def main():
    """命令行接口"""
    if len(sys.argv) < 5:
        print("用法: python compute_generrna_scores_conda.py <fasta_file> <ckpt_path> <tokenizer_path> <device> [normalize]")
        print("示例: python compute_generrna_scores_conda.py data.fasta model_updated.pt tokenizer cuda false")
        sys.exit(1)

    fasta_file = sys.argv[1]
    ckpt_path = sys.argv[2]
    tokenizer_path = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else 'cuda'
    # 可选参数: normalize (默认为False)
    normalize = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False

    try:
        log_likelihoods = compute_log_likelihoods_via_conda(
            fasta_file, ckpt_path, tokenizer_path, device, normalize=normalize
        )

        # 输出结果（JSON格式）
        result = {
            'success': True,
            'log_likelihoods': log_likelihoods,
            'num_sequences': len(log_likelihoods)
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        # 输出错误信息（JSON格式）
        result = {
            'success': False,
            'error': str(e),
            'log_likelihoods': [],
            'num_sequences': 0
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
