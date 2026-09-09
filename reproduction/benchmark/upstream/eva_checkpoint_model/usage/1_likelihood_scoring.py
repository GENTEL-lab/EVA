#!/usr/bin/env python3
"""
RNAGen模型 - 序列Log-Likelihood打分

功能：
1. 输入一条RNA序列，输出log-likelihood打分
2. 支持条件前缀（RNA类型和/或谱系信息）
3. 支持交互式模式和批量处理模式

使用方法：
    # 交互式模式
    python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 --interactive

    # 单条序列打分
    python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 --sequence "AUGCAUGC"

    # 带条件前缀
    python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 \
        --sequence "AUGCAUGC" --rna_type mRNA --lineage "d__bacteria"

    # 批量处理
    python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 \
        --input_file sequences.txt --output_file scores.txt
"""

import argparse
import csv
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# 导入工具函数
from usage_helpers import (
    load_model,
    build_conditional_prefix,
    validate_rna_sequence,
    prepare_input_ids,
    extract_rna_sequence,
    get_available_rna_types,
    print_usage_header
)


def compute_sequence_likelihood(
    model,
    tokenizer,
    sequence: str,
    device: str = 'cpu'
) -> float:
    """
    计算序列的log-likelihood

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        sequence: 输入序列（可包含条件前缀）
        device: 设备

    Returns:
        平均log-likelihood分数
    """
    # 准备输入
    inputs = prepare_input_ids(tokenizer, sequence, device)

    # 前向传播（使用autocast确保数据类型匹配）
    with torch.no_grad():
        # 获取模型数据类型
        model_dtype = next(model.parameters()).dtype
        
        # 如果模型是 BFloat16，使用 autocast
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

    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # 计算log-likelihood
    # 对于每个位置，计算实际token的log概率
    log_probs = F.log_softmax(logits, dim=-1)

    # 获取实际token的log概率
    # 注意：预测位置i的token是input_ids[i+1]
    input_ids = inputs['input_ids'][0]  # [seq_len]
    token_log_probs = []

    for i in range(len(input_ids) - 1):
        # 预测位置i的token是input_ids[i+1]
        predicted_token = input_ids[i + 1]
        log_prob = log_probs[0, i, predicted_token].item()
        token_log_probs.append(log_prob)

    # 计算平均log-likelihood
    avg_log_likelihood = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0

    return avg_log_likelihood


def score_single_sequence(
    model,
    tokenizer,
    sequence: str,
    rna_type: str = None,
    lineage: str = None,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    对单条序列打分

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        sequence: RNA序列
        rna_type: RNA类型（可选）
        lineage: 谱系信息（可选）
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        包含分数和元数据的字典
    """
    # 验证序列
    if not validate_rna_sequence(sequence):
        raise ValueError(f"无效的RNA序列: {sequence}")

    # 构建条件前缀
    prefix = build_conditional_prefix(rna_type=rna_type, lineage=lineage)

    # 完整序列
    full_sequence = prefix + sequence

    if verbose:
        print(f"\n输入序列: {sequence}")
        if prefix:
            print(f"条件前缀: {prefix}")
        print(f"完整输入: {full_sequence}")

    # 计算分数
    score = compute_sequence_likelihood(model, tokenizer, full_sequence, device)

    if verbose:
        print(f"Log-likelihood: {score:.4f}")

    return {
        'sequence': sequence,
        'prefix': prefix,
        'full_sequence': full_sequence,
        'log_likelihood': score,
        'rna_type': rna_type,
        'lineage': lineage
    }


def interactive_mode(model, tokenizer, device: str):
    """
    交互式模式

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        device: 设备
    """
    print("\n进入交互式模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("-" * 80)

    while True:
        print("\n请输入RNA序列（只包含AUGC）:")
        sequence = input("> ").strip()

        if sequence.lower() in ['quit', 'exit', 'q']:
            print("退出交互式模式")
            break

        if sequence.lower() == 'help':
            print("\n使用说明:")
            print("1. 输入RNA序列（只包含AUGC）")
            print("2. 可选：输入RNA类型（如mRNA, rRNA等）")
            print("3. 可选：输入谱系信息")
            print("\n支持的RNA类型:")
            for rna_type in get_available_rna_types():
                print(f"  - {rna_type}")
            continue

        if not sequence:
            continue

        # 询问是否使用条件前缀
        print("\n是否使用RNA类型？(留空跳过)")
        rna_type = input("RNA类型: ").strip() or None

        print("\n是否使用谱系信息？(留空跳过)")
        lineage = input("谱系: ").strip() or None

        # 打分
        try:
            result = score_single_sequence(
                model, tokenizer, sequence,
                rna_type=rna_type,
                lineage=lineage,
                device=device,
                verbose=True
            )
        except Exception as e:
            print(f"错误: {e}")
            continue


def batch_mode(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    device: str
):
    """
    批量处理模式

    输入文件格式（每行）:
        序列 [TAB] RNA类型 [TAB] 谱系

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        input_file: 输入文件路径
        output_file: 输出文件路径
        device: 设备
    """
    print(f"\n批量处理模式")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    results = []

    # 读取输入文件
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 解析行
            parts = line.split('\t')
            sequence = parts[0].strip()
            rna_type = parts[1].strip() if len(parts) > 1 else None
            lineage = parts[2].strip() if len(parts) > 2 else None

            # 处理空字符串
            if rna_type == '':
                rna_type = None
            if lineage == '':
                lineage = None

            print(f"\n处理第 {line_num} 条序列...")

            try:
                result = score_single_sequence(
                    model, tokenizer, sequence,
                    rna_type=rna_type,
                    lineage=lineage,
                    device=device,
                    verbose=False
                )
                results.append(result)
                print(f"  序列: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
                print(f"  分数: {result['log_likelihood']:.4f}")
            except Exception as e:
                print(f"  错误: {e}")
                continue

    # 写入输出文件（CSV格式）
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        
        # 写入表头
        csv_writer.writerow(['序列', 'RNA类型', '谱系', 'Log-Likelihood'])

        # 写入结果
        for result in results:
            csv_writer.writerow([
                result['sequence'],
                result['rna_type'] or '',
                result['lineage'] or '',
                f"{result['log_likelihood']:.4f}"
            ])

    print(f"\n完成！共处理 {len(results)} 条序列")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='RNAGen模型 - 序列Log-Likelihood打分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 --interactive

  # 单条序列打分
  python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 --sequence "AUGCAUGC"

  # 带条件前缀
  python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 \\
      --sequence "AUGCAUGC" --rna_type mRNA --lineage "d__bacteria"

  # 批量处理
  python 1_likelihood_scoring.py --checkpoint ../checkpoint_100M_1018 \\
      --input_file sequences.txt --output_file scores.txt
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint目录路径'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='设备类型 (默认: auto)'
    )

    # 单条序列模式
    parser.add_argument(
        '--sequence',
        type=str,
        help='RNA序列（只包含AUGC）'
    )

    parser.add_argument(
        '--rna_type',
        type=str,
        help='RNA类型（如mRNA, rRNA等）'
    )

    parser.add_argument(
        '--lineage',
        type=str,
        help='谱系信息（如d__bacteria;p__proteobacteria）'
    )

    # 交互式模式
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='启用交互式模式'
    )

    # 批量处理模式
    parser.add_argument(
        '--input_file',
        type=str,
        help='输入文件路径（批量处理）'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help='输出文件路径（批量处理）'
    )

    args = parser.parse_args()

    # 打印头部
    print_usage_header(
        "RNAGen - 序列Log-Likelihood打分",
        "计算RNA序列的log-likelihood分数"
    )

    # 加载模型
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device

    # 根据模式运行
    if args.interactive:
        # 交互式模式
        interactive_mode(model, tokenizer, str(device))

    elif args.input_file and args.output_file:
        # 批量处理模式
        batch_mode(model, tokenizer, args.input_file, args.output_file, str(device))

    elif args.sequence:
        # 单条序列模式
        result = score_single_sequence(
            model, tokenizer, args.sequence,
            rna_type=args.rna_type,
            lineage=args.lineage,
            device=str(device),
            verbose=True
        )

    else:
        print("错误: 请指定运行模式")
        print("  --interactive: 交互式模式")
        print("  --sequence: 单条序列模式")
        print("  --input_file 和 --output_file: 批量处理模式")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
