#!/usr/bin/env python3
"""
RNAGen模型 - 自回归序列生成

功能：
1. 从头生成RNA序列（仅提供条件前缀）
2. 续写生成（提供开头的一部分序列）
3. 支持条件前缀（RNA类型和/或谱系信息）
4. 支持交互式模式和批量处理模式

使用方法：
    # 交互式模式
    python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 --interactive

    # 从头生成（仅条件前缀）
    python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \
        --rna_type mRNA --max_length 100

    # 续写生成
    python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \
        --prefix_sequence "AUGCAUGC" --max_new_tokens 50

    # 批量生成
    python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \
        --input_file prompts.txt --output_file generated.txt
"""

import argparse
import csv
import sys
import torch
from pathlib import Path

# 导入工具函数（从 usage_helpers.py）
from usage_helpers import (
    load_model,
    build_conditional_prefix,
    validate_rna_sequence,
    prepare_input_ids,
    decode_sequence,
    extract_rna_sequence,
    get_available_rna_types,
    print_usage_header
)


def generate_sequence(
    model,
    tokenizer,
    config,
    prefix_sequence: str = "",
    rna_type: str = None,
    lineage: str = None,
    max_length: int = 200,
    max_new_tokens: int = None,
    # Beam Search参数
    use_beam_search: bool = False,
    num_beams: int = 5,
    chunk_size: int = 10,
    # 采样参数
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    device: str = 'cpu',
    verbose: bool = True
) -> list:
    """
    自回归生成RNA序列

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        prefix_sequence: 前缀序列（可选，用于续写）
        rna_type: RNA类型（可选）
        lineage: 谱系信息（可选）
        max_length: 最大生成长度
        max_new_tokens: 最大新生成token数（与max_length二选一）
        use_beam_search: 是否使用Chunk-based Beam Search
        num_beams: Beam数量（仅用于beam search）
        chunk_size: Chunk大小（仅用于beam search）
        temperature: 温度参数（越大越随机）
        top_p: nucleus sampling参数
        top_k: top-k sampling参数
        num_return_sequences: 生成序列数量
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        生成的序列列表
    """
    # 验证前缀序列
    if prefix_sequence and not validate_rna_sequence(prefix_sequence):
        raise ValueError(f"无效的前缀序列: {prefix_sequence}")

    # 构建条件前缀
    conditional_prefix = build_conditional_prefix(rna_type=rna_type, lineage=lineage)

    # 完整输入序列
    input_sequence = conditional_prefix + prefix_sequence

    if verbose:
        print(f"\n生成参数:")
        if conditional_prefix:
            print(f"  条件前缀: {conditional_prefix}")
        if prefix_sequence:
            print(f"  前缀序列: {prefix_sequence}")
        else:
            print(f"  模式: 从头生成")
        print(f"  最大长度: {max_length}")
        if max_new_tokens:
            print(f"  最大新token数: {max_new_tokens}")

        # 根据生成方法打印不同参数
        if use_beam_search:
            print(f"  生成方法: Chunk-based Beam Search")
            print(f"  Beam数量: {num_beams}")
            print(f"  Chunk大小: {chunk_size}")
            print(f"  温度: {temperature} (用于chunk内部生成)")
        else:
            print(f"  生成方法: 采样")
            print(f"  温度: {temperature}")
            print(f"  Top-p: {top_p}")
            print(f"  Top-k: {top_k}")

        print(f"  生成数量: {num_return_sequences}")

    # 准备输入
    inputs = prepare_input_ids(tokenizer, input_sequence, device)

    # 获取EOS和PAD token ID
    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>")

    # 创建output_token_mask（只允许输出RNA碱基和EOS）
    # 使用模型的vocab_size而不是tokenizer的，避免维度不匹配
    vocab_size = config.vocab_size  # 使用模型的114，而不是tokenizer的112
    output_token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    # 允许的token：AUGC和<eos>
    allowed_tokens = ['A', 'U', 'G', 'C', '<eos>']
    for token in allowed_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            output_token_mask[token_id] = True

    if verbose and not use_beam_search:
        print(f"\n开始生成...")

    # 生成
    with torch.no_grad():
        if use_beam_search:
            # 使用Chunk-based Beam Search
            generated_ids = model.chunk_beam_search_generate(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                sequence_ids=inputs['sequence_ids'],
                num_beams=num_beams,
                chunk_size=chunk_size,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                output_token_mask=output_token_mask,
                verbose=verbose
            )
        else:
            # 使用原有的采样方法
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                sequence_ids=inputs['sequence_ids'],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                output_token_mask=output_token_mask
            )

    # 解码生成的序列
    results = []
    for i, gen_ids in enumerate(generated_ids):
        # 解码完整序列
        full_sequence = decode_sequence(tokenizer, gen_ids, remove_special_tokens=True)

        # 提取纯RNA序列
        rna_sequence = extract_rna_sequence(full_sequence)

        results.append({
            'full_sequence': full_sequence,
            'rna_sequence': rna_sequence,
            'length': len(rna_sequence),
            'conditional_prefix': conditional_prefix,
            'prefix_sequence': prefix_sequence,
            'rna_type': rna_type,
            'lineage': lineage
        })

        if verbose:
            print(f"\n生成序列 {i+1}:")
            print(f"  完整序列: {full_sequence}")
            print(f"  RNA序列: {rna_sequence}")
            print(f"  长度: {len(rna_sequence)}")

    return results


def interactive_mode(model, tokenizer, config, device: str):
    """
    交互式模式

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        config: 模型配置
        device: 设备
    """
    print("\n进入交互式模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("-" * 80)

    while True:
        print("\n选择生成模式:")
        print("1. 从头生成（仅条件前缀）")
        print("2. 续写生成（提供前缀序列）")
        mode = input("> ").strip()

        if mode.lower() in ['quit', 'exit', 'q']:
            print("退出交互式模式")
            break

        if mode.lower() == 'help':
            print("\n使用说明:")
            print("1. 选择生成模式（从头生成或续写）")
            print("2. 可选：输入RNA类型和谱系信息")
            print("3. 设置生成参数（长度、温度等）")
            print("\n支持的RNA类型:")
            for rna_type in get_available_rna_types():
                print(f"  - {rna_type}")
            continue

        if mode not in ['1', '2']:
            print("无效的选择，请输入1或2")
            continue

        # 获取条件前缀
        print("\n是否使用RNA类型？(留空跳过)")
        rna_type = input("RNA类型: ").strip() or None

        print("\n是否使用谱系信息？(留空跳过)")
        lineage = input("谱系: ").strip() or None

        # 获取前缀序列（续写模式）
        prefix_sequence = ""
        if mode == '2':
            print("\n请输入前缀序列（只包含AUGC）:")
            prefix_sequence = input("> ").strip()

        # 获取生成参数
        print("\n生成参数设置:")
        max_length_str = input("最大长度 (默认200): ").strip()
        max_length = int(max_length_str) if max_length_str else 200

        temperature_str = input("温度 (默认1.0): ").strip()
        temperature = float(temperature_str) if temperature_str else 1.0

        num_seqs_str = input("生成数量 (默认1): ").strip()
        num_return_sequences = int(num_seqs_str) if num_seqs_str else 1

        # 生成
        try:
            results = generate_sequence(
                model, tokenizer, config,
                prefix_sequence=prefix_sequence,
                rna_type=rna_type,
                lineage=lineage,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                device=device,
                verbose=True
            )
        except Exception as e:
            print(f"错误: {e}")
            continue


def batch_mode(
    model,
    tokenizer,
    config,
    input_file: str,
    output_file: str,
    max_length: int,
    temperature: float,
    device: str
):
    """
    批量处理模式

    输入文件格式（每行）:
        前缀序列 [TAB] RNA类型 [TAB] 谱系

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        config: 模型配置
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_length: 最大生成长度
        temperature: 温度参数
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
            # 只移除尾部换行符，不移除制表符
            line = line.rstrip('\n\r')
            # 跳过空行和注释行（检查strip后的结果）
            if not line.strip() or line.strip().startswith('#'):
                continue

            # 解析行（使用原始line，保留制表符）
            parts = line.split('\t')
            prefix_sequence = parts[0].strip() if parts[0].strip() else ""
            rna_type = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            lineage = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None

            print(f"\n处理第 {line_num} 条...")

            try:
                gen_results = generate_sequence(
                    model, tokenizer, config,
                    prefix_sequence=prefix_sequence,
                    rna_type=rna_type,
                    lineage=lineage,
                    max_length=max_length,
                    temperature=temperature,
                    device=device,
                    verbose=False
                )
                results.extend(gen_results)
                print(f"  前缀: {prefix_sequence[:30]}{'...' if len(prefix_sequence) > 30 else ''}")
                print(f"  生成长度: {gen_results[0]['length']}")
            except Exception as e:
                print(f"  错误: {e}")
                continue

    # 写入输出文件（CSV格式）
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        
        # 写入表头
        csv_writer.writerow(['前缀序列', 'RNA类型', '谱系', '生成序列', '长度'])

        # 写入结果
        for result in results:
            csv_writer.writerow([
                result['prefix_sequence'],
                result.get('rna_type') or '',
                result.get('lineage') or '',
                result['rna_sequence'],
                result['length']
            ])

    print(f"\n完成！共生成 {len(results)} 条序列")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='RNAGen模型 - 自回归序列生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 --interactive

  # 从头生成
  python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --rna_type mRNA --max_length 100

  # 续写生成
  python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --prefix_sequence "AUGCAUGC" --max_new_tokens 50

  # 批量生成
  python 2_autoregressive_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --input_file prompts.txt --output_file generated.txt
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

    # 生成参数
    parser.add_argument(
        '--prefix_sequence',
        type=str,
        default="",
        help='前缀序列（用于续写，留空则从头生成）'
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

    parser.add_argument(
        '--max_length',
        type=int,
        default=200,
        help='最大生成长度 (默认: 200)'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        help='最大新生成token数（与max_length二选一）'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='温度参数 (默认: 1.0)'
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p采样参数 (默认: 0.9)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k采样参数 (默认: 50)'
    )

    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='生成序列数量 (默认: 1)'
    )

    # Beam Search参数
    parser.add_argument(
        '--use_beam_search',
        action='store_true',
        help='使用Chunk-based Beam Search而不是采样'
    )

    parser.add_argument(
        '--num_beams',
        type=int,
        default=5,
        help='Beam数量 (默认: 5)'
    )

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=10,
        help='每个chunk的token数 (默认: 10)'
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
        "RNAGen - 自回归序列生成",
        "生成RNA序列（从头生成或续写）"
    )

    # 加载模型
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device

    # 根据模式运行
    if args.interactive:
        # 交互式模式
        interactive_mode(model, tokenizer, config, str(device))

    elif args.input_file and args.output_file:
        # 批量处理模式
        batch_mode(
            model, tokenizer, config,
            args.input_file, args.output_file,
            args.max_length, args.temperature,
            str(device)
        )

    else:
        # 单次生成模式
        results = generate_sequence(
            model, tokenizer, config,
            prefix_sequence=args.prefix_sequence,
            rna_type=args.rna_type,
            lineage=args.lineage,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            use_beam_search=args.use_beam_search,
            num_beams=args.num_beams,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            device=str(device),
            verbose=True
        )


if __name__ == "__main__":
    main()
