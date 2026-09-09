#!/usr/bin/env python3
"""
RNAGen模型 - 片段重新生成（Infilling）

功能：
1. 选定完整序列的某个区域
2. 参考上下文对指定区域进行重新生成
3. 支持多个区域同时重新生成
4. 支持条件前缀（RNA类型和/或谱系信息）
5. 支持交互式模式和批量处理模式

使用方法：
    # 交互式模式
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 --interactive

    # 单区域重新生成（保持原长度）
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \
        --sequence "AUGCAUGCAUGCAUGC" --start 4 --end 8

    # 单区域重新生成（指定目标长度）
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \
        --sequence "AUGCAUGCAUGCAUGC" --start 4 --end 8 --target_length 6

    # 多区域重新生成（保持原长度）
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \
        --sequence "AUGCAUGCAUGCAUGC" --regions "4-8,12-16"

    # 多区域重新生成（指定不同目标长度）
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \
        --sequence "AUGCAUGCAUGCAUGC" --regions "4-8-6,12-16-3"

    # 批量处理
    python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \
        --input_file infill_tasks.txt --output_file infilled.txt
"""

import argparse
import csv
import sys
import torch
from pathlib import Path
from typing import List, Tuple

# 导入工具函数（从 usage_helpers.py）
from usage_helpers import (
    load_model,
    build_conditional_prefix,
    validate_rna_sequence,
    decode_sequence,
    extract_rna_sequence,
    get_available_rna_types,
    print_usage_header
)


def create_infilling_input(
    sequence: str,
    regions: List[Tuple[int, int, int]],
    tokenizer
) -> Tuple[str, List[str]]:
    """
    创建infilling任务的输入

    Args:
        sequence: 完整序列
        regions: 需要重新生成的区域列表 [(start, end, target_length), ...]
                target_length为目标长度，可以和(end-start)不同
        tokenizer: tokenizer实例

    Returns:
        (masked_sequence, original_spans)
        masked_sequence: 带mask的序列
        original_spans: 原始片段列表
    """
    # 验证区域
    for start, end, target_length in regions:
        if start < 0 or end > len(sequence) or start >= end:
            raise ValueError(f"无效的区域: ({start}, {end}), 序列长度: {len(sequence)}")
        if target_length <= 0:
            raise ValueError(f"无效的目标长度: {target_length}")

    # 检查区域是否重叠
    sorted_regions = sorted(regions, key=lambda x: x[0])
    for i in range(len(sorted_regions) - 1):
        if sorted_regions[i][1] > sorted_regions[i + 1][0]:
            raise ValueError(f"区域重叠: {sorted_regions[i][:2]} 和 {sorted_regions[i + 1][:2]}")

    # 构建masked序列和GLM格式
    # GLM格式: 序列[GLM]start-end-target_length;...
    original_spans = []
    glm_spans = []

    for start, end, target_length in sorted_regions:
        original_span = sequence[start:end]
        original_spans.append(original_span)
        glm_spans.append(f"{start}-{end}-{target_length}")

    # 创建GLM格式字符串
    glm_string = sequence + "[GLM]" + ";".join(glm_spans)

    return glm_string, original_spans


def infill_sequence(
    model,
    tokenizer,
    config,
    sequence: str,
    regions: List[Tuple[int, int, int]],
    rna_type: str = None,
    lineage: str = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    device: str = 'cpu',
    verbose: bool = True
) -> list:
    """
    对序列的指定区域进行重新生成

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        config: 模型配置
        sequence: 完整序列
        regions: 需要重新生成的区域列表 [(start, end, target_length), ...]
        rna_type: RNA类型（可选）
        lineage: 谱系信息（可选）
        temperature: 温度参数
        top_p: nucleus sampling参数
        top_k: top-k sampling参数
        num_return_sequences: 生成序列数量
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        生成结果列表
    """
    # 验证序列
    if not validate_rna_sequence(sequence):
        raise ValueError(f"无效的RNA序列: {sequence}")

    # 创建infilling输入
    glm_string, original_spans = create_infilling_input(sequence, regions, tokenizer)

    if verbose:
        print(f"\nInfilling任务:")
        print(f"  原始序列: {sequence}")
        print(f"  序列长度: {len(sequence)}")
        print(f"  重新生成区域:")
        for i, (start, end, target_length) in enumerate(regions):
            print(f"    区域 {i+1}: [{start}:{end}] (长度 {end-start}) -> 目标长度 {target_length}")
            print(f"    原始片段: '{original_spans[i]}'")

    # 构建条件前缀
    conditional_prefix = build_conditional_prefix(rna_type=rna_type, lineage=lineage)

    # 完整输入
    full_input = conditional_prefix + glm_string

    if verbose:
        if conditional_prefix:
            print(f"  条件前缀: {conditional_prefix}")
        print(f"  GLM格式: {glm_string}")
        print(f"\n生成参数:")
        print(f"  温度: {temperature}")
        print(f"  Top-p: {top_p}")
        print(f"  Top-k: {top_k}")
        print(f"  生成数量: {num_return_sequences}")

    # 准备输入（使用batch_preparer的GLM处理逻辑）
    # 注意：这里需要使用模型的batch_preparer来正确处理GLM格式
    # 为了简化，我们直接使用模型的generate方法

    # 由于模型的generate方法需要特定的输入格式，
    # 我们需要手动构建GLM格式的输入

    # 导入batch_preparer
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # 这里简化处理：直接使用自回归生成来模拟infilling
    # 实际应该使用GLM的span filling机制
    # 但由于模型代码中的GLM处理比较复杂，这里提供一个简化版本

    print("\n警告: 当前实现使用简化的infilling方法")
    print("完整的GLM span filling功能需要更复杂的输入准备")

    # 简化方法：对每个区域单独处理
    results = []

    for seq_idx in range(num_return_sequences):
        # 创建新序列（考虑长度变化需要重建）
        infilled_spans = []  # 存储每个区域重新生成的片段

        # 对每个区域进行重新生成
        for region_idx, (start, end, target_length) in enumerate(regions):
            # 获取上下文
            left_context = sequence[:start]
            right_context = sequence[end:]

            # 构建prompt（使用左侧上下文）
            prompt = conditional_prefix + left_context

            # 准备输入
            from usage_helpers import prepare_input_ids
            inputs = prepare_input_ids(tokenizer, prompt, device)

            # 获取token mask
            # 使用模型的vocab_size而不是tokenizer的，避免维度不匹配
            vocab_size = config.vocab_size  # 使用模型的114，而不是tokenizer的112
            output_token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            allowed_tokens = ['A', 'U', 'G', 'C', '<eos_span>']
            for token in allowed_tokens:
                token_id = tokenizer.token_to_id(token)
                if token_id is not None:
                    output_token_mask[token_id] = True

            # 生成
            eos_token_id = tokenizer.token_to_id("<eos_span>")
            pad_token_id = tokenizer.token_to_id("<pad>")

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    sequence_ids=inputs['sequence_ids'],
                    max_new_tokens=target_length + 10,  # 允许一些余量
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=1,
                    do_sample=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    output_token_mask=output_token_mask
                )

            # 解码生成的片段
            generated_text = decode_sequence(tokenizer, generated_ids[0], remove_special_tokens=True)
            generated_span = extract_rna_sequence(generated_text)

            # 移除prompt部分，只保留新生成的部分
            if generated_span.startswith(left_context):
                generated_span = generated_span[len(left_context):]

            # 截取到目标长度
            generated_span = generated_span[:target_length]

            # 如果生成的长度不够，用N填充
            if len(generated_span) < target_length:
                generated_span += 'N' * (target_length - len(generated_span))

            infilled_spans.append(generated_span)

        # 重建序列（考虑长度变化）
        new_sequence_parts = []
        current_pos = 0
        
        for region_idx, (start, end, target_length) in enumerate(regions):
            # 添加区域前的原始序列
            new_sequence_parts.append(sequence[current_pos:start])
            # 添加重新生成的片段
            new_sequence_parts.append(infilled_spans[region_idx])
            # 更新位置
            current_pos = end
        
        # 添加最后一个区域后的原始序列
        new_sequence_parts.append(sequence[current_pos:])
        
        new_sequence_str = ''.join(new_sequence_parts)

        results.append({
            'original_sequence': sequence,
            'infilled_sequence': new_sequence_str,
            'regions': regions,
            'original_spans': original_spans,
            'infilled_spans': infilled_spans,
            'conditional_prefix': conditional_prefix,
            'rna_type': rna_type,
            'lineage': lineage
        })

        if verbose:
            print(f"\n生成结果 {seq_idx + 1}:")
            print(f"  原始序列: {sequence} (长度: {len(sequence)})")
            print(f"  重新生成: {new_sequence_str} (长度: {len(new_sequence_str)})")
            # 计算每个区域在新序列中的位置
            current_pos = 0
            for i, (start, end, target_length) in enumerate(regions):
                # 计算新序列中该区域的起始位置
                new_start = start + sum(regions[j][2] - (regions[j][1] - regions[j][0]) for j in range(i))
                new_end = new_start + target_length
                
                original = original_spans[i]
                new_span = infilled_spans[i]
                print(f"  区域 {i+1}:")
                print(f"    原始位置: [{start}:{end}] (长度: {end-start})")
                print(f"    新的位置: [{new_start}:{new_end}] (长度: {target_length})")
                print(f"    原始片段: {original}")
                print(f"    新的片段: {new_span}")

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
        print("\n请输入完整的RNA序列:")
        sequence = input("> ").strip()

        if sequence.lower() in ['quit', 'exit', 'q']:
            print("退出交互式模式")
            break

        if sequence.lower() == 'help':
            print("\n使用说明:")
            print("1. 输入完整的RNA序列")
            print("2. 指定需要重新生成的区域（起始和结束位置）")
            print("3. 可选：输入RNA类型和谱系信息")
            print("4. 设置生成参数")
            print("\n示例:")
            print("  序列: AUGCAUGCAUGCAUGC")
            print("  区域: 4-8 (重新生成第4到第8位)")
            print("  多区域: 4-8,12-16")
            continue

        if not sequence:
            continue

        # 显示序列和位置
        print(f"\n序列: {sequence}")
        print(f"位置: {''.join(str(i % 10) for i in range(len(sequence)))}")

        # 获取区域
        print("\n请输入需要重新生成的区域:")
        print("格式1: start-end (保持原长度)")
        print("格式2: start-end-target_length (指定目标长度)")
        print("多区域: start1-end1-length1,start2-end2-length2")
        regions_str = input("> ").strip()

        try:
            # 解析区域
            regions = []
            for region_str in regions_str.split(','):
                parts = region_str.strip().split('-')
                if len(parts) == 2:
                    # 格式: start-end，默认target_length = end-start
                    start, end = map(int, parts)
                    target_length = end - start
                elif len(parts) == 3:
                    # 格式: start-end-target_length
                    start, end, target_length = map(int, parts)
                else:
                    raise ValueError(f"无效的区域格式: {region_str}")
                regions.append((start, end, target_length))
        except Exception as e:
            print(f"错误: 无效的区域格式 - {e}")
            continue

        # 获取条件前缀
        print("\n是否使用RNA类型？(留空跳过)")
        rna_type = input("RNA类型: ").strip() or None

        print("\n是否使用谱系信息？(留空跳过)")
        lineage = input("谱系: ").strip() or None

        # 获取生成参数
        print("\n生成参数设置:")
        temperature_str = input("温度 (默认1.0): ").strip()
        temperature = float(temperature_str) if temperature_str else 1.0

        num_seqs_str = input("生成数量 (默认1): ").strip()
        num_return_sequences = int(num_seqs_str) if num_seqs_str else 1

        # 生成
        try:
            results = infill_sequence(
                model, tokenizer, config, sequence, regions,
                rna_type=rna_type,
                lineage=lineage,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                device=device,
                verbose=True
            )
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            continue


def batch_mode(
    model,
    tokenizer,
    config,
    input_file: str,
    output_file: str,
    temperature: float,
    device: str
):
    """
    批量处理模式

    输入文件格式（每行）:
        序列 [TAB] 区域(start-end-length或start1-end1-length1,start2-end2-length2) [TAB] RNA类型 [TAB] 谱系
        注意：如果不指定length，则默认为end-start（保持原长度）

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        config: 模型配置
        input_file: 输入文件路径
        output_file: 输出文件路径
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
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 解析行
            parts = line.split('\t')
            sequence = parts[0].strip()
            regions_str = parts[1].strip() if len(parts) > 1 else ""
            rna_type = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            lineage = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None

            # 解析区域
            try:
                regions = []
                for region_str in regions_str.split(','):
                    parts = region_str.strip().split('-')
                    if len(parts) == 2:
                        # 格式: start-end，默认target_length = end-start
                        start, end = map(int, parts)
                        target_length = end - start
                    elif len(parts) == 3:
                        # 格式: start-end-target_length
                        start, end, target_length = map(int, parts)
                    else:
                        raise ValueError(f"无效的区域格式: {region_str}")
                    regions.append((start, end, target_length))
            except Exception as e:
                print(f"第 {line_num} 行: 无效的区域格式 - {e}")
                continue

            print(f"\n处理第 {line_num} 条...")

            try:
                infill_results = infill_sequence(
                    model, tokenizer, config, sequence, regions,
                    rna_type=rna_type,
                    lineage=lineage,
                    temperature=temperature,
                    device=device,
                    verbose=False
                )
                results.extend(infill_results)
                print(f"  序列长度: {len(sequence)}")
                print(f"  区域数量: {len(regions)}")
            except Exception as e:
                print(f"  错误: {e}")
                continue

    # 写入输出文件（CSV格式）
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        
        # 写入表头
        csv_writer.writerow(['原始序列', '区域', '原始片段', '重设计片段', 'RNA类型', '谱系', '重新生成序列'])

        # 写入结果
        for result in results:
            regions_str = ','.join(f"{s}-{e}-{tl}" for s, e, tl in result['regions'])
            original_spans_str = ','.join(result['original_spans'])
            infilled_spans_str = ','.join(result.get('infilled_spans', result['original_spans']))
            csv_writer.writerow([
                result['original_sequence'],
                regions_str,
                original_spans_str,
                infilled_spans_str,
                result.get('rna_type') or '',
                result.get('lineage') or '',
                result['infilled_sequence']
            ])

    print(f"\n完成！共处理 {len(results)} 条序列")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='RNAGen模型 - 片段重新生成（Infilling）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 --interactive

  # 单区域重新生成（保持原长度）
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --sequence "AUGCAUGCAUGCAUGC" --start 4 --end 8

  # 单区域重新生成（指定目标长度）
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --sequence "AUGCAUGCAUGCAUGC" --start 4 --end 8 --target_length 6

  # 多区域重新生成（保持原长度）
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --sequence "AUGCAUGCAUGCAUGC" --regions "4-8,12-16"

  # 多区域重新生成（指定不同目标长度）
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --sequence "AUGCAUGCAUGCAUGC" --regions "4-8-6,12-16-3"

  # 批量处理
  python 3_infilling_generation.py --checkpoint ../checkpoint_100M_1018 \\
      --input_file infill_tasks.txt --output_file infilled.txt
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

    # 序列和区域
    parser.add_argument(
        '--sequence',
        type=str,
        help='完整的RNA序列'
    )

    parser.add_argument(
        '--start',
        type=int,
        help='重新生成区域的起始位置（单区域模式）'
    )

    parser.add_argument(
        '--end',
        type=int,
        help='重新生成区域的结束位置（单区域模式）'
    )

    parser.add_argument(
        '--regions',
        type=str,
        help='多个区域（格式: start1-end1-length1,start2-end2-length2 或 start1-end1,start2-end2）'
    )

    parser.add_argument(
        '--target_length',
        type=int,
        help='重新生成区域的目标长度（与--start/--end配合使用，可选）'
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

    # 生成参数
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
        "RNAGen - 片段重新生成（Infilling）",
        "对序列的指定区域进行重新生成"
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
            args.temperature,
            str(device)
        )

    elif args.sequence:
        # 单次infilling模式
        # 解析区域
        if args.regions:
            regions = []
            for region_str in args.regions.split(','):
                parts = region_str.strip().split('-')
                if len(parts) == 2:
                    start, end = map(int, parts)
                    target_length = end - start
                elif len(parts) == 3:
                    start, end, target_length = map(int, parts)
                else:
                    print(f"错误: 无效的区域格式: {region_str}")
                    sys.exit(1)
                regions.append((start, end, target_length))
        elif args.start is not None and args.end is not None:
            # 检查是否指定了目标长度
            if hasattr(args, 'target_length') and args.target_length is not None:
                target_length = args.target_length
            else:
                target_length = args.end - args.start
            regions = [(args.start, args.end, target_length)]
        else:
            print("错误: 请指定区域（--start 和 --end，或 --regions）")
            sys.exit(1)

        results = infill_sequence(
            model, tokenizer, config, args.sequence, regions,
            rna_type=args.rna_type,
            lineage=args.lineage,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            device=str(device),
            verbose=True
        )

    else:
        print("错误: 请指定运行模式")
        print("  --interactive: 交互式模式")
        print("  --sequence: 单次infilling模式")
        print("  --input_file 和 --output_file: 批量处理模式")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
