#!/usr/bin/env python3
"""
FASTA序列打分脚本
使用Evo2模型对FASTA文件中的DNA序列进行打分，输出CSV文件
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
from tqdm import tqdm

# 导入Evo2模型
from evo2.models import Evo2


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用Evo2模型对FASTA文件中的DNA序列打分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.fasta output.csv
  %(prog)s input.fasta output.csv --model evo2_7b
  %(prog)s input.fasta output.csv --model evo2_1b_base --batch-size 8
        """
    )

    parser.add_argument(
        'input_fasta',
        type=str,
        help='输入的FASTA文件路径'
    )

    parser.add_argument(
        'output_csv',
        type=str,
        help='输出的CSV文件路径'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='evo2_7b_base',
        choices=['evo2_1b_base', 'evo2_7b_base', 'evo2_7b', 'evo2_40b', 'evo2_40b_base'],
        help='选择使用的Evo2模型 (默认: evo2_7b_base)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='批处理大小 (默认: 1)'
    )

    parser.add_argument(
        '--reduce-method',
        type=str,
        default='mean',
        choices=['mean', 'sum'],
        help='分数计算方式: mean (平均) 或 sum (求和) (默认: mean)'
    )

    parser.add_argument(
        '--no-rc',
        action='store_true',
        help='禁用反向互补平均（默认会对正向和反向互补序列都打分后取平均）'
    )

    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='禁用torch.compile优化（首次运行可用此选项快速测试，但推理会较慢）'
    )

    parser.add_argument(
        '--local-path',
        type=str,
        default=None,
        help='本地模型checkpoint路径（如果不指定则从HuggingFace下载）'
    )

    return parser.parse_args()


def read_fasta_sequences(fasta_file: str) -> List[Tuple[str, str, int]]:
    """
    读取FASTA文件中的所有序列

    自动将 RNA 序列（含 U）转换为 DNA 序列（T）

    返回: [(序列ID, 序列内容, 序列长度), ...]
    """
    sequences = []

    try:
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                seq_id = record.id
                # 转换为大写并将 U 替换为 T（RNA → DNA）
                seq_str = str(record.seq).upper().replace('U', 'T')
                seq_len = len(seq_str)
                sequences.append((seq_id, seq_str, seq_len))

        if not sequences:
            raise ValueError(f"FASTA文件中没有找到序列: {fasta_file}")

        print(f"从 {fasta_file} 中读取了 {len(sequences)} 条序列")
        return sequences

    except FileNotFoundError:
        print(f"错误: 找不到文件 {fasta_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取FASTA文件时出错: {e}", file=sys.stderr)
        sys.exit(1)


def score_sequences_batch(
    sequences: List[Tuple[str, str, int]],
    model: Evo2,
    batch_size: int,
    reduce_method: str,
    average_rc: bool
) -> List[Tuple[str, int, float]]:
    """
    批量对序列打分

    返回: [(序列ID, 序列长度, 分数), ...]
    """
    results = []

    # 提取序列ID、序列内容和长度
    seq_ids = [s[0] for s in sequences]
    seq_strings = [s[1] for s in sequences]
    seq_lengths = [s[2] for s in sequences]

    print(f"\n使用模型打分中...")
    print(f"参数: batch_size={batch_size}, reduce_method={reduce_method}, average_reverse_complement={average_rc}")
    print("注意：首次调用会触发模型编译，可能需要几分钟...")

    try:
        # 调用Evo2的score_sequences方法
        scores = model.score_sequences(
            seqs=seq_strings,
            batch_size=batch_size,
            prepend_bos=False,
            reduce_method=reduce_method,
            average_reverse_complement=average_rc
        )

        # 组合结果
        for seq_id, seq_len, score in zip(seq_ids, seq_lengths, scores):
            results.append((seq_id, seq_len, score))

        return results

    except Exception as e:
        print(f"错误: 打分过程中出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def write_csv_output(output_file: str, results: List[Tuple[str, int, float]]):
    """
    将结果写入CSV文件
    """
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            writer.writerow(['sequence_id', 'sequence_length', 'score'])

            # 写入数据
            for seq_id, seq_len, score in results:
                writer.writerow([seq_id, seq_len, f'{score:.6f}'])

        print(f"\n结果已保存到: {output_file}")
        print(f"共处理 {len(results)} 条序列")

    except Exception as e:
        print(f"错误: 写入CSV文件时出错: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()

    # 切换到 evo2 目录（确保能找到 configs/ 目录）
    import os
    evo2_dir = '/data4/huangyanjie/rna_benchmark/evo2_test/evo2'
    if os.path.exists(evo2_dir):
        os.chdir(evo2_dir)
        print(f"工作目录: {os.getcwd()}")

    # 禁用torch.compile（如果用户指定）
    if args.no_compile:
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        print("⚠️  已禁用 torch.compile 优化（快速测试模式）")

    # 禁用 FP8（A100 不支持，需要 H100 或更高）
    os.environ['TE_FP8_ENABLED'] = '0'
    os.environ['TRANSFORMERENGINE_FP8_ENABLED'] = '0'
    # Patch yaml.load to force use_fp8_input_projections=False on A100
    import yaml as _yaml
    _orig_yaml_load = _yaml.load
    def _patched_yaml_load(stream, Loader=None):
        data = _orig_yaml_load(stream, Loader=Loader)
        if isinstance(data, dict) and 'use_fp8_input_projections' in data:
            data['use_fp8_input_projections'] = False
        return data
    _yaml.load = _patched_yaml_load
    print("⚠️  已禁用 FP8 推理（A100 GPU 不支持）")

    # 检查输入文件是否存在
    if not Path(args.input_fasta).exists():
        print(f"错误: 输入文件不存在: {args.input_fasta}", file=sys.stderr)
        sys.exit(1)

    print(f"=" * 60)
    print(f"Evo2 序列打分工具")
    print(f"=" * 60)
    print(f"输入文件: {args.input_fasta}")
    print(f"输出文件: {args.output_csv}")
    print(f"使用模型: {args.model}")
    print(f"=" * 60)

    # 读取FASTA序列
    print("\n[1/3] 读取FASTA文件...")
    sequences = read_fasta_sequences(args.input_fasta)

    # 加载模型
    print(f"\n[2/3] 加载模型 {args.model}...")
    if args.local_path:
        print(f"使用本地checkpoint: {args.local_path}")
    else:
        print("(首次使用会自动从HuggingFace下载，可能需要一些时间)")
    print("\n⚠️  重要提示：")
    print("   - 首次运行会进行模型编译优化（torch.compile）")
    print("   - 这个过程可能需要 5-15 分钟，请耐心等待")
    print("   - 编译完成后会自动缓存，后续运行会很快")
    print("   - 如果看起来卡住了，不要担心，正在编译中...\n")
    try:
        model = Evo2(model_name=args.model, local_path=args.local_path)
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 对序列打分
    print(f"\n[3/3] 对序列打分...")
    average_rc = not args.no_rc
    results = score_sequences_batch(
        sequences=sequences,
        model=model,
        batch_size=args.batch_size,
        reduce_method=args.reduce_method,
        average_rc=average_rc
    )

    # 写入CSV
    write_csv_output(args.output_csv, results)

    print(f"\n{'=' * 60}")
    print("完成!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
