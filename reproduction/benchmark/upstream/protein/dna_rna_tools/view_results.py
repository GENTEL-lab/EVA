#!/usr/bin/env python3
"""
查看Likelihood计算结果的便捷工具

使用方法:
    # 查看最新的结果
    python view_results.py

    # 查看指定的结果文件
    python view_results.py output/likelihood_result_20251020_071810.txt

    # 查看所有结果
    python view_results.py --all
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def find_latest_result(output_dir='output'):
    """查找最新的结果文件"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    result_files = list(output_path.glob('likelihood_result_*.txt'))
    if not result_files:
        return None

    # 按修改时间排序，返回最新的
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    return latest


def find_all_results(output_dir='output'):
    """查找所有结果文件"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    result_files = list(output_path.glob('likelihood_result_*.txt'))
    # 按修改时间排序（最新的在前）
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files


def view_result(result_file):
    """查看单个结果文件"""
    result_path = Path(result_file)

    if not result_path.exists():
        print(f"❌ 结果文件不存在: {result_file}")
        return False

    print("=" * 60)
    print(f"📊 Likelihood计算结果")
    print("=" * 60)
    print(f"文件: {result_path}")
    print(f"时间: {datetime.fromtimestamp(result_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # 读取并显示内容
    with open(result_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)

    print("-" * 60)

    # 查找对应的配置文件和流程总结
    timestamp = result_path.stem.replace('likelihood_result_', '')
    config_file = result_path.parent / f'config_likelihood_{timestamp}.yaml'
    summary_file = result_path.parent / f'pipeline_summary_{timestamp}.txt'
    log_file = result_path.parent / f'likelihood_log_{timestamp}.log'

    print("\n📁 相关文件:")
    if config_file.exists():
        print(f"  配置文件: {config_file}")
    if summary_file.exists():
        print(f"  流程总结: {summary_file}")
    if log_file.exists():
        print(f"  运行日志: {log_file}")

    return True


def view_summary(summary_file):
    """查看流程总结"""
    summary_path = Path(summary_file)

    if not summary_path.exists():
        return

    print("\n" + "=" * 60)
    print("📝 流程总结")
    print("=" * 60)

    with open(summary_path, 'r', encoding='utf-8') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description='查看Likelihood计算结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'result_file',
        nargs='?',
        help='结果文件路径（不指定则查看最新的）'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='列出所有结果文件'
    )

    parser.add_argument(
        '--output_dir',
        default='output',
        help='输出目录（默认：output）'
    )

    parser.add_argument(
        '--with_summary',
        action='store_true',
        help='同时显示流程总结'
    )

    args = parser.parse_args()

    # 列出所有结果
    if args.all:
        print("=" * 60)
        print("📋 所有Likelihood计算结果")
        print("=" * 60)

        results = find_all_results(args.output_dir)

        if not results:
            print("未找到任何结果文件")
            print(f"\n提示: 请先运行likelihood计算")
            print("  python protein_to_rna_likelihood_pipeline.py --length 30 --auto_run")
            return

        for i, result in enumerate(results, 1):
            timestamp = datetime.fromtimestamp(result.stat().st_mtime)
            size = result.stat().st_size
            print(f"\n{i}. {result.name}")
            print(f"   时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   大小: {size} bytes")
            print(f"   路径: {result}")

        print("\n" + "=" * 60)
        print(f"共找到 {len(results)} 个结果文件")
        print("\n查看具体结果:")
        print(f"  python view_results.py {results[0]}")
        return

    # 查看指定或最新的结果
    if args.result_file:
        result_file = args.result_file
    else:
        result_file = find_latest_result(args.output_dir)
        if not result_file:
            print("❌ 未找到任何结果文件")
            print(f"\n💡 提示:")
            print("1. 请先运行likelihood计算:")
            print("   python protein_to_rna_likelihood_pipeline.py --length 30 --auto_run")
            print("\n2. 或者手动运行:")
            print("   cd /lnvme/home/huang_yan_jie/rna_benchmark/rnagen/scripts")
            print("   ./run_likelihood.sh <配置文件路径>")
            print("\n3. 查看所有配置文件:")
            print("   ls -lht output/config_*.yaml")
            return

        print(f"📌 查看最新的结果文件\n")

    # 显示结果
    success = view_result(result_file)

    # 显示流程总结
    if success and args.with_summary:
        result_path = Path(result_file)
        timestamp = result_path.stem.replace('likelihood_result_', '')
        summary_file = result_path.parent / f'pipeline_summary_{timestamp}.txt'
        if summary_file.exists():
            view_summary(summary_file)

    print()


if __name__ == '__main__':
    main()
