#!/usr/bin/env python3
"""
为已有的评估结果生成可视化图表
适用于评估完成后手动生成图表的场景
"""

import sys
import json
import argparse
from pathlib import Path

# 添加eval目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from scripts.result_aggregator import ResultAggregator


def generate_plots_for_experiment(experiment_dir: Path, stage: int):
    """
    为指定实验目录生成图表

    Args:
        experiment_dir: 实验目录路径
        stage: 训练阶段 (1, 2, 3)
    """
    print(f"📊 为实验生成图表: {experiment_dir.name}")
    print(f"阶段: Stage {stage}")
    print("=" * 60)

    # 检查目录是否存在
    if not experiment_dir.exists():
        print(f"❌ 错误: 目录不存在: {experiment_dir}")
        return False

    # 初始化结果聚合器
    aggregator = ResultAggregator(experiment_dir, stage=stage)

    # 加载所有checkpoint结果
    checkpoints_dir = experiment_dir / 'checkpoints'
    if not checkpoints_dir.exists():
        print(f"❌ 错误: checkpoints目录不存在: {checkpoints_dir}")
        return False

    all_results = []
    for ckpt_dir in sorted(checkpoints_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        metrics_file = ckpt_dir / 'checkpoint_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
                    print(f"  ✓ 加载: {ckpt_dir.name}")
            except Exception as e:
                print(f"  ✗ 加载失败 {ckpt_dir.name}: {e}")

    if not all_results:
        print("❌ 没有找到有效的checkpoint结果")
        return False

    print(f"\n✅ 共加载 {len(all_results)} 个checkpoint结果")

    # 生成图表
    print("\n🎨 生成图表...")
    try:
        aggregator.generate_plots(all_results)
        print(f"\n✅ 图表已保存到: {aggregator.plots_dir}")

        # 列出生成的图表
        plot_files = list(aggregator.plots_dir.glob('*.png'))
        if plot_files:
            print("\n生成的图表文件:")
            for plot_file in sorted(plot_files):
                size_kb = plot_file.stat().st_size / 1024
                print(f"  - {plot_file.name} ({size_kb:.1f} KB)")

        return True
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='为评估结果生成可视化图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为单个实验生成图表
  python3 generate_plots_from_results.py \\
    --experiment-dir experiments/exp_20251002_090241_stage2_distributed_eval \\
    --stage 2

  # 为最新的stage2实验生成图表
  python3 generate_plots_from_results.py --stage 2 --latest

  # 批量为所有stage1实验生成图表
  python3 generate_plots_from_results.py --stage 1 --all
        """
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        help='实验目录路径（绝对路径或相对于eval目录）'
    )
    parser.add_argument(
        '--stage',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='训练阶段 (1, 2, 3)'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='为最新的实验生成图表'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='为所有匹配的实验生成图表'
    )

    args = parser.parse_args()

    # 确定基础目录
    script_dir = Path(__file__).parent
    base_dir = script_dir / 'experiments'

    if not base_dir.exists():
        print(f"❌ 错误: experiments目录不存在: {base_dir}")
        return 1

    # 确定要处理的实验目录
    experiment_dirs = []

    if args.experiment_dir:
        # 单个指定目录
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.is_absolute():
            exp_dir = base_dir / exp_dir
        experiment_dirs = [exp_dir]

    elif args.latest:
        # 最新的stage实验
        pattern = f"exp_*_stage{args.stage}_*"
        matches = sorted(base_dir.glob(pattern), reverse=True)
        if matches:
            experiment_dirs = [matches[0]]
        else:
            print(f"❌ 没有找到匹配的实验: {pattern}")
            return 1

    elif args.all:
        # 所有stage实验
        pattern = f"exp_*_stage{args.stage}_*"
        experiment_dirs = sorted(base_dir.glob(pattern))
        if not experiment_dirs:
            print(f"❌ 没有找到匹配的实验: {pattern}")
            return 1

    else:
        print("❌ 错误: 必须指定 --experiment-dir, --latest 或 --all")
        parser.print_help()
        return 1

    # 处理所有实验
    success_count = 0
    for exp_dir in experiment_dirs:
        print("\n" + "=" * 60)
        if generate_plots_for_experiment(exp_dir, args.stage):
            success_count += 1
        print()

    # 总结
    print("=" * 60)
    print(f"✅ 完成: {success_count}/{len(experiment_dirs)} 个实验成功生成图表")
    return 0 if success_count == len(experiment_dirs) else 1


if __name__ == '__main__':
    exit(main())
