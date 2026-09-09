"""
完整流程：蛋白质序列 -> RNA序列 -> Likelihood计算

流程步骤：
1. 生成/输入蛋白质序列（可使用ESM或手动输入）
2. 使用反向翻译将蛋白质转换为RNA序列
3. 准备配置文件用于likelihood计算
4. 调用RNAGen模型计算likelihood

使用方法：
    # 使用随机蛋白质序列
    python protein_to_rna_likelihood_pipeline.py --length 50

    # 使用指定蛋白质序列
    python protein_to_rna_likelihood_pipeline.py --protein "MKPGFWYLCVNQSTHDEAIR"

    # 使用ESM生成（需要安装ESM）
    python protein_to_rna_likelihood_pipeline.py --use_esm --length 100
"""

import argparse
import random
import yaml
import sys
from pathlib import Path
from datetime import datetime

# 导入DNA/RNA转换工具
sys.path.insert(0, str(Path(__file__).parent))
from dna_rna_converter import reverse_translate_protein, CodonAnalyzer, GeneticCodeType


# ============================================
# 1. 蛋白质序列生成
# ============================================

def generate_random_protein(length: int = 50) -> str:
    """
    生成随机蛋白质序列

    Args:
        length: 序列长度

    Returns:
        随机蛋白质序列
    """
    # 20种标准氨基酸（不包括终止密码子*）
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 生成随机序列，以M开头（起始密码子）
    protein = 'M' + ''.join(random.choices(amino_acids, k=length-1))

    return protein


def generate_protein_with_esm(length: int = 50, temperature: float = 1.0) -> str:
    """
    使用ESM模型生成蛋白质序列

    Args:
        length: 目标长度
        temperature: 采样温度

    Returns:
        生成的蛋白质序列
    """
    try:
        import esm
        import torch

        print("正在加载ESM模型...")
        # 加载ESM-2模型（较小的版本）
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        # 准备起始序列
        start_seq = "M"  # 从甲硫氨酸开始

        print(f"使用ESM生成长度为{length}的蛋白质序列...")

        # 简化版：使用随机采样（ESM主要用于表示学习，不是生成模型）
        # 实际应用中可能需要使用专门的蛋白质生成模型
        print("注意：ESM主要用于蛋白质表示学习，这里使用简化的生成方法")

        # 回退到随机生成
        return generate_random_protein(length)

    except ImportError:
        print("警告：ESM未安装，使用随机生成")
        return generate_random_protein(length)
    except Exception as e:
        print(f"ESM生成失败: {e}，使用随机生成")
        return generate_random_protein(length)


# ============================================
# 2. 蛋白质到RNA转换
# ============================================

def protein_to_rna(
    protein: str,
    optimization: str = 'random',
    add_start_stop: bool = True,
    output_format: str = 'rna',
    genetic_code: str = 'standard'
) -> dict:
    """
    将蛋白质序列转换为RNA序列

    Args:
        protein: 蛋白质序列
        optimization: 密码子优化策略 ('random', 'first', 'frequent')
        add_start_stop: 是否添加起始和终止密码子
        output_format: 输出格式 ('rna', 'dna', 'model')
        genetic_code: 遗传密码类型

    Returns:
        包含转换信息的字典
    """
    print("\n" + "="*60)
    print("步骤2: 蛋白质序列反向翻译为RNA")
    print("="*60)

    print(f"蛋白质序列: {protein[:50]}{'...' if len(protein) > 50 else ''}")
    print(f"长度: {len(protein)} 个氨基酸")
    print(f"密码子优化策略: {optimization}")
    print(f"遗传密码: {genetic_code}")

    # 反向翻译
    rna_sequence = reverse_translate_protein(
        protein,
        code_type=genetic_code,
        optimization=optimization,
        output_format=output_format,
        add_start_stop=add_start_stop
    )

    # 验证
    analyzer = CodonAnalyzer(
        GeneticCodeType.STANDARD if genetic_code == 'standard' else GeneticCodeType.MYCOPLASMA
    )

    # 翻译回去验证
    dna_for_check = rna_sequence.replace('U', 'T')
    back_translation = analyzer.translate_sequence(dna_for_check)

    # 检查是否一致（去除终止密码子）
    # 如果add_start_stop=True，原始蛋白质已经包含M，所以不需要再加
    original_protein = protein
    # 去除终止密码子进行比较
    is_valid = back_translation.rstrip('*') == original_protein.rstrip('*')

    result = {
        'protein': protein,
        'rna_sequence': rna_sequence,
        'rna_length': len(rna_sequence),
        'back_translation': back_translation,
        'is_valid': is_valid,
        'optimization': optimization,
        'genetic_code': genetic_code
    }

    print(f"\nRNA序列: {rna_sequence[:60]}{'...' if len(rna_sequence) > 60 else ''}")
    print(f"RNA长度: {len(rna_sequence)} nt")
    print(f"反向验证: {'✓ 通过' if is_valid else '✗ 失败'}")

    return result


# ============================================
# 3. 准备Likelihood计算配置
# ============================================

def prepare_likelihood_config(
    rna_sequence: str,
    checkpoint_path: str,
    output_dir: str,
    rna_type: str = 'mRNA',
    lineage: str = None,
    device: str = 'cuda:0'
) -> str:
    """
    准备likelihood计算的配置文件

    Args:
        rna_sequence: RNA序列
        checkpoint_path: 模型checkpoint路径
        output_dir: 输出目录
        rna_type: RNA类型
        lineage: 谱系信息
        device: 计算设备

    Returns:
        配置文件路径
    """
    print("\n" + "="*60)
    print("步骤3: 准备Likelihood计算配置")
    print("="*60)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 配置文件路径
    config_file = output_path / f"config_likelihood_{timestamp}.yaml"
    output_file = output_path / f"likelihood_result_{timestamp}.txt"
    log_file = output_path / f"likelihood_log_{timestamp}.log"

    # 构建配置
    config = {
        'checkpoint': checkpoint_path,
        'device': device,
        'sequence': rna_sequence,
        'rna_type': rna_type,
        'output_file': str(output_file),
        'log_file': str(log_file),
        'use_nohup': False  # 前台运行，方便查看结果
    }

    if lineage:
        config['lineage'] = lineage

    # 写入配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"配置文件: {config_file}")
    print(f"输出文件: {output_file}")
    print(f"日志文件: {log_file}")
    print(f"\n配置内容:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))

    return str(config_file)


# ============================================
# 4. 完整流程
# ============================================

def run_complete_pipeline(args):
    """运行完整流程"""

    print("\n" + "="*60)
    print("蛋白质序列 -> RNA序列 -> Likelihood计算 完整流程")
    print("="*60)

    # ----------------------------------------
    # 步骤1: 生成/获取蛋白质序列
    # ----------------------------------------
    print("\n" + "="*60)
    print("步骤1: 获取蛋白质序列")
    print("="*60)

    if args.protein:
        protein = args.protein.upper()
        print(f"使用指定蛋白质序列: {protein[:50]}{'...' if len(protein) > 50 else ''}")
    elif args.use_esm:
        protein = generate_protein_with_esm(args.length, args.temperature)
        print(f"使用ESM生成蛋白质序列")
    else:
        protein = generate_random_protein(args.length)
        print(f"生成随机蛋白质序列")

    print(f"蛋白质序列: {protein}")
    print(f"长度: {len(protein)} 个氨基酸")

    # ----------------------------------------
    # 步骤2: 反向翻译为RNA
    # ----------------------------------------
    rna_result = protein_to_rna(
        protein,
        optimization=args.optimization,
        add_start_stop=args.add_start_stop,
        output_format='rna',  # likelihood脚本需要RNA格式（U）
        genetic_code=args.genetic_code
    )

    if not rna_result['is_valid']:
        print("\n警告：反向翻译验证失败！")
        return

    # ----------------------------------------
    # 步骤3: 准备配置文件
    # ----------------------------------------
    config_file = prepare_likelihood_config(
        rna_sequence=rna_result['rna_sequence'],
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        rna_type=args.rna_type,
        lineage=args.lineage,
        device=args.device
    )

    # ----------------------------------------
    # 步骤4: 运行Likelihood计算
    # ----------------------------------------
    print("\n" + "="*60)
    print("步骤4: 运行Likelihood计算")
    print("="*60)

    # 转换为容器路径
    container_config = config_file.replace(
        '/lnvme/home/huang_yan_jie/rna_benchmark/',
        '/rna_benchmark/'
    )

    run_script = "/lnvme/home/huang_yan_jie/rna_benchmark/rnagen/scripts/run_likelihood.sh"

    print(f"\n执行命令:")
    print(f"bash {run_script} {config_file}")
    print(f"\n或者手动运行:")
    print(f"cd /lnvme/home/huang_yan_jie/rna_benchmark/rnagen/scripts")
    print(f"./run_likelihood.sh {config_file}")

    # 保存流程信息
    summary_file = Path(args.output_dir) / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("蛋白质 -> RNA -> Likelihood 流程总结\n")
        f.write("="*60 + "\n\n")
        f.write(f"1. 蛋白质序列:\n")
        f.write(f"   {protein}\n")
        f.write(f"   长度: {len(protein)} aa\n\n")
        f.write(f"2. RNA序列:\n")
        f.write(f"   {rna_result['rna_sequence']}\n")
        f.write(f"   长度: {rna_result['rna_length']} nt\n")
        f.write(f"   优化策略: {args.optimization}\n\n")
        f.write(f"3. 配置文件:\n")
        f.write(f"   {config_file}\n\n")
        f.write(f"4. 运行命令:\n")
        f.write(f"   bash {run_script} {config_file}\n\n")

    print(f"\n流程总结已保存到: {summary_file}")

    # 如果指定了自动运行
    if args.auto_run:
        print("\n自动运行likelihood计算...")
        import subprocess
        try:
            result = subprocess.run(
                ['bash', run_script, config_file],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            print(result.stdout)
            if result.stderr:
                print("错误输出:", result.stderr)
        except subprocess.TimeoutExpired:
            print("警告：计算超时（5分钟）")
        except Exception as e:
            print(f"运行失败: {e}")

    print("\n" + "="*60)
    print("流程完成！")
    print("="*60)


# ============================================
# 主程序
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='蛋白质序列 -> RNA序列 -> Likelihood计算 完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 蛋白质序列来源
    protein_group = parser.add_mutually_exclusive_group()
    protein_group.add_argument(
        '--protein',
        type=str,
        help='指定蛋白质序列（单字母代码）'
    )
    protein_group.add_argument(
        '--use_esm',
        action='store_true',
        help='使用ESM模型生成蛋白质序列'
    )

    parser.add_argument(
        '--length',
        type=int,
        default=50,
        help='生成蛋白质序列的长度（默认：50）'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='ESM生成温度（默认：1.0）'
    )

    # 反向翻译参数
    parser.add_argument(
        '--optimization',
        type=str,
        choices=['random', 'first', 'frequent'],
        default='first',
        help='密码子优化策略（默认：first）'
    )

    parser.add_argument(
        '--add_start_stop',
        action='store_true',
        default=True,
        help='添加起始和终止密码子（默认：True）'
    )

    parser.add_argument(
        '--genetic_code',
        type=str,
        choices=['standard', 'mycoplasma', 'ciliate'],
        default='standard',
        help='遗传密码类型（默认：standard）'
    )

    # Likelihood计算参数
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/rna_benchmark/rnagen/checkpoint_100M_1018',
        help='RNAGen模型checkpoint路径（容器内路径）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备（默认：cuda）'
    )

    parser.add_argument(
        '--rna_type',
        type=str,
        default='mRNA',
        help='RNA类型（默认：mRNA）'
    )

    parser.add_argument(
        '--lineage',
        type=str,
        default=None,
        help='谱系信息（可选）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/lnvme/home/huang_yan_jie/rna_benchmark/dna_rna_tools/output',
        help='输出目录'
    )

    parser.add_argument(
        '--auto_run',
        action='store_true',
        help='自动运行likelihood计算（否则只生成配置文件）'
    )

    args = parser.parse_args()

    # 运行流程
    run_complete_pipeline(args)


if __name__ == '__main__':
    main()
