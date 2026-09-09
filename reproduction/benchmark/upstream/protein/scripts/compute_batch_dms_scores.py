#!/usr/bin/env python3
"""
批处理DMS评估脚本 - 宿主机Wrapper
一次加载模型，处理多个DMS文件
"""

import sys
import argparse
import pandas as pd
import numpy as np
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict
from scipy.stats import spearmanr


def get_species_lineage_prefix(taxon: str, source_organism: str) -> str:
    """获取物种前缀"""
    if taxon == "Prokaryote":
        if "Escherichia" in source_organism or "coli" in source_organism:
            return "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_coli"
        return "d__Bacteria"
    elif taxon == "Eukaryote":
        if "Homo sapiens" in source_organism or "Human" in source_organism:
            return "d__Eukaryota;p__Chordata;c__Mammalia;o__Primates;f__Hominidae;g__Homo;s__Homo_sapiens"
        return "d__Eukaryota"
    elif taxon == "Virus":
        return "d__Viruses"
    return ""


def reverse_translate_protein(protein_seq: str, codon_table: Dict[str, str]) -> str:
    """反向翻译蛋白质序列为RNA"""
    return ''.join(codon_table.get(aa, 'NNN') for aa in protein_seq)


def get_most_frequent_codon_table() -> Dict[str, str]:
    """返回最常用密码子表"""
    return {
        'A': 'GCG', 'C': 'UGC', 'D': 'GAU', 'E': 'GAA',
        'F': 'UUC', 'G': 'GGC', 'H': 'CAC', 'I': 'AUC',
        'K': 'AAA', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
        'P': 'CCG', 'Q': 'CAG', 'R': 'CGU', 'S': 'AGC',
        'T': 'ACC', 'V': 'GUG', 'W': 'UGG', 'Y': 'UAC',
        '*': 'UAA'
    }


def prepare_dms_sequences(dms_file: str, codon_optimization: str, add_lineage_prefix: bool, tmp_dir: str) -> tuple:
    """准备DMS序列数据"""
    # 1. 读取DMS文件
    df = pd.read_csv(dms_file)

    if 'mutant' not in df.columns or 'DMS_score' not in df.columns:
        raise ValueError("DMS文件必须包含 'mutant' 和 'DMS_score' 列")

    # 2. 获取物种信息
    taxon = df['taxon'].iloc[0] if 'taxon' in df.columns else "Unknown"
    source_organism = df['source_organism'].iloc[0] if 'source_organism' in df.columns else "Unknown"

    # 3. 反向翻译
    codon_table = get_most_frequent_codon_table()
    rna_sequences = []

    for protein_seq in df['mutant']:
        rna_seq = reverse_translate_protein(protein_seq, codon_table)
        if add_lineage_prefix:
            lineage = get_species_lineage_prefix(taxon, source_organism)
            if lineage:
                rna_seq = f"{lineage}|{rna_seq}"
        rna_sequences.append(rna_seq)

    # 4. 保存为JSON
    basename = os.path.splitext(os.path.basename(dms_file))[0]
    seq_json = os.path.join(tmp_dir, f"sequences_{basename}.json")

    with open(seq_json, 'w') as f:
        json.dump({'sequences': rna_sequences}, f)

    return seq_json, df, taxon, source_organism


def compute_spearman(dms_scores: List[float], model_scores: List[float]) -> tuple:
    """计算Spearman相关系数"""
    valid_indices = [i for i, (d, m) in enumerate(zip(dms_scores, model_scores))
                     if not (np.isnan(d) or np.isnan(m) or np.isinf(d) or np.isinf(m))]

    if len(valid_indices) < 2:
        return np.nan, np.nan, 0

    valid_dms = [dms_scores[i] for i in valid_indices]
    valid_model = [model_scores[i] for i in valid_indices]

    corr, pval = spearmanr(valid_dms, valid_model)
    return corr, pval, len(valid_indices)


def main():
    parser = argparse.ArgumentParser(description='批处理DMS评估 - 一次加载模型')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--dms_files', required=True, nargs='+', help='DMS文件列表')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--codon_optimization', default='most_frequent', help='密码子优化策略')
    parser.add_argument('--reduce_method', default='mean', help='reduce方法')
    parser.add_argument('--add_lineage_prefix', action='store_true', help='添加物种前缀')
    parser.add_argument('--docker_container', default='rnagen5', help='Docker容器名')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] 批处理模式：一次加载模型处理 {len(args.dms_files)} 个DMS文件")
    print(f"[INFO] 模型: {args.model_path}")
    print(f"[INFO] 设备: {args.device}")
    print(f"[INFO] Batch size: {args.batch_size}")

    # 1. 准备临时目录
    with tempfile.TemporaryDirectory(prefix='batch_dms_') as tmp_dir:
        # 将tmp_dir映射到容器内
        container_tmp_dir = f"/rna-multiverse/benchmark_protein_dms/tmp/{os.path.basename(tmp_dir)}"

        # 2. 准备所有DMS文件的序列
        print(f"\n[1/{len(args.dms_files)+3}] 准备序列数据...")
        prepared_files = []
        dms_data = {}

        for i, dms_file in enumerate(args.dms_files):
            print(f"  [{i+1}/{len(args.dms_files)}] {os.path.basename(dms_file)}")
            try:
                seq_json, df, taxon, source_organism = prepare_dms_sequences(
                    dms_file, args.codon_optimization, args.add_lineage_prefix, tmp_dir
                )
                prepared_files.append(seq_json)
                dms_data[seq_json] = {
                    'df': df,
                    'taxon': taxon,
                    'source_organism': source_organism,
                    'dms_file': dms_file
                }
            except Exception as e:
                print(f"  [ERROR] 处理失败: {e}")
                continue

        if not prepared_files:
            print("[ERROR] 没有成功准备任何序列文件")
            sys.exit(1)

        # 3. 创建文件列表JSON
        files_list_json = os.path.join(tmp_dir, 'dms_files_list.json')
        # 转换为容器内路径
        container_prepared_files = [
            f"{container_tmp_dir}/{os.path.basename(f)}" for f in prepared_files
        ]
        with open(files_list_json, 'w') as f:
            json.dump(container_prepared_files, f)

        container_files_list = f"{container_tmp_dir}/dms_files_list.json"
        container_output_dir = f"{container_tmp_dir}/outputs"
        os.makedirs(os.path.join(tmp_dir, 'outputs'), exist_ok=True)

        # 4. 调用Docker批处理脚本（只加载一次模型！）
        print(f"\n[2/{len(args.dms_files)+3}] 调用Docker批处理...")
        cmd = [
            'docker', 'exec', args.docker_container,
            'bash', '-c',
            f'cd /rna-multiverse/rnagen && '
            f'python3 /rna-multiverse/benchmark_protein_dms/scripts/compute_batch_dms_in_docker.py '
            f'{container_files_list} '
            f'{args.model_path.replace("/data4/huangyanjie/rna_benchmark", "/rna-multiverse")} '
            f'{args.device} '
            f'{container_output_dir} '
            f'{args.reduce_method} '
            f'{args.batch_size}'
        ]

        print(f"[DEBUG] 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Docker执行失败")
            print(f"[DEBUG] stdout: {result.stdout}")
            print(f"[DEBUG] stderr: {result.stderr}")
            sys.exit(1)

        # 5. 解析结果
        try:
            summary = json.loads(result.stdout.strip().split('\n')[-1])
            if not summary.get('success'):
                print(f"[ERROR] 批处理失败: {summary.get('error')}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 解析结果失败: {e}")
            print(f"[DEBUG] stdout: {result.stdout}")
            sys.exit(1)

        # 6. 处理每个文件的结果
        print(f"\n[3/{len(args.dms_files)+3}] 计算Spearman相关系数...")
        final_results = []

        for seq_json in prepared_files:
            basename = os.path.basename(seq_json).replace('sequences_', '').replace('.json', '')
            output_json = os.path.join(tmp_dir, 'outputs', f"{basename}_output.json")

            if not os.path.exists(output_json):
                print(f"  [WARNING] 输出文件不存在: {output_json}")
                continue

            with open(output_json, 'r') as f:
                ll_result = json.load(f)

            if not ll_result.get('success'):
                print(f"  [ERROR] {basename} 计算失败")
                continue

            # 获取DMS数据
            data_info = dms_data[seq_json]
            df = data_info['df']
            dms_scores = df['DMS_score'].tolist()
            model_scores = ll_result['log_likelihoods']

            # 计算Spearman
            corr, pval, valid_count = compute_spearman(dms_scores, model_scores)

            # 保存结果CSV
            output_csv = os.path.join(args.output_dir, f"{basename}_batch.csv")
            result_df = df.copy()
            result_df['model_score'] = model_scores
            result_df.to_csv(output_csv, index=False)

            # 保存统计JSON
            stats = {
                'dms_file': data_info['dms_file'],
                'model_path': args.model_path,
                'total_sequences': len(dms_scores),
                'valid_predictions': valid_count,
                'spearman_correlation': corr,
                'p_value': pval,
                'codon_optimization': args.codon_optimization,
                'reduce_method': args.reduce_method,
                'device': args.device,
                'add_lineage_prefix': args.add_lineage_prefix,
                'taxon': data_info['taxon'],
                'source_organism': data_info['source_organism'],
                'processing_time': ll_result.get('elapsed_time', 0)
            }

            stats_json = os.path.join(args.output_dir, f"{basename}_batch_stats.json")
            with open(stats_json, 'w') as f:
                json.dump(stats, f, indent=2)

            final_results.append(stats)

            print(f"  ✓ {basename}: Spearman={corr:.4f}, p={pval:.4f}, valid={valid_count}/{len(dms_scores)}")

        # 7. 输出汇总
        print(f"\n{'='*60}")
        print(f"批处理完成！")
        print(f"{'='*60}")
        print(f"模型加载时间: {summary['model_load_time']:.2f}秒")
        print(f"总处理时间: {summary['total_processing_time']:.2f}秒")
        print(f"平均每文件: {summary['average_time_per_file']:.2f}秒")
        print(f"成功处理: {len(final_results)}/{len(args.dms_files)} 个文件")
        print(f"{'='*60}")

        # 保存批处理汇总
        batch_summary = {
            'model_path': args.model_path,
            'device': args.device,
            'batch_size': args.batch_size,
            'model_load_time': summary['model_load_time'],
            'total_processing_time': summary['total_processing_time'],
            'average_time_per_file': summary['average_time_per_file'],
            'total_files': len(args.dms_files),
            'successful_files': len(final_results),
            'results': final_results
        }

        summary_file = os.path.join(args.output_dir, 'batch_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)

        print(f"\n汇总保存至: {summary_file}")


if __name__ == '__main__':
    main()
