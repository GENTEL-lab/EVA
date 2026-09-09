#!/usr/bin/env python3
"""
ProteinGym DMS评估 - Docker Wrapper（宿主机运行）

此脚本运行在宿主机，通过docker exec调用容器内的计算脚本
负责：
1. 加载DMS数据（蛋白质序列）
2. 反向翻译为RNA序列
3. 调用容器内脚本计算log-likelihood
4. 计算Spearman相关系数
5. 保存结果

用法：
    python compute_dms_scores_docker.py --model_path <model> --dms_file <dms> --output <output> [options]

示例：
    python compute_dms_scores_docker.py \
        --model_path /data4/huangyanjie/rna_benchmark/new_model/scaling_tiny_6e18_v11_20251110_232136/checkpoint-5906/model_only \
        --dms_file /data4/huangyanjie/DMS_ProteinGym_substitutions/BLAT_ECOLX_Firnberg_2014.csv \
        --output results.csv \
        --device cuda:6
"""

import argparse
import csv
import json
import os
import sys
import subprocess
import shutil
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr
import pandas as pd


# Docker配置
DOCKER_CONTAINER_NAME = "rnagen5"
DOCKER_TEMP_DIR = "/rna-multiverse/benchmark_protein_dms/tmp"
HOST_TEMP_DIR = "/data4/huangyanjie/rna_benchmark/benchmark_protein_dms/tmp"

# 默认参考文件路径
DEFAULT_REFERENCE_FILE = "/data4/huangyanjie/rna_benchmark/ProteinGym/reference_files/DMS_substitutions.csv"

# Lineage TSV文件路径
LINEAGE_TSV_FILE = "/data4/huangyanjie/rna_benchmark/rnagen/lineage_greengenes.tsv"

# 全局缓存：物种名到谱系的映射
_LINEAGE_CACHE = None
_LINEAGE_BY_TAXID = None


def load_lineage_tsv():
    """
    加载lineage TSV文件并创建物种名到谱系的映射

    Returns:
        (species_to_lineage, taxid_to_lineage)
        - species_to_lineage: 物种名(小写) -> 谱系字符串
        - taxid_to_lineage: taxid -> 谱系字符串
    """
    global _LINEAGE_CACHE, _LINEAGE_BY_TAXID

    if _LINEAGE_CACHE is not None:
        return _LINEAGE_CACHE, _LINEAGE_BY_TAXID

    species_to_lineage = {}
    taxid_to_lineage = {}

    try:
        with open(LINEAGE_TSV_FILE, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # 跳过表头

            for row in reader:
                if len(row) < 2:
                    continue

                taxid = row[0].strip()
                lineage = row[1].strip()

                # 将大写格式转换为小写格式: D__ -> d__, P__ -> p__, etc.
                lineage_lower = lineage.replace('D__', 'd__').replace('P__', 'p__').replace(
                    'C__', 'c__').replace('O__', 'o__').replace('F__', 'f__').replace(
                    'G__', 'g__').replace('S__', 's__')

                # 存储taxid映射
                taxid_to_lineage[taxid] = lineage_lower

                # 从lineage中提取物种名 (S__后面的部分)
                if 'S__' in lineage or 's__' in lineage_lower:
                    parts = lineage_lower.split('s__')
                    if len(parts) > 1:
                        species_name = parts[1].strip()
                        if species_name:
                            # 存储多种形式的物种名
                            species_lower = species_name.lower().replace('_', ' ').replace("'", "")
                            species_to_lineage[species_lower] = lineage_lower
                            # 也存储原始格式（带下划线）
                            species_to_lineage[species_name.lower()] = lineage_lower

        print(f"[INFO] 已加载 {len(taxid_to_lineage)} 条物种谱系信息", file=sys.stderr)
        _LINEAGE_CACHE = species_to_lineage
        _LINEAGE_BY_TAXID = taxid_to_lineage

    except Exception as e:
        print(f"[WARNING] 加载lineage TSV文件失败: {e}", file=sys.stderr)
        _LINEAGE_CACHE = {}
        _LINEAGE_BY_TAXID = {}

    return _LINEAGE_CACHE, _LINEAGE_BY_TAXID


def check_docker_container():
    """检查docker容器是否运行"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        if DOCKER_CONTAINER_NAME in result.stdout:
            return True
        else:
            print(f"错误: Docker容器 '{DOCKER_CONTAINER_NAME}' 未运行", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"错误: 无法检查docker状态: {e}", file=sys.stderr)
        return False


def get_species_from_reference(dms_file: str, reference_file: str = DEFAULT_REFERENCE_FILE) -> Tuple[str, str]:
    """
    从参考文件中读取物种信息

    Args:
        dms_file: DMS CSV文件路径
        reference_file: 参考文件路径（DMS_substitutions.csv）

    Returns:
        (taxon, source_organism)
        - taxon: 物种类群 (Virus/Prokaryote/Eukaryote/Human)
        - source_organism: 具体物种名称（如 Escherichia coli, Homo sapiens）
    """
    dms_filename = os.path.basename(dms_file)

    try:
        # 读取参考文件
        ref_df = pd.read_csv(reference_file)

        # 查找匹配的行
        match = ref_df[ref_df['DMS_filename'] == dms_filename]

        if len(match) == 0:
            print(f"  警告: 在参考文件中未找到 {dms_filename}，将使用文件名推断物种", file=sys.stderr)
            # 回退到文件名推断
            return get_species_from_filename(dms_file), 'Unknown'

        # 提取物种信息
        taxon = match.iloc[0]['taxon']
        source_organism = match.iloc[0]['source_organism']

        return taxon, source_organism

    except Exception as e:
        print(f"  警告: 读取参考文件失败: {e}，将使用文件名推断物种", file=sys.stderr)
        return get_species_from_filename(dms_file), 'Unknown'


def get_species_from_filename(dms_file: str) -> str:
    """
    从DMS文件名提取物种信息（回退方法）

    文件名格式示例:
    - BLAT_ECOLX_Firnberg_2014.csv -> ECOLX (E. coli)
    - ARGR_ECOLI_Tsuboyama_2023_1AOY.csv -> ECOLI (E. coli)
    - ACE2_HUMAN_Chan_2020.csv -> HUMAN (Human)
    - AMIE_PSEAE_Wrenbeck_2017.csv -> PSEAE (P. aeruginosa)
    - BBC1_YEAST_Tsuboyama_2023_1TG0.csv -> YEAST (S. cerevisiae)
    """
    basename = os.path.basename(dms_file)
    parts = basename.split('_')

    # 通常物种代码在第二个位置
    if len(parts) >= 2:
        species_code = parts[1].upper()
        return species_code

    return 'UNKNOWN'


def get_lineage_for_species(taxon: str, source_organism: str) -> str:
    """
    根据物种信息获取GTDB风格的谱系字符串（从TSV文件查找）

    Args:
        taxon: 物种类群 (Virus/Prokaryote/Eukaryote/Human)
        source_organism: 具体物种名称

    Returns:
        GTDB风格的谱系字符串，格式：d__Domain;p__Phylum;...;s__Species|
    """
    # 加载TSV文件
    species_to_lineage, taxid_to_lineage = load_lineage_tsv()

    source_lower = source_organism.lower().strip()

    # 1. 首先尝试精确匹配物种名
    if source_lower in species_to_lineage:
        lineage = species_to_lineage[source_lower]
        # 添加结尾的 | 符号
        if not lineage.endswith('|'):
            lineage = lineage + '|'
        return lineage

    # 2. 尝试模糊匹配：查找包含物种名关键词的条目
    # 提取关键词（如 "Escherichia coli" -> ["escherichia", "coli"]）
    keywords = source_lower.split()

    # 尝试匹配属名+种名的组合
    for species_name, lineage in species_to_lineage.items():
        # 检查是否所有关键词都在物种名中
        if all(kw in species_name for kw in keywords) and len(keywords) >= 2:
            if not lineage.endswith('|'):
                lineage = lineage + '|'
            print(f"[INFO] 物种 '{source_organism}' 模糊匹配到: {species_name}", file=sys.stderr)
            return lineage

    # 3. 对于常见物种名的变体进行特殊处理
    species_aliases = {
        'e. coli': 'escherichia coli',
        'e.coli': 'escherichia coli',
        's. cerevisiae': 'saccharomyces cerevisiae',
        's.cerevisiae': 'saccharomyces cerevisiae',
        'p. aeruginosa': 'pseudomonas aeruginosa',
        'p.aeruginosa': 'pseudomonas aeruginosa',
        'h. sapiens': 'homo sapiens',
        'h.sapiens': 'homo sapiens',
    }

    # 检查别名
    for alias, full_name in species_aliases.items():
        if alias in source_lower:
            if full_name in species_to_lineage:
                lineage = species_to_lineage[full_name]
                if not lineage.endswith('|'):
                    lineage = lineage + '|'
                print(f"[INFO] 物种别名 '{source_organism}' 映射到: {full_name}", file=sys.stderr)
                return lineage

    # 4. 按taxon分类使用默认值
    default_lineages = {
        'Human': 'homo sapiens',
        'Prokaryote': 'escherichia coli',
        'Eukaryote': 'saccharomyces cerevisiae',
        'Virus': 'escherichia coli',  # 病毒使用宿主的谱系，默认E. coli
    }

    if taxon in default_lineages:
        default_species = default_lineages[taxon]
        if default_species in species_to_lineage:
            lineage = species_to_lineage[default_species]
            if not lineage.endswith('|'):
                lineage = lineage + '|'
            print(f"[INFO] 未找到 '{source_organism}'，使用taxon '{taxon}' 的默认物种: {default_species}", file=sys.stderr)
            return lineage

    # 5. 最终回退：使用E. coli
    if 'escherichia coli' in species_to_lineage:
        lineage = species_to_lineage['escherichia coli']
        if not lineage.endswith('|'):
            lineage = lineage + '|'
        print(f"[WARNING] 未找到 '{source_organism}'，回退到 Escherichia coli", file=sys.stderr)
        return lineage

    # 6. 如果TSV也没有E. coli，使用硬编码的默认值
    print(f"[WARNING] TSV文件中未找到任何匹配，使用硬编码默认值", file=sys.stderr)
    return "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli|"


def get_codon_table_for_species(taxon: str, source_organism: str) -> Dict[str, str]:
    """
    根据物种信息获取最优密码子表

    Args:
        taxon: 物种类群 (Virus/Prokaryote/Eukaryote/Human)
        source_organism: 具体物种名称

    Returns:
        密码子表字典
    """
    # E. coli密码子表
    ecoli_table = {
        'A': 'GCG', 'C': 'UGC', 'D': 'GAU', 'E': 'GAA',
        'F': 'UUU', 'G': 'GGC', 'H': 'CAU', 'I': 'AUU',
        'K': 'AAA', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
        'P': 'CCG', 'Q': 'CAG', 'R': 'CGU', 'S': 'AGC',
        'T': 'ACC', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
        '*': 'UAA'
    }

    # Human密码子表
    human_table = {
        'A': 'GCC', 'C': 'UGC', 'D': 'GAC', 'E': 'GAG',
        'F': 'UUC', 'G': 'GGC', 'H': 'CAC', 'I': 'AUC',
        'K': 'AAG', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
        'P': 'CCC', 'Q': 'CAG', 'R': 'AGG', 'S': 'AGC',
        'T': 'ACC', 'V': 'GUG', 'W': 'UGG', 'Y': 'UAC',
        '*': 'UGA'
    }

    # Yeast密码子表 (Saccharomyces cerevisiae)
    yeast_table = {
        'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAA',
        'F': 'UUU', 'G': 'GGU', 'H': 'CAU', 'I': 'AUU',
        'K': 'AAG', 'L': 'UUG', 'M': 'AUG', 'N': 'AAU',
        'P': 'CCU', 'Q': 'CAA', 'R': 'AGA', 'S': 'UCU',
        'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
        '*': 'UAA'
    }

    # P. aeruginosa密码子表
    pseudomonas_table = {
        'A': 'GCC', 'C': 'UGC', 'D': 'GAC', 'E': 'GAA',
        'F': 'UUC', 'G': 'GGC', 'H': 'CAC', 'I': 'AUC',
        'K': 'AAG', 'L': 'CUG', 'M': 'AUG', 'N': 'AAC',
        'P': 'CCG', 'Q': 'CAG', 'R': 'CGC', 'S': 'UCC',
        'T': 'ACC', 'V': 'GUC', 'W': 'UGG', 'Y': 'UAC',
        '*': 'UAA'
    }

    # 病毒密码子表（使用宿主生物的密码子表）
    # 对于流感、HIV等病毒，使用人类密码子表
    virus_table = human_table

    # 根据taxon和source_organism选择密码子表
    source_lower = source_organism.lower()

    # 优先匹配具体物种
    if 'escherichia coli' in source_lower or 'e. coli' in source_lower:
        return ecoli_table
    elif 'homo sapiens' in source_lower:
        return human_table
    elif 'saccharomyces cerevisiae' in source_lower:
        return yeast_table
    elif 'pseudomonas aeruginosa' in source_lower:
        return pseudomonas_table

    # 按taxon分类选择
    if taxon == 'Human':
        return human_table
    elif taxon == 'Prokaryote':
        # 默认使用E. coli（大多数细菌实验）
        return ecoli_table
    elif taxon == 'Eukaryote':
        # 默认使用酵母
        return yeast_table
    elif taxon == 'Virus':
        # 病毒使用宿主的密码子表，默认为人类
        return virus_table
    else:
        # 默认使用E. coli
        return ecoli_table


def reverse_translate_protein(
    protein_seq: str,
    optimization: str = 'first',
    taxon: Optional[str] = None,
    source_organism: Optional[str] = None,
    add_lineage_prefix: bool = False
) -> str:
    """
    将蛋白质序列反向翻译为RNA序列

    Args:
        protein_seq: 蛋白质序列（氨基酸）
        optimization: 密码子优化策略 ('first', 'most_frequent')
        taxon: 物种类群（用于most_frequent策略和lineage前缀）
        source_organism: 具体物种名称（用于most_frequent策略和lineage前缀）
        add_lineage_prefix: 是否添加lineage前缀（用于rnagen模型）

    Returns:
        RNA序列（可能包含lineage前缀）
    """
    # 简化版密码子表（标准遗传密码）- 每个氨基酸取第一个密码子
    codon_table_first = {
        'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAA',
        'F': 'UUU', 'G': 'GGU', 'H': 'CAU', 'I': 'AUU',
        'K': 'AAA', 'L': 'UUA', 'M': 'AUG', 'N': 'AAU',
        'P': 'CCU', 'Q': 'CAA', 'R': 'CGU', 'S': 'UCU',
        'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
        '*': 'UAA'  # 终止密码子
    }

    # 选择密码子表
    if optimization == 'most_frequent':
        if taxon and source_organism:
            codon_table = get_codon_table_for_species(taxon, source_organism)
        else:
            # 默认使用E. coli表
            codon_table = get_codon_table_for_species('Prokaryote', 'Escherichia coli')
    else:
        codon_table = codon_table_first

    rna_seq = []
    for aa in protein_seq:
        if aa in codon_table:
            rna_seq.append(codon_table[aa])
        else:
            # 未知氨基酸，使用N的密码子
            rna_seq.append('AAU')

    rna_sequence = ''.join(rna_seq)

    # 如果需要添加lineage前缀
    # 根据 lineage_model_input_format.md 文档，正确格式为：
    # 标准格式: |d__bacteria;p__...;s__species;<rna_mRNA>|{rna_sequence}
    # 注意：方向标记 5/3 在 docker worker 中添加
    if add_lineage_prefix and taxon and source_organism:
        lineage = get_lineage_for_species(taxon, source_organism)
        # lineage 已经以 | 结尾，需要移除并重新格式化
        lineage_clean = lineage.rstrip('|')
        # 格式: |{lineage};<rna_mRNA>|{sequence}
        # 注意：lineage 末尾不需要分号，<rna_mRNA> 前需要分号
        rna_sequence = f"|{lineage_clean};<rna_mRNA>|{rna_sequence}"

    return rna_sequence


def load_dms_data(dms_file: str, max_sequences: int = None) -> Tuple[List[str], List[float]]:
    """
    加载DMS数据

    Returns:
        (protein_sequences, dms_scores)
    """
    protein_sequences = []
    dms_scores = []

    with open(dms_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_sequences and i >= max_sequences:
                break
            protein_sequences.append(row['mutated_sequence'])
            dms_scores.append(float(row['DMS_score']))

    return protein_sequences, dms_scores


def compute_log_likelihoods_via_docker(
    rna_sequences: List[str],
    checkpoint_path: str,
    device: str = 'cuda:0',
    reduce_method: str = 'mean',
    docker_container: str = DOCKER_CONTAINER_NAME
) -> List[float]:
    """
    通过docker调用计算log-likelihood

    Args:
        rna_sequences: RNA序列列表
        checkpoint_path: 宿主机checkpoint路径
        device: 计算设备
        reduce_method: 归约方式
        docker_container: Docker容器名称

    Returns:
        log_likelihoods列表
    """
    # 1. 检查docker容器
    if not check_docker_container():
        raise RuntimeError(f"Docker容器 '{docker_container}' 未运行")

    # 2. 准备临时目录
    os.makedirs(HOST_TEMP_DIR, exist_ok=True)

    # 3. 创建临时JSON文件（包含RNA序列）
    temp_json = os.path.join(HOST_TEMP_DIR, f"sequences_{os.getpid()}.json")
    with open(temp_json, 'w') as f:
        json.dump({'sequences': rna_sequences}, f)

    docker_json_path = f"{DOCKER_TEMP_DIR}/sequences_{os.getpid()}.json"

    # 4. 复制worker脚本到docker可访问位置
    script_dir = Path(__file__).parent
    local_script = script_dir / "compute_dms_ll_in_docker.py"

    if not local_script.exists():
        raise FileNotFoundError(f"Worker脚本不存在: {local_script}")

    target_script = os.path.join(HOST_TEMP_DIR, "compute_dms_ll_in_docker.py")
    shutil.copy(local_script, target_script)
    docker_script_path = f"{DOCKER_TEMP_DIR}/compute_dms_ll_in_docker.py"

    # 5. 转换checkpoint路径：宿主机 -> 容器内
    if checkpoint_path.startswith('/data4/huangyanjie/rna_benchmark/'):
        docker_checkpoint_path = checkpoint_path.replace(
            '/data4/huangyanjie/rna_benchmark/', '/rna-multiverse/'
        )
    else:
        docker_checkpoint_path = checkpoint_path

    try:
        # 6. 准备输出JSON文件路径
        temp_output_json = os.path.join(HOST_TEMP_DIR, f"output_{os.getpid()}.json")
        docker_output_json = f"{DOCKER_TEMP_DIR}/output_{os.getpid()}.json"

        # 7. 调用docker exec运行计算
        # 注意：cd到rnagen项目目录
        cmd = [
            'docker', 'exec', docker_container,
            'bash', '-c',
            f'cd /rna-multiverse/rnagen && '
            f'python3 {docker_script_path} {docker_json_path} {docker_checkpoint_path} {device} {docker_output_json} {reduce_method}'
        ]

        print(f"[INFO] 调用Docker计算log-likelihood...")
        print(f"[INFO] 容器: {docker_container}")
        print(f"[INFO] 序列数: {len(rna_sequences)}")
        print(f"[INFO] 模型: {docker_checkpoint_path}")
        print(f"[INFO] 设备: {device}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # 8. 从文件读取JSON结果（避免docker exec stdout大小限制）
        try:
            # 等待文件生成
            max_wait = 5
            for _ in range(max_wait):
                if os.path.exists(temp_output_json):
                    break
                time.sleep(1)

            if not os.path.exists(temp_output_json):
                raise FileNotFoundError(f"输出文件未生成: {temp_output_json}")

            with open(temp_output_json, 'r') as f:
                output = json.load(f)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Docker stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"无法读取docker返回的JSON: {e}")

        # DEBUG: 打印stderr以查看warning信息
        if result.stderr:
            print(f"[DEBUG] Docker stderr:\n{result.stderr}", file=sys.stderr)

        # 8. 检查是否成功
        if not output.get('success', False):
            error_msg = output.get('error', 'Unknown error')
            raise RuntimeError(f"Docker计算失败: {error_msg}")

        return output['log_likelihoods']

    finally:
        # 9. 清理临时文件
        if os.path.exists(temp_json):
            os.remove(temp_json)
        if 'temp_output_json' in locals() and os.path.exists(temp_output_json):
            os.remove(temp_output_json)


def parse_args():
    parser = argparse.ArgumentParser(description='ProteinGym DMS评估 - 新模型Docker Wrapper')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径（宿主机）')
    parser.add_argument('--dms_file', type=str, required=True, help='DMS CSV文件路径（宿主机）')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备 (default: cuda:0)')
    parser.add_argument('--max_sequences', type=int, default=None, help='最多处理的序列数（用于测试）')
    parser.add_argument('--codon_optimization', type=str, default='first',
                        choices=['first', 'random', 'most_frequent'], help='密码子优化策略')
    parser.add_argument('--reduce_method', type=str, default='mean',
                        choices=['mean', 'sum'], help='Log-likelihood归约方式')
    parser.add_argument('--reference_file', type=str, default=DEFAULT_REFERENCE_FILE,
                        help=f'参考文件路径 (default: {DEFAULT_REFERENCE_FILE})')
    parser.add_argument('--add_lineage_prefix', action='store_true',
                        help='是否添加lineage前缀到RNA序列（用于rnagen模型）')
    parser.add_argument('--docker_container', type=str, default=DOCKER_CONTAINER_NAME,
                        help=f'Docker容器名称 (default: {DOCKER_CONTAINER_NAME})')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ProteinGym DMS评估 - 新模型Docker Wrapper")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"DMS文件: {args.dms_file}")
    print(f"输出文件: {args.output}")
    print(f"设备: {args.device}")
    print("=" * 80)

    # 1. 加载DMS数据
    print("\n[1/5] 加载DMS数据...")
    protein_sequences, dms_scores = load_dms_data(args.dms_file, args.max_sequences)
    print(f"  加载了 {len(protein_sequences)} 个突变序列")

    # 1.5 从参考文件读取物种信息（用于most_frequent策略和lineage前缀）
    taxon = None
    source_organism = None
    if args.codon_optimization == 'most_frequent' or args.add_lineage_prefix:
        print(f"  从参考文件读取物种信息: {args.reference_file}")
        taxon, source_organism = get_species_from_reference(args.dms_file, args.reference_file)
        print(f"  物种类群: {taxon}")
        print(f"  具体物种: {source_organism}")
        if args.add_lineage_prefix:
            lineage = get_lineage_for_species(taxon, source_organism)
            print(f"  Lineage前缀: {lineage}")

    # 2. 反向翻译为RNA
    print(f"\n[2/5] 反向翻译为RNA序列 (策略: {args.codon_optimization}, lineage前缀: {args.add_lineage_prefix})...")
    rna_sequences = [
        reverse_translate_protein(seq, args.codon_optimization, taxon, source_organism, args.add_lineage_prefix)
        for seq in protein_sequences
    ]
    print(f"  完成 {len(rna_sequences)} 条RNA序列的翻译")
    if args.add_lineage_prefix and len(rna_sequences) > 0:
        print(f"  示例序列（前100字符）: {rna_sequences[0][:100]}")

    # 3. 调用Docker计算log-likelihood
    print(f"\n[3/5] 调用Docker计算log-likelihood (reduce: {args.reduce_method})...")
    model_scores = compute_log_likelihoods_via_docker(
        rna_sequences,
        args.model_path,
        args.device,
        args.reduce_method,
        args.docker_container
    )
    print(f"  计算完成")

    # 4. 计算Spearman相关系数
    print("\n[4/5] 计算Spearman相关系数...")
    valid_pairs = [(dms, model) for dms, model in zip(dms_scores, model_scores) if not np.isnan(model)]

    if len(valid_pairs) < 2:
        print(f"  警告: 有效预测数量不足 ({len(valid_pairs)}个)，仍然保存结果")
        spearman_corr = float('nan')
        p_value = float('nan')
    else:
        valid_dms_scores = [p[0] for p in valid_pairs]
        valid_model_scores = [p[1] for p in valid_pairs]
        spearman_corr, p_value = spearmanr(valid_dms_scores, valid_model_scores)

    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    print(f"总序列数: {len(protein_sequences)}")
    print(f"有效预测数: {len(valid_pairs)}")
    print(f"Spearman相关系数: {spearman_corr:.4f}")
    print(f"P值: {p_value:.6f}")
    print("=" * 80)

    # 5. 保存结果
    print(f"\n[5/5] 保存结果到 {args.output}...")

    # 保存详细结果CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_sequence', 'rna_sequence', 'DMS_score', 'model_score'])
        for i in range(len(protein_sequences)):
            writer.writerow([
                protein_sequences[i],
                rna_sequences[i],
                dms_scores[i],
                model_scores[i]
            ])

    # 保存统计信息JSON
    stats_file = args.output.replace('.csv', '_stats.json')
    stats = {
        'dms_file': args.dms_file,
        'model_path': args.model_path,
        'total_sequences': len(protein_sequences),
        'valid_predictions': len(valid_pairs),
        'spearman_correlation': float(spearman_corr),
        'p_value': float(p_value),
        'codon_optimization': args.codon_optimization,
        'reduce_method': args.reduce_method,
        'device': args.device,
        'add_lineage_prefix': args.add_lineage_prefix,
        'taxon': taxon if taxon else 'N/A',
        'source_organism': source_organism if source_organism else 'N/A'
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  结果已保存到: {args.output}")
    print(f"  统计信息已保存到: {stats_file}")
    print("\n评估完成!")


if __name__ == '__main__':
    main()
