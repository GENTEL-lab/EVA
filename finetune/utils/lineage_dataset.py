"""
Lineage-based RNA数据集处理
支持基于Greengenes谱系字符串的条件序列生成和补全
用于新的2阶段训练系统
"""

import os
import re
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import math
import numpy as np

# 使用 eva 模块的 tokenizer
from eva.lineage_tokenizer import get_lineage_rna_tokenizer

logger = logging.getLogger(__name__)


# ========== SpanConfig 定义 ==========
@dataclass
class SpanConfig:
    """Span采样配置"""
    max_coverage_ratios: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.5, 0.8])
    coverage_probs: List[float] = field(default_factory=lambda: [0.28, 0.30, 0.28, 0.14])
    span_distributions: List[Tuple[float, float]] = field(default_factory=lambda: [(10, 5), (20, 10), (50, 20)])
    max_num_spans: int = 10
    allow_overlap: bool = False
    min_gap_between_spans: int = 0
    span_id_range: Tuple[int, int] = (0, 49)
    fixed_span_id: Optional[int] = None
    fixed_span_length: Optional[int] = None
    fixed_regions: Optional[List[Dict[str, Any]]] = None
    fixed_region_ratio: float = 0.0


# ========== 序列标准化函数 ==========
def normalize_rna_sequence(sequence: str, warn_non_standard: bool = True) -> Tuple[str, Dict[str, int]]:
    """
    标准化RNA序列

    Args:
        sequence: 原始序列
        warn_non_standard: 是否警告非标准碱基

    Returns:
        (标准化后的序列, 统计信息)
    """
    stats = {"t_to_u_count": 0, "non_standard_count": 0}

    # 转大写
    sequence = sequence.upper()

    # T -> U 转换
    original_t_count = sequence.count('T')
    sequence = sequence.replace('T', 'U')
    stats["t_to_u_count"] = original_t_count

    # 只保留标准碱基
    standard_bases = set('AUGC')
    clean_sequence = []
    for base in sequence:
        if base in standard_bases:
            clean_sequence.append(base)
        else:
            stats["non_standard_count"] += 1
            if warn_non_standard:
                pass  # 静默处理

    return ''.join(clean_sequence), stats


def _process_sequence_chunk(offsets: List[int], data_file: str, lineage_mapper, processor) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    并行处理序列chunk的工作函数

    Args:
        offsets: 序列偏移量列表
        data_file: FASTA文件路径
        lineage_mapper: 谱系映射器
        processor: 序列处理器

    Returns:
        (索引列表, 分类统计)
    """
    index = []
    stats = {
        "no_lineage": 0,
        "no_rna_type": 0,
        "too_short": 0,
        "total_processed": 0,
    }

    with open(data_file, 'r', encoding='utf-8') as f:
        for offset in offsets:
            f.seek(offset)

            # 读取header
            header = f.readline().strip()

            # 读取序列
            seq_lines = []
            while True:
                line = f.readline()
                if not line or line.startswith('>'):
                    break
                seq_lines.append(line.strip())

            # 处理序列
            stats["total_processed"] += 1
            seq = ''.join(seq_lines)

            # 过滤太短的序列（这个仍然要过滤）
            if len(seq) < 20:
                stats["too_short"] += 1
                continue

            # 解析header
            taxid, rna_type = _parse_header_simple(header)

            # 检查RNA类型（不过滤，只统计）
            rna_token = processor._get_rna_type_token(rna_type)
            if rna_token is None:
                stats["no_rna_type"] += 1

            # 检查谱系（不过滤，只统计）
            has_lineage = False
            if lineage_mapper:
                lineage = lineage_mapper.get_lineage(taxid)
                has_lineage = lineage is not None

            if not has_lineage:
                stats["no_lineage"] += 1

            # 添加到索引（保留所有样本，包括缺少谱系或RNA类型的）
            index.append({
                "offset": offset,
                "seq_len": len(seq),
                "taxid": taxid,
                "rna_type": rna_type,
            })

    return index, stats


def _parse_header_simple(header: str) -> Tuple[str, str]:
    """简化的header解析函数（用于并行处理）"""
    taxid = "unknown"
    rna_type = "unknown"

    try:
        parts = header.strip('>').split('|')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 'taxid' or key == 'species_taxid':
                    taxid = value.strip()
                elif key == 'rna_type':
                    rna_type = value.strip()
    except:
        pass

    return taxid, rna_type


def clean_lineage(lineage: str) -> str:
    """
    清理谱系字符串，移除无关标注信息

    处理规则:
    1. 移除括号及其内容: (in: xxx), (nom. ined.) 等分类学标注
    2. 移除方括号但保留内容: [Acholeplasma] -> Acholeplasma
    3. 移除单引号: 'Candidatus' -> Candidatus
    4. 空格替换为下划线: Homo sapiens -> Homo_sapiens
    5. 清理多余空白

    Args:
        lineage: 原始谱系字符串

    Returns:
        清理后的谱系字符串
    """
    # 1. 移除括号及其内容 (xxx)
    lineage = re.sub(r'\s*\([^)]*\)', '', lineage)

    # 2. 移除方括号但保留内容 [Genus] -> Genus
    lineage = re.sub(r'\[([^\]]+)\]', r'\1', lineage)

    # 3. 移除单引号 'Genus' -> Genus
    lineage = lineage.replace("'", "")

    # 4. 空格替换为下划线
    lineage = lineage.replace(" ", "_")

    # 5. 清理多余空白
    lineage = re.sub(r'\s+', '', lineage)

    # 6. 移除物种名末尾的编号（提升泛化能力）
    # 示例: actinobacterium_220105 -> actinobacterium
    #       bacterium_41040-1 -> bacterium
    #       sp._ABC123 -> sp.
    # 6a. 原有规则：移除下划线开头的后缀
    lineage = re.sub(r'_[A-Z0-9][A-Z0-9\-._]*(?=;|$)', '', lineage)

    # 6b. 移除菌株编号：O157, H7, O157:H7等（在转小写前处理）
    lineage = re.sub(r'_?[OH]\d+(?::\w+)?', '', lineage, flags=re.IGNORECASE)

    # 6c. 移除serovar/serotype后缀及其内容
    lineage = re.sub(r'_serovar_\w+', '', lineage, flags=re.IGNORECASE)

    # 6d. 移除sp.后的编号（sp._ABC123 -> sp.）
    lineage = re.sub(r'(sp\.)_?[A-Z0-9]+', r'\1', lineage, flags=re.IGNORECASE)

    # 6e. 移除冒号（菌株编号分隔符）
    lineage = lineage.replace(':', '')

    # 7. 转小写（避免与RNA碱基大写AUCG混淆）
    lineage = lineage.lower()

    # 8. 移除罕见特殊符号（%, #, +, ,, &, ", >, /, .）
    # 原因: 这些符号占0.64%的<unk>，且会干扰后续的编号清理
    # 必须在移除数字之前执行，避免如 #WM-A1% 中的#影响模式匹配
    # "/" 和 "." 仅在0.05%的谱系中使用，主要用于地理标记和缩写编号
    lineage = re.sub(r'[%#+,&">\/.]', '', lineage)

    # 9. 移除所有数字（提升泛化能力，避免<unk> token）
    # 原因: 99.36%的<unk>来自数字（扫描372,528条谱系的结果）
    # 示例: bacterium_2c -> bacterium_c, actinobacterium_15a-1 -> actinobacterium_a-
    lineage = re.sub(r'\d+', '', lineage)

    # 10. 移除编号残留的字母部分
    # 10a. 移除 _短字母-字母 格式（如 _wm-a, _a-, _cp-）
    lineage = re.sub(r'_[a-z]{1,4}-[a-z]*(?=;|$)', '', lineage)
    # 10b. 移除末尾 _1-3个字母 的短编号（如 _cpa, _bc, _a）
    lineage = re.sub(r'_[a-z]{1,3}(?=;|$)', '', lineage)

    # 11. 移除_sp.后缀（sp.表示未鉴定到种的属级分类）
    lineage = re.sub(r'_sp\.(?=;|$)', '', lineage)

    # 11b. 移除bacterium后的编号（如bacterium_xpnta -> bacterium）
    # 对于未鉴定的细菌，只保留bacterium，不保留菌株编号
    lineage = re.sub(r'bacterium_[^;]+', 'bacterium', lineage)

    # 12. 清理多余下划线（保留层级前缀的双下划线）
    lineage = re.sub(r'_{3,}', '__', lineage)  # 合并3个及以上的下划线为双下划线（保留层级前缀）
    # 只移除"非双下划线一部分"的末尾单个下划线
    # (?<!_) 表示前面不是下划线，这样可以保留空值层级前缀如p__
    lineage = re.sub(r'(?<!_)_(?=;|$)', '', lineage)  # 移除末尾单个下划线（但保留双下划线）

    # 13. 修剪尾部空层级（保留中间缺失层级）
    # 修剪规则：
    # - 移除尾部连续的空层级（如 c__、o__）
    # - 保留中间的空层级（如 d__bacteria;p__;c__firmicutes 中的 p__）
    # - 只保留到最后一个有具体内容的层级
    if ';' in lineage:
        levels = lineage.split(';')

        # 从后向前找到第一个非空层级
        # 空层级判断：格式为 "前缀__"（如 c__、o__、s__ 等）
        last_non_empty_idx = -1
        for i in range(len(levels) - 1, -1, -1):
            # 检查是否为空层级：匹配 "单字母__" 或 "多字母__"（没有后续内容）
            if not re.match(r'^[a-z]+__$', levels[i]):
                last_non_empty_idx = i
                break

        # 截断到最后一个非空层级
        if last_non_empty_idx >= 0:
            lineage = ';'.join(levels[:last_non_empty_idx + 1])
        else:
            # 所有层级都是空的，返回空字符串
            lineage = ''

    return lineage


@dataclass
class SpanInfo:
    """单个Span的信息"""
    start: int       # 起始位置（序列索引）
    length: int      # 长度
    span_id: int     # <span_0> ~ <span_49>

    @property
    def end(self) -> int:
        """Span结束位置（不包含）"""
        return self.start + self.length

    def overlaps_with(self, other: 'SpanInfo', min_gap: int = 0) -> bool:
        """
        检查是否与另一个span重叠（考虑最小间隔）

        Args:
            other: 另一个SpanInfo对象
            min_gap: 最小间隔（0=可紧邻）

        Returns:
            True如果重叠，False如果不重叠
        """
        # 考虑间隔：[self.start - min_gap, self.end + min_gap)
        return not (self.end + min_gap <= other.start or
                    other.end + min_gap <= self.start)


class LineageMapper:
    """物种谱系映射器 - 从lineage_greengenes.tsv加载taxid到谱系的映射"""

    def __init__(self, lineage_file: str):
        """
        初始化谱系映射器

        Args:
            lineage_file: lineage_greengenes.tsv文件路径
        """
        self.lineage_file = lineage_file
        self.taxid_to_lineage = self._load_lineage_mapping()
        logger.info(f"谱系映射器初始化完成: {len(self.taxid_to_lineage)} 条映射")

    def _load_lineage_mapping(self) -> Dict[str, str]:
        """加载taxid到谱系字符串的映射"""
        mapping = {}

        try:
            with open(self.lineage_file, 'r', encoding='utf-8') as f:
                header = f.readline()  # 跳过header

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 2:
                        taxid = parts[0].strip()
                        lineage = parts[1].strip()
                        mapping[taxid] = lineage

            logger.info(f"成功加载 {len(mapping)} 条谱系映射")
            return mapping

        except Exception as e:
            logger.error(f"加载谱系文件失败: {e}")
            return {}

    def get_lineage(self, taxid: str) -> Optional[str]:
        """
        获取taxid对应的谱系字符串

        Args:
            taxid: 物种taxid

        Returns:
            Greengenes格式的谱系字符串，如果找不到返回None
        """
        return self.taxid_to_lineage.get(taxid)


class LineageRNASequenceProcessor:
    """
    基于谱系的RNA序列处理器
    支持Greengenes谱系字符串编码
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 2048,
        lineage_file: Optional[str] = None,
        use_direction_tokens: bool = True,
        add_bos_token: bool = True,  # 是否添加<bos> token（默认True保持兼容）
        use_lineage_prefix: bool = True,  # 是否使用谱系前缀（默认True保持兼容）
        use_rna_type_prefix: bool = True,  # 是否使用RNA类型前缀（默认True保持兼容）
        enable_reverse_augmentation: bool = True,  # 是否启用序列反转数据增强（默认True）
        fuzzy_factor: float = 0.2,  # 模糊因子（默认0.2）
        deterministic: bool = False,  # 是否使用确定性模式（评估时关闭随机性）
        pretrain_ratio: float = 0.0,  # pretraining任务比例（0.0-1.0，无前缀的纯序列）
        fixed_lineage: Optional[str] = None,  # 固定谱系字符串（用于微调特定物种）
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_direction_tokens = use_direction_tokens  # 是否使用5/3方向标记
        self.add_bos_token = add_bos_token  # 是否添加<bos> token
        self.use_lineage_prefix = use_lineage_prefix  # 是否使用谱系前缀
        self.use_rna_type_prefix = use_rna_type_prefix  # 是否使用RNA类型前缀
        self.enable_reverse_augmentation = enable_reverse_augmentation  # 是否启用序列反转
        self.fuzzy_factor = fuzzy_factor  # 模糊因子
        self.deterministic = deterministic  # 确定性模式（评估时关闭随机性）
        self.pretrain_ratio = pretrain_ratio  # pretraining任务比例
        self.fixed_lineage = fixed_lineage  # 固定谱系（微调时使用）

        # 初始化谱系映射器
        if fixed_lineage:
            # 使用固定谱系时，不需要谱系映射器
            logger.info(f"使用固定谱系: {fixed_lineage}")
            self.lineage_mapper = None
        elif lineage_file and os.path.exists(lineage_file):
            self.lineage_mapper = LineageMapper(lineage_file)
        else:
            if use_lineage_prefix:
                raise FileNotFoundError(f'Lineage conditioning requested but no lineage_file/fixed_lineage is available: {lineage_file}')
            self.lineage_mapper = None

        # RNA类型映射（与MultiTaskRNASequenceProcessor保持一致）
        self.rna_type_mapping = {
            "mRNA": "<rna_mRNA>",
            "mrna": "<rna_mRNA>",
            "rRNA": "<rna_rRNA>",
            "rrna": "<rna_rRNA>",
            "LSU_rRNA": "<rna_rRNA>",
            "18S_rRNA": "<rna_rRNA>",
            "16S_rRNA": "<rna_rRNA>",
            "tRNA": "<rna_tRNA>",
            "trna": "<rna_tRNA>",
            "sRNA": "<rna_sRNA>",
            "srna": "<rna_sRNA>",
            "lncRNA": "<rna_lncRNA>",
            "lncrna": "<rna_lncRNA>",
            "circRNA": "<rna_circRNA>",
            "circrna": "<rna_circRNA>",
            "viral_RNA": "<rna_viral_RNA>",
            "viral_rna": "<rna_viral_RNA>",
            "miRNA": "<rna_miRNA>",
            "mirna": "<rna_miRNA>",
            "snoRNA": "<rna_snoRNA>",
            "snorna": "<rna_snoRNA>",
            "snoRNA_C_D_box": "<rna_snoRNA>",
            "snoRNA_H_ACA_box": "<rna_snoRNA>",
            "snRNA": "<rna_snRNA>",
            "snrna": "<rna_snRNA>",
            "piRNA": "<rna_piRNA>",
            "pirna": "<rna_piRNA>",
            "ribozyme": "<rna_ribozyme>",
            "scaRNA": "<rna_scaRNA>",
            "scarna": "<rna_scaRNA>",
            "Y_RNA": "<rna_Y_RNA>",
            "y_rna": "<rna_Y_RNA>",
            "vault_RNA": "<rna_vault_RNA>",
            "vault_rna": "<rna_vault_RNA>",
        }

        # 排除的RNA类型
        self.excluded_rna_types = {
            "ncRNA", "ncrna", "misc_RNA", "misc_rna", "other", "null",
            "pseudogene", "IG_V_pseudogene", "TR_V_pseudogene", "TR_J_pseudogene",
            "IG_C_pseudogene", "IG_J_pseudogene", "IG_D_pseudogene", "rRNA_pseudogene",
            "transcribed_unitary_pseudogene",
            "pri_miRNA", "pre_miRNA", "miRNA_loop", "miRNA_primary_transcript",
            "nonsense_mediated_decay", "non_stop_decay", "retained_intron",
            "processed_transcript",
            "immunoglobulin_gene", "T_cell_receptor_gene", "IG_LV_gene",
            "unknown_likely_coding", "protein_coding_CDS_not_defined",
            "protein_coding_LoF", "TEC", "CDS", "scRNA", "antisense_RNA"
        }

    def _get_rna_type_token(self, rna_type: str) -> Optional[str]:
        """获取RNA类型对应的token"""
        if rna_type in self.excluded_rna_types or rna_type.lower() in self.excluded_rna_types:
            return None

        token = self.rna_type_mapping.get(rna_type)
        if token is None:
            token = self.rna_type_mapping.get(rna_type.lower())

        return token

    def _format_lineage_prefix(self, lineage: Optional[str], rna_token: Optional[str]) -> str:
        """
        格式化谱系前缀

        根据谱系和RNA类型的存在情况，以及配置选项，灵活格式化前缀：
        - 同时有谱系与RNA类型 → |谱系;<RNA类型>|
        - 只有谱系 → |谱系|
        - 只有RNA类型 → |<RNA类型>|
        - 谱系与RNA类型都没有 → 空字符串

        配置选项:
        - use_lineage_prefix=False 时不添加物种谱系信息
        - use_rna_type_prefix=False 时不添加RNA类型信息

        Args:
            lineage: 谱系字符串（可以为None）
            rna_token: RNA类型token（可以为None）

        Returns:
            格式化后的前缀字符串
        """
        # 根据配置过滤
        effective_lineage = lineage if self.use_lineage_prefix else None
        effective_rna_token = rna_token if self.use_rna_type_prefix else None

        # 情况1: 同时有谱系和RNA类型
        if effective_lineage and effective_rna_token:
            return f"|{effective_lineage};{effective_rna_token}|"

        # 情况2: 只有谱系
        elif effective_lineage and not effective_rna_token:
            return f"|{effective_lineage}|"  # 注意：末尾无分号

        # 情况3: 只有RNA类型
        elif not effective_lineage and effective_rna_token:
            return f"|{effective_rna_token}|"

        # 情况4: 都没有
        else:
            return ""  # 完全空字符串

    def process_generation_sample(
        self,
        sequence: str,
        lineage: Optional[str],
        rna_type: Optional[str],
        reverse_sequence: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        处理序列生成样本（Stage 1）

        格式: <bos>|谱系信息;<rna_类型>|5[序列]3<eos>

        当 pretrain_ratio > 0 时，有一定比例的样本使用无前缀格式：
        格式: <bos>5[序列]3<eos>

        Args:
            sequence: RNA序列
            lineage: Greengenes谱系字符串（可以为None）
            rna_type: RNA类型（可以为None）
            reverse_sequence: 是否反转序列（数据增强）

        Returns:
            处理后的样本字典，如果样本无效返回None
        """
        # 🆕 根据 pretrain_ratio 决定是否使用前缀（pretraining 任务）
        is_pretrain_sample = random.random() < self.pretrain_ratio

        # 检查RNA类型（允许为None或无效）
        rna_token = None
        if rna_type and not is_pretrain_sample:
            rna_token = self._get_rna_type_token(rna_type)
            # 注意：即使rna_token是None也继续处理，不再直接return None

        # 清理谱系字符串（如果存在且不是pretrain样本）
        cleaned_lineage = None
        if lineage and self.use_lineage_prefix and not is_pretrain_sample:
            cleaned_lineage = clean_lineage(lineage)

        # 得到完整前缀（pretrain样本返回空字符串）
        if is_pretrain_sample:
            lineage_prefix = ""  # pretraining: 无前缀
        else:
            lineage_prefix = self._format_lineage_prefix(cleaned_lineage, rna_token)

        # 计算前缀的实际token数量（处理空前缀的情况）
        if lineage_prefix:
            prefix_token_ids = self.tokenizer.encode(lineage_prefix)
            len_prefix_tokens = len(prefix_token_ids)
        else:
            len_prefix_tokens = 0  # 没有前缀

        # 截断序列（根据实际token数量动态调整预留空间）
        # 预留空间 = <bos>(1) + 前缀tokens + 方向标记(0/2) + <eos>(1)
        if self.use_direction_tokens:
            reserved_tokens = 1 + len_prefix_tokens + 2 + 1
        else:
            reserved_tokens = 1 + len_prefix_tokens + 1

        if len(sequence) > self.max_seq_length - reserved_tokens:
            sequence = sequence[:self.max_seq_length - reserved_tokens]

        # 添加方向标记（如果启用）
        if self.use_direction_tokens:
            sequence = "5" + sequence + "3"
            if reverse_sequence:
                sequence = sequence[::-1]  # 结果: 3...sequence...5
        else:
            # 序列反转（不添加方向标记）
            if reverse_sequence:
                sequence = sequence[::-1]

        # 构建完整序列
        if self.add_bos_token:
            formatted_seq = f"<bos>{lineage_prefix}{sequence}<eos>"
        else:
            # 兼容旧模型：不添加<bos>
            formatted_seq = f"{lineage_prefix}{sequence}<eos>"

        # Tokenize
        input_ids = self.tokenizer.encode(formatted_seq)
        labels = input_ids.copy()

        # 将<unk> token位置的labels设为-100（不计算loss，避免loss=inf）
        # 原因：N核苷酸会被编码为<unk>，而output_token_mask只允许A/U/G/C/<eos>
        #      如果<unk>参与loss计算，会导致所有logits为-inf，从而产生loss=inf
        unk_token_id = self.tokenizer.token_to_id("<unk>")
        if unk_token_id is not None:
            for i, token_id in enumerate(input_ids):
                if token_id == unk_token_id:
                    labels[i] = -100

        # 条件部分（前缀）不参与loss计算
        try:
            # 判断是否有|...|格式的前缀（包括3种情况：|谱系;RNA类型|、|谱系|、|RNA类型|）
            if '|' in formatted_seq:
                # 统一处理：找到第二个'|'，mask从0到第二个'|'（包含<bos>和整个|...|部分）
                pipe_positions = [i for i, token_id in enumerate(input_ids)
                                if self.tokenizer.decode([token_id]) == '|']

                if len(pipe_positions) >= 2:
                    # mask从0到第二个'|'（包含）
                    prefix_end = pipe_positions[1]
                    for i in range(0, prefix_end + 1):
                        labels[i] = -100

                    # mask第一个方向标签（如果启用了方向标记）
                    # 原因：第一个方向标签是随机决定的（数据增强50%概率反转），
                    #      条件信息中没有任何依据可以预测它应该是'5'还是'3'
                    if self.use_direction_tokens and prefix_end + 1 < len(labels):
                        next_token = self.tokenizer.decode([input_ids[prefix_end + 1]])
                        if next_token in ['5', '3']:
                            labels[prefix_end + 1] = -100
                else:
                    # 如果|符号不足2个，至少mask <bos>（容错处理）
                    if self.add_bos_token:
                        labels[0] = -100
            else:
                # 完全没有前缀（谱系和RNA类型都没有）
                # 只mask <bos>（如果有的话）
                if self.add_bos_token:
                    labels[0] = -100

                # 即使没有谱系前缀，如果启用了方向标记，第一个token可能是方向标签
                # 同样需要mask（因为它是随机决定的）
                if self.use_direction_tokens and len(input_ids) > 0:
                    # 确定第一个非<bos>位置
                    first_content_idx = 1 if self.add_bos_token else 0
                    if first_content_idx < len(input_ids):
                        first_token = self.tokenizer.decode([input_ids[first_content_idx]])
                        if first_token in ['5', '3']:
                            labels[first_content_idx] = -100
        except Exception as e:
            logger.warning(f"处理条件前缀时出错: {e}")

        # Stage 1额外保护：排除<eos_span>（虽然数据中不应出现，但作为额外保护）
        eos_span_token_id = self.tokenizer.token_to_id("<eos_span>")
        if eos_span_token_id is not None:
            for i, token_id in enumerate(input_ids):
                if token_id == eos_span_token_id:
                    labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "position_ids": torch.arange(len(input_ids), dtype=torch.long),
            "sequence_ids": torch.zeros(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_type": "lineage_generation",
            "sequence_length": len(sequence),
        }

    def _process_completion_sample_multi_span(
        self,
        sequence: str,
        lineage: Optional[str],
        rna_type: Optional[str],
        span_config: SpanConfig,
        reverse_sequence: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        处理多span补全样本（新逻辑）

        格式: <bos_glm>|谱系|AB<span_3>CD<span_17>EF<eos><span_3>XXX<span_17>YY<eos_span>

        当 pretrain_ratio > 0 时，有一定比例的样本使用无前缀格式
        """
        # 🆕 根据 pretrain_ratio 决定是否使用前缀（pretraining 任务）
        is_pretrain_sample = random.random() < self.pretrain_ratio

        # 检查RNA类型（允许为None或无效）
        rna_token = None
        if rna_type and not is_pretrain_sample:
            rna_token = self._get_rna_type_token(rna_type)

        # 清理谱系字符串（如果存在且不是pretrain样本）
        cleaned_lineage = None
        if lineage and self.use_lineage_prefix and not is_pretrain_sample:
            cleaned_lineage = clean_lineage(lineage)

        # 计算前缀长度用于序列截断
        if is_pretrain_sample:
            lineage_prefix = ""  # pretraining: 无前缀
            len_prefix_tokens = 0
        else:
            lineage_prefix = self._format_lineage_prefix(cleaned_lineage, rna_token)
            if lineage_prefix:
                len_prefix_tokens = len(self.tokenizer.encode(lineage_prefix))
            else:
                len_prefix_tokens = 0

        # 截断序列（预留空间给前缀、<eos>、<eos_span>等）
        # 预留：<bos_glm>(1) + prefix + <eos>(1) + <eos_span>(1) + 额外buffer(10)
        reserved_tokens = 1 + len_prefix_tokens + 1 + 1 + 10
        if len(sequence) > self.max_seq_length - reserved_tokens:
            sequence = sequence[:self.max_seq_length - reserved_tokens]

        # 如果需要反向，先反转整个序列
        if reverse_sequence:
            sequence = sequence[::-1]

        # ✅ 修复：采样多个span，并处理fixed_span_length导致的异常
        try:
            spans = self._sample_multiple_spans(len(sequence), span_config)
        except ValueError as e:
            # 评估时：fixed_span_length太长，跳过该序列
            if "fixed_span_length" in str(e):
                logger.debug(f"跳过序列: {e}")
                return None
            else:
                raise  # 其他错误继续抛出

        if not spans:
            # 没有采样到span，返回None
            logger.warning(f"未采样到任何span，序列长度={len(sequence)}")
            return None

        # 构建多span GLM格式
        formatted_seq, input_ids, labels = self._build_multi_span_glm(
            sequence, cleaned_lineage, rna_token, spans, reverse_sequence
        )

        # 计算position_ids
        position_ids = self._calculate_glm_position_ids_multi_span(
            input_ids, spans, fuzzy_factor=0.2
        )

        # 计算span统计信息
        total_span_length = sum(s.length for s in spans)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "position_ids": position_ids,
            "sequence_ids": torch.zeros(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_type": "lineage_completion_multi_span",
            "sequence_length": len(sequence),
            "num_spans": len(spans),
            "total_span_length": total_span_length,
        }

    def _sample_multiple_spans(
        self,
        seq_len: int,
        span_config: SpanConfig
    ) -> List[SpanInfo]:
        """
        采样多个span（遵循预算约束）

        🔄 修正：使用"先过滤分布再采样"策略（Downgrade Strategy）
        避免大分布被强行截断成小碎片。

        采样逻辑：
        1. Filter: 过滤掉 μ > remaining_budget 的分布
        2. Sample: 从合法分布中随机选一个，采样长度
        3. Fallback: 如果没有合法分布，直接用剩余预算

        Args:
            seq_len: 序列长度
            span_config: Span配置

        Returns:
            按起始位置排序的span列表
        """
        import random

        # ===== 🆕 固定长度模式（用于eval消融实验） =====
        if span_config.fixed_span_length is not None:
            fixed_length = span_config.fixed_span_length

            # 序列太短，无法采样该长度的span，返回空列表（跳过评估）
            if fixed_length > seq_len:
                return []

            # 随机选择起始位置
            max_start = seq_len - fixed_length
            if max_start < 0:
                return []

            start_pos = random.randint(0, max_start)

            # 选择span_id（优先使用固定值）
            span_id = span_config.fixed_span_id if span_config.fixed_span_id is not None else random.randint(*span_config.span_id_range)

            # 返回单个固定长度的span
            return [SpanInfo(start=start_pos, length=fixed_length, span_id=span_id)]

        # ===== 🆕 固定区域模式（用于指定区域GLM训练） =====
        if span_config.fixed_regions and random.random() < span_config.fixed_region_ratio:
            return self._create_fixed_region_spans(seq_len, span_config)

        # ===== 🔄 修正逻辑：高斯分布多span采样（用于训练） =====
        # 步骤1: 采样覆盖比例，计算总预算
        coverage_ratio = random.choices(
            span_config.max_coverage_ratios,
            weights=span_config.coverage_probs
        )[0]
        total_budget = int(coverage_ratio * seq_len)

        if total_budget < 1:
            return []  # 预算不足，不采样

        # 步骤2: 循环采样span
        spans = []
        remaining_budget = total_budget

        for _ in range(span_config.max_num_spans):
            # 检查预算
            if remaining_budget < 4: #如果预算只剩渣渣（例如 < 4bp），就不硬凑了
                break

            # 🔄 修正：先过滤分布，再采样（Downgrade Strategy）
            # 2.1 Filter: 过滤掉 μ > remaining_budget 的分布
            available_distributions = [
                (mu, sigma) for mu, sigma in span_config.span_distributions
                if mu * 0.8 <= remaining_budget #宽松过滤：只要均值不超过预算太多（比如预算是均值的80%以上），可以尝试采样再截断
            ]

            # 2.2 Fallback: 如果没有合法分布，直接使用剩余预算
            if not available_distributions:
                # 使用剩余预算作为长度
                sampled_length = remaining_budget
                if sampled_length < 1:
                    break
            else:
                # 2.3 Sample: 从合法分布中随机选一个
                mu, sigma = random.choice(available_distributions)

                # 2.4 从高斯分布采样长度
                sampled_length = int(random.gauss(mu, sigma))

                # 2.5 截断到[1, remaining_budget]（不再截断到seq_len，因为预算已经限制了）
                sampled_length = max(1, min(sampled_length, remaining_budget))

            # 2.6 寻找可放置位置
            max_attempts = 500  # 防止死循环
            placed = False

            for attempt in range(max_attempts):
                # 随机起始位置
                max_start = seq_len - sampled_length
                if max_start < 0:
                    break

                start_pos = random.randint(0, max_start)

                # 创建候选span
                candidate = SpanInfo(
                    start=start_pos,
                    length=sampled_length,
                    span_id=span_config.fixed_span_id if span_config.fixed_span_id is not None else random.randint(*span_config.span_id_range)
                )

                # 检查重叠
                if span_config.allow_overlap:
                    overlaps = False
                else:
                    overlaps = any(
                        candidate.overlaps_with(existing, span_config.min_gap_between_spans)
                        for existing in spans
                    )

                if not overlaps:
                    spans.append(candidate)
                    remaining_budget -= sampled_length
                    placed = True
                    break

            if not placed:
                # 找不到可放置位置，停止采样
                break

        # 步骤3: 按起始位置排序
        spans.sort(key=lambda s: s.start)

        return spans

    def _create_fixed_region_spans(
        self,
        seq_len: int,
        span_config: SpanConfig
    ) -> List[SpanInfo]:
        """
        根据固定区域配置创建span（随机选择一个区域）

        Args:
            seq_len: 序列长度
            span_config: Span配置（包含fixed_regions）

        Returns:
            包含单个固定区域span的列表
        """
        import random

        if not span_config.fixed_regions:
            return []

        # 随机选择一个固定区域
        region = random.choice(span_config.fixed_regions)
        start = region.get('start', 0)
        length = region.get('length', 100)

        # 处理负数起始位置（从末尾计算）
        if start < 0:
            start = max(0, seq_len + start)

        # 确保不超出序列范围
        if start >= seq_len:
            return []

        actual_length = min(length, seq_len - start)
        if actual_length <= 0:
            return []

        # 分配span_id（优先使用固定值）
        span_id = span_config.fixed_span_id if span_config.fixed_span_id is not None else random.randint(*span_config.span_id_range)

        return [SpanInfo(start=start, length=actual_length, span_id=span_id)]

    def _build_multi_span_glm(
        self,
        sequence: str,
        lineage: Optional[str],
        rna_token: Optional[str],
        spans: List[SpanInfo],
        reverse_sequence: bool = False
    ) -> Tuple[str, List[int], List[int]]:
        """
        构建多span GLM格式

        格式：
        - 正向：|lineage|5AB<span_3>CD<span_17>EF3<eos><span_3>XXX<span_17>YY<eos_span>
        - 反向：|lineage|3BA<span_3>DC<span_17>FE5<eos><span_3>XXX<span_17>YY<eos_span>

        Args:
            sequence: 原始序列
            lineage: 谱系字符串（可选）
            rna_token: RNA类型token（可选）
            spans: Span列表（已按位置排序）
            reverse_sequence: 是否为反向序列（影响方向标记5/3的位置）

        Returns:
            (formatted_seq, input_ids, labels)
        """
        # 构建前缀
        lineage_prefix = self._format_lineage_prefix(lineage, rna_token)

        # ===== 构建输入部分：prefix + suffix嵌套 =====
        parts = []
        last_end = 0

        # 添加起始方向标记（如果启用）
        if self.use_direction_tokens:
            if reverse_sequence:
                parts.append("3")  # 反向序列：3'端在起始位置
            else:
                parts.append("5")  # 正向序列：5'端在起始位置

        for span in spans:
            # 添加中间可见部分
            visible_part = sequence[last_end:span.start]
            parts.append(visible_part)

            # 添加span占位符
            parts.append(f"<span_{span.span_id}>")

            last_end = span.end

        # 添加最后的可见部分
        parts.append(sequence[last_end:])

        # 添加结束方向标记（如果启用）
        if self.use_direction_tokens:
            if reverse_sequence:
                parts.append("5")  # 反向序列：5'端在结束位置
            else:
                parts.append("3")  # 正向序列：3'端在结束位置

        input_part = ''.join(parts)

        # ===== 构建生成部分：所有span内容，每个span后都有<eos_span> =====
        generation_parts = []
        for span in spans:
            span_content = sequence[span.start:span.end]
            # 每个span内容后都加<eos_span>，让模型学会何时停止生成当前span
            generation_parts.append(f"<span_{span.span_id}>{span_content}<eos_span>")

        generation_part = ''.join(generation_parts)

        # ===== 完整序列 =====
        if self.add_bos_token:
            formatted_seq = f"<bos_glm>{lineage_prefix}{input_part}<eos>{generation_part}"
        else:
            formatted_seq = f"{lineage_prefix}{input_part}<eos>{generation_part}"

        # ===== Tokenize =====
        input_ids = self.tokenizer.encode(formatted_seq)

        # ===== 构建Labels（只在生成部分的span内容计算loss） =====
        labels = self._build_multi_span_labels(input_ids, spans, lineage_prefix)

        return formatted_seq, input_ids, labels

    def _build_multi_span_labels(
        self,
        input_ids: List[int],
        spans: List[SpanInfo],
        lineage_prefix: str
    ) -> List[int]:
        """
        构建labels：只在生成部分的span内容计算loss

        🛑 修正：<eos_span> 必须保留计算loss，否则模型不知道何时停止生成！

        逻辑：
        1. 输入部分（lineage + prefix + suffix嵌套）: 全部-100
        2. <eos>: -100
        3. 生成部分的<span_id>: -100（提示符，不计算loss）
        4. 生成部分的span内容: 保留（计算loss）
        5. <eos_span>: 保留 ✅（必须计算loss，让模型学会停止！）

        Args:
            input_ids: tokenized序列
            spans: Span列表
            lineage_prefix: 谱系前缀字符串

        Returns:
            labels列表
        """
        labels = input_ids.copy()

        # 找到关键token ID
        eos_token_id = self.tokenizer.token_to_id("<eos>")
        eos_span_token_id = self.tokenizer.token_to_id("<eos_span>")

        # 构建span_id到token_id的映射
        span_token_ids = {
            self.tokenizer.token_to_id(f"<span_{s.span_id}>")
            for s in spans
        }

        # 找到<eos>位置（输入部分和生成部分的分界）
        eos_position = next(
            (i for i, tid in enumerate(input_ids) if tid == eos_token_id),
            None
        )

        if eos_position is None:
            # 容错：找不到<eos>，全部mask
            logger.warning("未找到<eos> token，全部mask")
            return [-100] * len(input_ids)

        # 步骤1: Mask输入部分（0 到 <eos>，包含<eos>）
        for i in range(eos_position + 1):
            labels[i] = -100

        # 步骤2: 处理生成部分（<eos>之后）
        for i in range(eos_position + 1, len(labels)):
            token_id = input_ids[i]

            if token_id in span_token_ids:
                # 🛑 <span_id> 是提示符，Mask掉
                labels[i] = -100
            elif token_id == eos_span_token_id:
                # ✅ <eos_span> 保留计算loss（必须让模型学会停止！）
                pass  # 保持原样，不设为-100
            # else: 保留span内容（真实token用于计算loss）

        # 步骤3: 额外保护 - 将<unk> token标记为-100（避免loss=inf）
        unk_token_id = self.tokenizer.token_to_id("<unk>")
        if unk_token_id is not None:
            for i in range(len(labels)):
                if input_ids[i] == unk_token_id:
                    labels[i] = -100

        return labels

    def _calculate_glm_position_ids_multi_span(
        self,
        input_ids: List[int],
        spans: List[SpanInfo],
        fuzzy_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算多span GLM的跳跃式position_ids

        🔄 修正：使用几何分布 (Geometric Distribution) 计算 fuzzy_jump
        根据 ProGen3 论文：
        - δ 必须是非负数（告诉模型"有更多空余位置"，训练 Early Stopping）
        - δ 服从几何分布：mean_delta = 0.2 * span_length
        - jump_width = span_length + δ ≥ span_length

        Args:
            input_ids: tokenized序列
            spans: Span列表（已按位置排序）
            fuzzy_factor: 模糊因子（可选，如果为None则使用self.fuzzy_factor）

        Returns:
            position_ids张量
        """
        import numpy as np

        # 使用默认的fuzzy_factor（来自实例属性）
        if fuzzy_factor is None:
            fuzzy_factor = self.fuzzy_factor

        position_ids = []
        current_pos = 0

        # 构建span_id到span对象的映射
        span_id_to_info = {s.span_id: s for s in spans}
        span_id_to_token = {
            s.span_id: self.tokenizer.token_to_id(f"<span_{s.span_id}>")
            for s in spans
        }
        token_to_span_id = {v: k for k, v in span_id_to_token.items()}

        # 记录每个span在输入部分的起始position
        span_start_positions = {}

        # 找到<eos>位置（输入部分和生成部分的分界）
        eos_token_id = self.tokenizer.token_to_id("<eos>")
        eos_position = next(
            (i for i, tid in enumerate(input_ids) if tid == eos_token_id),
            None
        )

        if eos_position is None:
            logger.warning("未找到<eos> token，使用顺序position")
            return torch.arange(len(input_ids), dtype=torch.long)

        # ===== 阶段1: 输入部分（prefix + suffix嵌套） =====
        for i in range(eos_position + 1):
            token_id = input_ids[i]

            # 检查是否是span标记
            if token_id in token_to_span_id:
                span_id = token_to_span_id[token_id]
                span_info = span_id_to_info[span_id]

                # 记录span的起始position
                span_start_positions[span_id] = current_pos
                position_ids.append(current_pos)

                # 跳过 1 + L 个位置（ProGen3逻辑）
                # 1个span标记位置 + span内容长度 + δ
                current_pos += 1  # span标记占1个位置

                # 计算 jump_width
                if self.deterministic:
                    # 评估模式：确定性，不加 δ
                    jump_width = span_info.length
                else:
                    # 训练模式：使用几何分布采样 δ
                    # mean_delta = 0.2 * span_length（fuzzy_factor 默认为 0.2）
                    # p = 1 / (mean_delta + 1)
                    # delta = np.random.geometric(p) - 1（确保从 0 开始）
                    mean_delta = fuzzy_factor * span_info.length
                    p = 1.0 / (mean_delta + 1.0)
                    delta = np.random.geometric(p) - 1  # 从 0 开始
                    jump_width = span_info.length + delta

                jump_width = max(1, jump_width)  # 确保至少为1
                current_pos += jump_width
            else:
                # 普通token：正常递增
                position_ids.append(current_pos)
                current_pos += 1

        # ===== 阶段2: 生成部分（依次填充各个span） =====
        i = eos_position + 1
        while i < len(input_ids):
            token_id = input_ids[i]

            # 检查是否是span标记（生成部分的span起始）
            if token_id in token_to_span_id:
                span_id = token_to_span_id[token_id]

                # 重置到该span的起始位置（与ProGen3对齐）
                # 两个<span_id>必须共享同一个position
                if span_id in span_start_positions:
                    current_pos = span_start_positions[span_id]
                else:
                    # 容错：如果没找到起始位置，继续递增
                    logger.warning(f"未找到span_{span_id}的起始位置")

                position_ids.append(current_pos)
                current_pos += 1
            else:
                # span内容：继续递增
                position_ids.append(current_pos)
                current_pos += 1

            i += 1

        return torch.tensor(position_ids, dtype=torch.long)

    def _sample_span_parameters(self, seq_len: int, span_config: SpanConfig) -> Tuple[int, int]:
        """
        采样span的起始位置和长度

        ✅ 修复：支持fixed_span_length参数用于评估

        Args:
            seq_len: 序列长度
            span_config: Span采样配置

        Returns:
            (span_start, span_length)

        Raises:
            ValueError: 当fixed_span_length超过序列长度时（评估模式下）
        """
        # ✅ 修复：检查是否使用固定span长度（用于评估）
        if hasattr(span_config, 'fixed_span_length') and span_config.fixed_span_length is not None:
            # 评估模式：使用固定span长度
            span_length = span_config.fixed_span_length

            # 检查span长度是否超过序列长度
            if span_length >= seq_len:
                # 评估时：span太长，跳过该序列
                raise ValueError(f"fixed_span_length={span_length} >= seq_len={seq_len}, skipping sequence")

            # 随机采样起始位置（确保span在序列范围内）
            max_start = max(0, seq_len - span_length)
            span_start = random.randint(0, max_start)
        else:
            # 训练模式：随机采样span长度
            # 随机选择一个长度比例
            ratio_idx = random.choices(
                range(len(span_config.ratio_probs)),
                weights=span_config.ratio_probs
            )[0]

            max_length_ratio = span_config.max_length_ratios[ratio_idx]
            span_mu = span_config.span_mus[ratio_idx]
            span_sigma = span_config.span_sigmas[ratio_idx]

            # 计算最大span长度
            max_span_length = int(seq_len * max_length_ratio)

            # 从高斯分布采样span长度
            span_length = int(random.gauss(span_mu, span_sigma))
            span_length = max(1, min(span_length, max_span_length))

            # 随机采样起始位置
            max_start = max(0, seq_len - span_length)
            span_start = random.randint(0, max_start)

        return span_start, span_length

    def _calculate_glm_position_ids(
        self,
        input_ids: List[int],
        span_length: int,
        target_length: int,
        fuzzy_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算GLM格式的position_ids（跳跃式）

        Args:
            input_ids: 输入token ids
            span_length: 原始span长度
            target_length: 目标生成长度
            fuzzy_factor: 模糊因子（可选，如果为None则使用self.fuzzy_factor）

        Returns:
            position_ids张量
        """
        # 使用默认的fuzzy_factor（来自实例属性）
        if fuzzy_factor is None:
            fuzzy_factor = self.fuzzy_factor

        span_token_id = self.tokenizer.token_to_id("<span_0>")
        eos_token_id = self.tokenizer.token_to_id("<eos>")

        position_ids = []
        current_pos = 0
        in_suffix = False
        in_generation = False

        span_positions = [i for i, tid in enumerate(input_ids) if tid == span_token_id]
        eos_position = next((i for i, tid in enumerate(input_ids) if tid == eos_token_id), None)

        for i, token_id in enumerate(input_ids):
            if not in_suffix and not in_generation:
                # 前缀部分：正常递增
                position_ids.append(current_pos)
                current_pos += 1

                # 检查是否到达第一个<span_0>
                if len(span_positions) > 0 and i == span_positions[0]:
                    in_suffix = True

            elif in_suffix and not in_generation:
                # suffix部分：position跳跃
                if i == span_positions[0]:
                    # <span_0> token
                    position_ids.append(current_pos)
                    # ✅ 修复：跳过 1 + L 个位置（ProGen3逻辑）
                    # 1个span标记位置 + span内容长度
                    current_pos += 1  # span标记占1个位置

                    # 计算fuzzy跳跃（span内容长度）
                    if self.deterministic:
                        # ✅ 确定性模式（评估时）：不使用fuzzy，确保可重现
                        target_jump = target_length
                    else:
                        # 训练模式：使用fuzzy，增强泛化
                        target_jump = int(target_length * (1.0 + random.uniform(-fuzzy_factor, fuzzy_factor)))
                    target_jump = max(1, target_jump)
                    current_pos += target_jump  # 再跳span内容长度
                else:
                    position_ids.append(current_pos)
                    current_pos += 1

                # 检查是否到达<eos>
                if eos_position is not None and i == eos_position:
                    in_suffix = False
                    in_generation = True
                    # ✅ 修复：重置到span起始位置（不加1，与ProGen3对齐）
                    # 两个<span_0>必须共享同一个position
                    if len(span_positions) > 0:
                        span_start_pos = position_ids[span_positions[0]]
                        current_pos = span_start_pos  # 不加1！

            else:
                # 生成部分：从span起始位置继续
                position_ids.append(current_pos)
                current_pos += 1

        return torch.tensor(position_ids, dtype=torch.long)


class LineageRNADataset(Dataset):
    """
    基于谱系的RNA数据集
    支持两种训练模式：序列生成（Stage 1）和序列补全（Stage 2）
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: Any,
        lineage_file: str,
        max_seq_length: int = 2048,
        mode: str = "generation",  # "generation" 或 "completion" 或 "mixed"
        span_config: Optional[SpanConfig] = None,
        glm_probability: float = 0.333,  # GLM采样概率（仅在mode='mixed'时生效）
        max_samples: Optional[int] = None,
        use_direction_tokens: bool = True,
        add_bos_token: bool = True,
        use_lineage_prefix: bool = True,
        use_rna_type_prefix: bool = True,  # 是否使用RNA类型前缀（默认True保持兼容）
        enable_reverse_augmentation: bool = True,
        fuzzy_factor: float = 0.2,  # 模糊因子
        deterministic: bool = False,  # 确定性模式
        pretrain_ratio: float = 0.0,  # pretraining任务比例（0.0-1.0，无前缀的纯序列）
        fixed_lineage: Optional[str] = None,  # 固定谱系字符串（用于微调特定物种）
    ):
        """
        初始化数据集

        Args:
            data_file: FASTA数据文件路径
            tokenizer: RNA tokenizer
            lineage_file: lineage_greengenes.tsv文件路径
            max_seq_length: 最大序列长度
            mode: 训练模式 ("generation" 或 "completion")
            span_config: Span采样配置（仅completion模式需要）
            max_samples: 最大样本数
            use_direction_tokens: 是否使用5/3方向标记
            add_bos_token: 是否添加<bos> token（兼容旧模型）
            use_lineage_prefix: 是否使用谱系前缀（默认True）
            use_rna_type_prefix: 是否使用RNA类型前缀（默认True）
            enable_reverse_augmentation: 是否启用序列反转数据增强（默认True）
            fuzzy_factor: 模糊因子（训练时使用）
            deterministic: 确定性模式（评估时设为True）
            pretrain_ratio: pretraining任务比例（0.0-1.0，无前缀的纯序列）
            fixed_lineage: 固定谱系字符串（用于微调特定物种，如病毒）
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.lineage_file = lineage_file
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.span_config = span_config or SpanConfig()
        self.glm_probability = glm_probability
        self.max_samples = max_samples
        self.use_direction_tokens = use_direction_tokens
        self.add_bos_token = add_bos_token
        self.use_lineage_prefix = use_lineage_prefix
        self.use_rna_type_prefix = use_rna_type_prefix
        self.fuzzy_factor = fuzzy_factor
        self.deterministic = deterministic
        self.pretrain_ratio = pretrain_ratio
        self.fixed_lineage = fixed_lineage

        # 初始化处理器
        self.processor = LineageRNASequenceProcessor(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            lineage_file=lineage_file,
            use_direction_tokens=use_direction_tokens,
            add_bos_token=add_bos_token,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
            fuzzy_factor=fuzzy_factor,
            deterministic=deterministic,
            pretrain_ratio=pretrain_ratio,
            fixed_lineage=fixed_lineage,
        )

        # 加载数据
        self.samples = self._load_data()

        logger.info(f"LineageRNADataset初始化完成:")
        logger.info(f"  - 模式: {mode}")
        logger.info(f"  - 样本数: {len(self.samples)}")
        logger.info(f"  - 谱系文件: {lineage_file}")

    def _load_data(self) -> List[Tuple[str, Optional[str], Optional[str], str]]:
        """
        加载FASTA数据

        Returns:
            List of (sequence, lineage, rna_type, taxid)
            其中lineage和rna_type可以为None
        """
        samples = []
        total_t_to_u = 0

        # 详细的分类统计
        filter_stats = {
            "no_lineage": 0,  # 没有谱系信息（不过滤，只统计）
            "no_rna_type": 0,  # 没有有效RNA类型（不过滤，只统计）
            "too_short": 0,  # 序列太短（会被过滤）
            "total_processed": 0,  # 总处理数
        }

        # 记录样本分类示例（用于诊断）
        sample_examples = {
            "no_lineage": [],  # 最多记录10个
            "no_rna_type": [],  # 最多记录10个
        }

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                current_seq = []
                current_header = ""

                for line in f:
                    line = line.strip()

                    if line.startswith('>'):
                        # 处理前一个序列
                        if current_seq and current_header:
                            seq = ''.join(current_seq)
                            filter_stats["total_processed"] += 1

                            if len(seq) < 20:
                                filter_stats["too_short"] += 1
                            else:
                                normalized_seq, stats = normalize_rna_sequence(seq, warn_non_standard=False)
                                total_t_to_u += stats["t_to_u_count"]

                                if len(normalized_seq) >= 20:
                                    # 解析header获取taxid和RNA类型
                                    taxid, rna_type = self._parse_header(current_header)

                                    # 检查RNA类型（不过滤，将无效类型视为None）
                                    rna_token = self.processor._get_rna_type_token(rna_type)
                                    actual_rna_type = rna_type if rna_token is not None else None

                                    if actual_rna_type is None:
                                        filter_stats["no_rna_type"] += 1
                                        if len(sample_examples["no_rna_type"]) < 10:
                                            sample_examples["no_rna_type"].append({
                                                "header": current_header[:100],
                                                "taxid": taxid,
                                                "rna_type": rna_type,
                                                "seq_len": len(normalized_seq)
                                            })

                                    # 获取谱系信息（不过滤，将缺失视为None）
                                    lineage = None
                                    if self.processor.fixed_lineage:
                                        lineage = self.processor.fixed_lineage
                                    elif self.processor.lineage_mapper:
                                        lineage = self.processor.lineage_mapper.get_lineage(taxid)
                                        if not lineage:
                                            filter_stats["no_lineage"] += 1
                                            if len(sample_examples["no_lineage"]) < 10:
                                                sample_examples["no_lineage"].append({
                                                    "header": current_header[:100],
                                                    "taxid": taxid,
                                                    "rna_type": rna_type,
                                                    "seq_len": len(normalized_seq)
                                                })
                                    else:
                                        filter_stats["no_lineage"] += 1

                                    # 添加样本（允许lineage和actual_rna_type为None）
                                    samples.append((normalized_seq, lineage, actual_rna_type, taxid))
                                else:
                                    filter_stats["too_short"] += 1

                        current_seq = []
                        current_header = line

                    elif line:
                        current_seq.append(line)

                # 处理最后一个序列
                if current_seq and current_header:
                    seq = ''.join(current_seq)
                    filter_stats["total_processed"] += 1

                    if len(seq) < 20:
                        filter_stats["too_short"] += 1
                    else:
                        normalized_seq, stats = normalize_rna_sequence(seq, warn_non_standard=False)
                        total_t_to_u += stats["t_to_u_count"]

                        if len(normalized_seq) >= 20:
                            taxid, rna_type = self._parse_header(current_header)

                            # 检查RNA类型（不过滤，将无效类型视为None）
                            rna_token = self.processor._get_rna_type_token(rna_type)
                            actual_rna_type = rna_type if rna_token is not None else None

                            if actual_rna_type is None:
                                filter_stats["no_rna_type"] += 1
                                if len(sample_examples["no_rna_type"]) < 10:
                                    sample_examples["no_rna_type"].append({
                                        "header": current_header[:100],
                                        "taxid": taxid,
                                        "rna_type": rna_type,
                                        "seq_len": len(normalized_seq)
                                    })

                            # 获取谱系信息（不过滤，将缺失视为None）
                            lineage = None
                            if self.processor.fixed_lineage:
                                lineage = self.processor.fixed_lineage
                            elif self.processor.lineage_mapper:
                                lineage = self.processor.lineage_mapper.get_lineage(taxid)
                                if not lineage:
                                    filter_stats["no_lineage"] += 1
                                    if len(sample_examples["no_lineage"]) < 10:
                                        sample_examples["no_lineage"].append({
                                            "header": current_header[:100],
                                            "taxid": taxid,
                                            "rna_type": rna_type,
                                            "seq_len": len(normalized_seq)
                                        })
                            else:
                                filter_stats["no_lineage"] += 1

                            # 添加样本（允许lineage和actual_rna_type为None）
                            samples.append((normalized_seq, lineage, actual_rna_type, taxid))
                        else:
                            filter_stats["too_short"] += 1

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

        # 输出详细统计信息
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Lineage数据集加载统计:")
        logger.info(f"{'='*60}")
        logger.info(f"总处理序列数: {filter_stats['total_processed']:,}")
        logger.info(f"✅ 保留样本数: {len(samples):,}")
        logger.info(f"📋 样本分类:")
        logger.info(f"   - 没有谱系信息: {filter_stats['no_lineage']:,}")
        logger.info(f"   - 没有有效RNA类型: {filter_stats['no_rna_type']:,}")
        logger.info(f"❌ 过滤统计:")
        logger.info(f"   - 序列太短(<20): {filter_stats['too_short']:,}")

        if total_t_to_u > 0:
            logger.info(f"\n🧬 序列标准化: T→U转换 {total_t_to_u:,} 次")

        # 输出样本分类示例
        if sample_examples["no_lineage"]:
            logger.info(f"\n📋 没有谱系信息的样本示例 (前{len(sample_examples['no_lineage'])}个):")
            for i, sample in enumerate(sample_examples["no_lineage"], 1):
                logger.info(f"   {i}. taxid={sample['taxid']}, rna_type={sample['rna_type']}, len={sample['seq_len']}")
                logger.info(f"      header={sample['header']}")

        if sample_examples["no_rna_type"]:
            logger.info(f"\n📋 没有有效RNA类型的样本示例 (前{len(sample_examples['no_rna_type'])}个):")
            for i, sample in enumerate(sample_examples["no_rna_type"], 1):
                logger.info(f"   {i}. rna_type='{sample['rna_type']}', taxid={sample['taxid']}, len={sample['seq_len']}")

        logger.info(f"{'='*60}\n")

        # 限制样本数（用于调试或评估）
        if self.max_samples and len(samples) > self.max_samples:
            # ✅ 使用随机采样而非顺序采样，确保数据分布与训练集一致
            import random
            logger.info(f"  从 {len(samples)} 个样本中随机采样 {self.max_samples} 个")
            random.seed(42)  # 固定随机种子，确保评估结果可复现
            samples = random.sample(samples, self.max_samples)

        return samples

    def _parse_header(self, header: str) -> Tuple[str, str]:
        """
        解析FASTA header提取taxid和RNA类型

        支持格式:
        - >seq_id|taxid=12345|rna_type=mRNA
        - >seq_id|species_taxid=12345|rna_type=mRNA
        - >seq_id|rna_type:mrna|taxid:12345
        """
        taxid = "unknown"
        rna_type = "unknown"

        try:
            parts = header.strip('>').split('|')

            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key == 'taxid' or key == 'species_taxid':
                        taxid = value.strip()
                    elif key == 'rna_type':
                        rna_type = value.strip()
                elif ':' in part:
                    if part.startswith('taxid:') or part.startswith('species_taxid:'):
                        taxid = part.split(':')[1].strip()
                    elif part.startswith('rna_type:'):
                        rna_type = part.split('rna_type:')[1].strip()

        except Exception as e:
            logger.warning(f"解析header失败: {e}")

        return taxid, rna_type

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本，使用循环避免递归深度问题"""
        max_attempts = 10  # 最多尝试10次避免无限循环
        attempts = 0

        while attempts < max_attempts:
            sample_idx = (idx + attempts) % len(self.samples)
            sequence, lineage, rna_type, taxid = self.samples[sample_idx]

            # 根据配置决定是否反转序列（数据增强）
            if self.processor.enable_reverse_augmentation:
                reverse_sequence = random.random() < 0.5
            else:
                reverse_sequence = False  # 禁用反转（如legacy评估模式）

            # 根据模式处理样本
            if self.mode == "mixed":
                # Mixed模式：根据概率随机选择CLM或GLM
                if random.random() < self.glm_probability:
                    # GLM模式（多span补全）
                    sample = self.processor._process_completion_sample_multi_span(
                        sequence, lineage, rna_type, self.span_config, reverse_sequence=reverse_sequence
                    )
                    # Never silently change a sampled GLM objective into CLM.
                    if sample is None:
                        raise ValueError(f'GLM sample construction failed at row {idx}; adjust explicit span configuration')
                else:
                    # CLM模式（标准生成）
                    sample = self.processor.process_generation_sample(
                        sequence, lineage, rna_type, reverse_sequence=reverse_sequence
                    )
            elif self.mode == "generation":
                sample = self.processor.process_generation_sample(
                    sequence, lineage, rna_type, reverse_sequence=reverse_sequence
                )
            elif self.mode == "completion":
                sample = self.processor._process_completion_sample_multi_span(
                    sequence, lineage, rna_type, self.span_config, reverse_sequence=reverse_sequence
                )
            else:
                raise ValueError(f"未知模式: {self.mode}")

            # 如果处理成功（没有被过滤），返回结果
            if sample is not None:
                return sample

            raise ValueError(f'Sample construction failed at row {idx}; no replacement row was used')

        # 如果尝试多次仍被过滤，返回一个默认样本
        logger.warning(f"索引 {idx} 附近的样本均被过滤，返回默认样本")
        return {
            "input_ids": torch.tensor([self.tokenizer.token_to_id("<pad>")], dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
            "position_ids": torch.zeros(1, dtype=torch.long),
            "sequence_ids": torch.zeros(1, dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
            "task_type": "fallback",
            "sequence_length": 0,
        }


class ChunkedLineageRNADataset(Dataset):
    """
    内存高效的分chunk流式RNA数据集

    通过索引+按需加载的方式支持超大数据集（如239GB训练集）

    特性：
    - 索引构建：首次扫描FASTA文件记录每条序列的文件偏移量
    - 索引缓存：将索引保存为.index文件，后续训练直接加载
    - 按需加载：训练时根据索引动态读取序列，不预加载全部数据
    - LRU缓存：缓存最近访问的序列，平衡内存占用和读取性能
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: Any,
        lineage_file: str,
        max_seq_length: int = 2048,
        mode: str = "generation",
        span_config: Optional[SpanConfig] = None,
        glm_probability: float = 0.333,  # GLM采样概率（仅在mode='mixed'时生效）
        max_samples: Optional[int] = None,
        cache_size: int = 100000,  # LRU缓存大小
        force_rebuild_index: bool = False,  # 强制重建索引
        use_direction_tokens: bool = True,
        add_bos_token: bool = True,
        use_lineage_prefix: bool = True,
        use_rna_type_prefix: bool = True,  # 是否使用RNA类型前缀（默认True保持兼容）
        enable_reverse_augmentation: bool = True,
        fuzzy_factor: float = 0.2,  # 模糊因子
        deterministic: bool = False,  # 确定性模式
        pretrain_ratio: float = 0.0,  # pretraining任务比例（0.0-1.0，无前缀的纯序列）
        fixed_lineage: Optional[str] = None,  # 固定谱系字符串（用于微调特定物种）
    ):
        """
        初始化chunked数据集

        Args:
            data_file: FASTA数据文件路径
            tokenizer: RNA tokenizer
            lineage_file: lineage_greengenes.tsv文件路径
            max_seq_length: 最大序列长度
            mode: 训练模式 ("generation" 或 "completion")
            span_config: Span采样配置（仅completion模式需要）
            max_samples: 最大样本数（用于调试）
            cache_size: LRU缓存大小（缓存最近访问的序列数）
            force_rebuild_index: 是否强制重建索引
            use_direction_tokens: 是否使用5/3方向标记
            add_bos_token: 是否添加<bos> token（兼容旧模型）
            use_lineage_prefix: 是否使用谱系前缀（默认True）
            use_rna_type_prefix: 是否使用RNA类型前缀（默认True）
            enable_reverse_augmentation: 是否启用序列反转数据增强（默认True）
            fuzzy_factor: 模糊因子（训练时使用）
            deterministic: 确定性模式（评估时设为True）
            pretrain_ratio: pretraining任务比例（0.0-1.0，无前缀的纯序列）
            fixed_lineage: 固定谱系字符串（用于微调特定物种，如病毒）
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.lineage_file = lineage_file
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.span_config = span_config or SpanConfig()
        self.glm_probability = glm_probability
        self.max_samples = max_samples
        self.cache_size = cache_size
        self.use_direction_tokens = use_direction_tokens
        self.add_bos_token = add_bos_token
        self.use_lineage_prefix = use_lineage_prefix
        self.use_rna_type_prefix = use_rna_type_prefix
        self.fuzzy_factor = fuzzy_factor
        self.deterministic = deterministic
        self.pretrain_ratio = pretrain_ratio
        self.fixed_lineage = fixed_lineage

        # 初始化处理器
        self.processor = LineageRNASequenceProcessor(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            lineage_file=lineage_file,
            use_direction_tokens=use_direction_tokens,
            add_bos_token=add_bos_token,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
            fuzzy_factor=fuzzy_factor,
            deterministic=deterministic,
            pretrain_ratio=pretrain_ratio,
            fixed_lineage=fixed_lineage,
        )

        # 索引文件路径
        self.index_file = data_file + '.index'

        # 构建或加载索引
        self.index = self._load_or_build_index(force_rebuild_index)

        # 限制样本数（用于调试或评估）
        if max_samples and len(self.index) > max_samples:
            # ✅ 使用随机采样而非顺序采样，确保数据分布与训练集一致
            import random
            logger.info(f"  从 {len(self.index)} 个样本中随机采样 {max_samples} 个")
            random.seed(42)  # 固定随机种子，确保评估结果可复现
            sampled_indices = random.sample(range(len(self.index)), max_samples)
            self.index = [self.index[i] for i in sorted(sampled_indices)]  # 保持文件顺序以提高读取效率

        # LRU缓存（缓存已处理的样本）
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(f"ChunkedLineageRNADataset初始化完成:")
        logger.info(f"  - 模式: {mode}")
        logger.info(f"  - 样本数: {len(self.index)}")
        logger.info(f"  - 谱系文件: {lineage_file}")
        logger.info(f"  - LRU缓存大小: {cache_size}")

    def _load_or_build_index(self, force_rebuild: bool) -> List[Dict[str, Any]]:
        """
        加载或构建索引（支持分布式训练）

        索引格式：
        [
            {
                "offset": 文件偏移量,
                "seq_len": 序列长度,
                "taxid": 物种taxid,
                "rna_type": RNA类型
            },
            ...
        ]
        """
        import pickle
        import hashlib
        import torch.distributed as dist
        import time

        # 获取分布式信息
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        # 检查索引文件是否存在且有效
        if not force_rebuild and os.path.exists(self.index_file):
            try:
                if rank == 0:
                    logger.info(f"加载索引文件: {self.index_file}")
                
                with open(self.index_file, 'rb') as f:
                    index_data = pickle.load(f)

                # 验证索引有效性
                if (index_data.get('data_file') == self.data_file and
                    index_data.get('version') == '1.0'):

                    index = index_data['index']
                    if rank == 0:
                        logger.info(f"✅ 索引加载成功: {len(index):,} 条序列")

                        # 输出缓存统计
                        filter_stats = index_data.get('filter_stats', {})
                        if filter_stats:
                            logger.info(f"索引统计:")
                            logger.info(f"  - 总处理序列: {filter_stats.get('total_processed', 0):,}")
                            logger.info(f"  - 保留样本: {len(index):,}")
                            logger.info(f"  - 没有谱系信息: {filter_stats.get('no_lineage', 0):,}")
                            logger.info(f"  - 没有有效RNA类型: {filter_stats.get('no_rna_type', 0):,}")
                            logger.info(f"  - 序列太短(<20): {filter_stats.get('too_short', 0):,}")

                    return index
                else:
                    if rank == 0:
                        logger.warning("索引文件版本不匹配或数据文件已更改，重建索引")

            except Exception as e:
                if rank == 0:
                    logger.warning(f"加载索引失败: {e}，重建索引")

        # 分布式索引构建：只让 rank 0 构建，其他 rank 等待
        if is_distributed:
            if rank == 0:
                # Rank 0: 构建并保存索引
                logger.info(f"🔨 [Rank 0] 开始构建索引: {self.data_file}")
                index = self._build_index()
                self._save_index(index)
                logger.info(f"✅ [Rank 0] 索引构建完成")
            
            # 同步：确保 rank 0 完成索引构建
            if world_size > 1:
                dist.barrier()
            
            # 其他 rank: 等待并加载索引
            if rank != 0:
                max_retries = 10
                retry_delay = 2  # 秒
                
                for attempt in range(max_retries):
                    if os.path.exists(self.index_file):
                        try:
                            with open(self.index_file, 'rb') as f:
                                index_data = pickle.load(f)
                            index = index_data['index']
                            logger.info(f"✅ [Rank {rank}] 索引加载成功: {len(index):,} 条序列")
                            return index
                        except Exception as e:
                            logger.warning(f"[Rank {rank}] 加载索引失败 (尝试 {attempt+1}/{max_retries}): {e}")
                            time.sleep(retry_delay)
                    else:
                        logger.warning(f"[Rank {rank}] 等待索引文件创建 (尝试 {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                
                raise RuntimeError(f"[Rank {rank}] 无法加载索引文件: {self.index_file}")
            
            return index
        
        else:
            # 非分布式环境：直接构建
            logger.info(f"开始构建索引: {self.data_file}")
            index = self._build_index()
            self._save_index(index)
            return index
    
    def _save_index(self, index: List[Dict[str, Any]]):
        """保存索引到文件"""
        import pickle
        
        try:
            index_data = {
                'version': '1.0',
                'data_file': self.data_file,
                'index': index,
                'filter_stats': self._filter_stats,
            }

            # 原子性写入：先写临时文件，再重命名
            temp_index_file = self.index_file + '.tmp'
            with open(temp_index_file, 'wb') as f:
                # 使用 protocol=4 保证跨 Python 3.7+ 版本兼容
                pickle.dump(index_data, f, protocol=4)
            
            # 原子性重命名（避免多进程同时写入导致损坏）
            os.replace(temp_index_file, self.index_file)
            
            logger.info(f"✅ 索引已保存: {self.index_file} (pickle protocol=4)")
        except Exception as e:
            logger.warning(f"保存索引失败: {e}")

        return index

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        构建索引：扫描FASTA文件记录每条序列的偏移量
        支持多进程并行加速

        Returns:
            索引列表
        """
        import multiprocessing as mp
        from functools import partial

        # 获取CPU核心数
        num_workers = min(mp.cpu_count(), 144)  # 最多使用144个核心

        logger.info(f"开始并行扫描FASTA文件构建索引（{num_workers}个进程）...")

        # 第一步：快速扫描找到所有序列的起始偏移量
        logger.info("步骤1: 快速扫描定位所有序列...")
        sequence_offsets = self._find_all_sequence_offsets()
        logger.info(f"找到 {len(sequence_offsets):,} 条序列")

        # 第二步：并行处理序列验证和过滤
        logger.info(f"步骤2: 并行验证序列（{num_workers}个进程）...")

        # 将序列分块
        chunk_size = max(1, len(sequence_offsets) // num_workers)
        chunks = [sequence_offsets[i:i+chunk_size]
                  for i in range(0, len(sequence_offsets), chunk_size)]

        # 并行处理（带进度显示）
        with mp.Pool(processes=num_workers) as pool:
            worker_func = partial(
                _process_sequence_chunk,
                data_file=self.data_file,
                lineage_mapper=self.processor.lineage_mapper,
                processor=self.processor
            )

            # 使用imap_unordered获取实时进度
            from tqdm import tqdm
            results = []
            with tqdm(total=len(chunks), desc="处理进度", unit="chunk") as pbar:
                for result in pool.imap_unordered(worker_func, chunks):
                    results.append(result)
                    pbar.update(1)
                    # 显示已处理的序列数
                    total_processed = sum(r[1]['total_processed'] for r in results)
                    total_valid = sum(len(r[0]) for r in results)
                    pbar.set_postfix({
                        '已处理': f'{total_processed:,}',
                        '有效': f'{total_valid:,}'
                    })

        # 合并结果
        index = []
        filter_stats = {
            "no_lineage": 0,
            "no_rna_type": 0,
            "too_short": 0,
            "total_processed": 0,
        }

        for chunk_index, chunk_stats in results:
            index.extend(chunk_index)
            for key in filter_stats:
                filter_stats[key] += chunk_stats[key]

        self._filter_stats = filter_stats

        # 输出统计信息
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 索引构建完成:")
        logger.info(f"{'='*60}")
        logger.info(f"总处理序列数: {self._filter_stats['total_processed']:,}")
        logger.info(f"✅ 保留样本数: {len(index):,}")
        logger.info(f"📋 样本分类:")
        logger.info(f"   - 没有谱系信息: {self._filter_stats['no_lineage']:,}")
        logger.info(f"   - 没有有效RNA类型: {self._filter_stats['no_rna_type']:,}")
        logger.info(f"❌ 过滤统计:")
        logger.info(f"   - 序列太短(<20): {self._filter_stats['too_short']:,}")
        logger.info(f"{'='*60}\n")

        return index

    def _find_all_sequence_offsets(self) -> List[int]:
        """快速扫描找到所有'>'开头的行的偏移量"""
        import os
        from tqdm import tqdm

        offsets = []

        # 获取文件大小用于进度显示
        file_size = os.path.getsize(self.data_file)

        with open(self.data_file, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="扫描序列", unit='B', unit_scale=True) as pbar:
                last_pos = 0
                while True:
                    pos = f.tell()
                    line = f.readline()

                    if not line:
                        break

                    if line.startswith('>'):
                        offsets.append(pos)

                    # 更新进度
                    pbar.update(pos - last_pos)
                    last_pos = pos

        return offsets

    def _process_sequence_for_index(
        self,
        header: str,
        seq_lines: List[str],
        offset: int,
        index: List[Dict[str, Any]]
    ):
        """
        处理单个序列用于索引构建

        Args:
            header: FASTA header
            seq_lines: 序列行列表
            offset: 文件偏移量
            index: 索引列表（原地修改）
        """
        self._filter_stats["total_processed"] += 1

        # 计算序列长度
        seq = ''.join(seq_lines)
        seq_len = len(seq)

        # 过滤太短的序列（这个仍然要过滤）
        if seq_len < 20:
            self._filter_stats["too_short"] += 1
            return

        # 解析header
        taxid, rna_type = self._parse_header(header)

        # 检查RNA类型（不过滤，只统计）
        rna_token = self.processor._get_rna_type_token(rna_type)
        if rna_token is None:
            self._filter_stats["no_rna_type"] += 1

        # 检查是否有谱系信息（不过滤，只统计）
        has_lineage = False
        if self.processor.fixed_lineage:
            lineage = self.processor.fixed_lineage
            has_lineage = True
        elif self.processor.lineage_mapper:
            lineage = self.processor.lineage_mapper.get_lineage(taxid)
            has_lineage = lineage is not None

        if not has_lineage:
            self._filter_stats["no_lineage"] += 1

        # 添加到索引（保留所有样本，包括缺少谱系或RNA类型的）
        index.append({
            "offset": offset,
            "seq_len": seq_len,
            "taxid": taxid,
            "rna_type": rna_type,
        })

    def _parse_header(self, header: str) -> Tuple[str, str]:
        """
        解析FASTA header提取taxid和RNA类型

        支持格式:
        - >seq_id|taxid=12345|rna_type=mRNA
        - >seq_id|species_taxid=12345|rna_type=mRNA
        """
        taxid = "unknown"
        rna_type = "unknown"

        try:
            parts = header.strip('>').split('|')

            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key == 'taxid' or key == 'species_taxid':
                        taxid = value.strip()
                    elif key == 'rna_type':
                        rna_type = value.strip()

        except Exception as e:
            logger.warning(f"解析header失败: {e}")

        return taxid, rna_type

    def _read_sequence_at_offset(self, offset: int) -> Tuple[str, str, str]:
        """
        从指定偏移量读取序列

        Args:
            offset: 文件偏移量

        Returns:
            (sequence, taxid, rna_type)
        """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            f.seek(offset)

            # 读取header
            header = f.readline().strip()

            # 读取序列（直到下一个header或文件末尾）
            seq_lines = []
            while True:
                pos = f.tell()
                line = f.readline()

                if not line or line.startswith('>'):
                    break

                seq_lines.append(line.strip())

            sequence = ''.join(seq_lines)
            taxid, rna_type = self._parse_header(header)

            return sequence, taxid, rna_type

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本（带LRU缓存）

        Args:
            idx: 样本索引

        Returns:
            处理后的样本字典
        """
        # 检查缓存
        if idx in self._cache:
            self._cache_hits += 1
            # 将访问的项移到末尾（LRU）
            self._cache.move_to_end(idx)
            return self._cache[idx]

        self._cache_misses += 1

        # 从磁盘读取
        index_entry = self.index[idx]
        offset = index_entry['offset']
        taxid = index_entry['taxid']
        rna_type = index_entry['rna_type']

        # 读取序列
        sequence, _, _ = self._read_sequence_at_offset(offset)

        # 标准化序列（使用文件顶部定义的函数）
        normalized_seq, _ = normalize_rna_sequence(sequence, warn_non_standard=False)

        # 获取谱系信息（允许为None）
        lineage = None
        if self.processor.fixed_lineage:
            lineage = self.processor.fixed_lineage
        elif self.processor.lineage_mapper:
            lineage = self.processor.lineage_mapper.get_lineage(taxid)

        # 检查RNA类型（将无效类型视为None）
        rna_token = self.processor._get_rna_type_token(rna_type)
        actual_rna_type = rna_type if rna_token is not None else None

        # 根据配置决定是否反转序列（数据增强）
        if self.processor.enable_reverse_augmentation:
            reverse_sequence = random.random() < 0.5
        else:
            reverse_sequence = False  # 禁用反转（如legacy评估模式）

        # 根据模式处理样本（允许lineage和actual_rna_type为None）
        if self.mode == "mixed":
            # Mixed模式：根据概率随机选择CLM或GLM
            if random.random() < self.glm_probability:
                # GLM模式（多span补全）
                sample = self.processor._process_completion_sample_multi_span(
                    normalized_seq, lineage, actual_rna_type, self.span_config, reverse_sequence=reverse_sequence
                )
                # Never silently change a sampled GLM objective into CLM.
                if sample is None:
                    raise ValueError(f'GLM sample construction failed at row {idx}; adjust explicit span configuration')
            else:
                # CLM模式（标准生成）
                sample = self.processor.process_generation_sample(
                    normalized_seq, lineage, actual_rna_type, reverse_sequence=reverse_sequence
                )
        elif self.mode == "generation":
            sample = self.processor.process_generation_sample(
                normalized_seq, lineage, actual_rna_type, reverse_sequence=reverse_sequence
            )
        elif self.mode == "completion":
            sample = self.processor._process_completion_sample_multi_span(
                normalized_seq, lineage, actual_rna_type, self.span_config, reverse_sequence=reverse_sequence
            )
        else:
            raise ValueError(f"未知模式: {self.mode}")

        # 如果处理失败，返回fallback
        if sample is None:
            raise ValueError(f'Sample construction failed at row {idx}; no padding fallback was used')

        # 添加到缓存
        self._cache[idx] = sample

        # 维护缓存大小（LRU淘汰）
        if len(self._cache) > self.cache_size:
            # 移除最旧的项（FIFO，位于开头）
            self._cache.popitem(last=False)

        return sample

    def _get_fallback_sample(self) -> Dict[str, Any]:
        """返回fallback样本（用于错误恢复）"""
        return {
            "input_ids": torch.tensor([self.tokenizer.token_to_id("<pad>")], dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
            "position_ids": torch.zeros(1, dtype=torch.long),
            "sequence_ids": torch.zeros(1, dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
            "task_type": "fallback",
            "sequence_length": 0,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_accesses = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_accesses": total_accesses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
        }


def create_lineage_dataset(
    data_file: str,
    tokenizer: Any,
    lineage_file: str,
    mode: str = "generation",
    max_seq_length: int = 2048,
    span_config: Optional[SpanConfig] = None,
    glm_probability: float = 0.333,  # GLM采样概率（仅在mode='mixed'时生效）
    max_samples: Optional[int] = None,
    use_chunked: bool = True,  # 是否使用chunked模式
    cache_size: int = 10000,  # LRU缓存大小
    force_rebuild_index: bool = False,  # 是否强制重建索引
    use_direction_tokens: bool = True,  # 是否使用5/3方向标记
    add_bos_token: bool = True,  # 是否添加<bos> token（默认True保持兼容）
    use_lineage_prefix: bool = True,  # 是否使用谱系前缀（默认True保持兼容）
    use_rna_type_prefix: bool = True,  # 是否使用RNA类型前缀（默认True保持兼容）
    enable_reverse_augmentation: bool = True,  # 是否启用序列反转数据增强（默认True）
    fuzzy_factor: float = 0.2,  # 模糊因子（默认0.2）
    deterministic: bool = False,  # 是否使用确定性模式（评估时关闭随机性）
    pretrain_ratio: float = 0.0,  # pretraining任务比例（0.0-1.0，无前缀的纯序列）
    fixed_lineage: Optional[str] = None,  # 固定谱系字符串（用于微调特定物种）
) -> Dataset:
    """
    创建基于谱系的RNA数据集

    Args:
        data_file: FASTA数据文件
        tokenizer: RNA tokenizer
        lineage_file: 谱系映射文件
        mode: 训练模式 ("generation" 或 "completion")
        max_seq_length: 最大序列长度
        span_config: Span配置（completion模式）
        max_samples: 最大样本数
        use_chunked: 是否使用内存高效的chunked模式（推荐用于大数据集）
        cache_size: LRU缓存大小（仅chunked模式）
        force_rebuild_index: 是否强制重建索引（仅chunked模式）
        use_direction_tokens: 是否使用5/3方向标记
        add_bos_token: 是否添加<bos> token（兼容旧模型）
        use_lineage_prefix: 是否使用谱系前缀（默认True）
        use_rna_type_prefix: 是否使用RNA类型前缀（默认True）
        enable_reverse_augmentation: 是否启用序列反转数据增强（默认True）
        fuzzy_factor: 模糊因子（训练时使用，用于增强泛化）
        deterministic: 是否使用确定性模式（评估时设为True，关闭随机性确保可重现）
        pretrain_ratio: pretraining任务比例（0.0-1.0，无前缀的纯序列）

    Returns:
        LineageRNADataset 或 ChunkedLineageRNADataset 实例
    """
    if use_chunked:
        logger.info("使用ChunkedLineageRNADataset（内存高效模式）")
        return ChunkedLineageRNADataset(
            data_file=data_file,
            tokenizer=tokenizer,
            lineage_file=lineage_file,
            max_seq_length=max_seq_length,
            mode=mode,
            span_config=span_config,
            glm_probability=glm_probability,
            max_samples=max_samples,
            cache_size=cache_size,
            force_rebuild_index=force_rebuild_index,
            use_direction_tokens=use_direction_tokens,
            add_bos_token=add_bos_token,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
            fuzzy_factor=fuzzy_factor,
            deterministic=deterministic,
            pretrain_ratio=pretrain_ratio,
            fixed_lineage=fixed_lineage,
        )
    else:
        logger.info("使用LineageRNADataset（原始全量加载模式）")
        return LineageRNADataset(
            data_file=data_file,
            tokenizer=tokenizer,
            lineage_file=lineage_file,
            max_seq_length=max_seq_length,
            mode=mode,
            span_config=span_config,
            glm_probability=glm_probability,
            max_samples=max_samples,
            use_direction_tokens=use_direction_tokens,
            add_bos_token=add_bos_token,
            use_lineage_prefix=use_lineage_prefix,
            use_rna_type_prefix=use_rna_type_prefix,
            enable_reverse_augmentation=enable_reverse_augmentation,
            fuzzy_factor=fuzzy_factor,
            deterministic=deterministic,
            pretrain_ratio=pretrain_ratio,
            fixed_lineage=fixed_lineage,
        )
