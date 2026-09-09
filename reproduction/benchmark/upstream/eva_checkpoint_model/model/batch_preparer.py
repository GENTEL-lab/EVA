"""
RNA批处理准备器
统一的批处理准备器，支持CLM和GLM格式，专门优化RNA序列处理
"""

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from utils import distributed as dist
from model.tokenizer import get_rna_tokenizer, END_OF_SPAN_TOKEN

# 通用序列模式（用于兼容性）
CLM_PATTERN = re.compile(r"^[A-Z]+$")
GLM_PATTERN = re.compile(r"^[A-Z]+\[GLM\](?:\d+\-\d+\-\d+;)*\d+\-\d+\-\d+;?$")

# RNA序列模式
RNA_CLM_PATTERN = re.compile(r"^[AUGC]+$")
RNA_GLM_PATTERN = re.compile(r"^[AUGC]+\[GLM\](?:\d+\-\d+\-\d+;)*\d+\-\d+\-\d+;?$")


@dataclass
class DataPrepConfig:
    """数据准备配置"""
    fuzzy_span_len_factor: float = 0.2
    max_glm_spans: int = 50


class RNAGenBatchPreparer:
    """
    RNA序列批处理准备器
    融合了基础功能和RNA特化功能，支持CLM和GLM格式
    
    注意：当前实现假设所有序列都是1->2方向，传入反向序列会导致编码错误
    """
    
    def __init__(
        self,
        data_prep_config: Optional[DataPrepConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """初始化批处理准备器"""
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.rng = rng or np.random.default_rng(0)
        
        # 使用RNA专用tokenizer
        self.tokenizer = get_rna_tokenizer()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        
        if self.pad_token_id is None:
            raise ValueError("RNA tokenizer必须包含<pad>标记")
    
    # ==================== 基础批处理功能 ====================
    
    def get_batch_kwargs(
        self,
        sequences: list[str],
        device: torch.device = torch.device("cpu"),
        reverse: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        准备批量序列用于模型输入
        
        NOTE: 此函数假设所有序列都是1->2方向
        传入反向序列会导致编码错误
        """
        sequence_encodings = [self.prepare_singleseq(sequence, reverse) for sequence in sequences]
        padded_encodings = self.pad_encodings(sequence_encodings)
        padded_encodings = {k: v.to(device=device, non_blocking=True) for k, v in padded_encodings.items()}
        
        return padded_encodings
    
    def pad_encodings(self, sequence_encodings: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        对编码序列进行padding

        注意：labels 使用 -100 作为 padding 值，这是 PyTorch CrossEntropyLoss 的标准 ignore_index
        这样可以确保 padding 位置不会被计入损失计算
        """
        padding_value = {
            "input_ids": self.pad_token_id,  # 0 - 用于标识 padding 位置
            "labels": -100,                   # -100 - PyTorch 标准的 ignore_index
            "position_ids": 0,
            "sequence_ids": 0,
        }
        padded_batch = {}
        for key, padding_value in padding_value.items():
            padded_batch[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings],
                batch_first=True,
                padding_value=padding_value,
            ).to(dtype=torch.long)

        return padded_batch
    
    def get_generation_kwargs(self, sequence: str, reverse_sequences: bool) -> dict[str, torch.Tensor]:
        """
        准备生成任务的输入
        
        NOTE: 此函数假设序列和上下文都是1->2方向
        传入反向序列会导致编码错误
        """
        single_seq_encoding = self.prepare_singleseq(sequence, reverse_sequences)
        prefix_length = single_seq_encoding["metadata"]["prefix_length"]
        
        input_ids = single_seq_encoding["input_ids"][:prefix_length]
        sequence_ids = single_seq_encoding["sequence_ids"][:prefix_length]
        position_ids = single_seq_encoding["position_ids"][:prefix_length]
        
        return {
            "input_ids": input_ids.unsqueeze(0).to(dist.get_device()),
            "sequence_ids": sequence_ids.unsqueeze(0).to(dist.get_device()),
            "position_ids": position_ids.unsqueeze(0).to(dist.get_device()),
        }
    
    def prepare_singleseq(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        """
        准备单个序列
        自动识别CLM或GLM格式
        
        NOTE: 此函数假设序列是1->2方向
        """
        # 优先检查RNA GLM格式
        if self.is_rna_glm_instance(sequence):
            return self.prepare_glm(sequence, reverse_sequence)
        # 然后检查通用GLM格式（兼容性）
        elif is_glm_instance(sequence):
            return self.prepare_glm(sequence, reverse_sequence)
        else:
            return self.prepare_clm(sequence, reverse_sequence)
    
    # ==================== CLM格式处理 ====================
    
    def prepare_clm(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        """
        准备CLM格式的序列
        格式：1 + 序列 + 2
        """
        # 添加方向标记
        sequence = "1" + sequence + "2"
        
        if reverse_sequence:
            sequence = sequence[::-1]
        
        # 使用tokenizer编码
        tokens = self.tokenizer.encode(f"<bos>{sequence}<eos>").ids
        
        return {
            "input_ids": torch.tensor(tokens),
            "labels": torch.tensor(tokens),
            "position_ids": torch.arange(len(tokens)),
            "sequence_ids": torch.zeros(len(tokens)),
            # 生成时移除<1/2><eos>
            "metadata": {"prefix_length": len(tokens) - 2},
        }
    
    # ==================== GLM格式处理 ====================
    
    def prepare_glm(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        """准备GLM格式的序列，支持span masking"""
        sequence, masking_info = get_spans_to_mask(sequence)
        spans_to_mask = sorted(masking_info.keys())
        remaining_spans = get_remaining_spans_from_infill_spans(spans_to_mask, len(sequence))
        tokens = list(sequence)
        
        if reverse_sequence:
            tokens = tokens[::-1]
            spans_to_mask = [(len(tokens) - e, len(tokens) - s) for s, e in spans_to_mask]
            remaining_spans = [(len(tokens) - e, len(tokens) - s) for s, e in remaining_spans]
            masking_info = {(len(tokens) - e, len(tokens) - s): L for (s, e), L in masking_info.items()}
        
        infill_span_ids = self.rng.choice(self.data_prep_config.max_glm_spans, len(spans_to_mask))
        infill_span_ids = [f"<span_{i}>" for i in infill_span_ids]
        
        all_spans = sorted(
            [(*x, True, infill_span_ids[i]) for i, x in enumerate(spans_to_mask)]
            + [(*x, False, "") for x in remaining_spans]
        )
        
        prefix_tokens, suffix_tokens = [], []
        prefix_pos_ids, suffix_pos_ids = [], []
        
        pos_id_start = 0 + 2  # 0 for <bos>, 1 for <term_1>
        for s, e, is_infill_span, span_id in all_spans:
            if is_infill_span:
                # 如果是infill span，添加到suffix并在prefix中用span_id替换
                span_suffix_tokens = [span_id] + tokens[s:e] + [END_OF_SPAN_TOKEN]
                suffix_tokens.extend(span_suffix_tokens)
                suffix_pos_ids.extend(list(range(pos_id_start, pos_id_start + len(span_suffix_tokens))))
                
                prefix_tokens.append(span_id)
                prefix_pos_ids.append(pos_id_start)
                pos_id_start += 1
                
                # infill长度L可能与span长度不同（例如用于miniaturization）
                L = masking_info[(s, e)]
                fuzzy_diff = np.floor(L * self.data_prep_config.fuzzy_span_len_factor)
                pos_id_start += L + int(fuzzy_diff)
            else:
                # 如果是remaining span，添加到prefix
                prefix = tokens[s:e]
                prefix_tokens.extend(deepcopy(prefix))
                prefix_pos_ids.extend(list(range(pos_id_start, pos_id_start + len(prefix))))
                pos_id_start += len(prefix)
        
        term_1, term_2 = ("1", "2") if not reverse_sequence else ("2", "1")
        
        prefix_tokens = ["<bos_glm>", term_1] + prefix_tokens + [term_2, "<eos>"]
        prefix_labels = ["<pad>"] * len(prefix_tokens)
        prefix_pos_ids = [0, 1] + prefix_pos_ids + [pos_id_start, pos_id_start + 1]
        
        input_tokens = prefix_tokens + suffix_tokens
        labels = prefix_labels + [x if not x.startswith("<span_") else "<pad>" for x in suffix_tokens]
        pos_ids = prefix_pos_ids + suffix_pos_ids
        
        input_tokens_ids = self.tokenizer.encode("".join(input_tokens)).ids
        labels_ids = self.tokenizer.encode("".join(labels)).ids
        
        return {
            "input_ids": torch.tensor(input_tokens_ids),
            "labels": torch.tensor(labels_ids),
            "position_ids": torch.tensor(pos_ids),
            "sequence_ids": torch.zeros(len(input_tokens_ids)),
            # +1 for first span_id in suffix which we want to keep
            "metadata": {"prefix_length": len(prefix_pos_ids) + 1},
        }
    
    # ==================== RNA特化功能 ====================
    
    def parse_fasta_header(self, description: str) -> dict[str, Any]:
        """
        解析FASTA header中的信息
        格式：>seq_id|rna_type:mrna|species:ailuropoda_melanoleuca|L=长度
        """
        metadata = {}
        
        # 解析L=长度信息
        length_match = re.search(r'L=(\d+)', description)
        if length_match:
            metadata['target_length'] = int(length_match.group(1))
        
        # 解析RNA类型
        if 'rna_type:' in description:
            try:
                rna_part = description.split('rna_type:')[1].split('|')[0]
                metadata['rna_type'] = rna_part.strip()
            except:
                metadata['rna_type'] = 'mrna'  # 默认值
        else:
            metadata['rna_type'] = 'mrna'
        
        # 解析物种
        if 'species:' in description:
            try:
                species_part = description.split('species:')[1].split('|')[0]
                metadata['species'] = species_part.strip()
            except:
                metadata['species'] = 'unknown'  # 默认值
        else:
            metadata['species'] = 'unknown'
        
        return metadata
    
    def prepare_rna_sequence(self, sequence: str, target_length: Optional[int] = None) -> str:
        """
        准备RNA序列用于模型输入
        支持长度控制前缀
        """
        # 验证RNA序列
        if not RNA_CLM_PATTERN.match(sequence):
            raise ValueError(f"无效的RNA序列: {sequence}")
        
        # 如果指定了目标长度，可以在此处添加长度控制逻辑
        # 当前版本直接返回序列，长度控制留待后续实现
        return sequence
    
    def is_rna_glm_instance(self, sequence: str) -> bool:
        """检查是否为RNA GLM实例"""
        return RNA_GLM_PATTERN.match(sequence) is not None
    
    def prepare_from_fasta_record(self, record: SeqRecord, reverse_sequence: bool = False) -> dict[str, Any]:
        """
        从FASTA记录准备训练数据
        解析header中的长度信息
        """
        sequence = str(record.seq).upper()
        
        # 解析header中的元数据
        metadata = self.parse_fasta_header(record.description)
        
        # 准备序列
        prepared_sequence = self.prepare_rna_sequence(sequence, metadata.get('target_length'))
        
        # 生成训练数据
        result = self.prepare_singleseq(prepared_sequence, reverse_sequence)
        
        # 添加原始元数据
        result["metadata"].update({
            "sequence_id": record.id,
            "original_length": len(sequence),
            **metadata
        })
        
        return result
    
    def get_batch_kwargs_from_fasta(
        self, 
        fasta_records: list[SeqRecord], 
        device: torch.device = torch.device("cpu"),
        reverse: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        从FASTA记录批量准备训练数据
        """
        sequence_encodings = []
        
        for record in fasta_records:
            try:
                encoding = self.prepare_from_fasta_record(record, reverse)
                sequence_encodings.append(encoding)
            except ValueError as e:
                print(f"跳过无效序列 {record.id}: {e}")
                continue
        
        if not sequence_encodings:
            raise ValueError("没有有效的RNA序列")
        
        # 使用padding方法
        padded_encodings = self.pad_encodings(sequence_encodings)
        padded_encodings = {k: v.to(device=device, non_blocking=True) for k, v in padded_encodings.items()}
        
        return padded_encodings
    
    # ==================== 验证功能 ====================
    
    def assert_valid_instance(self, sequence: str) -> None:
        """验证序列格式（通用或RNA）"""
        if len(sequence) == 0:
            return
        
        # 检查RNA格式
        if RNA_CLM_PATTERN.match(sequence) or RNA_GLM_PATTERN.match(sequence):
            return
        
        # 检查通用格式（兼容性）
        if CLM_PATTERN.match(sequence) or GLM_PATTERN.match(sequence):
            return
        
        raise ValueError(f"序列不是有效的CLM或GLM格式: {sequence}")
    
    def assert_valid_rna_instance(self, sequence: str) -> None:
        """验证RNA序列格式"""
        if len(sequence) == 0:
            return
        
        if not RNA_CLM_PATTERN.match(sequence) and not RNA_GLM_PATTERN.match(sequence):
            raise ValueError(f"无效的RNA序列格式: {sequence}")


# ==================== 辅助函数 ====================

def assert_valid_instance(sequence: str) -> None:
    """验证序列实例（模块级函数，用于兼容性）"""
    if len(sequence) == 0:
        return
    
    if not CLM_PATTERN.match(sequence) and not GLM_PATTERN.match(sequence):
        raise ValueError(f"Sequence is not a valid CLM or GLM instance: {sequence}")


def is_glm_instance(sequence: str) -> bool:
    """检查是否为GLM实例（模块级函数，用于兼容性）"""
    return GLM_PATTERN.match(sequence) is not None


def get_spans_to_mask(sequence: str) -> tuple[str, dict[tuple[int, int], int]]:
    """从GLM格式字符串中提取span信息"""
    spans = {}
    sequence, spans_str = sequence.split("[GLM]")
    spans_str = spans_str.strip(";")
    for span in spans_str.split(";"):
        s, e, length = span.split("-")
        spans[(int(s), int(e))] = int(length)
    return sequence, spans


def prepare_glm_string_from_spans(spans: dict[tuple[int, int], int]) -> str:
    """从span字典创建GLM格式字符串"""
    return "[GLM]" + ";".join(f"{s}-{e}-{v}" for (s, e), v in spans.items())


def get_remaining_spans_from_infill_spans(
    infill_spans: list[tuple[int, int]], num_tokens: int
) -> list[tuple[int, int]]:
    """从infill spans计算remaining spans"""
    remaining_spans = []
    start = 0
    for s, e in infill_spans:
        assert s >= 0 and e <= num_tokens, f"Span {s}-{e} is invalid for sequence of length {num_tokens}."
        if start < s:
            remaining_spans.append((start, s))
        start = e
    if start < num_tokens:
        remaining_spans.append((start, num_tokens))
    
    return remaining_spans


# ==================== 数据集处理器 ====================

class RNADatasetProcessor:
    """RNA数据集处理器"""
    
    def __init__(self, batch_preparer: RNAGenBatchPreparer):
        self.batch_preparer = batch_preparer
    
    def load_fasta_file(self, fasta_path: str, max_sequences: Optional[int] = None) -> list[SeqRecord]:
        """加载FASTA文件"""
        sequences = []
        
        with open(fasta_path, 'r') as f:
            for i, record in enumerate(SeqIO.parse(f, 'fasta')):
                if max_sequences and i >= max_sequences:
                    break
                sequences.append(record)
        
        print(f"从 {fasta_path} 加载了 {len(sequences)} 条RNA序列")
        return sequences
    
    def create_training_batches(
        self, 
        fasta_path: str, 
        batch_size: int,
        max_sequences: Optional[int] = None,
        device: torch.device = torch.device("cpu")
    ):
        """创建训练批次"""
        sequences = self.load_fasta_file(fasta_path, max_sequences)
        
        # 分批处理
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            try:
                batch_data = self.batch_preparer.get_batch_kwargs_from_fasta(
                    batch_sequences, device=device
                )
                yield batch_data, batch_sequences
            except ValueError as e:
                print(f"跳过批次 {i//batch_size + 1}: {e}")
                continue


if __name__ == "__main__":
    # 测试RNA批处理准备器
    print("=== RNA批处理准备器测试 ===")
    
    # 创建批处理准备器
    batch_preparer = RNAGenBatchPreparer()
    
    # 测试单个序列
    test_sequence = "AUGCUAGCUAGCUAGC"
    result = batch_preparer.prepare_clm(test_sequence, reverse_sequence=False)
    
    print(f"测试序列: {test_sequence}")
    print(f"编码后的tokens: {result['input_ids']}")
    print(f"解码验证: {batch_preparer.tokenizer.decode(result['input_ids'].tolist())}")
    
    # 测试FASTA header解析
    test_header = "rna_00000001 L=156"
    metadata = batch_preparer.parse_fasta_header(test_header)
    print(f"Header解析: {test_header} -> {metadata}")
    
    print("测试完成！")