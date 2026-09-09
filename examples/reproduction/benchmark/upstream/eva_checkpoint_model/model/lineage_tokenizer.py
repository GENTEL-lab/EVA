"""
Lineage-based RNA专用tokenizer
支持Greengenes谱系字符串编码，词汇表精简无物种token
"""

import json
import os
import re
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE

# RNA特殊标记
RNA_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<bos_glm>",
    "<eos_span>",
    "<unk>",
]

# GLM span标记（保留完整50个，支持未来多span训练）
GLM_SPAN_TOKENS = [f"<span_{i}>" for i in range(50)]

# RNA类型token（15种明确功能的RNA类型）
RNA_TYPE_TOKENS = [
    "<rna_mRNA>",        # 信使RNA
    "<rna_rRNA>",        # 核糖体RNA
    "<rna_tRNA>",        # 转运RNA
    "<rna_sRNA>",        # 小RNA
    "<rna_lncRNA>",      # 长链非编码RNA
    "<rna_circRNA>",     # 环状RNA
    "<rna_viral_RNA>",   # 病毒RNA
    "<rna_miRNA>",       # 微小RNA
    "<rna_snoRNA>",      # 小核仁RNA
    "<rna_snRNA>",       # 小核RNA
    "<rna_piRNA>",       # PIWI结合RNA
    "<rna_ribozyme>",    # 核酶
    "<rna_scaRNA>",      # 小Cajal体RNA
    "<rna_Y_RNA>",       # Y RNA
    "<rna_vault_RNA>",   # Vault RNA
]

# Greengenes谱系层级前缀标记（小写，避免与RNA碱基大写AUCG混淆）
LINEAGE_LEVEL_TOKENS = [
    "d__",  # Domain (域)
    "p__",  # Phylum (门)
    "c__",  # Class (纲)
    "o__",  # Order (目)
    "f__",  # Family (科)
    "g__",  # Genus (属)
    "s__",  # Species (种)
]

# 谱系特殊字符（数据预处理后需要的字符）
# 说明：谱系字符串经过clean_lineage()预处理，移除了括号标注、方括号、单引号、斜杠、点号，空格替换为下划线
# 因此只需保留核心分隔符和连字符
LINEAGE_SPECIAL_CHARS = [
    ";",  # 层级分隔符（必需）
    "|",  # 谱系前缀边界符（必需）
    "_",  # 下划线（必需，用于D__等层级前缀，以及空格替换后的物种名）
    "-",  # 连字符（常见于物种名）
]

# RNA碱基（作为单独的token，优先定义，仅用于RNA序列）
RNA_BASES = ["A", "U", "G", "C"]

# 序列方向标记（用于标识RNA序列的5'和3'端）
DIRECTION_TOKENS = ["5", "3"]

# 字母字符 - 谱系全部小写，只需小写字母（大写AUCG已在RNA_BASES中）
ALPHANUMERIC_CHARS = [chr(i) for i in range(ord('a'), ord('z') + 1)]  # a-z (26个)

# 完整词汇表
def build_lineage_rna_vocab():
    """构建Lineage RNA词汇表（无物种token，无任务token）"""
    vocab_list = (
        RNA_SPECIAL_TOKENS +
        GLM_SPAN_TOKENS +
        RNA_TYPE_TOKENS +
        LINEAGE_LEVEL_TOKENS +
        LINEAGE_SPECIAL_CHARS +
        RNA_BASES +
        DIRECTION_TOKENS +      # 序列方向标记（5'和3'端）
        ALPHANUMERIC_CHARS      # 字母字符放在最后
    )
    return vocab_list

LINEAGE_RNA_VOCAB = build_lineage_rna_vocab()

END_OF_SPAN_TOKEN = "<eos_span>"
PAD_TOKEN_ID = 0


class LineageRNATokenizer:
    """Lineage-based RNA序列专用tokenizer"""

    def __init__(self):
        self.tokenizer = self._create_tokenizer()

    def _create_tokenizer(self) -> Tokenizer:
        """创建Lineage RNA tokenizer

        注意：使用BPE模型但不提供merges规则，实现字符级tokenization。
        由于我们使用自定义的encode()方法逐字符匹配token（而非BPE的标准tokenize流程），
        需要将所有token声明为special，这样：
        1. BPE模型不会尝试合并字符（因为special tokens不参与合并）
        2. 避免保存时出现"vocab contains holes"警告
        3. 与我们的逐字符编码逻辑一致：每个字符都应被"特殊处理"（保持独立）
        """
        # 构建词汇表
        vocab = {token: idx for idx, token in enumerate(LINEAGE_RNA_VOCAB)}
        merges = []  # 不提供合并规则，强制字符级tokenization

        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))

        # 将所有token声明为special，确保它们不会被BPE尝试合并
        # 这在我们的字符级tokenization场景下是合理的
        all_special_tokens = LINEAGE_RNA_VOCAB.copy()
        tokenizer.add_special_tokens(all_special_tokens)

        # 配置padding
        tokenizer.enable_padding(
            direction="right",
            pad_id=PAD_TOKEN_ID,
            pad_type_id=0,
            pad_token="<pad>"
        )

        print(f"Lineage RNA tokenizer创建完成，词汇表大小: {len(LINEAGE_RNA_VOCAB)}")
        return tokenizer

    def encode(self, sequence: str) -> List[int]:
        """
        编码RNA序列（逐字符编码）

        Args:
            sequence: 包含谱系信息和RNA序列的字符串

        Returns:
            token ID列表
        """
        # 逐字符编码
        token_ids = []
        i = 0
        while i < len(sequence):
            # 优先匹配多字符token（特殊标记、层级前缀等）
            matched = False

            # 检查RNA类型token（最长优先）
            for rna_token in sorted(RNA_TYPE_TOKENS, key=len, reverse=True):
                if sequence[i:i+len(rna_token)] == rna_token:
                    token_id = self.token_to_id(rna_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(rna_token)
                        matched = True
                        break

            if matched:
                continue

            # 检查GLM span标记
            for span_token in GLM_SPAN_TOKENS:
                if sequence[i:i+len(span_token)] == span_token:
                    token_id = self.token_to_id(span_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(span_token)
                        matched = True
                        break

            if matched:
                continue

            # 检查其他特殊标记
            for special_token in RNA_SPECIAL_TOKENS:
                if sequence[i:i+len(special_token)] == special_token:
                    token_id = self.token_to_id(special_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(special_token)
                        matched = True
                        break

            if matched:
                continue

            # 检查谱系层级前缀（D__, P__, C__等）
            for level_token in LINEAGE_LEVEL_TOKENS:
                if sequence[i:i+len(level_token)] == level_token:
                    token_id = self.token_to_id(level_token)
                    if token_id is not None:
                        token_ids.append(token_id)
                        i += len(level_token)
                        matched = True
                        break

            if matched:
                continue

            # 单字符编码
            char = sequence[i]
            token_id = self.token_to_id(char)
            if token_id is not None:
                token_ids.append(token_id)
            else:
                # 未知字符使用<unk>
                unk_id = self.token_to_id("<unk>")
                if unk_id is not None:
                    token_ids.append(unk_id)
            i += 1

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """解码token序列（无空格拼接）"""
        tokens = [self.id_to_token(tid) for tid in token_ids if self.id_to_token(tid) is not None]
        return "".join(tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """获取token对应的ID"""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """获取ID对应的token"""
        return self.tokenizer.id_to_token(token_id)

    @property
    def vocab_size(self) -> int:
        """词汇表大小（返回实际词典大小，去重后）"""
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        """返回词汇表大小（兼容Hugging Face接口）"""
        return self.vocab_size

    def get_output_token_ids(self) -> List[int]:
        """
        返回模型实际需要预测的token ID列表（通用版本，包含所有结束标记）

        在条件生成任务中，模型只需要预测RNA序列本身和结束标记，
        不需要预测条件token（谱系信息、RNA类型等）。

        Returns:
            可输出token的ID列表
        """
        output_tokens = [
            "A", "U", "G", "C",           # RNA碱基（4个）
            "<eos>",                       # Stage 1序列生成结束标记
            "<eos_span>",                  # Stage 2序列补全结束标记
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def get_stage1_output_token_ids(self) -> List[int]:
        """
        返回Stage 1序列生成任务的输出token ID列表

        Stage 1需要预测RNA碱基、方向标记和<eos>，不需要<eos_span>

        Returns:
            可输出token的ID列表 (A, U, G, C, 5, 3, <eos>)
        """
        output_tokens = [
            "A", "U", "G", "C",    # RNA碱基（4个）
            "5", "3",              # 序列方向标记（5'和3'端）
            "<eos>",               # Stage 1序列生成结束标记
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def get_stage2_output_token_ids(self) -> List[int]:
        """
        返回Stage 2序列补全任务的输出token ID列表

        Stage 2只需要预测RNA碱基和<eos_span>，不需要<eos>

        Returns:
            可输出token的ID列表 (A, U, G, C, <eos_span>)
        """
        output_tokens = [
            "A", "U", "G", "C",    # RNA碱基（4个）
            "<eos_span>",          # Stage 2序列补全结束标记
        ]
        token_ids = []
        for token in output_tokens:
            token_id = self.token_to_id(token)
            if token_id is not None:
                token_ids.append(token_id)
        return token_ids

    def save(self, filepath: str):
        """保存tokenizer（向后兼容）"""
        self.tokenizer.save(filepath)

    def save_pretrained(self, save_directory: str):
        """保存tokenizer到HuggingFace格式"""
        import json

        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)

        # 保存tokenizer文件
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        # 创建词汇表文件 - 使用实际tokenizer的词汇表，而不是硬编码重建
        vocab = self.tokenizer.get_vocab()
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # 创建special_tokens_map.json
        special_tokens_map = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>"
        }
        special_tokens_path = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

        # 创建tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": "LineageRNATokenizer",
            "auto_map": {
                "AutoTokenizer": ["lineage_tokenizer.py", "LineageRNATokenizer"]
            },
            "vocab_size": self.vocab_size,
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "rna_bases": RNA_BASES,
            "special_tokens": RNA_SPECIAL_TOKENS,
            "glm_span_tokens": GLM_SPAN_TOKENS,
            "rna_type_tokens": RNA_TYPE_TOKENS,
            "lineage_level_tokens": LINEAGE_LEVEL_TOKENS,
            "lineage_special_chars": LINEAGE_SPECIAL_CHARS,
            "mode": "lineage",
            "description": "Lineage-based tokenizer without species tokens or task tokens"
        }
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        print(f"LineageRNATokenizer已保存到: {save_directory}")

    @classmethod
    def from_file(cls, filepath: str) -> 'LineageRNATokenizer':
        """从文件加载tokenizer（向后兼容）"""
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(filepath)
        return instance

    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'LineageRNATokenizer':
        """从HuggingFace格式加载tokenizer"""
        import json

        # 检查必要文件
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer文件不存在: {tokenizer_path}")

        # 创建实例
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(tokenizer_path)

        return instance


def create_lineage_rna_tokenizer_json(output_path: str):
    """创建并保存Lineage RNA tokenizer的JSON文件"""
    tokenizer = LineageRNATokenizer()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存tokenizer
    tokenizer.save(output_path)

    print(f"Lineage RNA tokenizer已保存到: {output_path}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊标记: {RNA_SPECIAL_TOKENS[:8]}...")
    print(f"RNA碱基: {RNA_BASES}")
    print(f"谱系层级标记: {LINEAGE_LEVEL_TOKENS}")
    print(f"谱系特殊字符: {LINEAGE_SPECIAL_CHARS}")

    return tokenizer


def get_lineage_rna_tokenizer(use_direction_tokens: bool = True) -> LineageRNATokenizer:
    """获取Lineage RNA tokenizer实例

    Args:
        use_direction_tokens: 是否使用5/3方向标记
            - True: 加载包含5/3方向标记的tokenizer (vocab_size=114)
            - False: 加载旧版tokenizer，不包含方向标记 (vocab_size=112)

    Returns:
        LineageRNATokenizer实例
    """
    if use_direction_tokens:
        # 使用新tokenizer（包含5/3方向标记，vocab_size=114）
        tokenizer_path = os.path.join(os.path.dirname(__file__), "lineage_tokenizer.json")
    else:
        # 使用旧tokenizer（不含方向标记，vocab_size=112）
        tokenizer_path = os.path.join(os.path.dirname(__file__), "lineage_tokenizer_old.json")

    if os.path.exists(tokenizer_path):
        tokenizer = LineageRNATokenizer.from_file(tokenizer_path)
        print(f"已加载Lineage RNA tokenizer: {os.path.basename(tokenizer_path)} (vocab_size={tokenizer.vocab_size})")
        return tokenizer
    else:
        if use_direction_tokens:
            # 如果新tokenizer文件不存在，创建它
            print(f"创建新的Lineage RNA tokenizer: {tokenizer_path}")
            return create_lineage_rna_tokenizer_json(tokenizer_path)
        else:
            # 旧tokenizer文件不存在是严重错误
            raise FileNotFoundError(
                f"旧版tokenizer文件不存在: {tokenizer_path}\n"
                "请确保已备份旧版tokenizer为 lineage_tokenizer_vocab134.json"
            )


if __name__ == "__main__":
    # 测试Lineage RNA tokenizer
    tokenizer = LineageRNATokenizer()

    # 测试序列（谱系全小写，RNA序列大写）
    test_sequences = [
        "|d__eukaryota;p__chordata;c__mammalia;<rna_mRNA>|AUGCUAGCUAGC<eos>",
        "|d__bacteria;p__;c__;o__;f__;g__escherichia;s__escherichia_coli;<rna_rRNA>|AUCGAUCG<eos>",
        "AUGCUAGC",  # 纯RNA序列
    ]

    print("=== Lineage RNA Tokenizer测试 ===")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print()

    for seq in test_sequences:
        encoded = tokenizer.encode(seq)
        decoded = tokenizer.decode(encoded)
        is_correct = seq == decoded

        print(f"原序列: {seq}")
        print(f"编码长度: {len(encoded)}")
        print(f"解码结果: {decoded}")
        print(f"编码正确: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print(f"  差异: 原始长度={len(seq)}, 解码长度={len(decoded)}")
        print("-" * 80)

    # 保存tokenizer
    output_path = os.path.join(os.path.dirname(__file__), "lineage_tokenizer.json")
    create_lineage_rna_tokenizer_json(output_path)
