"""
RNA专用tokenizer
支持AUGC碱基和特殊标记
"""

import json
import os
import re
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence
from tokenizers.trainers import BpeTrainer

# RNA特殊标记
RNA_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>", 
    "<eos>",
    "<bos_glm>",
    "<eos_span>",
    "<unk>",
    "1",  # 正向开始标记
    "2",  # 正向结束标记
]

# GLM span标记（最多50个）
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

def normalize_species_name(species_name):
    """
    将物种名称标准化为token格式

    规则：
    1. 转为小写
    2. 空格转下划线
    3. 移除特殊字符，保留字母、数字、下划线
    4. 连续下划线合并为单个
    5. 去除首尾下划线
    """
    # 转小写
    normalized = species_name.lower()

    # 空格转下划线
    normalized = normalized.replace(' ', '_')

    # 移除特殊字符，只保留字母、数字、下划线
    normalized = re.sub(r'[^a-z0-9_]', '_', normalized)

    # 连续下划线合并为单个
    normalized = re.sub(r'_+', '_', normalized)

    # 去除首尾下划线
    normalized = normalized.strip('_')

    return normalized

def load_species_tokens(json_file_path=None):
    """
    从training_tokens.json加载物种列表并生成标准化token

    Args:
        json_file_path: JSON文件路径，默认为相对路径

    Returns:
        List[str]: 物种token列表
    """
    if json_file_path is None:
        # 默认路径：相对于tokenizer.py的位置
        current_dir = os.path.dirname(__file__)
        json_file_path = os.path.join(current_dir, "..", "data", "tree", "training_tokens.json")
        json_file_path = os.path.abspath(json_file_path)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        species_list = data.get('tokens', [])
        print(f"从 {json_file_path} 加载了 {len(species_list)} 个物种")

        # 生成标准化的物种token
        species_tokens = []
        for species_name in species_list:
            normalized = normalize_species_name(species_name)
            token = f"<species_{normalized}>"
            species_tokens.append(token)

        print(f"生成了 {len(species_tokens)} 个物种token")
        return species_tokens

    except FileNotFoundError:
        print(f"警告: 物种文件 {json_file_path} 不存在，使用空物种列表")
        return []
    except Exception as e:
        print(f"加载物种文件时出错: {e}，使用空物种列表")
        return []

# 动态加载物种token
SPECIES_TOKENS = load_species_tokens()

# 任务类型token
TASK_TOKENS = ["<task_seq>", "<task_rna>", "<task_species>", "<task_clm>"]

# RNA碱基
RNA_BASES = ["A", "U", "G", "C"]

# 完整词汇表（动态构建）
def build_rna_vocab():
    """构建完整的RNA词汇表"""
    return RNA_SPECIAL_TOKENS + GLM_SPAN_TOKENS + RNA_TYPE_TOKENS + SPECIES_TOKENS + TASK_TOKENS + RNA_BASES

RNA_VOCAB = build_rna_vocab()

END_OF_SPAN_TOKEN = "<eos_span>"
PAD_TOKEN_ID = 0


class RNATokenizer:
    """RNA序列专用tokenizer"""
    
    def __init__(self):
        self.tokenizer = self._create_tokenizer()
    
    def _create_tokenizer(self) -> Tokenizer:
        """创建RNA tokenizer"""
        # 使用BPE模型，但词汇表预定义
        tokenizer = Tokenizer(BPE())

        # 设置预分词器：逐字符分割
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=" ", behavior="removed"),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])

        # 重新构建词汇表（确保使用最新的物种token）
        current_vocab = build_rna_vocab()
        vocab = {token: idx for idx, token in enumerate(current_vocab)}
        merges = []  # RNA不需要合并规则，每个碱基独立

        tokenizer.model = BPE(vocab=vocab, merges=merges)

        # 设置特殊标记
        tokenizer.add_special_tokens(RNA_SPECIAL_TOKENS + GLM_SPAN_TOKENS + RNA_TYPE_TOKENS + SPECIES_TOKENS + TASK_TOKENS)

        # 配置padding
        tokenizer.enable_padding(
            direction="right",
            pad_id=PAD_TOKEN_ID,
            pad_type_id=0,
            pad_token="<pad>"
        )

        print(f"RNA tokenizer创建完成，词汇表大小: {len(current_vocab)}")
        return tokenizer
    
    def encode(self, sequence: str) -> List[int]:
        """编码RNA序列"""
        return self.tokenizer.encode(sequence).ids
    
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
        """词汇表大小"""
        return self.tokenizer.get_vocab_size()
    
    def __len__(self) -> int:
        """返回词汇表大小（兼容Hugging Face接口）"""
        return self.tokenizer.get_vocab_size()
    
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
        
        # 创建词汇表文件
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
            "tokenizer_class": "RNATokenizer",
            "auto_map": {
                "AutoTokenizer": ["rna_tokenizer.py", "RNATokenizer"]
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
            "species_tokens": SPECIES_TOKENS,
            "task_tokens": TASK_TOKENS,
            "species_count": len(SPECIES_TOKENS),
            "dynamic_species_loading": True
        }
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        print(f"RNATokenizer已保存到: {save_directory}")
    
    @classmethod
    def from_file(cls, filepath: str) -> 'RNATokenizer':
        """从文件加载tokenizer（向后兼容）"""
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(filepath)
        return instance
    
    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'RNATokenizer':
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


def create_rna_tokenizer_json(output_path: str):
    """创建并保存RNA tokenizer的JSON文件"""
    tokenizer = RNATokenizer()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存tokenizer
    tokenizer.save(output_path)
    
    print(f"RNA tokenizer已保存到: {output_path}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊标记: {RNA_SPECIAL_TOKENS[:8]}...")
    print(f"RNA碱基: {RNA_BASES}")
    
    return tokenizer


def get_rna_tokenizer() -> RNATokenizer:
    """获取RNA tokenizer实例"""
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    
    if os.path.exists(tokenizer_path):
        return RNATokenizer.from_file(tokenizer_path)
    else:
        # 如果文件不存在，创建新的tokenizer
        print(f"创建新的RNA tokenizer: {tokenizer_path}")
        return create_rna_tokenizer_json(tokenizer_path)


if __name__ == "__main__":
    # 测试RNA tokenizer
    tokenizer = RNATokenizer()
    
    # 测试序列
    test_sequences = [
        "1AUGCUAGCUAGC2",
        "<bos>1AUGCUAGCUAGC2<eos>",
        "AUGCUAGC[GLM]1-5-4",
    ]
    
    print("=== RNA Tokenizer测试 ===")
    for seq in test_sequences:
        encoded = tokenizer.encode(seq)
        decoded = tokenizer.decode(encoded)
        print(f"原序列: {seq}")
        print(f"编码: {encoded}")
        print(f"解码: {decoded}")
        print(f"编码正确: {seq == decoded}")
        print("-" * 50)
    
    # 保存tokenizer
    output_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    create_rna_tokenizer_json(output_path)