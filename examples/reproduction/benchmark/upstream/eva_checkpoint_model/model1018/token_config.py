"""
RNA Tokenizer 共享配置
定义所有 tokenizer 使用的通用 token 常量
"""

# RNA特殊标记（核心标记）
RNA_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<bos_glm>",
    "<eos_span>",
    "<unk>",
]

# GLM span标记（最多50个，用于序列补全任务）
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

# RNA碱基（4种核苷酸）
RNA_BASES = ["A", "U", "G", "C"]

# 常用常量
END_OF_SPAN_TOKEN = "<eos_span>"
PAD_TOKEN_ID = 0

# Token ID 映射（用于快速访问）
SPECIAL_TOKEN_IDS = {
    "pad": 0,
    "bos": 1,
    "eos": 2,
    "bos_glm": 3,
    "eos_span": 4,
    "unk": 5,
}

