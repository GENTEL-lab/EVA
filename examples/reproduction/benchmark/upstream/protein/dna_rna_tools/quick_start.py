"""
DNA/RNA转换工具 - 快速开始示例

这个文件展示了最常用的功能，帮助你快速上手
"""

from dna_rna_converter import (
    quick_dna_to_rna,
    analyze_codon_sequence,
    reverse_translate_protein,
    DNAToRNAConverter,
    CodonAnalyzer,
    GeneticCodeType,
)


def example_1_dna_to_rna():
    """示例1: DNA到RNA转换"""
    print("=" * 60)
    print("示例1: DNA到RNA转换")
    print("=" * 60)

    dna = "ATGCGATCGATCG"
    rna = quick_dna_to_rna(dna)

    print(f"DNA: {dna}")
    print(f"RNA: {rna}")
    print()


def example_2_analyze_gene():
    """示例2: 分析基因序列"""
    print("=" * 60)
    print("示例2: 分析基因编码序列")
    print("=" * 60)

    gene = "ATGAAACCCGGGTTTAAATAG"
    result = analyze_codon_sequence(gene)

    print(f"基因序列: {gene}")
    print(f"翻译结果: {result['translation']}")
    print(f"密码子数: {result['codon_count']}")
    print(f"序列有效: {result['is_valid']}")
    print()


def example_3_reverse_translation():
    """示例3: 反向翻译（氨基酸 -> RNA）"""
    print("=" * 60)
    print("示例3: 反向翻译（氨基酸 -> RNA）")
    print("=" * 60)

    # 基本反向翻译
    protein = "MKP"
    rna = reverse_translate_protein(protein, optimization='first')
    print(f"蛋白质: {protein}")
    print(f"RNA: {rna}")

    # 生成完整编码序列
    full_rna = reverse_translate_protein(
        "KPG",
        add_start_stop=True,
        optimization='first'
    )
    print(f"\n蛋白质片段: KPG")
    print(f"完整编码序列: {full_rna}")
    print(f"  起始密码子: {full_rna[:3]}")
    print(f"  终止密码子: {full_rna[-3:]}")
    print()


def example_4_codon_optimization():
    """示例4: 密码子优化策略"""
    print("=" * 60)
    print("示例4: 密码子优化策略")
    print("=" * 60)

    protein = "MKPGF"

    # 不同策略
    rna_first = reverse_translate_protein(protein, optimization='first')
    rna_freq = reverse_translate_protein(protein, optimization='frequent')

    print(f"蛋白质: {protein}")
    print(f"first策略:    {rna_first}")
    print(f"frequent策略: {rna_freq}")
    print()


def example_5_genetic_codes():
    """示例5: 不同遗传密码表"""
    print("=" * 60)
    print("示例5: 不同遗传密码表")
    print("=" * 60)

    protein = "MW"  # 甲硫氨酸 + 色氨酸

    # 标准遗传密码
    standard_rna = reverse_translate_protein(
        protein,
        code_type='standard',
        optimization='first'
    )

    # 支原体遗传密码
    mycoplasma_rna = reverse_translate_protein(
        protein,
        code_type='mycoplasma',
        optimization='first'
    )

    print(f"蛋白质: {protein}")
    print(f"标准遗传密码: {standard_rna}")
    print(f"支原体密码:   {mycoplasma_rna}")
    print()


def example_6_round_trip():
    """示例6: 往返翻译验证"""
    print("=" * 60)
    print("示例6: 往返翻译验证")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 原始蛋白质
    protein = "MKPGF"
    print(f"原始蛋白质: {protein}")

    # 反向翻译为RNA
    rna = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"反向翻译RNA: {rna}")

    # 翻译回蛋白质
    dna = rna.replace('U', 'T')
    back_translation = analyzer.translate_sequence(dna)
    print(f"翻译回蛋白质: {back_translation}")

    # 验证
    if back_translation == protein:
        print("✓ 往返翻译验证通过！")
    else:
        print("✗ 往返翻译验证失败！")
    print()


def example_7_codon_lookup():
    """示例7: 查询氨基酸的所有密码子"""
    print("=" * 60)
    print("示例7: 查询氨基酸的所有密码子")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    amino_acids = ['M', 'W', 'L', 'S']

    for aa in amino_acids:
        codons = analyzer.get_codons_for_amino_acid(aa)
        print(f"{aa}: {codons} (共{len(codons)}个)")
    print()


def example_8_for_evo2_model():
    """示例8: 为Evo 2模型准备数据"""
    print("=" * 60)
    print("示例8: 为Evo 2模型准备数据")
    print("=" * 60)

    converter = DNAToRNAConverter()

    # 场景1: DNA序列转换为模型输入格式
    dna = "ATGCGATCG"
    model_input = converter.convert_dna_to_rna_format(dna)
    print(f"DNA序列: {dna}")
    print(f"模型输入格式(T): {model_input}")

    # 场景2: 蛋白质反向翻译为模型格式
    protein = "MKP"
    model_rna = reverse_translate_protein(
        protein,
        output_format='model',
        optimization='first'
    )
    print(f"\n蛋白质: {protein}")
    print(f"模型格式(T): {model_rna}")

    # 场景3: 模型输出转换为标准RNA
    model_output = "ATGAAACCC"
    standard_rna = converter.convert_model_output_to_rna(model_output)
    print(f"\n模型输出(T): {model_output}")
    print(f"标准RNA(U): {standard_rna}")
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DNA/RNA转换工具 - 快速开始示例")
    print("=" * 60 + "\n")

    examples = [
        example_1_dna_to_rna,
        example_2_analyze_gene,
        example_3_reverse_translation,
        example_4_codon_optimization,
        example_5_genetic_codes,
        example_6_round_trip,
        example_7_codon_lookup,
        example_8_for_evo2_model,
    ]

    for example in examples:
        example()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
