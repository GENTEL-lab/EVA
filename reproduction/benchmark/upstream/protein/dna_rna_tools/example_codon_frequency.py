#!/usr/bin/env python3
"""
密码子频率优化示例

演示如何使用物种特异的密码子频率进行反向翻译
"""

from dna_rna_converter import reverse_translate_protein, CodonAnalyzer, GeneticCodeType
from codon_frequency_manager import CodonFrequencyManager, list_available_species


def example1_list_species():
    """示例1：列出所有可用物种"""
    print("=" * 60)
    print("示例1：列出所有可用物种")
    print("=" * 60)

    species = list_available_species()
    print(f"\n共有 {len(species)} 个物种的密码子频率数据：\n")

    for s in species:
        print(f"• {s['name']}")
        print(f"  ID: {s['species_id']}")
        print(f"  TaxID: {s['taxid']}")
        print()


def example2_compare_species():
    """示例2：比较不同物种的反向翻译结果"""
    print("=" * 60)
    print("示例2：比较不同物种的反向翻译结果")
    print("=" * 60)

    protein = "MKPGFWYLCV"
    print(f"\n蛋白质序列: {protein}\n")

    species_list = [
        'homo_sapiens',
        'escherichia_coli',
        'saccharomyces_cerevisiae'
    ]

    results = {}
    for species_id in species_list:
        manager = CodonFrequencyManager()
        table = manager.get_species_table(species_id)

        rna = reverse_translate_protein(
            protein,
            optimization='most_frequent',
            species_id=species_id,
            output_format='rna'
        )

        # 计算GC含量
        gc_count = rna.count('G') + rna.count('C')
        gc_content = (gc_count / len(rna)) * 100

        results[species_id] = {
            'name': table.name,
            'rna': rna,
            'gc': gc_content
        }

    # 显示结果
    for species_id, data in results.items():
        print(f"{data['name']}:")
        print(f"  RNA: {data['rna']}")
        print(f"  GC含量: {data['gc']:.2f}%")
        print()


def example3_codon_frequency_analysis():
    """示例3：分析特定氨基酸的密码子频率"""
    print("=" * 60)
    print("示例3：分析特定氨基酸的密码子频率")
    print("=" * 60)

    manager = CodonFrequencyManager()

    # 分析人类的亮氨酸密码子使用
    print("\n人类亮氨酸（L）的密码子使用频率：\n")
    human_table = manager.get_species_table('homo_sapiens')
    leu_codons = human_table.get_codons_for_amino_acid('L', sorted_by_frequency=True)

    for codon, freq in leu_codons:
        bar = '█' * int(freq * 50)  # 可视化
        print(f"  {codon}: {freq:.4f} ({freq*100:5.2f}%) {bar}")


def example4_optimization_strategies():
    """示例4：比较不同的优化策略"""
    print("=" * 60)
    print("示例4：比较不同的优化策略")
    print("=" * 60)

    protein = "MKPGFWYLCV"
    species_id = 'homo_sapiens'

    print(f"\n蛋白质序列: {protein}")
    print(f"物种: 人类 (Homo sapiens)\n")

    strategies = [
        ('first', '总是选择第一个密码子'),
        ('random', '随机选择密码子'),
        ('most_frequent', '选择最常用密码子'),
        ('weighted_random', '按频率加权随机选择'),
    ]

    for strategy, description in strategies:
        if strategy in ['most_frequent', 'weighted_random']:
            rna = reverse_translate_protein(
                protein,
                optimization=strategy,
                species_id=species_id,
                output_format='rna'
            )
        else:
            rna = reverse_translate_protein(
                protein,
                optimization=strategy,
                output_format='rna'
            )

        print(f"{description}:")
        print(f"  {rna}")
        print()


def example5_validate_translation():
    """示例5：验证反向翻译的正确性"""
    print("=" * 60)
    print("示例5：验证反向翻译的正确性")
    print("=" * 60)

    protein = "MKPGFWYLCVNQSTHDEAIR"
    print(f"\n原始蛋白质: {protein}\n")

    # 使用人类密码子优化
    rna = reverse_translate_protein(
        protein,
        optimization='most_frequent',
        species_id='homo_sapiens',
        output_format='rna'
    )

    print(f"反向翻译得到的RNA:\n{rna}\n")

    # 正向翻译回蛋白质
    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    dna = rna.replace('U', 'T')
    back_translated = analyzer.translate_sequence(dna)

    print(f"正向翻译回蛋白质: {back_translated}\n")

    # 验证
    if back_translated == protein:
        print("✓ 验证通过：反向翻译正确！")
    else:
        print("✗ 验证失败：翻译结果不匹配")


def example6_species_comparison():
    """示例6：比较不同物种对同一氨基酸的密码子偏好"""
    print("=" * 60)
    print("示例6：比较不同物种的密码子偏好")
    print("=" * 60)

    manager = CodonFrequencyManager()

    amino_acid = 'R'  # 精氨酸
    species_ids = ['homo_sapiens', 'escherichia_coli']

    print(f"\n比较不同物种对氨基酸 {amino_acid}（精氨酸）的密码子偏好：\n")

    comparison = manager.compare_codon_usage(species_ids, amino_acid)

    for species_id, codons in comparison.items():
        table = manager.get_species_table(species_id)
        print(f"{table.name}:")

        for codon, freq in codons:
            bar = '█' * int(freq * 40)
            print(f"  {codon}: {freq:.4f} ({freq*100:5.2f}%) {bar}")
        print()


def example7_real_world_workflow():
    """示例7：真实场景工作流程"""
    print("=" * 60)
    print("示例7：真实场景工作流程")
    print("=" * 60)

    # 假设我们要在人类细胞中表达一个蛋白质
    protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

    print(f"\n目标：在人类细胞中表达蛋白质")
    print(f"蛋白质序列（{len(protein)} aa）:\n{protein}\n")

    # 步骤1：使用人类密码子优化
    print("步骤1：使用人类最常用密码子进行反向翻译...")
    optimized_rna = reverse_translate_protein(
        protein,
        optimization='most_frequent',
        species_id='homo_sapiens',
        output_format='rna'
    )

    # 步骤2：分析序列特征
    print("\n步骤2：分析优化后的序列特征...")
    gc_count = optimized_rna.count('G') + optimized_rna.count('C')
    gc_content = (gc_count / len(optimized_rna)) * 100

    print(f"  RNA长度: {len(optimized_rna)} nt")
    print(f"  GC含量: {gc_content:.2f}%")

    # 步骤3：验证
    print("\n步骤3：验证反向翻译正确性...")
    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    dna = optimized_rna.replace('U', 'T')
    back_translated = analyzer.translate_sequence(dna)

    if back_translated == protein:
        print("  ✓ 验证通过")
    else:
        print("  ✗ 验证失败")

    # 步骤4：输出结果
    print("\n步骤4：输出优化后的序列...")
    print(f"\n优化后的RNA序列:\n{optimized_rna}\n")

    print("✓ 工作流程完成！序列已优化用于人类细胞表达。")


def main():
    """运行所有示例"""
    examples = [
        example1_list_species,
        example2_compare_species,
        example3_codon_frequency_analysis,
        example4_optimization_strategies,
        example5_validate_translation,
        example6_species_comparison,
        example7_real_world_workflow,
    ]

    print("\n" + "=" * 60)
    print("密码子频率优化示例")
    print("=" * 60 + "\n")

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            print()
        except Exception as e:
            print(f"\n✗ 示例 {i} 执行失败: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("所有示例执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
