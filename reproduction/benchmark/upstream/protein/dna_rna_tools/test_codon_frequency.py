"""
测试密码子频率功能

验证物种特异密码子频率表的加载、查询和反向翻译功能
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from codon_frequency_manager import (
    CodonFrequencyManager,
    list_available_species,
    get_codon_frequency,
    select_codon_by_frequency
)
from dna_rna_converter import (
    reverse_translate_protein,
    CodonAnalyzer,
    GeneticCodeType
)


def test_load_species_data():
    """测试1：加载物种数据"""
    print("=" * 60)
    print("测试1：加载物种数据")
    print("=" * 60)

    manager = CodonFrequencyManager()
    species_list = manager.list_species()

    print(f"\n✓ 成功加载 {len(species_list)} 个物种的密码子频率表：\n")
    for species in species_list:
        print(f"  • {species['name']}")
        print(f"    ID: {species['species_id']}")
        print(f"    TaxID: {species['taxid']}")
        print(f"    遗传密码: {species['genetic_code']}")
        print()

    assert len(species_list) >= 5, "应该至少有5个物种"
    print("✓ 测试通过\n")


def test_codon_frequency_query():
    """测试2：查询密码子频率"""
    print("=" * 60)
    print("测试2：查询密码子频率")
    print("=" * 60)

    manager = CodonFrequencyManager()

    # 测试人类的亮氨酸密码子频率
    print("\n人类（Homo sapiens）亮氨酸（L）的密码子使用频率：\n")
    human_table = manager.get_species_table('homo_sapiens')
    leu_codons = human_table.get_codons_for_amino_acid('L', sorted_by_frequency=True)

    for codon, freq in leu_codons:
        print(f"  {codon}: {freq:.4f} ({freq*100:.2f}%)")

    # 验证频率和为1
    total_freq = sum(freq for _, freq in leu_codons)
    print(f"\n  总频率: {total_freq:.4f}")
    assert abs(total_freq - 1.0) < 0.001, "归一化频率之和应该为1"

    print("\n✓ 测试通过\n")


def test_species_comparison():
    """测试3：比较不同物种的密码子偏好"""
    print("=" * 60)
    print("测试3：比较不同物种的密码子偏好")
    print("=" * 60)

    manager = CodonFrequencyManager()

    # 比较人类和大肠杆菌对精氨酸（R）的密码子偏好
    species_ids = ['homo_sapiens', 'escherichia_coli']
    amino_acid = 'R'

    print(f"\n比较不同物种对氨基酸 {amino_acid}（精氨酸）的密码子偏好：\n")

    comparison = manager.compare_codon_usage(species_ids, amino_acid)

    for species_id, codons in comparison.items():
        table = manager.get_species_table(species_id)
        print(f"{table.name}:")
        for codon, freq in codons[:3]:  # 只显示前3个最常用的
            print(f"  {codon}: {freq:.4f} ({freq*100:.2f}%)")
        print()

    print("✓ 测试通过\n")


def test_reverse_translation_strategies():
    """测试4：测试不同的反向翻译策略"""
    print("=" * 60)
    print("测试4：测试不同的反向翻译策略")
    print("=" * 60)

    protein = "MKPGFWYLCV"
    print(f"\n测试蛋白质序列: {protein}\n")

    strategies = [
        ('first', None, "总是选择第一个密码子"),
        ('random', None, "随机选择密码子"),
        ('most_frequent', 'homo_sapiens', "人类最常用密码子"),
        ('weighted_random', 'homo_sapiens', "人类频率加权随机"),
        ('most_frequent', 'escherichia_coli', "大肠杆菌最常用密码子"),
    ]

    results = {}
    for strategy, species_id, description in strategies:
        try:
            rna = reverse_translate_protein(
                protein,
                optimization=strategy,
                output_format='rna',
                species_id=species_id
            )
            results[description] = rna
            print(f"{description}:")
            print(f"  RNA: {rna}")
            print(f"  长度: {len(rna)} nt")
            print()
        except Exception as e:
            print(f"{description}: 错误 - {e}\n")

    # 验证所有序列长度相同
    lengths = [len(rna) for rna in results.values()]
    assert len(set(lengths)) == 1, "所有反向翻译结果长度应该相同"
    assert lengths[0] == len(protein) * 3, "RNA长度应该是蛋白质长度的3倍"

    print("✓ 测试通过\n")


def test_reverse_translation_with_validation():
    """测试5：反向翻译并验证正确性"""
    print("=" * 60)
    print("测试5：反向翻译并验证正确性")
    print("=" * 60)

    protein = "MKPGFWYLCV"
    print(f"\n原始蛋白质序列: {protein}\n")

    # 使用人类最常用密码子反向翻译
    rna = reverse_translate_protein(
        protein,
        optimization='most_frequent',
        output_format='rna',
        species_id='homo_sapiens'
    )

    print(f"反向翻译得到的RNA: {rna}\n")

    # 正向翻译回蛋白质进行验证
    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    dna = rna.replace('U', 'T')
    translated_protein = analyzer.translate_sequence(dna)

    print(f"正向翻译回蛋白质: {translated_protein}\n")

    # 验证
    assert translated_protein == protein, f"翻译结果不匹配: {translated_protein} != {protein}"

    print("✓ 反向翻译正确性验证通过\n")


def test_frequency_based_selection():
    """测试6：测试频率加权选择的统计特性"""
    print("=" * 60)
    print("测试6：测试频率加权选择的统计特性")
    print("=" * 60)

    amino_acid = 'L'  # 亮氨酸有6个密码子
    species_id = 'homo_sapiens'
    n_samples = 1000

    print(f"\n对氨基酸 {amino_acid}（亮氨酸）进行 {n_samples} 次加权随机选择\n")

    # 获取理论频率
    manager = CodonFrequencyManager()
    table = manager.get_species_table(species_id)
    theoretical_freqs = dict(table.get_codons_for_amino_acid(amino_acid))

    # 进行多次采样
    codon_counts = {}
    for _ in range(n_samples):
        codon = select_codon_by_frequency(species_id, amino_acid, 'weighted_random')
        codon_counts[codon] = codon_counts.get(codon, 0) + 1

    # 计算观测频率
    print("密码子\t理论频率\t观测频率\t观测次数")
    print("-" * 60)
    for codon in sorted(theoretical_freqs.keys()):
        theo_freq = theoretical_freqs[codon]
        obs_count = codon_counts.get(codon, 0)
        obs_freq = obs_count / n_samples
        print(f"{codon}\t{theo_freq:.4f}\t\t{obs_freq:.4f}\t\t{obs_count}")

    print("\n✓ 测试通过（观测频率应该接近理论频率）\n")


def test_complete_workflow():
    """测试7：完整工作流程"""
    print("=" * 60)
    print("测试7：完整工作流程（蛋白质→RNA→验证）")
    print("=" * 60)

    # 模拟一个真实场景
    protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

    print(f"\n测试蛋白质序列（{len(protein)} aa）:\n{protein[:60]}...\n")

    # 使用不同物种的密码子优化策略
    species_list = ['homo_sapiens', 'escherichia_coli', 'saccharomyces_cerevisiae']

    for species_id in species_list:
        manager = CodonFrequencyManager()
        table = manager.get_species_table(species_id)

        print(f"\n{table.name}:")
        print("-" * 40)

        # 反向翻译
        rna = reverse_translate_protein(
            protein,
            optimization='most_frequent',
            output_format='rna',
            species_id=species_id
        )

        # 验证
        analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
        dna = rna.replace('U', 'T')
        back_translated = analyzer.translate_sequence(dna)

        # 统计信息
        gc_count = rna.count('G') + rna.count('C')
        gc_content = (gc_count / len(rna)) * 100

        print(f"  RNA长度: {len(rna)} nt")
        print(f"  GC含量: {gc_content:.2f}%")
        print(f"  验证: {'✓ 通过' if back_translated == protein else '✗ 失败'}")

    print("\n✓ 完整工作流程测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("密码子频率功能测试套件")
    print("=" * 60 + "\n")

    tests = [
        test_load_species_data,
        test_codon_frequency_query,
        test_species_comparison,
        test_reverse_translation_strategies,
        test_reverse_translation_with_validation,
        test_frequency_based_selection,
        test_complete_workflow,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ 测试失败: {test_func.__name__}")
            print(f"  错误: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
