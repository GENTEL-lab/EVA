"""
反向翻译功能的测试和使用示例

测试氨基酸序列到RNA序列的反向翻译功能
"""

import sys
from dna_rna_converter import (
    CodonAnalyzer,
    GeneticCodeType,
    reverse_translate_protein,
)


def test_basic_reverse_translation():
    """测试基本的反向翻译功能"""
    print("=" * 60)
    print("测试1: 基本反向翻译（氨基酸 -> RNA）")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 测试单个氨基酸
    protein = "M"
    rna = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"氨基酸: {protein}")
    print(f"RNA序列: {rna}")
    assert 'U' in rna, "RNA格式应该包含U"

    # 测试多个氨基酸
    protein = "MKP"
    rna = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"\n氨基酸: {protein}")
    print(f"RNA序列: {rna}")
    print(f"长度: {len(rna)} (应该是 {len(protein) * 3})")
    assert len(rna) == len(protein) * 3, "RNA长度应该是氨基酸数量的3倍"

    # 验证反向翻译的正确性（翻译回去应该得到原始氨基酸）
    dna = rna.replace('U', 'T')
    back_translation = analyzer.translate_sequence(dna)
    print(f"反向验证: {rna} -> {back_translation}")
    assert back_translation == protein, "反向翻译后再翻译回去应该得到原始氨基酸"

    print("✓ 测试通过\n")


def test_optimization_strategies():
    """测试不同的密码子优化策略"""
    print("=" * 60)
    print("测试2: 密码子优化策略")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    protein = "MKPGF"

    # 测试不同策略
    strategies = ['first', 'frequent', 'random']

    for strategy in strategies:
        rna = analyzer.reverse_translate(protein, optimization=strategy, output_format='rna')
        print(f"{strategy:10s} 策略: {rna}")

        # 验证正确性
        dna = rna.replace('U', 'T')
        back = analyzer.translate_sequence(dna)
        assert back == protein, f"{strategy}策略的反向翻译验证失败"

    print("✓ 测试通过\n")


def test_output_formats():
    """测试不同的输出格式"""
    print("=" * 60)
    print("测试3: 不同输出格式")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    protein = "MKP"

    # RNA格式（U）
    rna = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"RNA格式(U):   {rna}")
    assert 'U' in rna and 'T' not in rna, "RNA格式应该只包含U"

    # DNA格式（T）
    dna = analyzer.reverse_translate(protein, optimization='first', output_format='dna')
    print(f"DNA格式(T):   {dna}")
    assert 'T' in dna and 'U' not in dna, "DNA格式应该只包含T"

    # 模型格式（T，用于Evo 2）
    model = analyzer.reverse_translate(protein, optimization='first', output_format='model')
    print(f"模型格式(T):  {model}")
    assert model == dna, "模型格式应该与DNA格式相同"

    print("✓ 测试通过\n")


def test_with_start_stop_codons():
    """测试自动添加起始和终止密码子"""
    print("=" * 60)
    print("测试4: 自动添加起始和终止密码子")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 不包含起始和终止的氨基酸序列
    protein = "KPG"

    # 不添加起始终止
    rna_without = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"原始蛋白: {protein}")
    print(f"不添加起始终止: {rna_without}")
    print(f"  长度: {len(rna_without)}")

    # 添加起始终止
    rna_with = analyzer.reverse_translate_with_start_stop(protein, optimization='first', output_format='rna')
    print(f"添加起始终止:   {rna_with}")
    print(f"  长度: {len(rna_with)}")
    print(f"  起始密码子: {rna_with[:3]}")
    print(f"  终止密码子: {rna_with[-3:]}")

    # 验证
    assert rna_with.startswith('AUG'), "应该以起始密码子AUG开始"
    assert len(rna_with) == len(rna_without) + 6, "应该多出6个碱基（起始+终止）"

    # 验证终止密码子
    dna = rna_with.replace('U', 'T')
    stop_codon = dna[-3:]
    assert analyzer.code_table.get(stop_codon) == '*', "最后应该是终止密码子"

    print("✓ 测试通过\n")


def test_different_genetic_codes():
    """测试不同遗传密码表的反向翻译"""
    print("=" * 60)
    print("测试5: 不同遗传密码表")
    print("=" * 60)

    # 包含色氨酸的蛋白质
    protein = "MW"  # 甲硫氨酸 + 色氨酸

    # 标准遗传密码
    standard_analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)
    standard_rna = standard_analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"标准遗传密码: {protein} -> {standard_rna}")

    # 支原体遗传密码（TGA编码色氨酸）
    mycoplasma_analyzer = CodonAnalyzer(GeneticCodeType.MYCOPLASMA)
    mycoplasma_rna = mycoplasma_analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"支原体密码:   {protein} -> {mycoplasma_rna}")

    # 验证两者都能正确翻译回去
    standard_back = standard_analyzer.translate_sequence(standard_rna.replace('U', 'T'))
    mycoplasma_back = mycoplasma_analyzer.translate_sequence(mycoplasma_rna.replace('U', 'T'))

    print(f"\n验证:")
    print(f"  标准: {standard_rna} -> {standard_back}")
    print(f"  支原体: {mycoplasma_rna} -> {mycoplasma_back}")

    assert standard_back == protein, "标准遗传密码验证失败"
    assert mycoplasma_back == protein, "支原体遗传密码验证失败"

    print("✓ 测试通过\n")


def test_convenience_function():
    """测试便捷函数"""
    print("=" * 60)
    print("测试6: 便捷函数 reverse_translate_protein()")
    print("=" * 60)

    protein = "MKP"

    # 基本使用
    rna = reverse_translate_protein(protein)
    print(f"基本使用: {protein} -> {rna}")

    # 指定优化策略
    rna_first = reverse_translate_protein(protein, optimization='first')
    print(f"first策略: {protein} -> {rna_first}")

    # 指定输出格式
    dna = reverse_translate_protein(protein, output_format='dna', optimization='first')
    print(f"DNA格式: {protein} -> {dna}")

    # 添加起始终止
    full_rna = reverse_translate_protein("KP", add_start_stop=True, optimization='first')
    print(f"添加起始终止: KP -> {full_rna}")
    assert full_rna.startswith('AUG'), "应该以AUG开始"

    # 不同遗传密码
    myco_rna = reverse_translate_protein(protein, code_type='mycoplasma', optimization='first')
    print(f"支原体密码: {protein} -> {myco_rna}")

    print("✓ 测试通过\n")


def test_special_amino_acids():
    """测试特殊氨基酸（终止密码子）"""
    print("=" * 60)
    print("测试7: 特殊氨基酸（终止密码子）")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 包含终止密码子的序列
    protein = "MKP*"  # * 表示终止

    rna = analyzer.reverse_translate(protein, optimization='first', output_format='rna')
    print(f"氨基酸序列: {protein}")
    print(f"RNA序列: {rna}")

    # 验证最后一个密码子是终止密码子
    dna = rna.replace('U', 'T')
    last_codon = dna[-3:]
    print(f"最后密码子: {last_codon}")
    assert analyzer.code_table.get(last_codon) == '*', "最后应该是终止密码子"

    print("✓ 测试通过\n")


def test_round_trip_translation():
    """测试往返翻译（DNA -> 蛋白质 -> DNA）"""
    print("=" * 60)
    print("测试8: 往返翻译验证")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 原始DNA序列
    original_dna = "ATGAAACCCGGGTTT"
    print(f"原始DNA: {original_dna}")

    # 正向翻译：DNA -> 蛋白质
    protein = analyzer.translate_sequence(original_dna)
    print(f"翻译为蛋白质: {protein}")

    # 反向翻译：蛋白质 -> DNA
    reconstructed_dna = analyzer.reverse_translate(protein, optimization='first', output_format='dna')
    print(f"反向翻译DNA: {reconstructed_dna}")

    # 再次正向翻译验证
    protein_check = analyzer.translate_sequence(reconstructed_dna)
    print(f"验证蛋白质: {protein_check}")

    assert protein == protein_check, "往返翻译后蛋白质应该相同"
    print("✓ 测试通过\n")


def test_get_codons_for_amino_acid():
    """测试获取氨基酸的所有密码子"""
    print("=" * 60)
    print("测试9: 获取氨基酸的所有密码子")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    test_cases = [
        ('M', 1, '甲硫氨酸（唯一密码子）'),
        ('W', 1, '色氨酸（唯一密码子）'),
        ('L', 6, '亮氨酸（6个密码子）'),
        ('S', 6, '丝氨酸（6个密码子）'),
        ('*', 3, '终止密码子（3个）'),
    ]

    for aa, expected_count, description in test_cases:
        codons = analyzer.get_codons_for_amino_acid(aa)
        print(f"{aa} ({description}): {codons}")
        assert len(codons) == expected_count, f"{aa}的密码子数量应该是{expected_count}"

    print("✓ 测试通过\n")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试10: 边界情况")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 空序列
    empty_rna = analyzer.reverse_translate("", optimization='first', output_format='rna')
    print(f"空序列: '' -> '{empty_rna}'")
    assert empty_rna == '', "空序列应该返回空字符串"

    # 单个氨基酸
    single_rna = analyzer.reverse_translate("M", optimization='first', output_format='rna')
    print(f"单个氨基酸: 'M' -> '{single_rna}'")
    assert len(single_rna) == 3, "单个氨基酸应该返回3个碱基"

    # 小写输入
    lower_rna = analyzer.reverse_translate("mkp", optimization='first', output_format='rna')
    upper_rna = analyzer.reverse_translate("MKP", optimization='first', output_format='rna')
    print(f"小写输入: 'mkp' -> '{lower_rna}'")
    print(f"大写输入: 'MKP' -> '{upper_rna}'")
    assert lower_rna == upper_rna, "大小写应该得到相同结果"

    # 无效氨基酸
    try:
        analyzer.reverse_translate("MXP", optimization='first', output_format='rna')
        print("✗ 应该抛出异常（无效氨基酸X）")
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")

    print("✓ 测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("反向翻译功能 - 完整测试套件")
    print("=" * 60 + "\n")

    tests = [
        test_basic_reverse_translation,
        test_optimization_strategies,
        test_output_formats,
        test_with_start_stop_codons,
        test_different_genetic_codes,
        test_convenience_function,
        test_special_amino_acids,
        test_round_trip_translation,
        test_get_codons_for_amino_acid,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ 测试出错: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


def show_usage_examples():
    """展示实际使用示例"""
    print("\n" + "=" * 60)
    print("反向翻译 - 实际使用示例")
    print("=" * 60 + "\n")

    print("示例1: 最简单的使用")
    print("-" * 40)
    protein = "MKP"
    rna = reverse_translate_protein(protein)
    print(f"蛋白质: {protein}")
    print(f"RNA:    {rna}\n")

    print("示例2: 指定密码子优化策略")
    print("-" * 40)
    protein = "MKPGF"
    rna_first = reverse_translate_protein(protein, optimization='first')
    rna_freq = reverse_translate_protein(protein, optimization='frequent')
    print(f"蛋白质:      {protein}")
    print(f"first策略:   {rna_first}")
    print(f"frequent策略: {rna_freq}\n")

    print("示例3: 生成完整的编码序列")
    print("-" * 40)
    protein = "KPGFWY"
    full_rna = reverse_translate_protein(protein, add_start_stop=True, optimization='first')
    print(f"蛋白质片段: {protein}")
    print(f"完整编码序列: {full_rna}")
    print(f"  起始: {full_rna[:3]}")
    print(f"  终止: {full_rna[-3:]}\n")

    print("示例4: 用于Evo 2模型")
    print("-" * 40)
    protein = "MKP"
    model_format = reverse_translate_protein(protein, output_format='model', optimization='first')
    print(f"蛋白质: {protein}")
    print(f"模型格式(T): {model_format}")
    print("（可直接用于Evo 2模型输入）\n")

    print("示例5: 使用类接口进行更复杂的操作")
    print("-" * 40)
    from dna_rna_converter import CodonAnalyzer, GeneticCodeType

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 查看某个氨基酸的所有可能密码子
    aa = 'L'
    codons = analyzer.get_codons_for_amino_acid(aa)
    print(f"氨基酸 {aa} 的所有密码子: {codons}")

    # 反向翻译
    protein = "MLKP"
    rna = analyzer.reverse_translate(protein, optimization='random', output_format='rna')
    print(f"反向翻译: {protein} -> {rna}\n")


if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()

    # 展示使用示例
    show_usage_examples()

    # 退出码
    sys.exit(0 if success else 1)
