"""
DNA/RNA转换工具的测试和使用示例

演示如何使用dna_rna_converter模块进行各种序列转换和分析操作
"""

import sys
from dna_rna_converter import (
    DNAToRNAConverter,
    GeneticCodeType,
    CodonAnalyzer,
    TranscriptBuilder,
    SequenceConverter,
    quick_dna_to_rna,
    analyze_codon_sequence
)


def test_basic_conversion():
    """测试基本的DNA到RNA转换"""
    print("=" * 60)
    print("测试1: 基本DNA到RNA转换")
    print("=" * 60)

    converter = DNAToRNAConverter()

    # 测试DNA序列
    dna = "ATGCGATCGATCG"

    # 转换为模型格式（保持T）
    model_format = converter.convert_dna_to_rna_format(dna)
    print(f"原始DNA序列:     {dna}")
    print(f"模型格式(T):     {model_format}")

    # 转换为标准RNA格式（T->U）
    rna = converter.convert_model_output_to_rna(model_format)
    print(f"标准RNA格式(U):  {rna}")

    # 使用便捷函数
    rna_quick = quick_dna_to_rna(dna, output_format='standard')
    print(f"快速转换结果:    {rna_quick}")

    assert rna == rna_quick, "转换结果应该一致"
    print("✓ 测试通过\n")


def test_reverse_complement():
    """测试反向互补功能"""
    print("=" * 60)
    print("测试2: 反向互补转换")
    print("=" * 60)

    converter = DNAToRNAConverter()

    dna = "ATGCGATCG"

    # 正向序列
    forward = converter.convert_dna_to_rna_format(dna, reverse_complement=False)
    print(f"正向序列:        {forward}")

    # 反向互补序列
    reverse = converter.convert_dna_to_rna_format(dna, reverse_complement=True)
    print(f"反向互补序列:    {reverse}")

    # 验证反向互补的正确性
    expected_rc = "CGATCGCAT"  # ATGCGATCG的反向互补
    assert reverse == expected_rc, f"反向互补错误: 期望{expected_rc}, 得到{reverse}"
    print("✓ 测试通过\n")


def test_codon_analysis():
    """测试密码子分析功能"""
    print("=" * 60)
    print("测试3: 密码子分析")
    print("=" * 60)

    # 标准遗传密码
    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 测试单个密码子翻译
    test_codons = [
        ('ATG', 'M', '起始密码子'),
        ('TAA', '*', '终止密码子'),
        ('TGG', 'W', '色氨酸'),
        ('TTT', 'F', '苯丙氨酸'),
    ]

    print("单个密码子翻译:")
    for codon, expected_aa, description in test_codons:
        aa = analyzer.translate_codon(codon)
        status = "✓" if aa == expected_aa else "✗"
        print(f"  {status} {codon} -> {aa} ({description})")
        assert aa == expected_aa, f"翻译错误: {codon} 应该是 {expected_aa}"

    # 测试完整序列翻译
    coding_seq = "ATGTTTTGGTAA"  # ATG(M) TTT(F) TGG(W) TAA(*)
    protein = analyzer.translate_sequence(coding_seq)
    expected_protein = "MFW*"
    print(f"\n完整序列翻译:")
    print(f"  DNA序列: {coding_seq}")
    print(f"  蛋白质:  {protein}")
    assert protein == expected_protein, f"翻译错误: 期望{expected_protein}, 得到{protein}"
    print("✓ 测试通过\n")


def test_mutation_types():
    """测试突变类型判断"""
    print("=" * 60)
    print("测试4: 突变类型判断")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    test_cases = [
        ('TTT', 'TTC', 'synonymous', '同义突变(都编码F)'),
        ('TTT', 'TAT', 'missense', '错义突变(F->Y)'),
        ('TGG', 'TAG', 'nonsense', '无义突变(W->*)'),
    ]

    for original, mutated, expected_type, description in test_cases:
        mutation_type = analyzer.check_mutation_type(original, mutated)
        status = "✓" if mutation_type == expected_type else "✗"
        print(f"  {status} {original}->{mutated}: {mutation_type} ({description})")
        assert mutation_type == expected_type, f"突变类型判断错误"

    print("✓ 测试通过\n")


def test_genetic_code_variants():
    """测试不同遗传密码表"""
    print("=" * 60)
    print("测试5: 不同遗传密码表")
    print("=" * 60)

    # TGA在不同遗传密码中的含义
    codon = "TGA"

    # 标准遗传密码: TGA = 终止
    standard = CodonAnalyzer(GeneticCodeType.STANDARD)
    standard_aa = standard.translate_codon(codon)
    print(f"标准遗传密码:   {codon} -> {standard_aa} (终止)")
    assert standard_aa == '*', "标准密码表中TGA应该是终止密码子"

    # 支原体遗传密码: TGA = 色氨酸
    mycoplasma = CodonAnalyzer(GeneticCodeType.MYCOPLASMA)
    mycoplasma_aa = mycoplasma.translate_codon(codon)
    print(f"支原体遗传密码: {codon} -> {mycoplasma_aa} (色氨酸)")
    assert mycoplasma_aa == 'W', "支原体密码表中TGA应该编码色氨酸"

    # TAA在纤毛虫中的含义
    codon2 = "TAA"
    ciliate = CodonAnalyzer(GeneticCodeType.CILIATE)
    ciliate_aa = ciliate.translate_codon(codon2)
    print(f"纤毛虫遗传密码: {codon2} -> {ciliate_aa} (谷氨酰胺)")
    assert ciliate_aa == 'Q', "纤毛虫密码表中TAA应该编码谷氨酰胺"

    print("✓ 测试通过\n")


def test_coding_sequence_validation():
    """测试编码序列验证"""
    print("=" * 60)
    print("测试6: 编码序列验证")
    print("=" * 60)

    analyzer = CodonAnalyzer(GeneticCodeType.STANDARD)

    # 有效的编码序列
    valid_seq = "ATGTTTTGGTAA"  # 起始ATG + 编码区 + 终止TAA
    is_valid, message = analyzer.validate_coding_sequence(valid_seq)
    print(f"有效序列: {valid_seq}")
    print(f"  结果: {is_valid}, {message}")
    assert is_valid, "应该是有效的编码序列"

    # 无效序列：缺少起始密码子
    invalid_seq1 = "TTGTTTTGGTAA"
    is_valid1, message1 = analyzer.validate_coding_sequence(invalid_seq1)
    print(f"\n无起始密码子: {invalid_seq1}")
    print(f"  结果: {is_valid1}, {message1}")
    assert not is_valid1, "应该检测到缺少起始密码子"

    # 无效序列：长度不是3的倍数
    invalid_seq2 = "ATGTTTTGG"
    is_valid2, message2 = analyzer.validate_coding_sequence(invalid_seq2)
    print(f"\n长度错误: {invalid_seq2}")
    print(f"  结果: {is_valid2}, {message2}")
    assert not is_valid2, "应该检测到长度错误"

    print("✓ 测试通过\n")


def test_transcript_building():
    """测试转录本构建"""
    print("=" * 60)
    print("测试7: 转录本构建与拼接")
    print("=" * 60)

    builder = TranscriptBuilder()

    # 模拟3个外显子
    exons = ["ATGCCC", "GGGTTT", "AAATAG"]

    # 使用@连接（相邻片段）
    transcript_adjacent = builder.join_exons(exons, use_adjacent=True)
    print(f"相邻拼接(@): {transcript_adjacent}")
    assert '@' in transcript_adjacent, "应该包含@符号"

    # 使用#连接（距离未知）
    transcript_unknown = builder.join_exons(exons, use_adjacent=False)
    print(f"距离未知(#): {transcript_unknown}")
    assert '#' in transcript_unknown, "应该包含#符号"

    # 添加启动子
    promoter = "A" * 20  # 简化的启动子序列
    augmented = builder.build_augmented_transcript(exons, promoter=promoter)
    print(f"\n增广转录本（含启动子）:")
    print(f"  长度: {len(augmented)}")
    print(f"  前30个字符: {augmented[:30]}...")
    assert augmented.startswith(promoter), "应该以启动子开始"

    print("✓ 测试通过\n")


def test_complete_workflow():
    """测试完整的工作流程"""
    print("=" * 60)
    print("测试8: 完整工作流程（模拟Evo 2使用场景）")
    print("=" * 60)

    converter = DNAToRNAConverter(GeneticCodeType.STANDARD)

    # 场景：处理一个基因的多个外显子
    exons = [
        "ATGAAACCC",  # 外显子1: ATG AAA CCC
        "GGGTTTCCC",  # 外显子2: GGG TTT CCC
        "AAATAG"      # 外显子3: AAA TAG(终止)
    ]

    promoter = "TATA" + "A" * 100  # 简化的启动子

    print("步骤1: 构建转录本")
    transcript = converter.process_transcript(
        exons=exons,
        promoter=promoter,
        add_flanking=False,
        to_rna_format=True
    )
    print(f"  转录本长度: {len(transcript)}")
    print(f"  前50个字符: {transcript[:50]}...")

    print("\n步骤2: 分析编码序列")
    # 提取编码区（去除启动子和特殊符号）
    coding_seq = ''.join(exons)
    analysis = converter.analyze_coding_sequence(coding_seq)
    print(f"  序列长度: {analysis['length']}")
    print(f"  密码子数: {analysis['codon_count']}")
    print(f"  起始密码子: {analysis['start_codon']}")
    print(f"  终止密码子: {analysis['stop_codon']}")
    print(f"  翻译结果: {analysis['translation']}")
    print(f"  序列有效性: {analysis['is_valid']}")

    print("\n步骤3: 生成训练样本（50%反向互补）")
    for i in range(5):
        sample = converter.generate_training_sample(coding_seq)
        is_reversed = sample != coding_seq.upper()
        print(f"  样本{i+1}: {'反向互补' if is_reversed else '正向'} - {sample[:20]}...")

    print("\n步骤4: 模型输出转换为标准RNA")
    model_output = transcript  # 模拟模型输出（T格式）
    standard_rna = converter.convert_model_output_to_rna(model_output)
    print(f"  模型输出(T): {model_output[:30]}...")
    print(f"  标准RNA(U): {standard_rna[:30]}...")
    assert 'U' in standard_rna and 'T' not in standard_rna, "应该将T转换为U"

    print("✓ 测试通过\n")


def test_convenience_functions():
    """测试便捷函数"""
    print("=" * 60)
    print("测试9: 便捷函数")
    print("=" * 60)

    dna = "ATGAAACCCGGGTAA"

    # 快速转换
    rna_standard = quick_dna_to_rna(dna, output_format='standard')
    rna_model = quick_dna_to_rna(dna, output_format='model')

    print(f"原始DNA:         {dna}")
    print(f"标准RNA格式(U): {rna_standard}")
    print(f"模型格式(T):     {rna_model}")

    assert 'U' in rna_standard, "标准格式应该包含U"
    assert 'T' in rna_model and 'U' not in rna_model, "模型格式应该只包含T"

    # 快速分析
    print("\n密码子序列分析:")
    analysis = analyze_codon_sequence(dna, code_type='standard')
    print(f"  长度: {analysis['length']}")
    print(f"  翻译: {analysis['translation']}")
    print(f"  有效性: {analysis['is_valid']}")

    print("✓ 测试通过\n")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试10: 边界情况处理")
    print("=" * 60)

    converter = DNAToRNAConverter()

    # 空序列
    try:
        result = converter.convert_dna_to_rna_format("")
        print(f"✓ 空序列处理: {result}")
    except Exception as e:
        print(f"✗ 空序列处理失败: {e}")

    # 包含小写字母
    mixed_case = "AtGcGaTcG"
    result = converter.convert_dna_to_rna_format(mixed_case)
    print(f"✓ 混合大小写: {mixed_case} -> {result}")
    assert result == "ATGCGATCG", "应该转换为大写"

    # 非3倍数长度的序列
    non_triplet = "ATGCC"
    analyzer = CodonAnalyzer()
    translation = analyzer.translate_sequence(non_triplet)
    print(f"✓ 非3倍数序列翻译: {non_triplet} -> {translation}")

    # 包含特殊符号的序列
    with_special = "ATG@CCC#GGG"
    try:
        result = converter.convert_dna_to_rna_format(with_special, validate=False)
        print(f"✓ 特殊符号序列: {with_special} -> {result}")
    except Exception as e:
        print(f"✗ 特殊符号处理失败: {e}")

    print("✓ 测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DNA/RNA转换工具 - 完整测试套件")
    print("=" * 60 + "\n")

    tests = [
        test_basic_conversion,
        test_reverse_complement,
        test_codon_analysis,
        test_mutation_types,
        test_genetic_code_variants,
        test_coding_sequence_validation,
        test_transcript_building,
        test_complete_workflow,
        test_convenience_functions,
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
    print("实际使用示例")
    print("=" * 60 + "\n")

    print("示例1: 简单的DNA到RNA转换")
    print("-" * 40)
    dna = "ATGCGATCGATCG"
    rna = quick_dna_to_rna(dna)
    print(f"DNA: {dna}")
    print(f"RNA: {rna}\n")

    print("示例2: 分析基因编码序列")
    print("-" * 40)
    gene = "ATGAAACCCGGGTTTAAATAG"
    result = analyze_codon_sequence(gene)
    print(f"基因序列: {gene}")
    print(f"翻译结果: {result['translation']}")
    print(f"密码子数: {result['codon_count']}")
    print(f"序列有效: {result['is_valid']}\n")

    print("示例3: 处理多外显子转录本")
    print("-" * 40)
    converter = DNAToRNAConverter()
    exons = ["ATGCCC", "GGGTTT", "AAATAG"]
    transcript = converter.process_transcript(exons)
    print(f"外显子: {exons}")
    print(f"转录本: {transcript}\n")

    print("示例4: 使用不同遗传密码表")
    print("-" * 40)
    seq = "ATGTGACCC"
    for code_type in ['standard', 'mycoplasma']:
        result = analyze_codon_sequence(seq, code_type=code_type)
        print(f"{code_type}: {seq} -> {result['translation']}")
    print()


if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()

    # 展示使用示例
    show_usage_examples()

    # 退出码
    sys.exit(0 if success else 1)
