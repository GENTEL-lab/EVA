"""
DNA/密码子到RNA转换工具

基于Evo 2模型的技术规范实现DNA与RNA序列的双向转换，
支持密码子一致性维护、多种遗传密码表、序列拼接和反向互补。

遵循KISS、YAGNI和SOLID原则：
- 单一职责：每个类只负责一项核心功能
- 简洁直接：避免过度设计
- 易于扩展：通过继承和组合支持新的遗传密码表
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
from pathlib import Path


class GeneticCodeType(Enum):
    """遗传密码表类型"""
    STANDARD = "standard"
    MYCOPLASMA = "mycoplasma"  # 支原体
    CILIATE = "ciliate"  # 纤毛虫


class GeneticCodeTable:
    """
    遗传密码表管理器
    单一职责：维护和查询不同物种的密码子-氨基酸映射关系
    """

    # 标准遗传密码表（使用T代替U，符合Evo 2规范）
    STANDARD_CODE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    # 支原体遗传密码（TGA编码色氨酸而非终止）
    MYCOPLASMA_CODE = STANDARD_CODE.copy()
    MYCOPLASMA_CODE['TGA'] = 'W'

    # 纤毛虫遗传密码（TAA和TAG编码谷氨酰胺）
    CILIATE_CODE = STANDARD_CODE.copy()
    CILIATE_CODE['TAA'] = 'Q'
    CILIATE_CODE['TAG'] = 'Q'

    @classmethod
    def get_code_table(cls, code_type: GeneticCodeType) -> Dict[str, str]:
        """获取指定类型的遗传密码表"""
        if code_type == GeneticCodeType.STANDARD:
            return cls.STANDARD_CODE
        elif code_type == GeneticCodeType.MYCOPLASMA:
            return cls.MYCOPLASMA_CODE
        elif code_type == GeneticCodeType.CILIATE:
            return cls.CILIATE_CODE
        else:
            raise ValueError(f"不支持的遗传密码类型: {code_type}")

    @classmethod
    def is_start_codon(cls, codon: str, code_type: GeneticCodeType = GeneticCodeType.STANDARD) -> bool:
        """判断是否为起始密码子（通常是ATG）"""
        return codon.upper() == 'ATG'

    @classmethod
    def is_stop_codon(cls, codon: str, code_type: GeneticCodeType = GeneticCodeType.STANDARD) -> bool:
        """判断是否为终止密码子（根据遗传密码类型）"""
        code_table = cls.get_code_table(code_type)
        return code_table.get(codon.upper(), '') == '*'


class SequenceConverter:
    """
    序列转换器基类
    单一职责：提供DNA/RNA碱基转换的核心功能
    """

    COMPLEMENT_MAP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    @staticmethod
    def dna_to_rna_format(sequence: str) -> str:
        """
        DNA序列转换为RNA格式（用于模型输入）
        按照Evo 2规范：保持T不变，因为模型统一使用DNA字母表
        """
        return sequence.upper().replace('U', 'T')

    @staticmethod
    def rna_format_to_rna(sequence: str) -> str:
        """
        模型输出格式转换为标准RNA序列
        按照Evo 2规范：将T替换为U
        """
        return sequence.upper().replace('T', 'U')

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """计算反向互补序列"""
        complement = ''.join(SequenceConverter.COMPLEMENT_MAP.get(base, base)
                            for base in sequence.upper())
        return complement[::-1]

    @staticmethod
    def validate_sequence(sequence: str, allow_special: bool = False) -> bool:
        """验证序列是否合法"""
        valid_bases = set('ATCG')
        if allow_special:
            valid_bases.update('#@')
        return all(base in valid_bases for base in sequence.upper())


class CodonAnalyzer:
    """
    密码子分析器
    单一职责：分析和验证密码子序列的一致性，支持正向翻译和反向翻译
    """

    def __init__(
        self,
        code_type: GeneticCodeType = GeneticCodeType.STANDARD,
        species_id: Optional[str] = None
    ):
        self.code_type = code_type
        self.code_table = GeneticCodeTable.get_code_table(code_type)
        # 构建反向密码子表（氨基酸 -> 密码子列表）
        self._reverse_table = self._build_reverse_table()

        # 物种特异密码子频率表（延迟加载）
        self.species_id = species_id
        self._frequency_table = None

    def _build_reverse_table(self) -> Dict[str, List[str]]:
        """构建反向密码子表：氨基酸 -> 密码子列表"""
        reverse_table = {}
        for codon, amino_acid in self.code_table.items():
            if amino_acid not in reverse_table:
                reverse_table[amino_acid] = []
            reverse_table[amino_acid].append(codon)
        return reverse_table

    def _get_frequency_table(self):
        """延迟加载密码子频率表"""
        if self._frequency_table is None and self.species_id is not None:
            try:
                from codon_frequency_manager import get_global_manager
                manager = get_global_manager()
                self._frequency_table = manager.get_species_table(self.species_id)
            except Exception as e:
                # 如果加载失败，记录警告但不中断
                print(f"警告：无法加载物种 {self.species_id} 的密码子频率表: {e}")
                self._frequency_table = None
        return self._frequency_table

    def get_codons_for_amino_acid(self, amino_acid: str) -> List[str]:
        """获取编码指定氨基酸的所有密码子"""
        return self._reverse_table.get(amino_acid.upper(), [])

    def translate_codon(self, codon: str) -> str:
        """翻译单个密码子为氨基酸"""
        if len(codon) != 3:
            raise ValueError(f"密码子长度必须为3，当前为{len(codon)}")
        return self.code_table.get(codon.upper(), 'X')  # X表示未知

    def translate_sequence(self, sequence: str, start_pos: int = 0) -> str:
        """
        翻译整个序列为氨基酸序列
        start_pos: 起始位置，用于指定阅读框（0, 1, 或 2）
        """
        if len(sequence) < 3:
            return ''

        amino_acids = []
        for i in range(start_pos, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if len(codon) == 3:
                amino_acids.append(self.translate_codon(codon))

        return ''.join(amino_acids)

    def check_mutation_type(self, original_codon: str, mutated_codon: str) -> str:
        """
        判断突变类型
        返回: 'synonymous'(同义), 'missense'(错义), 'nonsense'(无义)
        """
        original_aa = self.translate_codon(original_codon)
        mutated_aa = self.translate_codon(mutated_codon)

        if original_aa == mutated_aa:
            return 'synonymous'
        elif mutated_aa == '*':
            return 'nonsense'
        else:
            return 'missense'

    def validate_coding_sequence(self, sequence: str) -> Tuple[bool, str]:
        """
        验证编码序列的完整性
        返回: (是否有效, 错误信息)
        """
        if len(sequence) % 3 != 0:
            return False, f"序列长度{len(sequence)}不是3的倍数"

        if len(sequence) < 3:
            return False, "序列太短，至少需要一个密码子"

        # 检查起始密码子
        first_codon = sequence[:3].upper()
        if not GeneticCodeTable.is_start_codon(first_codon, self.code_type):
            return False, f"缺少起始密码子ATG，当前为{first_codon}"

        # 检查终止密码子
        last_codon = sequence[-3:].upper()
        if not GeneticCodeTable.is_stop_codon(last_codon, self.code_type):
            return False, f"缺少终止密码子，当前为{last_codon}"

        return True, "序列有效"

    def reverse_translate(
        self,
        protein_sequence: str,
        optimization: str = 'random',
        output_format: str = 'rna'
    ) -> str:
        """
        反向翻译：氨基酸序列 -> RNA/DNA序列

        Args:
            protein_sequence: 氨基酸序列（单字母代码）
            optimization: 密码子选择策略
                - 'random': 随机选择密码子
                - 'first': 总是选择第一个密码子
                - 'frequent': 选择常用密码子（简化版，选择字母序最小的）
                - 'weighted_random': 按物种频率加权随机选择（需要设置species_id）
                - 'most_frequent': 选择物种最常用密码子（需要设置species_id）
                - 'least_frequent': 选择物种最不常用密码子（需要设置species_id）
            output_format: 输出格式
                - 'rna': 标准RNA格式（U）
                - 'dna': DNA格式（T）
                - 'model': 模型格式（T，用于Evo 2）

        Returns:
            反向翻译得到的核酸序列
        """
        if not protein_sequence:
            return ''

        # 检查是否需要频率表
        frequency_based_strategies = ['weighted_random', 'most_frequent', 'least_frequent']
        if optimization in frequency_based_strategies:
            freq_table = self._get_frequency_table()
            if freq_table is None:
                raise ValueError(
                    f"策略 '{optimization}' 需要指定 species_id。"
                    f"请在创建 CodonAnalyzer 时传入 species_id 参数。"
                )

        codons = []
        for amino_acid in protein_sequence.upper():
            available_codons = self.get_codons_for_amino_acid(amino_acid)

            if not available_codons:
                raise ValueError(f"未知的氨基酸: {amino_acid}")

            # 根据优化策略选择密码子
            if optimization == 'random':
                selected_codon = random.choice(available_codons)
            elif optimization == 'first':
                selected_codon = available_codons[0]
            elif optimization == 'frequent':
                # 简化策略：选择字母序最小的（通常更常用）
                selected_codon = sorted(available_codons)[0]
            elif optimization in frequency_based_strategies:
                # 使用物种特异频率表
                freq_table = self._get_frequency_table()
                selected_codon = freq_table.select_codon(amino_acid, optimization)
            else:
                raise ValueError(f"不支持的优化策略: {optimization}")

            codons.append(selected_codon)

        # 拼接密码子
        sequence = ''.join(codons)

        # 根据输出格式转换
        if output_format == 'rna':
            return sequence.replace('T', 'U')
        elif output_format in ['dna', 'model']:
            return sequence
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

    def reverse_translate_with_start_stop(
        self,
        protein_sequence: str,
        optimization: str = 'random',
        output_format: str = 'rna'
    ) -> str:
        """
        反向翻译并自动添加起始和终止密码子

        Args:
            protein_sequence: 氨基酸序列（不包含起始M和终止*）
            optimization: 密码子选择策略
            output_format: 输出格式

        Returns:
            完整的编码序列（包含起始ATG和终止密码子）
        """
        # 移除可能存在的起始M和终止*
        protein = protein_sequence.strip().upper()
        if protein.startswith('M'):
            protein = protein[1:]
        if protein.endswith('*'):
            protein = protein[:-1]

        # 反向翻译主体序列
        if protein:
            body_sequence = self.reverse_translate(protein, optimization, 'dna')
        else:
            body_sequence = ''

        # 添加起始密码子ATG
        start_codon = 'ATG'

        # 添加终止密码子（根据遗传密码类型选择）
        stop_codons = self.get_codons_for_amino_acid('*')
        if optimization == 'random':
            stop_codon = random.choice(stop_codons)
        else:
            stop_codon = stop_codons[0]

        # 组合完整序列
        full_sequence = start_codon + body_sequence + stop_codon

        # 根据输出格式转换
        if output_format == 'rna':
            return full_sequence.replace('T', 'U')
        elif output_format in ['dna', 'model']:
            return full_sequence
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")


class TranscriptBuilder:
    """
    转录本构建器
    单一职责：处理序列拼接、增广和特殊符号插入
    """

    UNKNOWN_DISTANCE_SEPARATOR = '#'  # 连接距离未知的片段
    ADJACENT_SEPARATOR = '@'  # 连接相邻片段

    @staticmethod
    def join_exons(exons: List[str], use_adjacent: bool = True) -> str:
        """
        拼接外显子序列
        use_adjacent: True使用@（相邻），False使用#（距离未知）
        """
        separator = TranscriptBuilder.ADJACENT_SEPARATOR if use_adjacent else TranscriptBuilder.UNKNOWN_DISTANCE_SEPARATOR
        return separator.join(exons)

    @staticmethod
    def add_flanking_regions(exon: str, upstream: str = '', downstream: str = '') -> str:
        """为外显子添加侧翼序列"""
        parts = []
        if upstream:
            parts.append(upstream)
        parts.append(exon)
        if downstream:
            parts.append(downstream)
        return TranscriptBuilder.ADJACENT_SEPARATOR.join(parts)

    @staticmethod
    def build_augmented_transcript(
        exons: List[str],
        promoter: str = '',
        flanking_size: int = 32
    ) -> str:
        """
        构建增广转录本
        promoter: 启动子序列（建议1024bp）
        flanking_size: 每个外显子的侧翼大小（默认32bp）
        """
        parts = []

        # 添加启动子
        if promoter:
            parts.append(promoter)

        # 添加带侧翼的外显子
        for exon in exons:
            # 实际应用中，侧翼序列应从基因组中提取
            # 这里简化处理，仅拼接外显子本身
            parts.append(exon)

        return TranscriptBuilder.ADJACENT_SEPARATOR.join(parts)


class DNAToRNAConverter:
    """
    DNA到RNA转换器（主工具类）
    整合所有功能模块，提供统一的转换接口
    """

    def __init__(self, code_type: GeneticCodeType = GeneticCodeType.STANDARD):
        self.code_type = code_type
        self.sequence_converter = SequenceConverter()
        self.codon_analyzer = CodonAnalyzer(code_type)
        self.transcript_builder = TranscriptBuilder()

    def convert_dna_to_rna_format(
        self,
        dna_sequence: str,
        reverse_complement: bool = False,
        validate: bool = True
    ) -> str:
        """
        将DNA序列转换为模型可接受的RNA格式

        Args:
            dna_sequence: 输入的DNA序列
            reverse_complement: 是否取反向互补（模拟双链对称）
            validate: 是否验证序列合法性

        Returns:
            转换后的序列（T格式，用于模型输入）
        """
        if validate and not self.sequence_converter.validate_sequence(dna_sequence):
            raise ValueError("DNA序列包含非法字符")

        sequence = dna_sequence.upper()

        if reverse_complement:
            sequence = self.sequence_converter.reverse_complement(sequence)

        # 按照Evo 2规范，保持T不变
        return self.sequence_converter.dna_to_rna_format(sequence)

    def convert_model_output_to_rna(self, model_output: str) -> str:
        """
        将模型输出转换为标准RNA序列

        Args:
            model_output: 模型生成的序列（T格式）

        Returns:
            标准RNA序列（U格式）
        """
        return self.sequence_converter.rna_format_to_rna(model_output)

    def process_transcript(
        self,
        exons: List[str],
        promoter: str = '',
        add_flanking: bool = False,
        flanking_size: int = 32,
        to_rna_format: bool = True
    ) -> str:
        """
        处理完整的转录本

        Args:
            exons: 外显子序列列表
            promoter: 启动子序列
            add_flanking: 是否添加侧翼序列
            flanking_size: 侧翼大小
            to_rna_format: 是否转换为RNA格式（T表示）

        Returns:
            处理后的转录本序列
        """
        if add_flanking:
            transcript = self.transcript_builder.build_augmented_transcript(
                exons, promoter, flanking_size
            )
        else:
            transcript = self.transcript_builder.join_exons(exons)
            if promoter:
                transcript = promoter + TranscriptBuilder.ADJACENT_SEPARATOR + transcript

        if to_rna_format:
            transcript = self.sequence_converter.dna_to_rna_format(transcript)

        return transcript

    def analyze_coding_sequence(
        self,
        sequence: str,
        check_validity: bool = True
    ) -> Dict:
        """
        分析编码序列

        Returns:
            包含翻译结果、有效性等信息的字典
        """
        # 转换为统一格式
        seq = self.sequence_converter.dna_to_rna_format(sequence)

        result = {
            'sequence': seq,
            'length': len(seq),
            'codon_count': len(seq) // 3,
        }

        if check_validity:
            is_valid, message = self.codon_analyzer.validate_coding_sequence(seq)
            result['is_valid'] = is_valid
            result['validation_message'] = message

        if len(seq) >= 3:
            result['translation'] = self.codon_analyzer.translate_sequence(seq)
            result['start_codon'] = seq[:3]
            result['stop_codon'] = seq[-3:] if len(seq) >= 3 else ''

        return result

    def generate_training_sample(
        self,
        sequence: str,
        reverse_complement_prob: float = 0.5
    ) -> str:
        """
        生成训练样本（按照Evo 2的50%反向互补策略）

        Args:
            sequence: 输入序列
            reverse_complement_prob: 取反向互补的概率

        Returns:
            处理后的训练样本
        """
        should_reverse = random.random() < reverse_complement_prob
        return self.convert_dna_to_rna_format(
            sequence,
            reverse_complement=should_reverse
        )


# 便捷函数
def quick_dna_to_rna(dna: str, output_format: str = 'standard') -> str:
    """
    快速转换DNA到RNA

    Args:
        dna: DNA序列
        output_format: 'standard'(标准RNA，U) 或 'model'(模型格式，T)

    Returns:
        转换后的RNA序列
    """
    converter = DNAToRNAConverter()

    if output_format == 'model':
        return converter.convert_dna_to_rna_format(dna)
    else:
        model_format = converter.convert_dna_to_rna_format(dna)
        return converter.convert_model_output_to_rna(model_format)


def analyze_codon_sequence(sequence: str, code_type: str = 'standard') -> Dict:
    """
    快速分析密码子序列

    Args:
        sequence: DNA或RNA序列
        code_type: 'standard', 'mycoplasma', 或 'ciliate'

    Returns:
        分析结果字典
    """
    code_map = {
        'standard': GeneticCodeType.STANDARD,
        'mycoplasma': GeneticCodeType.MYCOPLASMA,
        'ciliate': GeneticCodeType.CILIATE,
    }

    converter = DNAToRNAConverter(code_map.get(code_type, GeneticCodeType.STANDARD))
    return converter.analyze_coding_sequence(sequence)


def reverse_translate_protein(
    protein: str,
    code_type: str = 'standard',
    optimization: str = 'random',
    output_format: str = 'rna',
    add_start_stop: bool = False,
    species_id: Optional[str] = None
) -> str:
    """
    快速反向翻译：氨基酸序列 -> RNA序列

    Args:
        protein: 氨基酸序列（单字母代码，如 "MKP"）
        code_type: 遗传密码类型 ('standard', 'mycoplasma', 'ciliate')
        optimization: 密码子选择策略
            - 'random': 随机选择密码子
            - 'first': 总是选择第一个密码子
            - 'frequent': 选择常用密码子
            - 'weighted_random': 按物种频率加权随机选择（需要species_id）
            - 'most_frequent': 选择物种最常用密码子（需要species_id）
            - 'least_frequent': 选择物种最不常用密码子（需要species_id）
        output_format: 输出格式
            - 'rna': 标准RNA格式（U）
            - 'dna': DNA格式（T）
            - 'model': 模型格式（T，用于Evo 2）
        add_start_stop: 是否自动添加起始和终止密码子
        species_id: 物种标识符（如 'homo_sapiens', 'escherichia_coli'）
                   用于频率加权策略

    Returns:
        反向翻译得到的RNA/DNA序列

    Examples:
        >>> reverse_translate_protein("MKP")
        'AUGAAACCC'  # 结果会因random而变化

        >>> reverse_translate_protein("KP", add_start_stop=True)
        'AUGAAACCCUAA'  # 自动添加起始ATG和终止UAA

        >>> reverse_translate_protein("MKP", optimization='most_frequent', species_id='homo_sapiens')
        'AUGAAGCCU'  # 使用人类最常用密码子
    """
    code_map = {
        'standard': GeneticCodeType.STANDARD,
        'mycoplasma': GeneticCodeType.MYCOPLASMA,
        'ciliate': GeneticCodeType.CILIATE,
    }

    analyzer = CodonAnalyzer(
        code_map.get(code_type, GeneticCodeType.STANDARD),
        species_id=species_id
    )

    if add_start_stop:
        return analyzer.reverse_translate_with_start_stop(
            protein, optimization, output_format
        )
    else:
        return analyzer.reverse_translate(
            protein, optimization, output_format
        )
