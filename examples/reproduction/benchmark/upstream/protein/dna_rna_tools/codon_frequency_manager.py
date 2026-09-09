"""
密码子使用频率管理器

基于物种特异的密码子使用频率表，提供加权密码子选择策略。
遵循SOLID原则：
- 单一职责：专注于密码子频率的加载、归一化和选择
- 开放封闭：易于扩展新物种和新的选择策略
- 接口隔离：提供清晰的频率查询和选择接口
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random


class CodonFrequencyTable:
    """
    密码子频率表
    单一职责：存储和查询单个物种的密码子使用频率
    """

    def __init__(
        self,
        species_id: str,
        name: str,
        taxid: int,
        genetic_code: str,
        codon_frequencies: Dict[str, float],
        description: str = ""
    ):
        self.species_id = species_id
        self.name = name
        self.taxid = taxid
        self.genetic_code = genetic_code
        self.description = description

        # 原始频率（每千个密码子）
        self.raw_frequencies = codon_frequencies.copy()

        # 归一化频率（按氨基酸分组，和为1）
        self.normalized_frequencies = self._normalize_frequencies()

    def _normalize_frequencies(self) -> Dict[str, Dict[str, float]]:
        """
        按氨基酸归一化密码子频率
        返回: {氨基酸: {密码子: 归一化频率}}
        """
        from dna_rna_converter import GeneticCodeTable, GeneticCodeType

        # 获取遗传密码表
        code_type_map = {
            'standard': GeneticCodeType.STANDARD,
            'mycoplasma': GeneticCodeType.MYCOPLASMA,
            'ciliate': GeneticCodeType.CILIATE
        }
        code_type = code_type_map.get(self.genetic_code, GeneticCodeType.STANDARD)
        code_table = GeneticCodeTable.get_code_table(code_type)

        # 按氨基酸分组
        aa_codons = {}
        for codon, freq in self.raw_frequencies.items():
            aa = code_table.get(codon, 'X')
            if aa not in aa_codons:
                aa_codons[aa] = {}
            aa_codons[aa][codon] = freq

        # 归一化每个氨基酸的密码子频率
        normalized = {}
        for aa, codons in aa_codons.items():
            total = sum(codons.values())
            if total > 0:
                normalized[aa] = {
                    codon: freq / total
                    for codon, freq in codons.items()
                }
            else:
                # 如果总频率为0，均匀分布
                n = len(codons)
                normalized[aa] = {
                    codon: 1.0 / n
                    for codon in codons
                }

        return normalized

    def get_codon_frequency(self, codon: str, normalized: bool = True) -> float:
        """
        获取单个密码子的频率

        Args:
            codon: 密码子序列
            normalized: 是否返回归一化频率（按氨基酸）

        Returns:
            密码子频率
        """
        codon = codon.upper()

        if normalized:
            # 查找归一化频率
            from dna_rna_converter import GeneticCodeTable, GeneticCodeType
            code_type_map = {
                'standard': GeneticCodeType.STANDARD,
                'mycoplasma': GeneticCodeType.MYCOPLASMA,
                'ciliate': GeneticCodeType.CILIATE
            }
            code_type = code_type_map.get(self.genetic_code, GeneticCodeType.STANDARD)
            code_table = GeneticCodeTable.get_code_table(code_type)
            aa = code_table.get(codon, 'X')

            return self.normalized_frequencies.get(aa, {}).get(codon, 0.0)
        else:
            return self.raw_frequencies.get(codon, 0.0)

    def get_codons_for_amino_acid(
        self,
        amino_acid: str,
        sorted_by_frequency: bool = True
    ) -> List[Tuple[str, float]]:
        """
        获取编码指定氨基酸的所有密码子及其频率

        Args:
            amino_acid: 氨基酸单字母代码
            sorted_by_frequency: 是否按频率降序排序

        Returns:
            [(密码子, 归一化频率), ...] 列表
        """
        aa = amino_acid.upper()
        codons_dict = self.normalized_frequencies.get(aa, {})

        codons_list = list(codons_dict.items())

        if sorted_by_frequency:
            codons_list.sort(key=lambda x: x[1], reverse=True)

        return codons_list

    def select_codon(
        self,
        amino_acid: str,
        strategy: str = 'weighted_random'
    ) -> str:
        """
        根据策略选择密码子

        Args:
            amino_acid: 氨基酸单字母代码
            strategy: 选择策略
                - 'weighted_random': 按频率加权随机选择
                - 'most_frequent': 选择最常用的密码子
                - 'least_frequent': 选择最不常用的密码子

        Returns:
            选中的密码子
        """
        codons = self.get_codons_for_amino_acid(amino_acid, sorted_by_frequency=True)

        if not codons:
            raise ValueError(f"未找到氨基酸 {amino_acid} 的密码子")

        if strategy == 'most_frequent':
            return codons[0][0]
        elif strategy == 'least_frequent':
            return codons[-1][0]
        elif strategy == 'weighted_random':
            # 加权随机选择
            codons_only = [c[0] for c in codons]
            weights = [c[1] for c in codons]
            return random.choices(codons_only, weights=weights, k=1)[0]
        else:
            raise ValueError(f"不支持的选择策略: {strategy}")


class CodonFrequencyManager:
    """
    密码子频率管理器
    单一职责：管理多物种的密码子频率表，提供统一的访问接口
    """

    def __init__(self, data_file: Optional[str] = None):
        """
        初始化管理器

        Args:
            data_file: 密码子频率数据文件路径（JSON格式）
                      如果为None，使用默认路径
        """
        if data_file is None:
            # 使用默认路径（与本模块同目录）
            module_dir = Path(__file__).parent
            data_file = module_dir / "codon_usage_tables.json"

        self.data_file = Path(data_file)
        self.species_tables: Dict[str, CodonFrequencyTable] = {}
        self.metadata: Dict = {}

        # 加载数据
        self._load_data()

    def _load_data(self):
        """从JSON文件加载密码子频率数据"""
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"密码子频率数据文件不存在: {self.data_file}"
            )

        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})

        # 加载每个物种的频率表
        species_data = data.get('species', {})
        for species_id, species_info in species_data.items():
            table = CodonFrequencyTable(
                species_id=species_id,
                name=species_info['name'],
                taxid=species_info['taxid'],
                genetic_code=species_info['genetic_code'],
                codon_frequencies=species_info['codons'],
                description=species_info.get('description', '')
            )
            self.species_tables[species_id] = table

    def list_species(self) -> List[Dict[str, any]]:
        """
        列出所有可用的物种

        Returns:
            物种信息列表
        """
        return [
            {
                'species_id': species_id,
                'name': table.name,
                'taxid': table.taxid,
                'genetic_code': table.genetic_code,
                'description': table.description
            }
            for species_id, table in self.species_tables.items()
        ]

    def get_species_table(self, species_id: str) -> CodonFrequencyTable:
        """
        获取指定物种的频率表

        Args:
            species_id: 物种标识符（如 'homo_sapiens'）

        Returns:
            密码子频率表对象
        """
        if species_id not in self.species_tables:
            available = ', '.join(self.species_tables.keys())
            raise ValueError(
                f"未找到物种 '{species_id}'。可用物种: {available}"
            )

        return self.species_tables[species_id]

    def has_species(self, species_id: str) -> bool:
        """检查是否存在指定物种的数据"""
        return species_id in self.species_tables

    def add_custom_species(
        self,
        species_id: str,
        name: str,
        taxid: int,
        genetic_code: str,
        codon_frequencies: Dict[str, float],
        description: str = ""
    ):
        """
        添加自定义物种的密码子频率表

        Args:
            species_id: 物种标识符
            name: 物种名称
            taxid: NCBI taxonomy ID
            genetic_code: 遗传密码类型
            codon_frequencies: 密码子频率字典
            description: 描述信息
        """
        table = CodonFrequencyTable(
            species_id=species_id,
            name=name,
            taxid=taxid,
            genetic_code=genetic_code,
            codon_frequencies=codon_frequencies,
            description=description
        )
        self.species_tables[species_id] = table

    def compare_codon_usage(
        self,
        species_ids: List[str],
        amino_acid: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        比较多个物种对同一氨基酸的密码子使用偏好

        Args:
            species_ids: 物种标识符列表
            amino_acid: 氨基酸单字母代码

        Returns:
            {物种ID: [(密码子, 频率), ...]}
        """
        result = {}
        for species_id in species_ids:
            table = self.get_species_table(species_id)
            result[species_id] = table.get_codons_for_amino_acid(
                amino_acid,
                sorted_by_frequency=True
            )
        return result


# 全局单例管理器（延迟初始化）
_global_manager: Optional[CodonFrequencyManager] = None


def get_global_manager() -> CodonFrequencyManager:
    """获取全局密码子频率管理器（单例模式）"""
    global _global_manager
    if _global_manager is None:
        _global_manager = CodonFrequencyManager()
    return _global_manager


def list_available_species() -> List[Dict[str, any]]:
    """快速列出所有可用物种"""
    manager = get_global_manager()
    return manager.list_species()


def get_codon_frequency(
    species_id: str,
    codon: str,
    normalized: bool = True
) -> float:
    """
    快速查询密码子频率

    Args:
        species_id: 物种标识符
        codon: 密码子序列
        normalized: 是否返回归一化频率

    Returns:
        密码子频率
    """
    manager = get_global_manager()
    table = manager.get_species_table(species_id)
    return table.get_codon_frequency(codon, normalized)


def select_codon_by_frequency(
    species_id: str,
    amino_acid: str,
    strategy: str = 'weighted_random'
) -> str:
    """
    根据物种频率选择密码子

    Args:
        species_id: 物种标识符
        amino_acid: 氨基酸单字母代码
        strategy: 选择策略

    Returns:
        选中的密码子
    """
    manager = get_global_manager()
    table = manager.get_species_table(species_id)
    return table.select_codon(amino_acid, strategy)
