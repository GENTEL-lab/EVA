# ncRNA_13datasets_spearman.csv 更新记录

## 2024-03-14 更新

### 背景
原有的 CSV 中有部分数据集（Domingo_2018_tRNA, Li_2016_tRNA, Janzen_2022_fam1b1_ribozyme, Kobori_2015_ribozyme_j12）的 Spearman 值是用 ncRNA/score 目录下的分数重新计算的，与原始 CSV 的来源不一致。

### 修改内容

将以下 4 个数据集的 `eva_1.4b_score` 更新为使用 `scores_original_data_wotag` 计算的 Spearman 值：

| 数据集 | 旧值 (ncRNA/score) | 新值 (scores_original_data_wotag) | 样本数 |
|--------|-------------------|----------------------------------|--------|
| Domingo_2018_tRNA | 0.225621 | **0.265759** | 4176 |
| Li_2016_tRNA | 0.353686 | **0.467907** | 65537 |
| Janzen_2022_fam1b1_ribozyme | -0.177129 | **-0.129395** | 1954 |
| Kobori_2015_ribozyme_j12 | -0.143593 | **-0.077802** | 256 |

### 数据来源

- **分数**: `/data/yanjie_huang/rna_benchmark/z_rnagym_70/scores_original_data_wotag/`
- **真实标签**: `/data/yanjie_huang/rna_benchmark/benchmark_rnagen/data/ground_truth_original/`

### 计算方法
```python
from scipy.stats import spearmanr

# 分数文件结构 (scores_original_data_wotag/{type}/{dataset}.json)
scores = [s['log_likelihood'] for s in score_data['scores']]
rho, _ = spearmanr(scores, intensities)
```

### 相关文件

- CSV 文件: `data/github_result/ncRNA_13datasets_spearman.csv`
- 备份文件: `data/github_result/ncRNA_13datasets_spearman_OLD.csv`
