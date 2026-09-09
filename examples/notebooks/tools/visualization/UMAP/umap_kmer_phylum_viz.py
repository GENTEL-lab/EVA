#!/usr/bin/env python3
"""
UMAP可视化脚本 - Phylum级别分类 + 缓存支持
"""
import sys
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
import os
import json
import argparse
from sklearn.preprocessing import normalize

# ======================================================================
# 配置参数
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# 路径设置（均相对于脚本所在目录）
mapping_file = os.path.join(script_dir, "taxid_phylum_mapping.json")
cache_dir = os.path.join(script_dir, "cache")
output_dir = os.path.join(script_dir, "pic")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# K-mer CSV 数据目录（可通过命令行 --data-dir 指定）
DEFAULT_DATA_DIR = os.path.join(script_dir, "data")

# UMAP参数（优化以促进三域分离和圆形分布）
UMAP_PARAMS = {
    'n_neighbors': 30,      # 增大以看到全局结构
    'min_dist': 0.8,        # 增大让点更分散
    'spread': 1.5,          # 调整整体形状
    'metric': 'cosine',
    'random_state': 42,
    'n_jobs': -1
}

# 采样设置（默认使用全部数据，可通过命令行参数覆盖）
DEFAULT_MAX_SAMPLES = None  # None表示使用全部数据
MAX_SAMPLES = DEFAULT_MAX_SAMPLES  # 当前使用的采样数量

# ======================================================================
# Domain Scaling方案配置
# ======================================================================
# 预定义的scaling方案（可以快速切换）
SCALING_SCHEMES = {
    'conservative': {
        'Archaea': 3,
        'Bacteria': 1.5,
        'Eukaryota': 1,
        'Unknown': 1
    },
    'moderate': {
        'Archaea': 5,
        'Bacteria': 2,
        'Eukaryota': 1,
        'Unknown': 1
    },
    'aggressive': {
        'Archaea': 8,
        'Bacteria': 3,
        'Eukaryota': 0.5,
        'Unknown': 1
    },
    'extreme': {
        'Archaea': 10,
        'Bacteria': 4,
        'Eukaryota': 0.3,
        'Unknown': 1
    }
}

# 默认使用的scaling方案
DEFAULT_SCALING_SCHEME = 'moderate'

# ======================================================================
# Phylum配色方案（新配色：Bacteria=绿，Eukaryota=蓝，Archaea=红）
# 使用更鲜艳饱和的颜色
# ======================================================================
PHYLUM_COLORS = {
    # Bacteria - 绿色系（更鲜艳）
    'Bacteria/Pseudomonadota': '#00AA00',      # 鲜艳深绿
    'Bacteria/Bacillota': '#33CC33',            # 鲜艳中绿
    'Bacteria/Bacteroidota': '#66FF66',         # 鲜艳浅绿
    'Bacteria/Actinomycetota': '#99FF99',       # 鲜艳极浅绿
    'Bacteria/Unknown': '#E8F5E9',              # 最浅绿（统一）
    'Bacteria/Other': '#E8F5E9',                # 最浅绿（统一）

    # Eukaryota - 蓝色系（更鲜艳）
    'Eukaryota/Ascomycota': '#0000DD',          # 鲜艳深蓝
    'Eukaryota/Basidiomycota': '#0033FF',       # 鲜艳中蓝
    'Eukaryota/Chordata': '#3366FF',            # 鲜艳皇家蓝
    'Eukaryota/Arthropoda': '#6699FF',          # 鲜艳矢车菊蓝
    'Eukaryota/Streptophyta': '#99CCFF',        # 鲜艳天空蓝
    'Eukaryota/Viridiplantae': '#99DDFF',       # 鲜艳亮天空蓝
    'Eukaryota/Unknown': '#99CCFF',             # 中等蓝（加深）
    'Eukaryota/Other': '#99CCFF',               # 中等蓝（加深）

    # Archaea - 红色系（更鲜艳）
    'Archaea/Thermoproteota': '#CC0000',        # 鲜艳深红
    'Archaea/Halobacteriota': '#FF0033',        # 鲜艳猩红
    'Archaea/Euryarchaeota': '#FF6666',         # 鲜艳番茄红
    'Archaea/Unknown': '#FFEBEE',               # 最浅红（统一）
    'Archaea/Other': '#FFEBEE',                 # 最浅红（统一）

    'Unknown': '#9E9E9E'                         # 灰色
}

def get_phylum_color(domain, phylum):
    """获取phylum对应的颜色"""
    # 清理phylum名称（移除Other_Phylum_前缀）
    if phylum.startswith('Other_Phylum_'):
        phylum_key = f"{domain}/Other"
    else:
        phylum_key = f"{domain}/{phylum}"

    return PHYLUM_COLORS.get(phylum_key, PHYLUM_COLORS.get(f"{domain}/Other", '#808080'))

def adjust_color_intensity(base_color_hex, intensity):
    """
    根据intensity调整颜色深浅
    intensity: 0-1, 0=浅色, 1=深色
    增强版：即使是低intensity也保持较高的饱和度
    """
    # 转换hex到RGB
    base_color_hex = base_color_hex.lstrip('#')
    r, g, b = tuple(int(base_color_hex[i:i+2], 16) for i in (0, 2, 4))

    # 调整intensity曲线，让最小值从0.3开始（保证最浅的颜色也有足够饱和度）
    intensity = 0.4 + intensity * 0.6  # 将[0,1]映射到[0.4,1]

    # 创建浅色版本（与白色混合，但混合比例更小）
    light_r, light_g, light_b = 255, 255, 255

    # 插值
    final_r = int(light_r + intensity * (r - light_r))
    final_g = int(light_g + intensity * (g - light_g))
    final_b = int(light_b + intensity * (b - light_b))

    return np.array([final_r, final_g, final_b]) / 255.0

# ======================================================================
# 缓存功能
# ======================================================================
def get_cache_key(scaling_scheme_name=None, custom_scaling=None):
    """
    生成缓存键（基于UMAP参数、数据量和scaling参数）

    Args:
        scaling_scheme_name: 预定义方案名称（如'moderate'）
        custom_scaling: 自定义scaling字典（如{'Archaea': 5, 'Bacteria': 2, ...}）
    """
    sample_str = "full" if MAX_SAMPLES is None else f"s{MAX_SAMPLES}"

    # 生成scaling标识
    if custom_scaling:
        scale_str = f"A{custom_scaling.get('Archaea', 1)}_B{custom_scaling.get('Bacteria', 1)}_E{custom_scaling.get('Eukaryota', 1)}"
    elif scaling_scheme_name:
        scale_str = scaling_scheme_name
    else:
        scale_str = DEFAULT_SCALING_SCHEME

    return f"umap_n{UMAP_PARAMS['n_neighbors']}_d{UMAP_PARAMS['min_dist']:.2f}_sp{UMAP_PARAMS['spread']:.2f}_{UMAP_PARAMS['metric']}_{sample_str}_{scale_str}"

def save_umap_cache(embeddings, taxid, seq_count, domains, phylums, scaling_scheme_name=None, custom_scaling=None):
    """保存UMAP结果到缓存"""
    cache_file = os.path.join(cache_dir, f"{get_cache_key(scaling_scheme_name, custom_scaling)}.npz")
    print(f"保存UMAP结果到缓存: {cache_file}")

    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        taxid=taxid,
        seq_count=seq_count,
        domains=domains,
        phylums=phylums,
        scaling_scheme_name=np.array([scaling_scheme_name if scaling_scheme_name else DEFAULT_SCALING_SCHEME]),
        custom_scaling=custom_scaling if custom_scaling else {},
        umap_params=UMAP_PARAMS,
        timestamp=np.array([pd.Timestamp.now().timestamp()])
    )

def load_umap_cache(scaling_scheme_name=None, custom_scaling=None):
    """从缓存加载UMAP结果"""
    cache_file = os.path.join(cache_dir, f"{get_cache_key(scaling_scheme_name, custom_scaling)}.npz")

    if not os.path.exists(cache_file):
        return None

    print(f"从缓存加载UMAP结果: {cache_file}")
    cached = np.load(cache_file, allow_pickle=True)

    return {
        'embeddings': cached['embeddings'],
        'taxid': cached['taxid'],
        'seq_count': cached['seq_count'],
        'domains': cached['domains'],
        'phylums': cached['phylums']
    }

# ======================================================================
# 主流程
# ======================================================================
def main(force_recompute=False, scaling_scheme=None, custom_scaling=None, data_dir=None):
    """
    主函数

    Args:
        force_recompute: 强制重新计算UMAP
        scaling_scheme: 使用预定义的scaling方案名称（如'moderate'）
        custom_scaling: 自定义scaling字典，如{'Archaea': 5, 'Bacteria': 2, 'Eukaryota': 1}
        data_dir: K-mer CSV 数据目录路径
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    print("=" * 70)
    print("UMAP Phylum可视化工具 v2.0")
    print("=" * 70)

    # 确定使用的scaling配置
    if custom_scaling:
        scaling_config = custom_scaling
        scaling_name = "custom"
        print(f"\n使用自定义scaling: {scaling_config}")
    elif scaling_scheme and scaling_scheme in SCALING_SCHEMES:
        scaling_config = SCALING_SCHEMES[scaling_scheme]
        scaling_name = scaling_scheme
        print(f"\n使用预定义scaling方案: {scaling_scheme}")
        print(f"  配置: {scaling_config}")
    else:
        scaling_config = SCALING_SCHEMES[DEFAULT_SCALING_SCHEME]
        scaling_name = DEFAULT_SCALING_SCHEME
        print(f"\n使用默认scaling方案: {DEFAULT_SCALING_SCHEME}")
        print(f"  配置: {scaling_config}")

    # 步骤1: 尝试从缓存加载
    if not force_recompute:
        cached_data = load_umap_cache(
            scaling_scheme_name=scaling_name if not custom_scaling else None,
            custom_scaling=custom_scaling
        )
        if cached_data is not None:
            print("✓ 使用缓存的UMAP结果，跳过数据加载和降维步骤")
            embeddings = cached_data['embeddings']
            taxid = cached_data['taxid']
            seq_count = cached_data['seq_count']
            domains = cached_data['domains']
            phylums = cached_data['phylums']

            # 跳转到可视化
            visualize_umap(embeddings, taxid, seq_count, domains, phylums, scaling_name)
            return

    # 步骤2: 加载taxonomy映射
    print(f"\n[1/6] 加载phylum映射: {mapping_file}")
    with open(mapping_file, 'r') as f:
        taxid_taxonomy_map = json.load(f)

    # 转换key为int
    taxid_taxonomy_map = {int(k): v for k, v in taxid_taxonomy_map.items()}
    print(f"  已加载 {len(taxid_taxonomy_map)} 个taxid的映射")

    # 步骤3: 读取并合并CSV文件
    print(f"\n[2/6] 读取CSV文件: {data_dir}")
    csv_pattern = os.path.join(data_dir, "species_kmer_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"  找到 {len(csv_files)} 个CSV文件")

    df_list = []
    for i, csv_file in enumerate(csv_files):
        print(f"  读取 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        df_list.append(df)

    data = pd.concat(df_list, ignore_index=True)
    print(f"  总共加载了 {len(data)} 条记录")

    # 步骤4: 提取特征并分配taxonomy
    print(f"\n[3/6] 提取特征并分配taxonomy")
    taxid = data.iloc[:, 0].values
    seq_count = data.iloc[:, 1].values
    kmer_features = data.iloc[:, 2:].values

    # 分配domain和phylum
    domains = []
    phylums = []
    for tid in taxid:
        tax_info = taxid_taxonomy_map.get(int(tid), {'domain': 'Unknown', 'phylum': 'Unknown'})
        domains.append(tax_info['domain'])
        phylums.append(tax_info['phylum'])

    domains = np.array(domains)
    phylums = np.array(phylums)

    # 统计
    print(f"\n=== 全数据集Domain分布 ===")
    unique_d, counts_d = np.unique(domains, return_counts=True)
    for d, c in zip(unique_d, counts_d):
        print(f"  {d}: {c} ({c/len(domains)*100:.1f}%)")

    print(f"\n=== 全数据集主要Phylum分布（前10） ===")
    phylum_full = [f"{d}/{p}" for d, p in zip(domains, phylums)]
    unique_p, counts_p = np.unique(phylum_full, return_counts=True)
    top_indices = np.argsort(counts_p)[::-1][:10]
    for idx in top_indices:
        print(f"  {unique_p[idx]}: {counts_p[idx]} ({counts_p[idx]/len(phylum_full)*100:.2f}%)")

    # 处理NaN/Inf
    kmer_features = np.nan_to_num(kmer_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 步骤5: 采样（使用全部数据）
    print(f"\n[4/6] 数据处理")
    if MAX_SAMPLES is not None and len(data) > MAX_SAMPLES:
        print(f"  随机采样 {MAX_SAMPLES} 条记录")
        np.random.seed(42)
        sample_idx = np.random.choice(len(data), size=MAX_SAMPLES, replace=False)
        taxid = taxid[sample_idx]
        seq_count = seq_count[sample_idx]
        kmer_features = kmer_features[sample_idx]
        domains = domains[sample_idx]
        phylums = phylums[sample_idx]
    else:
        print(f"  使用全部 {len(data)} 条记录")

    # 处理后统计
    print(f"\n=== 处理后Domain分布 ===")
    unique_d, counts_d = np.unique(domains, return_counts=True)
    for d, c in zip(unique_d, counts_d):
        print(f"  {d}: {c} ({c/len(domains)*100:.1f}%)")

    # Domain-specific scaling（增强以促进三域分离）
    print(f"\n[5/6] Domain-specific scaling + 归一化")
    print(f"  使用scaling配置: {scaling_config}")
    kmer_features_scaled = kmer_features.copy()
    for i, domain in enumerate(domains):
        scale_factor = scaling_config.get(domain, 1)
        kmer_features_scaled[i] *= scale_factor

    kmer_features_normalized = normalize(kmer_features_scaled, norm='l2')
    print("  特征归一化完成")

    # 步骤6: UMAP降维
    print(f"\n[6/6] UMAP降维")
    print(f"  参数: {UMAP_PARAMS}")
    umap_model = umap.UMAP(**UMAP_PARAMS)
    embeddings = umap_model.fit_transform(kmer_features_normalized)
    print(f"  UMAP降维完成: {embeddings.shape}")

    # 保存到缓存
    save_umap_cache(
        embeddings, taxid, seq_count, domains, phylums,
        scaling_scheme_name=scaling_name if not custom_scaling else None,
        custom_scaling=custom_scaling
    )

    # 可视化
    visualize_umap(embeddings, taxid, seq_count, domains, phylums, scaling_name)

def visualize_umap(embeddings, taxid, seq_count, domains, phylums, scaling_name=""):
    """UMAP可视化"""
    print(f"\n[可视化] 生成UMAP图")

    x, y = embeddings[:, 0], embeddings[:, 1]

    # seq_count归一化 (log scale)
    seq_count_log = np.log10(seq_count + 1)
    seq_count_normalized = (seq_count_log - seq_count_log.min()) / (seq_count_log.max() - seq_count_log.min() + 1e-10)

    # 创建图形
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor('white')

    # 按phylum分组绘制
    phylum_full = [f"{d}/{p}" for d, p in zip(domains, phylums)]
    unique_phylums = np.unique(phylum_full)

    # 统计各phylum数量用于排序
    phylum_counts = {}
    for phylum in unique_phylums:
        mask = np.array(phylum_full) == phylum
        phylum_counts[phylum] = mask.sum()

    # 按数量降序绘制（大的在底层）
    sorted_phylums = sorted(unique_phylums, key=lambda p: phylum_counts[p], reverse=True)

    legend_entries = []
    for phylum_key in sorted_phylums:
        mask = np.array(phylum_full) == phylum_key
        if mask.sum() == 0:
            continue

        # 跳过所有Unknown domain的点（灰色），不绘制
        if phylum_key.startswith('Unknown/'):
            continue

        # 跳过所有Other_Phylum的点（颜色太浅，接近灰白色），不绘制
        if '/Other_Phylum_' in phylum_key:
            continue

        x_phylum = x[mask]
        y_phylum = y[mask]
        counts_phylum = seq_count_normalized[mask]

        # 获取该phylum的domain和phylum名称
        domain, phylum_name = phylum_key.split('/', 1)
        base_color = get_phylum_color(domain, phylum_name)

        # 为每个点生成颜色（根据seq_count调整深浅）
        colors = np.array([adjust_color_intensity(base_color, c) for c in counts_phylum])

        plt.scatter(
            x_phylum, y_phylum,
            c=colors,
            s=6,                 # 点的大小（从12缩小到6）
            edgecolors='none',
            alpha=0.85,          # 提高透明度，让颜色更实
            label=f"{phylum_key} (n={mask.sum()})"
        )

        # 只为主要phylum添加图例
        if phylum_counts[phylum_key] > 100:  # 只显示样本数>100的phylum
            legend_entries.append((phylum_key, mask.sum(), base_color))

    # 不添加任何图例、标题或文字说明（用户在PPT中自己添加）

    # 保存图片（添加时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"umap_kmer_phylum_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图片已保存到: {save_path}")

    # 统计信息
    print("\n=== 统计信息 ===")
    print(f"  样本数量: {len(taxid)}")
    print(f"  Seq count范围: [{seq_count.min()}, {seq_count.max()}]")
    print(f"  UMAP坐标范围:")
    print(f"    X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"    Y: [{y.min():.2f}, {y.max():.2f}]")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UMAP Phylum可视化工具 - 支持自定义domain scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认moderate方案
  python %(prog)s

  # 使用预定义的aggressive方案
  python %(prog)s --scaling-scheme aggressive

  # 自定义scaling参数
  python %(prog)s --archaea-scale 8 --bacteria-scale 3 --eukaryota-scale 0.5

  # 强制重新计算UMAP
  python %(prog)s --force-recompute

  # 列出所有预定义方案
  python %(prog)s --list-schemes
        """
    )

    parser.add_argument('--force-recompute', action='store_true',
                       help='强制重新计算UMAP（忽略缓存）')

    parser.add_argument('--scaling-scheme', type=str,
                       choices=list(SCALING_SCHEMES.keys()),
                       help=f'使用预定义的scaling方案: {", ".join(SCALING_SCHEMES.keys())}')

    parser.add_argument('--archaea-scale', type=float,
                       help='Archaea domain的scaling系数（自定义）')
    parser.add_argument('--bacteria-scale', type=float,
                       help='Bacteria domain的scaling系数（自定义）')
    parser.add_argument('--eukaryota-scale', type=float,
                       help='Eukaryota domain的scaling系数（自定义）')
    parser.add_argument('--unknown-scale', type=float, default=1.0,
                       help='Unknown的scaling系数（自定义，默认1.0）')

    parser.add_argument('--list-schemes', action='store_true',
                       help='列出所有预定义的scaling方案')

    parser.add_argument('--data-dir', type=str, default=None,
                       help=f'K-mer CSV 数据目录路径（默认: <script_dir>/data）')

    args = parser.parse_args()

    # 列出方案
    if args.list_schemes:
        print("=" * 70)
        print("预定义的Domain Scaling方案:")
        print("=" * 70)
        for scheme_name, scheme_config in SCALING_SCHEMES.items():
            print(f"\n{scheme_name}:")
            for domain, scale in scheme_config.items():
                print(f"  {domain}: ×{scale}")
        print(f"\n默认方案: {DEFAULT_SCALING_SCHEME}")
        sys.exit(0)

    # 检查是否使用自定义scaling
    custom_scaling = None
    if any([args.archaea_scale, args.bacteria_scale, args.eukaryota_scale]):
        custom_scaling = {
            'Archaea': args.archaea_scale if args.archaea_scale else 1.0,
            'Bacteria': args.bacteria_scale if args.bacteria_scale else 1.0,
            'Eukaryota': args.eukaryota_scale if args.eukaryota_scale else 1.0,
            'Unknown': args.unknown_scale
        }

    main(
        force_recompute=args.force_recompute,
        scaling_scheme=args.scaling_scheme,
        custom_scaling=custom_scaling,
        data_dir=args.data_dir
    )
