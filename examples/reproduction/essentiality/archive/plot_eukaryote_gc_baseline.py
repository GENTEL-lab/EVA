#!/usr/bin/env python3
"""
Eukaryote comparison: RNAGen (8192) vs Evo2 + GC Content Baseline
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# Paths: 从脚本位置推导，支持 RNA_BENCHMARK_ROOT 环境变量覆盖
_script_dir = Path(__file__).resolve().parent
# 兼容: 脚本在 zeroshot_essentiality/ 或 package/scripts/
if (_script_dir / "eukaryote" / "output_8192").exists():
    _rnagen_root = Path(os.environ.get("RNA_BENCHMARK_ROOT", _script_dir.parent.parent))
    base_dir = _script_dir
else:
    # 脚本在 package/scripts/ 时: scripts->package->interpretability->rna_benchmark
    _rnagen_root = Path(os.environ.get("RNA_BENCHMARK_ROOT", _script_dir.parent.parent.parent.parent))
    base_dir = _rnagen_root / "interpretability" / "zeroshot_essentiality"
eukaryote_dir = base_dir / "eukaryote"
evo2_dir = _rnagen_root / "evo2_test" / "essentiality" / "output"

print("Loading data...")

# Load RNAGen 8192 results (with sequences)
with open(eukaryote_dir / "output" / "delta_ll_results_all_data.json") as f:
    data_with_seq = json.load(f)
print(f"Loaded original data with sequences: {len(data_with_seq)} genes")

# Build sequence lookup
seq_lookup = {(item['organism'], item['gene']): item['sequence'] for item in data_with_seq}

# Load RNAGen 8192 results
with open(eukaryote_dir / "output_8192" / "delta_ll_results_30M_8192.json") as f:
    rnagen_30m_8192 = json.load(f)

with open(eukaryote_dir / "output_8192" / "delta_ll_results_1400M_8192.json") as f:
    rnagen_1400m_8192 = json.load(f)

# Add sequences to 8192 results
for item in rnagen_30m_8192:
    item['sequence'] = seq_lookup.get((item['organism'], item['gene']), '')

for item in rnagen_1400m_8192:
    item['sequence'] = seq_lookup.get((item['organism'], item['gene']), '')

# Load Evo2 AUROC
with open(evo2_dir / "evo2_1b_eukaryote" / "auroc_results_evo2_1b_base_eukaryote.json") as f:
    evo2_1b_auroc = json.load(f)

with open(evo2_dir / "evo2_7b_eukaryote" / "auroc_results_evo2_7b_base_eukaryote.json") as f:
    evo2_7b_auroc = json.load(f)

with open(evo2_dir / "evo2_40b_eukaryote" / "auroc_results_evo2_40b_base_eukaryote.json") as f:
    evo2_40b_auroc = json.load(f)

print(f"Loaded EVA 21M:  {len(rnagen_30m_8192)} genes")
print(f"Loaded EVA 1.4B: {len(rnagen_1400m_8192)} genes")

# Calculate GC content
def calc_gc(seq):
    """Calculate GC content"""
    if not seq or len(seq) == 0:
        return 0.5
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

# Calculate AUROC per species for RNAGen
def calc_per_species_auroc(data):
    """Calculate AUROC for each species"""
    by_species = defaultdict(list)
    for item in data:
        by_species[item['organism']].append(item)

    aurocs = {}
    for species, items in by_species.items():
        labels = [1 if x['essential'] else 0 for x in items]
        scores = [x['delta_ll'] for x in items]

        if len(set(labels)) >= 2 and len(labels) >= 20:
            aurocs[species] = roc_auc_score(labels, scores)

    return aurocs

# Calculate GC content baseline per species
def calc_gc_baseline_auroc(data):
    """Use GC content as predictor"""
    by_species = defaultdict(list)
    for item in data:
        if item.get('sequence'):
            by_species[item['organism']].append(item)

    aurocs = {}
    for species, items in by_species.items():
        labels = [1 if x['essential'] else 0 for x in items]
        # Use GC content as score
        scores = [calc_gc(x['sequence']) for x in items]

        if len(set(labels)) >= 2 and len(labels) >= 20:
            try:
                aurocs[species] = roc_auc_score(labels, scores)
            except:
                aurocs[species] = 0.5

    return aurocs

print("\nCalculating per-species AUROC...")
rnagen_30m_auroc = calc_per_species_auroc(rnagen_30m_8192)
rnagen_1400m_auroc = calc_per_species_auroc(rnagen_1400m_8192)
gc_baseline_auroc = calc_gc_baseline_auroc(rnagen_1400m_8192)

# Get common species
species = sorted(rnagen_30m_auroc.keys())
print(f"Number of species: {len(species)}")

# Print GC baseline stats
print("\nGC Content Baseline AUROC per species:")
for sp in species:
    print(f"  {sp}: {gc_baseline_auroc.get(sp, 0.5):.4f}")

# Check GC content statistics
print("\nGC content statistics by species:")
for sp in species:
    sp_items = [x for x in rnagen_1400m_8192 if x['organism'] == sp and x.get('sequence')]
    essential = [calc_gc(x['sequence']) for x in sp_items if x['essential']]
    non_essential = [calc_gc(x['sequence']) for x in sp_items if not x['essential']]
    if essential and non_essential:
        print(f"  {sp}:")
        print(f"    Essential GC:     {np.mean(essential):.4f} ± {np.std(essential):.4f}")
        print(f"    Non-essential GC: {np.mean(non_essential):.4f} ± {np.std(non_essential):.4f}")

# Prepare data for plotting
results = []
for sp in species:
    result = {
        'organism': sp,
        'gc_baseline': gc_baseline_auroc.get(sp, 0.5),
        'evo2_1b': evo2_1b_auroc[sp]['auroc'],
        'evo2_7b': evo2_7b_auroc[sp]['auroc'],
        'evo2_40b': evo2_40b_auroc.get(sp, {}).get('auroc', np.nan),
        'rnagen_30m': rnagen_30m_auroc.get(sp, np.nan),
        'rnagen_1400m': rnagen_1400m_auroc.get(sp, np.nan),
    }
    results.append(result)

# Calculate means
mean_gc = np.nanmean([r['gc_baseline'] for r in results])
mean_evo1b = np.nanmean([r['evo2_1b'] for r in results])
mean_evo7b = np.nanmean([r['evo2_7b'] for r in results])
mean_evo40b = np.nanmean([r['evo2_40b'] for r in results])
mean_30m = np.nanmean([r['rnagen_30m'] for r in results])
mean_1400m = np.nanmean([r['rnagen_1400m'] for r in results])

print(f"\nMean AUROC:")
print(f"  GC Content Baseline: {mean_gc:.4f}")
print(f"  Evo2 1B:             {mean_evo1b:.4f}")
print(f"  Evo2 7B:             {mean_evo7b:.4f}")
print(f"  Evo2 40B:            {mean_evo40b:.4f}")
print(f"  EVA 21M (8192):   {mean_30m:.4f}")
print(f"  EVA 1.4B (8192):  {mean_1400m:.4f}")

# Model data: (display_name, key, mean, color)
# Updated colors to match the reference image
orange_color = (255/255, 164/255, 91/255)  # #FFA45B
blue_color = (115/255, 185/255, 231/255)   # #73B9E7
gray_color = (207/255, 208/255, 209/255)

# Order: Baseline at bottom, then Evo2, then EVA at top
model_data = [
    ('GC Content\nBaseline', 'gc_baseline', mean_gc, gray_color),
    ('Evo2 1B', 'evo2_1b', mean_evo1b, blue_color),
    ('Evo2 7B', 'evo2_7b', mean_evo7b, blue_color),
    ('Evo2 40B', 'evo2_40b', mean_evo40b, blue_color),
    ('EVA 21M', 'rnagen_30m', mean_30m, orange_color),
    ('EVA 1.4B', 'rnagen_1400m', mean_1400m, orange_color),
]

# Create figure
print("\nGenerating plot...")
# Load Arial font (old server style: fm.fontManager.addfont + rcParams)
font_candidates = [
    _rnagen_root.parent / "fonts" / "arial.ttf",
    _rnagen_root / "z_rnagym_70" / "RNAVerse" / "notebooks" / "design" / "fonts" / "arial.ttf",
]
for fp in font_candidates:
    if fp.exists():
        fm.fontManager.addfont(str(fp))
        break
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# Clean background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

y_positions = np.arange(len(model_data))

# Draw horizontal bars
for i, (name, key, mean, color) in enumerate(model_data):
    ax.barh(i, mean, height=0.75, color=color, alpha=0.8, edgecolor='none')

# Add strip plot (individual species points)
np.random.seed(42)
for i, (name, key, mean, color) in enumerate(model_data):
    values = [r[key] for r in results if not np.isnan(r[key])]
    jitter = np.random.uniform(-0.2, 0.2, len(values))
    ax.scatter(values, i + jitter, c='black', alpha=0.4, s=15, zorder=5)

# Reference line at 0.5
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

# Styling
ax.set_yticks(y_positions)
ax.set_yticklabels([m[0] for m in model_data], fontsize=24)
ax.set_xlabel('AUROC', fontsize=26)
ax.set_title('Zero-shot Gene Essentiality Prediction', fontsize=26)
ax.tick_params(axis='x', labelsize=20)
ax.set_xlim(0.4, 0.7)

# Black frame
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Grid
ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend: row1 = RNA/DNA models, row2 = Individual dataset (centered)
legend_row1 = [
    Patch(facecolor=orange_color, edgecolor='none', label='RNA Language Model'),
    Patch(facecolor=blue_color, edgecolor='none', label='DNA Language Model'),
]
legend_row2 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.4, markersize=10, linestyle='None', label='Individual dataset'),
]
leg1 = ax.legend(handles=legend_row1, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=18, frameon=False, labelspacing=0.5)
ax.add_artist(leg1)
ax.legend(handles=legend_row2, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=18, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

# 输出到 eukaryote/output_8192/（含 SVG）
output_dir = eukaryote_dir / "output_8192"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_dir / "figure_eukaryote_gc_baseline.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / "figure_eukaryote_gc_baseline.pdf", bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / "figure_eukaryote_gc_baseline.svg", bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_dir}/figure_eukaryote_gc_baseline.png/pdf/svg")

# Print detailed comparison
print("\nDetailed per-species comparison:")
print(f"{'Species':<40} {'GC Base':>8} {'Evo2 1B':>8} {'Evo2 7B':>8} {'Evo2 40B':>8} {'EVA 21M':>12} {'EVA 1.4B':>12}")
print("-" * 120)
for r in results:
    print(f"{r['organism']:<40} {r['gc_baseline']:>8.4f} {r['evo2_1b']:>8.4f} {r['evo2_7b']:>8.4f} {r['evo2_40b']:>8.4f} {r['rnagen_30m']:>12.4f} {r['rnagen_1400m']:>12.4f}")

# Save results
output_data = {
    'n_species': len(results),
    'summary': {
        'gc_baseline': {'mean': float(mean_gc)},
        'evo2_1b': {'mean': float(mean_evo1b)},
        'evo2_7b': {'mean': float(mean_evo7b)},
        'evo2_40b': {'mean': float(mean_evo40b)},
        'rnagen_30m': {'mean': float(mean_30m)},
        'rnagen_1400m': {'mean': float(mean_1400m)},
    },
    'per_species': results
}

with open(output_dir / "eukaryote_gc_baseline_results.json", 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_dir}/eukaryote_gc_baseline_results.json")
