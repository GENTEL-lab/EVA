#!/usr/bin/env python3
"""
RNA Composition Plot - Validation: Sequence Similarity vs. Data Enrichment
Replicates the style from the reference image showing API vs Enrichment Ratio.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20  # Base font size (doubled from 10)
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 22

# Data from the table
# RNA types categorized as Housekeeping (high API) and Info/Reg (low API)
data = {
    'tRNA': {'api': 0.2646, 'enrichment': 0.30, 'category': 'Housekeeping'},
    'snRNA': {'api': 0.2494, 'enrichment': 0.50, 'category': 'Housekeeping'},
    'rRNA': {'api': 0.1361, 'enrichment': 0.05, 'category': 'Housekeeping'},
    'snoRNA': {'api': 0.1041, 'enrichment': 0.44, 'category': 'Housekeeping'},
    'viral RNA': {'api': 0.0991, 'enrichment': 0.17, 'category': 'Housekeeping'},
    'sRNA': {'api': 0.1282, 'enrichment': 0.30, 'category': 'Housekeeping'},
    'piRNA': {'api': 0.1199, 'enrichment': 1.42, 'category': 'Housekeeping'},
    'miRNA': {'api': 0.0759, 'enrichment': 0.46, 'category': 'Info/Reg'},
    'circRNA': {'api': 0.0440, 'enrichment': 1.06, 'category': 'Info/Reg'},
    'mRNA': {'api': 0.0418, 'enrichment': 1.28, 'category': 'Info/Reg'},
    'lncRNA': {'api': 0.0315, 'enrichment': 2.32, 'category': 'Info/Reg'},
    'ncRNA': {'api': 0.0500, 'enrichment': 2.29, 'category': 'Info/Reg'},  # API estimated for ncRNA
}

# Separate data by category
housekeeping = {k: v for k, v in data.items() if v['category'] == 'Housekeeping'}
info_reg = {k: v for k, v in data.items() if v['category'] == 'Info/Reg'}

# Extract x and y values
x_hk = np.array([v['api'] for v in housekeeping.values()])
y_hk = np.array([v['enrichment'] for v in housekeeping.values()])
labels_hk = list(housekeeping.keys())

x_ir = np.array([v['api'] for v in info_reg.values()])
y_ir = np.array([v['enrichment'] for v in info_reg.values()])
labels_ir = list(info_reg.keys())

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 9))

# Plot Housekeeping (red circles, filled)
ax.scatter(x_hk, y_hk, c='#E74C3C', marker='o', s=200, alpha=0.8, label='Housekeeping', zorder=5, edgecolors='darkred', linewidths=2)

# Plot Info/Reg (blue circles, filled)
ax.scatter(x_ir, y_ir, c='#3498DB', marker='o', s=200, alpha=0.8, label='Info/Reg', zorder=5, edgecolors='darkblue', linewidths=2)

# Add labels for each point
for name, vals in data.items():
    if name == 'lncRNA':
        ax.annotate(name, (vals['api'], vals['enrichment']),
                    textcoords="offset points", xytext=(-30, 5), fontsize=18, alpha=0.85)
    elif name == 'ncRNA':
        ax.annotate(name, (vals['api'], vals['enrichment']),
                    textcoords="offset points", xytext=(5, -15), fontsize=18, alpha=0.85)
    else:
        ax.annotate(name, (vals['api'], vals['enrichment']),
                    textcoords="offset points", xytext=(5, 5), fontsize=18, alpha=0.85)

# Set log scale for y-axis
ax.set_yscale('log')

# Customize the plot
ax.set_xlabel('Internal Sequence Similarity (API)', fontsize=24)
ax.set_ylabel('Enrichment Ratio (Log Scale)', fontsize=24)
ax.set_title('Validation: Sequence Similarity vs. Data Enrichment', fontsize=14, pad=15)

# Set axis limits - original y range
ax.set_xlim(0, 0.30)
ax.set_ylim(0.01, 10)

# Add grid
ax.grid(True, linestyle='--', alpha=0.5)

# Add legend
ax.legend(loc='upper right', fontsize=22, framealpha=0.9)

# Add horizontal line at y=1 (no enrichment) - bold and prominent
ax.axhline(y=1, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax.text(0.28, 1.25, 'Enrichment = 1', fontsize=20, ha='right', color='black')

# Add shaded regions to highlight
ax.fill_between([0, 0.30], [1, 1], [10, 10], alpha=0.08, color='blue')
ax.fill_between([0, 0.30], [0.01, 0.01], [1, 1], alpha=0.08, color='red')

# Add region labels
ax.text(0.02, 5, 'Info/Reg\n(enriched)', fontsize=20, ha='left', va='center', color='#3498DB', alpha=0.7)
ax.text(0.02, 0.05, 'Housekeeping\n(depleted)', fontsize=20, ha='left', va='center', color='#E74C3C', alpha=0.7)

# Tight layout
plt.tight_layout()

# Set transparent background before saving
fig.patch.set_alpha(0)

# Save the figure next to this script
output_base = Path(__file__).resolve().parent / 'rna_api_enrichment'
plt.savefig(output_base.with_suffix('.svg'), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(output_base.with_suffix('.png'), dpi=300, bbox_inches='tight', transparent=True)

print("Plot saved successfully!")
print("Files saved:")
print(f"  - {output_base.with_suffix('.svg')}")
print(f"  - {output_base.with_suffix('.png')}")
