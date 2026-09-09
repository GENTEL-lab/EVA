#!/usr/bin/env python3
"""
Figure: SAE Feature Steering - Multiple RNA Types (v3 with high-success datasets)

8 cases with positive delta, covering different RNA types:
1. milena2021_cata U33G (Catalytic, +12)
2. andreasson2020_glms 11393 (Riboswitch, +12)
3. Janzen Fam21 Ribozyme 291 (Ribozyme, +12)
4. Janzen Fam31 Ribozyme 490 (Ribozyme, +9)
5. Janzen Fam1b1 Ribozyme 2 (Ribozyme, +9)
6. Domingo tRNA 9 (tRNA, +11)
7. Pepper Aptamer (+11)
8. Zuo Okra Aptamer (+8)
"""

import json
import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import font_manager, rcParams

# Set Arial font globally, larger sizes
ARIAL_FONT = "/data/yanjie_huang/enzyme1_server/fonts/arial.ttf"
font_manager.fontManager.addfont(ARIAL_FONT)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'

RNAplot_BIN = "/data/yanjie_huang/enzyme1_server/huangyanjie/miniconda3/bin/RNAplot"
TEMP_DIR = "/tmp/rnaplot_fig_multi_v3"
OUTPUT_DIR = "/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering/yanjie/figures"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

GREEN = "#27ae60"
RED = "#c0392b"
GRAY = "#bdc3c7"

RNA_COLORS = {
    'Ribozymes': '#A23B72',
    'Catalytic': '#9b59b6',
    'Riboswitch': '#16a085',
    'mRNA': '#F18F01',
    'tRNA': '#2E86AB',
    'Aptamer': '#17a2b8',
    'Ribozymes (fam21)': '#A23B72',
    'Ribozymes (fam31)': '#8e44ad',
    'Ribozymes (fam1b1)': '#6c5ce7',
    'Aptamer (Okra)': '#0984e3',
}

DATA_DIR = "/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering/rnafold_cases"

CASES = [
    # 1. milena catalytic RNA
    {
        "name": "milena_cata",
        "rna_type": "Catalytic",
        "rna_name": "Milena Catalytic RNA",
        "record_id": "U33G_fitness_0.01215",
        "cases_json": f"{DATA_DIR}/milena2021_cata_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/milena2021_cata_generation_v2_rows.csv",
        "steering_label": "0×",
        "feature_id": 252,
        "condition": "0x",
        "sample_idx": 6,
        "target_state": "mutant",
    },
    # 2. andreasson glmS riboswitch
    {
        "name": "andreasson_glms",
        "rna_type": "Riboswitch",
        "rna_name": "Andreasson glmS Riboswitch",
        "record_id": "11393_2_T54C,C71T_0.035997417",
        "cases_json": f"{DATA_DIR}/andreasson2020_glms_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/andreasson2020_glms_generation_v2_rows.csv",
        "steering_label": "5×",
        "feature_id": 8147,
        "condition": "5x",
        "sample_idx": 1,
        "target_state": "mutant",
    },
    # 3. janzen fam21 ribozyme
    {
        "name": "janzen_fam21",
        "rna_type": "Ribozymes (fam21)",
        "rna_name": "Janzen Fam21 Ribozyme",
        "record_id": "Janzen_2022_fam21_ribozyme_291",
        "cases_json": f"{DATA_DIR}/janzen_fam21_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/janzen_fam21_structure_switch_generation_case0_mutstate_paired_f7081_893_5929_7373_5397_147_n60_rows.csv",
        "steering_label": "0×",
        "feature_id": 5929,
        "condition": "0x",
        "sample_idx": 56,
        "target_state": "mutant",
    },
    # 4. janzen fam31 ribozyme
    {
        "name": "janzen_fam31",
        "rna_type": "Ribozymes (fam31)",
        "rna_name": "Janzen Fam31 Ribozyme",
        "record_id": "Janzen_2022_fam31_ribozyme_490",
        "case_index": 4,
        "cases_json": f"{DATA_DIR}/janzen_fam31_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/janzen_fam31_generation_v2_rows.csv",
        "steering_label": "2×",
        "feature_id": 677,
        "condition": "2x",
        "sample_idx": 7,
        "target_state": "mutant",
    },
    # 5. janzen fam1b1 ribozyme
    {
        "name": "janzen_fam1b1",
        "rna_type": "Ribozymes (fam1b1)",
        "rna_name": "Janzen Fam1b1 Ribozyme",
        "record_id": "Janzen_2022_fam1b1_ribozyme_2",
        "cases_json": f"{DATA_DIR}/janzen_fam1b1_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/janzen_fam1b1_structure_switch_generation_case0_mutstate_paired_f677_1709_7081_5929_n20_rows.csv",
        "steering_label": "1×",
        "feature_id": 677,
        "condition": "1x",
        "sample_idx": 9,
        "target_state": "mutant",
    },
    # 6. Domingo tRNA
    {
        "name": "domingo_trna",
        "rna_type": "tRNA",
        "rna_name": "Domingo tRNA 9",
        "record_id": "Domingo_2018_tRNA_9",
        "cases_json": f"{DATA_DIR}/domingo2018_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/domingo2018_structure_switch_generation_case0_wtstate_paired_f6227_147_4139_7609_6803_5929_n20_rows.csv",
        "steering_label": "10×",
        "feature_id": 6803,
        "condition": "10x",
        "sample_idx": 12,
        "target_state": "wt",
    },
    # 7. Pepper aptamer
    {
        "name": "pepper",
        "rna_type": "Aptamer",
        "rna_name": "Pepper Aptamer",
        "record_id": "pepper_C5A_A6C_A7C_A21G_C36U",
        "cases_json": f"{DATA_DIR}/chen2019_pepper_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/chen2019_pepper_structure_switch_generation_case0_wtstate_paired_f893_2008_5397_5929_863_n20_rows.csv",
        "steering_label": "0×",
        "feature_id": 2008,
        "condition": "0x",
        "sample_idx": 18,
        "target_state": "wt",
    },
    # 8. Zuo Okra aptamer
    {
        "name": "zuo_okra",
        "rna_type": "Aptamer (Okra)",
        "rna_name": "Zuo Okra Aptamer",
        "record_id": "Sequence_16",
        "cases_json": f"{DATA_DIR}/zuo2023_okra_structure_switch_cases.json",
        "gen_csv": f"{DATA_DIR}/zuo2023_okra_generation_v2_rows.csv",
        "steering_label": "5×",
        "feature_id": 6297,
        "condition": "5x",
        "sample_idx": 13,
        "target_state": "mutant",
    },
]


def run_rnaplot(sequence, structure, name):
    os.makedirs(TEMP_DIR, exist_ok=True)
    work_dir = f"{TEMP_DIR}/{name}"
    os.makedirs(work_dir, exist_ok=True)
    input_file = f"{work_dir}/input.txt"
    with open(input_file, "w") as f:
        f.write(f">{name}\n{sequence}\n{structure}\n")
    result = subprocess.run(
        [RNAplot_BIN, "-o", "svg", "-t", "1"],
        input=open(input_file).read(),
        capture_output=True, text=True, timeout=60, cwd=work_dir,
    )
    if result.returncode != 0:
        return None
    for fname in os.listdir(work_dir):
        if fname.endswith(".svg"):
            return f"{work_dir}/{fname}"
    return None


def colorize_svg(svg_path, wt_pairs, mut_pairs):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    pairs_group = root.find(f".//{{{SVG_NS}}}g[@id='pairs']")
    if pairs_group is None:
        return tree, {}
    wt_set = {frozenset(p) for p in wt_pairs}
    mut_set = {frozenset(p) for p in mut_pairs}
    stats = {"wt": 0, "mut": 0, "other": 0}
    for line in pairs_group.findall(f"{{{SVG_NS}}}line"):
        pid = line.get("id", "")
        if "," not in pid:
            continue
        try:
            a, b = int(pid.split(",")[0]), int(pid.split(",")[1])
        except ValueError:
            continue
        pair = frozenset({a, b})
        if pair in wt_set:
            line.set("stroke", GREEN); line.set("stroke-width", "4"); line.set("opacity", "1")
            stats["wt"] += 1
        elif pair in mut_set:
            line.set("stroke", RED); line.set("stroke-width", "4"); line.set("opacity", "1")
            stats["mut"] += 1
        else:
            line.set("stroke", GRAY); line.set("stroke-width", "1.5"); line.set("opacity", "0.4")
            stats["other"] += 1
    return tree, stats


def svg_to_png(svg_path, png_path, dpi=150):
    cairosvg.svg2png(url=svg_path, write_to=png_path, scale=dpi / 96.0)


def crop_to_content(img, pad=8, white_threshold=0.985):
    """Remove empty RNAplot canvas margins so structures fill each panel."""
    rgb = img[..., :3]
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        mask = (alpha > 0.02) & np.any(rgb < white_threshold, axis=-1)
    else:
        mask = np.any(rgb < white_threshold, axis=-1)
    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def add_zoomed_image(ax, img, zoom=0.30, x=0.5, y=0.52):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        (x, y),
        xycoords=ax.transAxes,
        frameon=False,
        pad=0,
        box_alignment=(0.5, 0.5),
    )
    ax.add_artist(ab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def create_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    COL_TITLES = ['Wild-type', 'No Steering', 'After Steering']
    COL_COLORS = ['#2c3e50', '#7f8c8d', '#27ae60']

    n_cases = len(CASES)
    cases_per_block = 4
    n_blocks = 2
    fig, axes = plt.subplots(cases_per_block, 3 * n_blocks, figsize=(18, 12))
    fig.patch.set_facecolor('white')

    for case_idx, case in enumerate(CASES):
        row = case_idx % cases_per_block
        col_offset = (case_idx // cases_per_block) * 3
        print(f"\n[{case_idx+1}/{n_cases}] {case['name']}...")

        # Load record
        with open(case["cases_json"]) as f:
            records = json.load(f)
        if "case_index" in case:
            record = records[case["case_index"]]
            if record["record_id"] != case["record_id"]:
                print(f"  Case index mismatch: {record['record_id']}")
                continue
        else:
            record = None
            for r in records:
                if r["record_id"] == case["record_id"]:
                    record = r
                    break
        if not record:
            print(f"  Record not found!")
            continue

        wt_pairs = record["diagnostic_wt_pairs_1"]
        mut_pairs = record["diagnostic_mutant_pairs_1"]
        wt_seq = record.get("wt_sequence") or record.get("wt_window") or record.get("sequence")
        wt_struct = record.get("wt_structure") or record.get("wt_structure_window")

        # Load generation data
        df = pd.read_csv(case["gen_csv"])
        sub = df[(df['record_id'] == case['record_id']) & (df['feature_id'] == case['feature_id'])]
        if "case_index" in case and "case_index" in sub.columns:
            sub = sub[sub['case_index'] == case['case_index']]

        ns = sub[(sub['condition'] == 'no_steer') & (sub['sample_idx'] == case['sample_idx'])]
        st = sub[(sub['condition'] == case['condition']) & (sub['sample_idx'] == case['sample_idx'])]

        if len(ns) == 0 or len(st) == 0:
            print(f"  Data not found! ns={len(ns)}, st={len(st)}")
            continue

        ns = ns.iloc[0]
        st = st.iloc[0]

        mut_seq = ns['full_sequence']
        mut_struct = ns['structure']
        steered_seq = st['full_sequence']
        steered_struct = st['structure']

        # Correct interpretation
        if case['target_state'] == 'mutant':
            ns_wt = ns['opposite_hits']
            ns_mut = ns['target_hits']
            st_wt = st['opposite_hits']
            st_mut = st['target_hits']
        else:
            ns_wt = ns['target_hits']
            ns_mut = ns['opposite_hits']
            st_wt = st['target_hits']
            st_mut = st['opposite_hits']

        delta = st_wt - ns_wt
        print(f"  {ns_wt}→{st_wt} WT ({delta:+.0f})")

        # Generate SVGs
        wt_svg = run_rnaplot(wt_seq, wt_struct, f"{case['name']}_wt")
        mut_svg = run_rnaplot(mut_seq, mut_struct, f"{case['name']}_mut")
        steered_svg = run_rnaplot(steered_seq, steered_struct, f"{case['name']}_st")

        # Colorize
        wt_tree, wt_stats = colorize_svg(wt_svg, wt_pairs, mut_pairs)
        mut_tree, mut_stats = colorize_svg(mut_svg, wt_pairs, mut_pairs)
        steered_tree, steered_stats = colorize_svg(steered_svg, wt_pairs, mut_pairs)

        # Save PNGs
        def save_png(tree, suffix):
            svg_f = f"{TEMP_DIR}/{case['name']}_{suffix}.svg"
            png_f = f"{TEMP_DIR}/{case['name']}_{suffix}.png"
            tree.write(svg_f, xml_declaration=True, encoding="utf-8")
            svg_to_png(svg_f, png_f, dpi=120)
            return png_f

        wt_png = save_png(wt_tree, "wt")
        mut_png = save_png(mut_tree, "mut")
        steered_png = save_png(steered_tree, "st")

        pngs = [wt_png, mut_png, steered_png]
        stats_list = [wt_stats, mut_stats, steered_stats]

        for col, (png, stats) in enumerate(zip(pngs, stats_list)):
            ax = axes[row, col_offset + col]
            ax.axis('off')
            ax.set_facecolor('white')

            if os.path.exists(png):
                img = mpimg.imread(png)
                img = crop_to_content(img, pad=10)
                add_zoomed_image(ax, img, zoom=0.30)

            # Column title (first row only)
            if row == 0:
                ax.set_title(COL_TITLES[col], fontsize=15,
                            pad=8, color=COL_COLORS[col])

            # Row label (first column)
            if col == 0:
                color = RNA_COLORS.get(case['rna_type'], '#555555')
                label_y = 0.08 if case['name'] in {'janzen_fam31', 'pepper'} else 0.96
                label_va = 'bottom' if case['name'] in {'janzen_fam31', 'pepper'} else 'top'
                ax.text(0.02, label_y, f"{case['rna_name']}\n({case['rna_type']})",
                       transform=ax.transAxes, ha='left', va=label_va,
                       fontsize=9, color=color)

            # Stats at bottom
            if col == 0:
                label = f"WT: {stats['wt']}"
            else:
                label = f"WT: {stats['wt']} | Mut: {stats['mut']}"
            ax.text(0.5, -0.08, label, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=9, clip_on=False,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', pad=0.25))

    # Legend
    legend_elements = [
        Patch(facecolor=GREEN, label=f'WT-specific pairs'),
        Patch(facecolor=RED, label=f'Mutant-specific pairs'),
        Patch(facecolor=GRAY, label='Other pairs'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
              bbox_to_anchor=(0.5, 0.005), frameon=True)

    fig.suptitle(
        'SAE Feature Steering Restores WT-like Secondary Structures Across Diverse RNA Types',
        fontsize=16, y=0.998
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.985], h_pad=0.55, w_pad=0.05)

    out_png = f"{OUTPUT_DIR}/figure3_multi_rna_v3.png"
    out_pdf = f"{OUTPUT_DIR}/figure3_multi_rna_v3.pdf"
    out_svg = f"{OUTPUT_DIR}/figure3_multi_rna_v3.svg"
    fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n✓ Saved: {out_png}")
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_svg}")


if __name__ == "__main__":
    create_figure()
