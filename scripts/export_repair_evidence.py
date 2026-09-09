"""Export this local repair's patch, source hashes and 49-label coverage update.

Does not commit, stage, push or modify the original inventory.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    def git(*cmd):
        return subprocess.check_output(['git', '-C', str(ROOT), *cmd])
    changed = git('diff', '--name-only', '-z').decode().strip('\0').split('\0')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().strip('\0').split('\0')
    files = sorted(p for p in set(changed + untracked) if p and (ROOT / p).is_file())
    patch = git('diff', '--binary', 'HEAD')
    for rel in sorted(p for p in untracked if p):
        diff = subprocess.run(['git', 'diff', '--no-index', '--binary', '--', '/dev/null', rel],
                              cwd=ROOT, capture_output=True)
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr.decode())
        patch += diff.stdout
    (args.output / 'eva_repair.patch').write_bytes(patch)
    manifest = {'baseline_commit': git('rev-parse', 'HEAD').decode().strip(),
                'source_files': {rel: sha(ROOT / rel) for rel in files},
                'patch_sha256': sha(args.output / 'eva_repair.patch'),
                'github_pushed': False, 'paper_results_reproduced': False}
    (args.output / 'repair_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with args.inventory.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 49 or len({r['original_label'] for r in rows}) != 49:
        raise ValueError('Expected exactly 49 unique original paper labels')
    related = {'fig:fig2', 'tab:stab_ncrna_benchmark'}
    numerical_updates = {
        'fig:sfig11': ('Archived raw-data recomputation: selected 401/401 likelihood groups, '
            'p=9.370116963725616e-68, d=1.70; eight fixed author-selected structure records verified. '
            'One historical Pepper no_steer/0x pair freshly matches archived sequences, structures '
            'and scores; not a fresh full SAE experiment. See sae/cpu_run_final and '
            'sae/historical_gpu_run/run1.'),
        'tab:stab_steering_generation': ('Archived CSV-to-LaTeX reconstruction matches 213/225. '
            'Raw-record generation success counts not fully reconstructed; original numbers retained. '
            'See sae/cpu_run_final/table_s30_report.json.'),
        'tab:stab_essentiality_species': ('Archived 42-species/139008-gene summary retained. '
            'Separately, ten full 95538-gene position-ablation groups were recomputed from stored '
            'predictions; not a fresh essentiality model inference. See essentiality/results.'),
        'fig:fig4': ('Available Source Data assembled: 83 aptamer records, independent experiment n=3 '
            'confirmed; Figure 4d has 200 values per distribution and SVG-verified box definitions. '
            'Not a new wet-lab experiment or complete Figure 4 reproduction.'),
        'fig:sfig8': ('Aptamer repeat Source Data and n=3 independent experiments documented. '
            'Structural-reference provenance remains incomplete; not a complete figure rerun.'),
        'tab:stab_broccoli_activity': ('Original repeat values preserved in Source Data; n=3 independent '
            'experiments confirmed. No new experiment performed.'),
        'tab:stab_pepper_activity': ('Original repeat values preserved in Source Data; n=3 independent '
            'experiments confirmed. No new experiment performed.'),
    }
    for row in rows:
        row['previous_runtime_validation'] = row.pop('runtime_validation')
        row['runtime_validation'] = (
            'Partial: complete EVA-21M/Milena assay executed; rho=0.9037308669414685 vs archived 0.5074679659247847. Figure/table NOT reproduced.'
            if row['original_label'] in related else
            'No full numerical rerun for this label. Shared workflow smoke tests do not validate this panel/table.')
        row['runtime_validation'] = numerical_updates.get(row['original_label'], row['runtime_validation'])
        if row['original_label'] in related:
            row['runtime_validation'] += (
                ' Historical 1.4B stored scores reproduce rho=0.84 arithmetically. '
                'A fresh 135-sequence historical 1.4B run gives rho=0.84; '
                'Correlations are displayed to two decimal places. '
                '133/135 header fitness values differ from positional labels; original label source '
                'unresolved. See benchmark/archive_arithmetic_verified.')
    (args.output / 'paper_result_coverage_updated.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Updated paper-result coverage (49 labels)', '',
             'Original source/data/calculation mappings are preserved in the companion JSON.',
             'Historical “not run by request” status is retained as previous_runtime_validation.', '',
             '| Label | Kind | Current numerical validation |', '|---|---|---|']
    for row in rows:
        lines.append(f"| {row['original_label']} | {row['kind']} | {row['runtime_validation']} |")
    (args.output / 'paper_result_coverage_updated.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'changed_source_files': len(files), 'paper_labels': len(rows),
                      'patch_sha256': manifest['patch_sha256']}, indent=2))


if __name__ == '__main__':
    main()
