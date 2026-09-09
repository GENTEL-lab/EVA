#!/usr/bin/env python3
"""CPU-only, strict benchmark coverage and keyed prediction evaluation.

No model execution, downloads, inferred labels, or objective substitutions.
Python >=3.10, standard library only. SVG charts are diagnostic, not paper art.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import statistics
import sys
from decimal import Decimal
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, context):
    if isinstance(value, bool):
        raise ValueError(f'{context}: boolean is not a measurement')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{context}: nonfinite value')
    return number


def unique(values, context):
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f'{context}: nonempty string identifiers required')
    if len(set(values)) != len(values):
        raise ValueError(f'{context}: duplicate identifiers')
    return values


def read_csv(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        unique(columns, str(path))
        if not set(required) <= set(columns):
            raise ValueError(f'{path}: missing columns {set(required) - set(columns)}')
        rows = list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'{path}: empty or ragged CSV')
    return columns, rows


def verified_file(root, item):
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest paths must stay inside the selected data root')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or escaping input: {relative}')
    expected = item.get('sha256', '')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(f'{relative}: a real lowercase SHA256 is required')
    if sha256(path) != expected:
        raise ValueError(f'{relative}: SHA256 mismatch')
    return path


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def spearman(x, y):
    x = [finite(v, 'prediction') for v in x]
    y = [finite(v, 'label') for v in y]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('Spearman requires equally sized vectors with at least 2 rows')
    a, b = ranks(x), ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    da, db = [v-ma for v in a], [v-mb for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in da) * math.fsum(v*v for v in db))
    if not denominator:
        raise ValueError('Spearman undefined for a constant vector')
    return math.fsum(u*v for u, v in zip(da, db)) / denominator


def keyed(path, column, sequence_policy):
    if sequence_policy not in ('exact', 'rna_t_to_u'):
        raise ValueError('Explicit supported sequence_policy required')
    _, rows = read_csv(path, ['variant_id', 'sequence', column])
    unique([r['variant_id'] for r in rows], str(path))
    for row in rows:
        sequence = row['sequence']
        if not sequence or any(c.isspace() for c in sequence):
            raise ValueError('Empty sequence or embedded whitespace')
        if sequence_policy == 'rna_t_to_u':
            sequence = sequence.upper().replace('T', 'U')
            if set(sequence) - set('ACGU'):
                raise ValueError('Noncanonical RNA under rna_t_to_u policy')
        row['sequence'] = sequence
        row[column] = finite(row[column], column)
    return {r['variant_id']: r for r in rows}


def audit_reference(config, data_root):
    groups = []
    unique([g['id'] for g in config['groups']], 'reference group')
    for group in config['groups']:
        models = unique(group['models'], 'models')
        datasets = unique(group['datasets'], 'datasets')
        path = verified_file(data_root, group['table'])
        columns, rows = read_csv(path, ['Model'])
        pairs, counts, model_counts, seen = {}, {}, {}, set()
        for row in rows:
            if group['layout'] == 'long':
                cells = [(row['Dataset'], row['Spearman'])]
            elif group['layout'] == 'wide':
                cells = [(name, row[name]) for name in columns if name != 'Model']
            else:
                raise ValueError('Unsupported table layout')
            for dataset, value in cells:
                key = (dataset, row['Model'])
                if key in seen:
                    raise ValueError(f'Duplicate reference pair: {key}')
                seen.add(key)
                if not value.strip():
                    continue
                rho = finite(value, str(key))
                if not -1 <= rho <= 1:
                    raise ValueError(f'Spearman outside [-1,1]: {key}')
                if 'Spearman_abs' in row and abs(finite(row['Spearman_abs'], 'abs') - abs(rho)) > 1e-14:
                    raise ValueError(f'Inconsistent stored absolute Spearman: {key}')
                if 'N_samples' in row:
                    n = int(row['N_samples'])
                    if n < 2:
                        raise ValueError('Invalid N_samples')
                    counts.setdefault(dataset, set()).add(n)
                    model_counts.setdefault(dataset, {})[row['Model']] = n
                pairs[key] = rho
        required = {(d, m) for d in datasets for m in models}
        extra = seen - required
        if extra:
            raise ValueError(f'Unexpected reference pairs: {sorted(extra)}')
        missing = sorted(required - set(pairs))
        inconsistent_n = {d: sorted(n) for d, n in counts.items() if len(n) != 1}
        summaries = []
        for model in models:
            values = [pairs[(d, model)] for d in datasets if (d, model) in pairs]
            summaries.append({'model': model, 'n_assays': len(values), 'expected_assays': len(datasets),
                              'mean_abs_spearman': statistics.mean(abs(v) for v in values) if len(values) == len(datasets) else None})
        groups.append({'id': group['id'], 'source_sha256': sha256(path),
                       'expected_pairs': len(required), 'observed_pairs': len(pairs),
                       'missing_pairs': [{'dataset': d, 'model': m} for d, m in missing],
                       'inconsistent_sample_counts': inconsistent_n,
                       'sample_counts_by_model_for_inconsistent_assays': {d: model_counts[d] for d in inconsistent_n},
                       'coverage_complete': not missing, 'sample_counts_consistent': not inconsistent_n,
                       'model_summary': summaries,
                       'complete': not missing and not inconsistent_n})
    return {'operation': 'audit_and_aggregate_published_summary_tables_not_model_inference',
            'groups': groups, 'complete': all(g['complete'] for g in groups),
            'paper_reproduction_claim': False}


def evaluate_manifest(manifest, data_root):
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    models = unique(manifest['models'], 'models')
    datasets = manifest['datasets']
    unique([d['id'] for d in datasets], 'datasets')
    required = {(d['id'], m) for d in datasets for m in models}
    predictions = {}
    for row in manifest['predictions']:
        key = (row['dataset'], row['model'])
        if key in predictions or key not in required:
            raise ValueError(f'Duplicate or unexpected prediction pair: {key}')
        predictions[key] = row
    missing = sorted(required - set(predictions))
    results, failures, unresolved = [], [], []
    for dataset in datasets:
        label_file = verified_file(data_root, dataset['labels'])
        labels = keyed(label_file, 'label', dataset['sequence_policy'])
        if dataset.get('source_status') != 'verified':
            unresolved.append(dataset['id'])
        for model in models:
            job = predictions.get((dataset['id'], model))
            if job is None:
                continue
            try:
                pred_file = verified_file(data_root, job['file'])
                rows = keyed(pred_file, 'score', dataset['sequence_policy'])
                if set(rows) != set(labels):
                    raise ValueError('Prediction/label ID sets differ; no row dropping permitted')
                if any(rows[k]['sequence'] != labels[k]['sequence'] for k in labels):
                    raise ValueError('Prediction/label sequence mismatch')
                rho = spearman([rows[k]['score'] for k in labels], [labels[k]['label'] for k in labels])
                reference = job.get('published_spearman')
                comparison = None
                if reference is not None:
                    if not isinstance(reference, str):
                        raise ValueError('Published value must be a decimal string preserving published precision')
                    expected = Decimal(reference)
                    if not expected.is_finite() or not -1 <= expected <= 1:
                        raise ValueError('Invalid published Spearman')
                    half_unit = Decimal(10) ** expected.as_tuple().exponent / 2
                    difference = Decimal(str(rho)) - expected
                    comparison = {'published': reference, 'difference': str(difference),
                                  'half_last_decimal_unit': str(half_unit),
                                  'matches_published_precision': abs(difference) <= half_unit}
                results.append({'dataset': dataset['id'], 'model': model, 'n': len(labels),
                                'spearman': rho, 'abs_spearman': abs(rho), 'comparison': comparison,
                                'predictions_sha256': sha256(pred_file), 'labels_sha256': sha256(label_file),
                                'sequence_policy': dataset['sequence_policy'],
                                'inference_provenance': job.get('inference_provenance'),
                                'inference_provenance_verified_by_this_tool': False})
            except (ValueError, OSError, KeyError) as error:
                failures.append({'dataset': dataset['id'], 'model': model, 'error': str(error)})
    complete = not missing and not failures
    summaries = []
    for model in models:
        values = [r['abs_spearman'] for r in results if r['model'] == model]
        summaries.append({'model': model, 'n_assays': len(values), 'expected_assays': len(datasets),
                          'mean_abs_spearman': statistics.mean(values) if len(values) == len(datasets) else None})
    all_compared = len(results) == len(required) and all(r['comparison'] is not None for r in results)
    matches = all_compared and all(r['comparison']['matches_published_precision'] for r in results)
    return {'operation': 'recompute_metrics_from_keyed_predictions_not_model_inference',
            'expected_pairs': len(required), 'completed_pairs': len(results), 'complete': complete,
            'missing_pairs': [{'dataset': d, 'model': m} for d, m in missing], 'failures': failures,
            'unresolved_label_sources': unresolved, 'results': results, 'model_summary': summaries,
            'all_references_compared': all_compared, 'all_match_published_precision': matches,
            'paper_reproduction_claim': False}


def plot_report(report, output):
    summaries = [(g['id'], g['model_summary']) for g in report['groups']] if 'groups' in report else [('evaluation', report['model_summary'])]
    for index, (name, rows) in enumerate(summaries):
        height, left, top, width = 135 + 34 * len(rows), 320, 75, 510
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="Arial, sans-serif" font-size="13" fill="#243342">',
               f'<text x="24" y="29" font-size="19">{html.escape(name)}: benchmark summary</text>',
               '<text x="24" y="51" fill="#5b6770">Cached metric analysis; no model inference. Counts show available / expected assays.</text>']
        for tick in range(6):
            x = left + width * tick / 5
            svg.extend([f'<path d="M{x},{top-12} V{height-54}" stroke="#e4e8ed"/>',
                        f'<text x="{x}" y="{height-33}" text-anchor="middle">{tick/5:.1f}</text>'])
        for i, row in enumerate(rows):
            y = top + 34 * i
            value = row['mean_abs_spearman']
            label = html.escape(f"{row['model']} ({row['n_assays']}/{row['expected_assays']})")
            svg.append(f'<text x="{left-12}" y="{y+15}" text-anchor="end">{label}</text>')
            if value is not None:
                length = width * value
                svg.extend([f'<rect x="{left}" y="{y}" width="{length}" height="22" fill="#557d9a"/>',
                            f'<text x="{left+length+6}" y="{y+15}">{value:.4f}</text>'])
            else:
                svg.append(f'<text x="{left+6}" y="{y+15}" fill="#994433">missing / incomplete</text>')
        svg.extend([f'<text x="{left+width/2}" y="{height-8}" text-anchor="middle">Mean absolute Spearman across assays</text>', '</g></svg>'])
        (output / f'summary_{index}.svg').write_text('\n'.join(svg) + '\n')


def write_metric_tables(report, manifest, output):
    """Emit plot-compatible tables only after every declared pair computed."""
    if not report['complete']:
        return
    rows = report['results']
    with (output / 'metrics_long.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Dataset', 'Model', 'Spearman', 'N_samples', 'Spearman_abs'])
        writer.writeheader()
        writer.writerows({'Dataset': r['dataset'], 'Model': r['model'], 'Spearman': r['spearman'],
                          'N_samples': r['n'], 'Spearman_abs': r['abs_spearman']} for r in rows)
    datasets = [d['id'] for d in manifest['datasets']]
    lookup = {(r['dataset'], r['model']): r['spearman'] for r in rows}
    with (output / 'metrics_wide.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Model', *datasets])
        writer.writeheader()
        writer.writerows({'Model': model, **{d: lookup[(d, model)] for d in datasets}} for model in manifest['models'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit-reference', 'evaluate'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new directory to retain prior evidence')
    config = json.loads(args.manifest.read_text())
    report = audit_reference(config, args.data_root) if args.command == 'audit-reference' else evaluate_manifest(config, args.data_root)
    report.update({'manifest_sha256': sha256(args.manifest), 'script_sha256': sha256(__file__),
                   'python': platform.python_version(), 'platform': platform.platform()})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.command == 'evaluate':
        write_metric_tables(report, config, args.output)
    if args.plot:
        plot_report(report, args.output)
    print(json.dumps({'complete': report['complete'], 'report': str(args.output / 'report.json')}))
    if not report['complete']:
        return 2
    if args.command == 'evaluate' and (report['unresolved_label_sources'] or not report['all_match_published_precision']):
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
