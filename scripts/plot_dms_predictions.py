"""Plot stored DMS predictions diagnostically, without inference or label correction."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

REQUIRED = ('source_row', 'sequence_id', 'sequence', 'experimental_value', 'log_likelihood')


def read_predictions(path):
    """Validate every row, preserve CSV order, and never silently exclude records."""
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)) or not set(REQUIRED).issubset(columns):
        raise ValueError('CSV must have unique headers and all required columns: ' + ', '.join(REQUIRED))
    rows, seen_ids, seen_indices = [], set(), set()
    for line_number, item in enumerate(reader, 2):
        if None in item or any(value is None for value in item.values()):
            raise ValueError(f'CSV row {line_number} has an inconsistent column count')
        if any(not item[k].strip() for k in REQUIRED):
            raise ValueError(f'CSV row {line_number} contains an empty required value')
        record_id = item['sequence_id']
        try:
            source_row = int(item['source_row'])
            experimental = float(item['experimental_value'])
            prediction = float(item['log_likelihood'])
        except ValueError as exc:
            raise ValueError(f'CSV row {line_number} has an invalid numeric value') from exc
        if source_row < 0 or source_row in seen_indices:
            raise ValueError('source_row values must be unique nonnegative integers')
        if record_id in seen_ids:
            raise ValueError(f'Duplicate sequence_id: {record_id}')
        if not math.isfinite(experimental) or not math.isfinite(prediction):
            raise ValueError(f'CSV row {line_number} has NaN or Inf')
        seen_ids.add(record_id)
        seen_indices.add(source_row)
        rows.append({
            'source_row': source_row, 'sequence_id': record_id,
            'sequence_sha256': hashlib.sha256(item['sequence'].encode()).hexdigest(),
            'experimental_value': experimental, 'log_likelihood': prediction,
        })
    if len(rows) < 2:
        raise ValueError('At least two prediction rows are required')
    for key in ('experimental_value', 'log_likelihood'):
        if len({row[key] for row in rows}) < 2:
            raise ValueError(f'Undefined Spearman: {key} is constant')
    return rows, hashlib.sha256(raw).hexdigest()


def plot_predictions(input_path, output_directory):
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output}')
    rows, input_hash = read_predictions(input_path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import spearmanr

    x = [r['experimental_value'] for r in rows]
    y = [r['log_likelihood'] for r in rows]
    rho = float(spearmanr(x, y).statistic)
    if not math.isfinite(rho):
        raise ValueError('Undefined Spearman correlation')
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    try:
        ax.scatter(x, y, s=25, color='#356A8A', alpha=0.75, linewidths=0)
        ax.set_xlabel('Experimental value from supplied CSV')
        ax.set_ylabel('Model log-likelihood')
        ax.set_title('Diagnostic plot of supplied predictions', fontsize=12, pad=14)
        ax.text(0.97, 0.04, f'n = {len(rows)}\nSpearman rho = {rho:.9f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9})
        ax.spines[['top', 'right']].set_visible(False)
        fig.text(0.5, 0.018, 'All input rows retained. Plotting does not change predictions or experimental values.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for extension in ('png', 'svg'):
            fig.savefig(output / f'diagnostic_predictions.{extension}', dpi=180)
    finally:
        plt.close(fig)
    report = {
        'status': 'STORED_PREDICTIONS_DIAGNOSTIC_NOT_PAPER_REPRODUCTION',
        'input': str(Path(input_path).resolve()), 'input_sha256': input_hash,
        'n': len(rows), 'spearman': rho, 'model_inference_performed': False,
        'label_provenance': 'Not assessed by the plotting step; supplied experimental values are unchanged.',
        'rows_filtered': 0, 'row_order': 'CSV input order, unchanged', 'rows': rows,
        'plot_scope': 'Diagnostic visualization of previously exported predictions; not Figure 2 reconstruction.',
        'versions': {'matplotlib': matplotlib.__version__, 'scipy': scipy.__version__},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args(argv)
    report = plot_predictions(args.input, args.output_directory)
    print(json.dumps({k: report[k] for k in ('status', 'n', 'spearman', 'input_sha256')}))


if __name__ == '__main__':
    main()
