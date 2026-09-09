"""Score a complete released ncRNA assay and compare with its archived correlation.

Labels in the published JSON are positional. Their association is checked against
the exact FASTA hash; this workflow preserves the original order and records IDs.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', default='Milena_2021_cata')
    p.add_argument('--reference-model', choices=['eva_21m_score', 'eva_1.4b_score'], default='eva_21m_score',
                   help='Select only the archived comparison row; this does not change the mean sequence-only scoring protocol. Use the historical runner for the recorded 1.4B sum protocol.')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rna-type', default=None)
    p.add_argument('--tolerance', type=float, default=None,
                   help='Explicit diagnostic tolerance; default is half the last published decimal place')
    args = p.parse_args()
    if args.batch_size < 1 or (args.tolerance is not None and (not math.isfinite(args.tolerance) or args.tolerance < 0)):
        p.error('batch size must be positive and tolerance must be finite and nonnegative')
    from scipy.stats import spearmanr
    import torch
    from tools.utils.io import read_fasta
    from tools.utils.model import ModelLoader
    from tools.utils.scorers.score_worker import score_in_batches
    from tools.utils.conditions.rna_types import get_rna_token

    fasta = ROOT / 'notebooks/prediction/data/ncRNA/fasta' / f'{args.dataset}.fasta'
    label_file = ROOT / 'notebooks/prediction/data/ncRNA/label' / f'{args.dataset}_intensities.json'
    table = ROOT / 'notebooks/prediction/data/ncRNA_13datasets_spearman.csv'
    labels = json.loads(label_file.read_text())
    if 'intensities' not in labels:
        raise ValueError('This assay has multiple readouts; explicitly resolve the paper readout first')
    y = labels['intensities']
    records = read_fasta(str(fasta))
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)) or len(records) != len(y) or len(y) != labels['num_sequences']:
        raise ValueError('Duplicate FASTA IDs or label/sequence counts differ')
    if not all(math.isfinite(float(v)) for v in y):
        raise ValueError('Non-finite experimental labels')
    with table.open() as f:
        refs = [r for r in csv.DictReader(f) if r['Dataset'] == args.dataset and r['Model'] == args.reference_model]
    if len(refs) != 1 or int(refs[0]['N_samples']) != len(y):
        raise ValueError('Expected exactly one reference with the same sample count')
    reference = float(refs[0]['Spearman'])
    tolerance = args.tolerance if args.tolerance is not None else float(
        Decimal(5).scaleb(Decimal(refs[0]['Spearman']).as_tuple().exponent - 1))
    sequences = [s.upper().replace('T', 'U') for _, s in records]
    if any(not s or set(s) - set('ACGU') for s in sequences):
        raise ValueError('Empty or noncanonical RNA sequence')
    prefix = f'|{get_rna_token(args.rna_type)}|' if args.rna_type else ''
    checkpoint_files = {n: digest(args.checkpoint / n) for n in ['config.json', 'tokenizer.json', 'model_weights.pt']}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    start = time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    model, tokenizer = ModelLoader(str(args.checkpoint)).load(device=args.device)
    scores = score_in_batches(model, tokenizer, [prefix + s for s in sequences], args.device,
                             batch_size=args.batch_size, reduce_method='mean', exclude_special_tokens=True)
    rho = float(spearmanr(scores, y).statistic)
    if not math.isfinite(rho):
        raise ValueError('Undefined Spearman correlation')
    with (args.output / 'predictions.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['source_row', 'sequence_id', 'sequence', 'experimental_value', 'log_likelihood'])
        w.writerows((i, rid, s, v, score) for i, ((rid, _), s, v, score) in enumerate(zip(records, sequences, y, scores)))
    def git(*cmd):
        if not (ROOT / '.git').exists():
            return 'unavailable (source archive without .git; see source_sha256)'
        return subprocess.check_output(['git', '-c', f'safe.directory={ROOT}', '-C', str(ROOT), *cmd], text=True).strip()
    sources = [Path('pyproject.toml')] + sorted(Path(p).relative_to(ROOT) for d in ['eva','tools','scripts'] for p in (ROOT / d).rglob('*.py'))
    report = {
        'dataset': args.dataset, 'n': len(y), 'spearman': rho, 'reference': reference,
        'reference_model': args.reference_model, 'absolute_difference': abs(rho - reference),
        'tolerance': tolerance, 'matches_archived_result': abs(rho - reference) <= tolerance,
        'comparison_basis': 'explicit diagnostic tolerance' if args.tolerance is not None else 'published decimal precision',
        'provenance_limit': 'This runner uses mean sequence-only scoring; reference-model selects only a comparison row. The released EVA-21M weight/config/tokenizer match recovered historical files, but their binding to the published sequence-to-label inputs is not established.',
        'seconds': time.perf_counter() - start,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
        'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
        'protocol': {'rna_type': args.rna_type, 'reduce': 'mean', 'excluded_targets': ['5','3','<eos>'],
                     'label_join': 'position in hash-identified released FASTA', 'batch_size': args.batch_size, 'seed': 42},
        'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [fasta, label_file, table]},
        'checkpoint_sha256': checkpoint_files,
        'git_commit': git('rev-parse','HEAD'), 'git_status': git('status','--porcelain'),
        'source_sha256': {str(p): digest(ROOT / p) for p in sources},
        'versions': {n: importlib.metadata.version(n) for n in ['torch','transformers','numpy','scipy','tokenizers']},
        'command': sys.argv,
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k,v in report.items() if k not in ['source_sha256','input_sha256','checkpoint_sha256','git_status']}, indent=2))
    return 0 if report['matches_archived_result'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
