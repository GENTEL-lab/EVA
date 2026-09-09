"""Fresh, version-bound EVA-1.4B CLM scoring of the current 135-row Milena assay.

Uses the bundled historical implementation and its recorded sum-score protocol.
Current labels are author-confirmed; archive metadata is not used as ground truth.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_historical_benchmark import read_fasta, finite_vector, align_archive, sha256, spearman

MANIFEST = ROOT / 'examples/reproduction/milena_14b/manifest.json'


def validate_inputs(root=ROOT, manifest_path=MANIFEST):
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['input_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError('Input changed after protocol freeze: ' + name)
    records = read_fasta(root / manifest['fasta'])
    payload = json.loads((root / manifest['labels']).read_text())
    labels = finite_vector(payload['intensities'], 'labels')
    if len(records) != manifest['n'] or len(labels) != manifest['n'] or payload['num_sequences'] != manifest['n']:
        raise ValueError('Expected complete 135-row assay')
    if any(set(row['sequence']) - set('ACGU') for row in records):
        raise ValueError('Noncanonical RNA')
    archive = json.loads((root / manifest['archive']).read_text())
    archived_scores = align_archive(records, archive)
    with (root / manifest['reference_table']).open() as f:
        rows = [r for r in csv.DictReader(f) if r['Dataset'] == manifest['dataset'] and r['Model'] == manifest['model']]
    if len(rows) != 1 or int(rows[0]['N_samples']) != manifest['n']:
        raise ValueError('Reference row/count mismatch')
    return manifest, records, labels, archived_scores, rows[0]['Spearman']


def align_fresh(records, path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    mapped = {}
    for row in rows:
        key = row['variant_id']
        if key in mapped:
            raise ValueError('Duplicate fresh prediction ID')
        mapped[key] = row
    if set(mapped) != {r['id'] for r in records}:
        raise ValueError('Fresh predictions do not cover exactly the frozen assay')
    for rec in records:
        if mapped[rec['id']]['sequence'].upper().replace('T','U') != rec['sequence']:
            raise ValueError('Fresh prediction sequence mismatch')
    return finite_vector([mapped[r['id']]['fresh_score'] for r in records], 'fresh scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-limit-gib', type=float, default=5)
    parser.add_argument('--check-only', action='store_true', help='Verify inputs/checkpoint without inference; never a reproduction pass')
    parser.add_argument('--strict-reference', action='store_true',
                        help='Return 2 after a valid run if its metric differs at the stored reference precision')
    args = parser.parse_args()
    if not args.device.startswith('cuda:') or not 0 < args.memory_limit_gib <= 5:
        parser.error('Select one CUDA device and a memory limit in (0, 5] GiB')
    manifest, records, labels, archive, reference_text = validate_inputs()
    checkpoint = args.checkpoint.resolve()
    hashes = {name: sha256(checkpoint/name) for name in manifest['checkpoint']['sha256']}
    if hashes != manifest['checkpoint']['sha256']:
        raise ValueError('Checkpoint files do not match the pinned public release')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status':'INPUTS_VERIFIED_ONLY','execution_status':'INPUTS_VERIFIED_ONLY',
              'comparison_status':'NOT_RUN','strict_reference':args.strict_reference,
              'fresh_inference':False,'n':len(records),
              'checkpoint':manifest['checkpoint'],'input_sha256':manifest['input_sha256'],
              'label_basis':manifest['label_basis'],'protocol':manifest['protocol'],
              'paper_result_reproduced':False}
    dest = args.output/'report.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    if args.check_only:
        print(json.dumps(report,indent=2))
        return 0
    command = [sys.executable, str(ROOT/'examples/reproduction/benchmark/run_historical_14b.py'),
               '--source-dir',str(ROOT/'examples/reproduction/benchmark/historical_14b'),
               '--checkpoint',str(checkpoint),'--archive',str(ROOT/manifest['archive']),
               '--output',str(args.output/'inference'),'--device',args.device,
               '--memory-limit-gib',str(args.memory_limit_gib),
               '--expected-weights-sha256',hashes['model_weights.pt']]
    start = time.monotonic()
    try:
        with (args.output/'inference.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        scores = align_fresh(records,args.output/'inference/predictions.csv')
        rho = spearman(scores,labels)
        reference = float(reference_text)
        tolerance = float(Decimal(5).scaleb(Decimal(reference_text).as_tuple().exponent-1))
        matched = abs(rho-reference) <= tolerance
        inference = json.loads((args.output/'inference/report.json').read_text())
        report.update(status='METRIC_MATCH' if matched else 'METRIC_DIFFERENCE',
                      comparison_status='MATCH' if matched else 'DIFFERENCE',
                      fresh_inference=True,spearman=rho,reference=reference,
                      absolute_difference=abs(rho-reference),reference_decimal=reference_text,
                      comparison_tolerance=tolerance,matches_reference_precision=matched,
                      comparison_basis="Stored reference CSV decimal precision; not a scientific acceptance tolerance",
                      archive_spearman=spearman(archive,labels),
                      prediction_rank_correlation=spearman(scores,archive),
                      archive_vector_exact=inference['exact_equal_count']==len(records),
                      paper_result_reproduced=False,
                      interpretation='Execution and reference comparison are reported separately. A successful run does not establish reproduction of every paper result. The documented Milena difference is retained as a numerical observation.',
                      inference_seconds=inference['scoring_seconds'],
                      peak_gpu_allocated_bytes=inference['cuda_peak_allocated_bytes'],
                      versions=inference['environment'],gpu=inference['gpu'])
        with (args.output/'predictions.csv').open('w') as f:
            writer=csv.writer(f)
            writer.writerow(['source_row','sequence_id','sequence','experimental_value','log_likelihood'])
            writer.writerows((i,r['id'],r['sequence'],y,s) for i,(r,y,s) in enumerate(zip(records,labels,scores)))
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_dms_predictions.py'),
                        '--input',str(args.output/'predictions.csv'),
                        '--output-directory',str(args.output/'plot')],check=True)
        report['execution_status'] = 'COMPLETED'
    except Exception as exc:
        report.update(status='FAILED',execution_status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['runner_sha256']=sha256(Path(__file__))
        dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    return 2 if args.strict_reference and not report['matches_reference_precision'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
