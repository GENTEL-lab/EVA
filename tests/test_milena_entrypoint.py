import csv
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import subprocess
import sys
import pytest
from scripts import reproduce_milena as runner
from scripts.reproduce_milena import validate_inputs, align_fresh, MANIFEST, ROOT

class MilenaEntrypointTests(unittest.TestCase):
    def test_frozen_current_complete_assay(self):
        manifest, records, labels, archive, reference = validate_inputs()
        self.assertEqual(len(records), 135)
        self.assertEqual(len(labels), len(archive))
        self.assertEqual(manifest['protocol']['reduce'], 'sum')
        self.assertEqual(manifest['checkpoint']['subdirectory'], 'EVA_1.4B_CLM')

    def test_changed_manifest_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(MANIFEST.read_text())
            manifest['input_sha256'][manifest['fasta']] = '0'*64
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                validate_inputs(ROOT,path)

    def test_predictions_align_by_id_and_sequence(self):
        records=[{'id':'a','sequence':'AU'},{'id':'b','sequence':'CG'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.csv'
            path.write_text('variant_id,sequence,fresh_score\nb,CG,2\na,AU,1\n')
            self.assertEqual(align_fresh(records,path),[1.,2.])
            for data in ['a,AU,1\na,AU,2\n','a,AU,1\n','a,AC,1\nb,CG,2\n','a,AU,nan\nb,CG,2\n']:
                path.write_text('variant_id,sequence,fresh_score\n'+data)
                with self.assertRaises(ValueError): align_fresh(records,path)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """Exercise CLI orchestration with a deterministic backend, without GPU weights."""
    records = [{'id': k, 'sequence': s} for k, s in [('a', 'AU'), ('b', 'CG'), ('c', 'AC')]]
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model_weights.pt').write_bytes(b'test checkpoint, not a model')
    manifest = {'checkpoint': {'sha256': {'model_weights.pt': hashlib.sha256(
        (checkpoint / 'model_weights.pt').read_bytes()).hexdigest()}},
        'input_sha256': {}, 'label_basis': 'synthetic test fixture',
        'protocol': {}, 'archive': 'unused-test-archive'}
    monkeypatch.setattr(runner, 'validate_inputs', lambda: (manifest, records, [1, 2, 3], [1, 2, 3], '1.000'))
    state = {'scores': [1, 3, 2], 'fail': None, 'calls': []}

    def backend(command, **kwargs):
        name = Path(command[1]).name
        state['calls'].append(name)
        if state['fail'] == name:
            raise subprocess.CalledProcessError(1, command)
        if name == 'run_historical_14b.py':
            out = Path(command[command.index('--output') + 1])
            out.mkdir()
            with (out / 'predictions.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['variant_id', 'sequence', 'fresh_score'])
                writer.writerows((r['id'], r['sequence'], score) for r, score in zip(records, state['scores']))
            (out / 'report.json').write_text(json.dumps({'exact_equal_count': 0,
                'scoring_seconds': 0.1, 'cuda_peak_allocated_bytes': 0,
                'environment': {}, 'gpu': 'synthetic backend'}))
        elif name == 'plot_dms_predictions.py':
            out = Path(command[command.index('--output-directory') + 1])
            out.mkdir()
            (out / 'diagnostic.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        else:
            raise AssertionError('Unexpected execution path: ' + name)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, 'run', backend)

    def invoke(*flags):
        output = tmp_path / 'output'
        monkeypatch.setattr(sys, 'argv', ['reproduce_milena.py', '--checkpoint', str(checkpoint),
                                          '--output', str(output), *flags])
        return runner.main(), json.loads((output / 'report.json').read_text())

    return invoke, state, tmp_path


@pytest.mark.parametrize('strict,match,expected', [(False, False, 0), (True, False, 2),
                                                 (False, True, 0), (True, True, 0)])
def test_execution_success_is_separate_from_reference_comparison(run_fixture, strict, match, expected):
    invoke, state, root = run_fixture
    if match:
        state['scores'] = [1, 2, 3]
    code, report = invoke(*(['--strict-reference'] if strict else []))
    assert code == expected
    assert report['execution_status'] == 'COMPLETED'
    assert report['comparison_status'] == ('MATCH' if match else 'DIFFERENCE')
    assert report['matches_reference_precision'] == match
    assert report['absolute_difference'] == (0 if match else 0.5)
    assert report['fresh_inference'] and not report['paper_result_reproduced']
    assert (root / 'output/predictions.csv').is_file()
    assert state['calls'] == ['run_historical_14b.py', 'plot_dms_predictions.py']


def test_check_only_never_claims_inference(run_fixture):
    invoke, state, _ = run_fixture
    code, report = invoke('--check-only', '--strict-reference')
    assert code == 0 and not state['calls']
    assert report['execution_status'] == 'INPUTS_VERIFIED_ONLY'
    assert report['comparison_status'] == 'NOT_RUN'
    assert not report['fresh_inference']


@pytest.mark.parametrize('failure', ['run_historical_14b.py', 'plot_dms_predictions.py'])
def test_backend_or_plot_failure_preserves_failure_report(run_fixture, failure):
    invoke, state, root = run_fixture
    state['fail'] = failure
    with pytest.raises(subprocess.CalledProcessError):
        invoke()
    report = json.loads((root / 'output/report.json').read_text())
    assert report['status'] == report['execution_status'] == 'FAILED'
    assert 'error' in report


@pytest.mark.parametrize('scores', [[1, 2], [1, float('nan'), 3], [2, 2, 2]])
def test_invalid_backend_output_is_not_a_success(run_fixture, scores):
    invoke, state, root = run_fixture
    state['scores'] = scores
    with pytest.raises(ValueError):
        invoke()
    assert json.loads((root / 'output/report.json').read_text())['execution_status'] == 'FAILED'
    assert state['calls'] == ['run_historical_14b.py']


def test_changed_checkpoint_fails_before_backend_launch(run_fixture):
    invoke, state, root = run_fixture
    (root / 'checkpoint/model_weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checkpoint files do not match'):
        invoke()
    assert not state['calls']
