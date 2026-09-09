"""Exercise source entry points after grouping tutorials and training resources."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('relative', [
    'design/design_paths.py', 'generation/generation_paths.py',
    'prediction/prediction_paths.py', 'interpretability_analysis/reproduce_paths.py',
])
def test_notebook_helpers_find_source_root(relative, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = ROOT / 'examples/notebooks' / relative
    spec = importlib.util.spec_from_file_location('notebook_paths', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.project_root() == ROOT
    if hasattr(module, 'font_path'):
        assert module.font_path().is_file()
    if hasattr(module, 'tools_dir'):
        assert (module.tools_dir() / 'predict.py').is_file()


def test_competitor_evaluator_from_outside_checkout(tmp_path):
    labels = tmp_path / 'labels.csv'
    predictions = tmp_path / 'predictions.csv'
    labels.write_text('variant_id,sequence,label\na,ACGU,1\nb,AGCU,2\nc,AUCG,3\n')
    predictions.write_text('variant_id,sequence,score\nc,AUCG,30\na,ACGU,10\nb,AGCU,20\n')
    output = tmp_path / 'metrics.json'
    subprocess.run([
        sys.executable, str(ROOT / 'examples/reproduction/benchmark/run_competitor.py'),
        'evaluate', '--predictions', str(predictions), '--labels', str(labels),
        '--output', str(output),
    ], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())['spearman'] == 1.0


def test_finetune_help_from_outside_checkout(tmp_path):
    run = subprocess.run([
        sys.executable, str(ROOT / 'training/finetune/train_finetune.py'), '--help',
    ], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert '--config' in run.stdout


def test_sae_shell_resolves_source_root(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    for name in ['config.json', 'tokenizer.json', 'model_weights.pt']:
        (checkpoint / name).write_text('path-resolution fixture only')
    env = {k: v for k, v in os.environ.items() if k not in {'EVA_REPO_ROOT', 'HF_MODEL_ROOT'}}
    env.update(SAE_CKPT_DIR=str(checkpoint), HF_DATA_FASTA=str(ROOT / 'tests/data/train_smoke.fasta'))
    helper = ROOT / 'examples/notebooks/interpretability_analysis/sae_repro_release/scripts/_resolve_paths.sh'
    run = subprocess.run([
        'bash', '-c', 'source "$1"; resolve_sae_paths && printf "%s" "$EVA_REPO_ROOT"',
        'path-check', str(helper),
    ], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert Path(run.stdout) == ROOT
