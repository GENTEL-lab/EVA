"""Evaluation-analysis dependency coverage and a real, CPU-only data-load smoke."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_dependencies_are_declared():
    requirements = (ROOT / "scripts/docker/requirements.txt").read_text().splitlines()
    project = (ROOT / "pyproject.toml").read_text()
    notebook = project.split("notebook = [", 1)[1].split("]", 1)[0]
    for name, version in (("pandas", "2.2.3"), ("seaborn", "0.13.2"),
                          ("scikit-learn", "1.5.2")):
        assert f"{name}=={version}" in requirements
        assert f'"{name}' in notebook


def test_analysis_cli_and_actual_json_csv_load(tmp_path):
    for module in ("pandas", "seaborn", "sklearn"):
        pytest.importorskip(module, reason="Install the notebook extra for analysis smoke")
    script = ROOT / "training/eval/scripts/analyze_results.py"
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"),
               PYTHONDONTWRITEBYTECODE="1")
    help_result = subprocess.run([sys.executable, str(script), "--help"],
                                 env=env, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    assert "--data_path" in help_result.stdout
    data = [{"checkpoint_name": f"checkpoint-{step}", "metrics":
             {"val_loss": loss, "val_perplexity": loss + 1, "total_tokens": 100},
             "evaluation_time": 1, "model_config": {"hidden_size": 32}}
            for step, loss in ((100, 3.0), (200, 2.0), (300, 1.0))]
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(data))
    program = """
import importlib.metadata as metadata
import runpy, sys
from pathlib import Path
from packaging.requirements import Requirement
from packaging.version import Version
for name in ('pandas', 'seaborn', 'scikit-learn'):
    for declaration in metadata.requires(name) or []:
        requirement = Requirement(declaration)
        if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
            actual = metadata.version(requirement.name)
            assert not requirement.specifier or Version(actual) in requirement.specifier, declaration
module = runpy.run_path(sys.argv[1])
analyzer = module['ResultAnalyzer']({'output_dir': sys.argv[3]})
frame = analyzer.load_experiment_data(sys.argv[2])
assert frame['step'].tolist() == [100, 200, 300]
assert analyzer.basic_statistics(frame)['loss_stats']['improvement'] == 2.0
csv_path = Path(sys.argv[3]) / 'metrics.csv'
frame.to_csv(csv_path, index=False)
loaded = analyzer.load_experiment_data(str(csv_path))
assert loaded['val_loss'].tolist() == [3.0, 2.0, 1.0]
print('JSON_ROWS=3 CSV_ROWS=3 LOSS_IMPROVEMENT=2.0 DEPENDENCY_CONSTRAINTS=PASS')
"""
    result = subprocess.run([sys.executable, "-c", program, str(script), str(source),
                             str(tmp_path / "analysis")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "DEPENDENCY_CONSTRAINTS=PASS" in result.stdout
