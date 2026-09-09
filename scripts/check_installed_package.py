"""Verify wheel imports, package data and installed CLIs outside the source checkout."""
import argparse
import importlib
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if Path.cwd().resolve().is_relative_to(source):
        raise ValueError('Run this check from outside the source checkout')
    locations = {}
    for name in ('eva', 'tools.generate', 'tools.predict', 'tools.directed_evolution'):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if location.is_relative_to(source):
            raise ValueError(f'{name} was imported from the source tree: {location}')
        locations[name] = str(location)
    tokenizer = importlib.resources.files('eva').joinpath('tokenizer.json')
    json.loads(tokenizer.read_text())
    data_files = [p.name for p in importlib.resources.files('tools.utils.data').iterdir() if p.name.endswith('.tsv')]
    if not data_files:
        raise ValueError('The installed wheel is missing lineage TSV data')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for cli in ('eva-generate', 'eva-predict', 'eva-evolve'):
        executable = shutil.which(cli)
        if not executable:
            raise ValueError('Missing installed CLI: ' + cli)
        subprocess.run([executable, '--help'], env=env, check=True, capture_output=True, text=True)
    print(json.dumps({'status': 'PASS', 'imports': locations, 'package_data': data_files,
                      'cli_help_checks': 3, 'model_inference': False}, indent=2))


if __name__ == '__main__':
    main()
