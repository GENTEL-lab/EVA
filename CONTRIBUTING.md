# Contributing to EVA

Open an issue before proposing a substantial architecture or scoring change.
Small documentation corrections and bug fixes can be submitted as pull requests.
Keep changes focused and explain the behavior they fix.

## Development setup

Use Python 3.10 or 3.11 in a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark,design,dev,notebook]'
python -m pip check
python -m pytest -q tests
python scripts/check_repository.py
```

These are CPU checks. Full MoE inference requires the
[GPU environment](docs/INSTALLATION.md). State the tests and environment used
in your pull request, and identify checks that were not run.

## Reporting a problem

Include the Git commit, Python/PyTorch/CUDA versions, OS, GPU model if relevant,
installation method, exact command and error log. Prefer a small synthetic or
openly available non-sensitive input. Remove credentials and confidential data.

## Reproducibility changes

Preserve IDs, checksums and scoring conventions. Do not fix a failure by
silently dropping samples, changing labels, substituting a model or changing
the objective. Describe deliberate API/protocol changes and test the affected
behavior. Historical snapshots and expected-result records retain their hashes;
add dated evidence instead of rewriting measurements. Do not commit checkpoints,
generated results or local environments.

## Package validation

Run `python -m build`, then test the built wheel outside the checkout as in CI.
The wheel contains `eva`, `tools` and package data, not the full training tree
or paper inputs. Use the complete source archive for paper reproduction.
External resource checks report access problems separately from CPU tests;
unavailable historical weights are not silently replaced with current releases.
