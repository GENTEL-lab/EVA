# Dense-control training and evaluation

The author dense-control implementation was recovered from the separate EVA_dense working tree. It uses the EVA transformer with one eager feed-forward expert per layer, one selected expert, expert-parallel size one and zero router auxiliary loss. This is the implementation used for the dense controls, not a substitution chosen to satisfy the missing historical model_dense imports.

## Released entry points

- [Dense training](../training/dense_pretrain/train_dense.py) now selects the recovered single-expert architecture explicitly and uses the public eva package.
- [Dense checkpoint loader](../training/eval/scripts/dense_model_loader.py) loads these single-expert checkpoints and rejects multi-expert checkpoints rather than changing their architecture.
- [Original source and configurations](../examples/reproduction/dense_control/original/) preserve the author working-tree files, including the separate original mid-training script. These original files preserve their original imports and paths; the integrated public entry is the training script above.
- [Source manifest](../examples/reproduction/dense_control/source_manifest.json) records original file hashes.

Use the normal installation guide. To run dense training, supply a training configuration and existing input data:

```sh
python training/dense_pretrain/train_dense.py --config /path/to/dense_config.yaml
```

The recovered configuration examples retain the original local training-data paths. Set train_file, lineage_file and output/log directories for your installation, and match the intended dataset/sampling protocol when reproducing the original experiment. The original sampled 30M training file is distinct from the complete OpenRNA corpus; an arbitrary subset is not equivalent.

## Correspondence to manuscript controls

The [manuscript architecture table](../examples/reproduction/dense_control/manuscript_dense_architectures.csv) identifies the dense control sizes. Three recovered mid-training checkpoint-6500 configurations match its 6.5M, 51.7M and 145.0M rows by layer count, hidden width, attention heads and FFN width. Their historical directory labels refer to the corresponding EVA comparison scales, not the dense parameter counts. [Checkpoint architecture evidence](../examples/reproduction/dense_control/checkpoint_architecture_evidence.json) records the matching configurations and their hashes. The table also lists a 437.0M dense control; that row is not represented by those three recovered checkpoint configurations.

## Validation

The container run passed all 25 focused dense-control and software-robustness tests. [Validation record](../examples/reproduction/dense_control/validation.json) records the scope. The focused tests exercise a real single-expert forward pass, gradient update and exact saved-weight reload, and reject a multi-expert checkpoint. They validate software behavior on a tiny synthetic model; the author separately confirms running dense training and reproducing the reported results. This update does not claim a new full-scale training run.

```sh
python -m unittest discover -s tests -p test_dense_control.py -v
```

The frozen v1.2.0 tag predates this recovery. Use the fixed commit containing this document for the repaired dense entry points; the archived v1.2.0 code remains unchanged.
