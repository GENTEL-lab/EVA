# Training entry points

Run from the repository root in the GPU environment described in
[the reproduction guide](../../docs/REPRODUCTION.md).

```bash
python training/pretrain/train_stage1.py --config training/configs/pretrain_smoke.yaml
```

This is a two-step synthetic engineering test, **not** a recovered paper recipe.
It uses the released MoE architecture at a small size and saves a DCP checkpoint,
tokenizer and configuration to `results/pretrain_smoke/final`.

The complete small-workflow test exercises mid-training and finetuning too:

```bash
python scripts/smoke_workflows.py --checkpoint checkpoint/EVA_21M --output results/workflow_smoke
```

For an explicitly configured run, use `training/midtrain/train_midtrain.py --config YOUR_CONFIG`.
Mid-training requires `training_config.resume_from_pretrain` pointing to a DCP
checkpoint and resets the optimizer/scheduler. Finetuning uses
`training/finetune/train_finetune.py --config YOUR_CONFIG` and accepts DCP or PT weights.

`data_config.mode` selects `generation`, `completion` or `mixed`.
`glm_probability` only controls the mix when mode is `mixed`; zero means CLM only.
RNA-type and lineage prefixes are independently configured by
`use_rna_type_prefix` and `use_lineage_prefix`. Directory names do not imply that
pretraining was exclusively CLM and mid-training exclusively GLM.
The exact original paper training configurations are not established by this repair.

Historical `scripts/lineage_training/*` and `configs/lineage_training/*` commands
were not included in the release and must not be used. The recovered author dense-control workflow uses a single eager FFN expert.
See [Dense control](../../docs/DENSE_CONTROL.md) for its source and validation.
