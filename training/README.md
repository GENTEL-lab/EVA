# Training

Pretraining, midtraining, fine-tuning and evaluation live together here.
Use the complete source checkout and the [GPU environment](../docs/INSTALLATION.md#gpu-runtime-docker).

| Task | Entry |
|---|---|
| Pretraining | [Training guide](pretrain/README.md) · `pretrain/train_stage1.py` · `dense_pretrain/train_dense.py` |
| Midtraining | `midtrain/train_midtrain.py` |
| Fine-tuning | [Aptamer walkthrough](finetune/aptamer/script/README.md) · `finetune/train_finetune.py` |
| Checkpoint evaluation | [Evaluation scripts and configurations](eval/) |

From the repository root, verify the training workflow with the small synthetic
example in the [reproduction guide](../docs/REPRODUCTION.md#training-and-fine-tuning).
Its configuration is [`configs/pretrain_smoke.yaml`](configs/pretrain_smoke.yaml).
This check exercises training and checkpoint reloads; paper-scale training needs
the corresponding data and configuration.

Fine-tuning commands use `python training/finetune/train_finetune.py --config PATH`.
Shared training helpers are under `common/` and `finetune/utils/`.
