# Aptamer finetuning (single GPU)

Use a running `eva-repro` container with the repository mounted at `/eva`, as
shown in the [reproduction guide](../../../../docs/REPRODUCTION.md).
The Python trainer is single-GPU; this wrapper does not implement distributed
training. `--gpu 0` means GPU 0 **inside** the container, even when it exposes host GPU 7.

Edit a copy of `experiment_config_template.yaml`: set the training FASTA,
output directory, checkpoint and architecture to match that checkpoint.
The bundled template is not evidence of the final paper hyperparameters.
RNA type is `data_config.rna_type_token` (the aptamer template uses `Y_RNA`);
there is no `--rna-type` CLI option. Template `glm_probability: 0` is CLM only.
YAML paths resolve from `/eva`, or may be absolute container paths.

```bash
bash training/finetune/aptamer/script/run_aptamer_finetuning.sh \
  --container eva-repro --gpu 0 --config results/my_aptamer/experiment_config.yaml --foreground
```

Omit `--foreground` to launch in the background and print the log path; launch
does not mean success. Positional experiment-directory and `--config` forms
remain supported. `CONTAINER_WORKDIR` overrides `/eva`; the source mount must
match. No LD_LIBRARY_PATH override is injected.

Inside a container, invoke Python directly, without nested Docker:

```bash
python training/finetune/train_finetune.py --config results/my_aptamer/experiment_config.yaml
```
