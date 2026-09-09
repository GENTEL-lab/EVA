"""Small real GPU workflow tests; synthetic data, NOT paper-result reproduction.

Exercise the released trainer classes, DCP/PT round trips and both SAE trainers.
All generated configs, weights and reports are confined to --output.
"""
import argparse
import copy
import gc
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-only', action='store_true', help='Run synthetic pretraining, midtraining and fine-tuning round trips without loading an external backbone')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.training_only and args.checkpoint is None:
        parser.error('--checkpoint is required unless --training-only is selected')
    import torch
    import yaml
    from training.pretrain.train_stage1 import LineageStage1Trainer
    from training.midtrain.train_midtrain import MidTrainingTrainer
    from finetune.train_finetune import FinetuneTrainer
    from training.eval.scripts.lineage_model_loader import load_lineage_model
    from tools.utils.model import ModelLoader
    from tools.utils.scorers.score_worker import score_in_batches

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'purpose': 'synthetic engineering smoke, not paper reproduction', 'workflows': {}}
    def record(name, result):
        report['workflows'][name] = result
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    base = yaml.safe_load((ROOT / 'config/training/pretrain_smoke.yaml').read_text())
    base['data_config']['train_file'] = str(ROOT / 'tests/data/train_smoke.fasta')
    lineage = args.output / 'lineage.tsv'
    lineage.write_text('taxid\tlineage\n9606\td__eukaryota;p__chordata;c__mammalia;o__primates;f__hominidae;g__homo;s__homo_sapiens\n')
    for name, cls in [('pretrain', LineageStage1Trainer), ('midtrain', MidTrainingTrainer), ('finetune', FinetuneTrainer)]:
        cfg = copy.deepcopy(base)
        dest = args.output / name
        cfg['training_config']['output_dir'] = str(dest)
        cfg['logging_config']['log_dir'] = str(dest / 'logs')
        if name != 'pretrain':
            cfg['training_config']['resume_from_pretrain'] = str(args.output / 'pretrain/final')
        if name == 'midtrain':
            cfg['data_config'].update(use_lineage_prefix=True, lineage_file=str(lineage), max_seq_length=256, mode='completion')
        if name == 'finetune':
            cfg['data_config'].update(rna_type_token='Y_RNA', pretrain_ratio=0.0, mode='completion')
        path = args.output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        trainer = cls(str(path))
        trainer.setup()
        initial = trainer.model.model.embed_tokens.weight.detach().clone()
        trainer.train_start_time = time.time()
        trainer.save_steps = None
        trainer.checkpoint_flops_milestones = []
        loss, _ = trainer.train_epoch(0, max_steps=2)
        assert trainer.global_step == 2 and math.isfinite(loss)
        assert not torch.equal(initial, trainer.model.model.embed_tokens.weight)
        trainer.save_checkpoint(1, is_final=True)
        if name == 'finetune':
            loaded, tok = ModelLoader(str(dest / 'final')).load(device='cuda:0')
        else:
            loaded, tok, _ = load_lineage_model(str(dest / 'final'), device='cuda:0')
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key].float(), value.float(), rtol=0, atol=0)
        record(name, {'optimizer_steps': trainer.global_step, 'loss': loss,
                      'embedding_updated': True, 'exact_weight_roundtrip': True,
                      'data_mode': cfg['data_config']['mode']})
        del trainer, loaded, initial
        gc.collect()
        torch.cuda.empty_cache()

    if args.training_only:
        print(json.dumps(report, indent=2))
        return

    model, tok = ModelLoader(str(args.checkpoint)).load(device='cuda:0')
    seqs = ['ACGU' * 8, 'ACGU' * 9]
    one = score_in_batches(model, tok, seqs, 'cuda:0', batch_size=1, exclude_special_tokens=True)
    two = score_in_batches(model, tok, seqs, 'cuda:0', batch_size=2, exclude_special_tokens=True)
    record('batch_consistency', {'batch1': one, 'batch2': two,
                                'max_absolute_difference': max(abs(a-b) for a,b in zip(one,two))})
    del model
    torch.cuda.empty_cache()
    module_path = ROOT / 'notebooks/interpretability_analysis/sae_repro_release/scripts/run_training.py'
    spec = importlib.util.spec_from_file_location('eva_sae_smoke', module_path)
    sae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sae)
    for mode, train in [('batch_topk', sae.train_batch_topk), ('sae_l1_penalty', sae.train_sae_l1_penalty)]:
        cfg = yaml.safe_load((module_path.parent.parent / f'configs/config_{mode}_smoke.yaml').read_text())
        cfg.update(layer=1, d_hidden=32, k=4, n_seqs=2, seq_batch=1, max_len=128,
                   batch_size=8, max_steps=2, log_every=1, save_every=1)
        cfg['paths'] = {'ckpt': str(args.checkpoint), 'data': base['data_config']['train_file'],
                        'output_root': str(args.output / mode)}
        (args.output / f'{mode}.yaml').write_text(yaml.safe_dump(cfg))
        train(cfg, torch.device('cuda:0'), logging.getLogger(mode))
        checkpoint = torch.load(args.output / mode / 'checkpoint_step2.pt', map_location='cpu', weights_only=False)
        reloaded = sae.SAE(256, 32)
        reloaded.load_state_dict(checkpoint['model_state_dict'], strict=True)
        assert checkpoint['step'] == 2
        assert all(torch.isfinite(v).all() for v in reloaded.state_dict().values())
        record(mode, {'optimizer_steps': 2, 'checkpoint_reloaded': True, 'finite_parameters': True,
                      'backbone': 'released EVA-21M, layer 1 (not the paper layer 13)'})
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
