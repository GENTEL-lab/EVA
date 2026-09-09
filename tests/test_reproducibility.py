import sys
from types import SimpleNamespace

import pytest

from tools import directed_evolution
from tools.utils.scorers import score_worker


def test_missing_viennarna_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, 'RNA', None)
    with pytest.raises(RuntimeError, match='ViennaRNA is required'):
        directed_evolution.calculate_mfe('GCGCUUCGCG')


def test_mfe_returns_energy_not_structure(monkeypatch):
    monkeypatch.setitem(sys.modules, 'RNA', SimpleNamespace(
        fold_compound=lambda seq: SimpleNamespace(mfe=lambda: ('(((...)))', -2.5))
    ))
    assert directed_evolution.calculate_mfe('GCGAAACGC') == -2.5


def test_real_viennarna():
    RNA = pytest.importorskip('RNA')
    seq = 'GGGAAACCC'
    assert directed_evolution.calculate_mfe(seq) == pytest.approx(RNA.fold_compound(seq).mfe()[1])


def test_batch_limit_order_and_options(monkeypatch):
    calls = []
    def score(model, tokenizer, sequences, device, **kwargs):
        calls.append((list(sequences), kwargs))
        return [float(s) for s in sequences]
    monkeypatch.setattr(score_worker, 'compute_batch_likelihood', score)
    assert score_worker.score_in_batches(None, None, ['1', '2', '3', '4', '5'],
        batch_size=2, reduce_method='sum', exclude_special_tokens=True) == [1, 2, 3, 4, 5]
    assert [len(c[0]) for c in calls] == [2, 2, 1]
    assert all(c[1] == {'reduce_method': 'sum', 'exclude_special_tokens': True} for c in calls)


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')], []])
def test_bad_predictions_fail(monkeypatch, values):
    monkeypatch.setattr(score_worker, 'compute_batch_likelihood', lambda *a, **k: values)
    with pytest.raises(ValueError, match='Invalid predictions'):
        score_worker.score_in_batches(None, None, ['A'])


def test_failure_does_not_change_scoring_path(monkeypatch):
    def score(*args, **kwargs):
        raise RuntimeError('required dependency missing')
    monkeypatch.setattr(score_worker, 'compute_batch_likelihood', score)
    with pytest.raises(RuntimeError, match='required dependency missing'):
        score_worker.score_in_batches(None, None, ['A'])


def test_invalid_batch_size():
    with pytest.raises(ValueError, match='batch_size'):
        score_worker.score_in_batches(None, None, ['A'], batch_size=0)


def test_bad_checkpoint_root(tmp_path):
    from tools.utils.model import ModelLoader
    with pytest.raises(FileNotFoundError, match='config.json'):
        ModelLoader(str(tmp_path))


def test_ambiguous_weights(tmp_path):
    from tools.utils.model import ModelLoader
    for name in ['config.json', 'tokenizer.json', 'first.pt', 'second.pt']:
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        ModelLoader(str(tmp_path))


@pytest.mark.parametrize('seqs', [[''], [None]])
def test_invalid_sequences(seqs):
    with pytest.raises(ValueError, match='nonempty'):
        score_worker.compute_batch_likelihood(None, None, seqs)


def test_invalid_reduction():
    with pytest.raises(ValueError, match='reduce_method'):
        score_worker.compute_batch_likelihood(None, None, ['ACGU'], reduce_method='median')


def test_optimization_prompt_not_double_wrapped():
    from tools.utils.conditions import GenerationCondition
    assert directed_evolution.format_sequence('ACGU', GenerationCondition(rna_type='tRNA')) == '|<rna_tRNA>|ACGU'


def test_mixed_glm_failure_does_not_change_objective():
    from finetune.utils.lineage_dataset import LineageRNADataset
    ds = object.__new__(LineageRNADataset)
    ds.samples = [('ACGU' * 8, None, 'tRNA', None)]
    ds.mode, ds.glm_probability, ds.span_config = 'mixed', 1.0, None
    ds.processor = SimpleNamespace(enable_reverse_augmentation=False,
        _process_completion_sample_multi_span=lambda *a, **kw: None,
        process_generation_sample=lambda *a, **kw: pytest.fail('GLM silently replaced by CLM'))
    with pytest.raises(ValueError, match='GLM sample construction failed'):
        ds[0]


@pytest.mark.parametrize('mode', ['mixed', 'completion'])
def test_glm_dataset_has_supervised_targets(mode):
    from pathlib import Path
    from eva.lineage_tokenizer import get_lineage_rna_tokenizer
    from finetune.utils.lineage_dataset import LineageRNADataset
    tok = get_lineage_rna_tokenizer()
    ds = LineageRNADataset(str(Path(__file__).parent / 'data/train_smoke.fasta'), tok,
        lineage_file=None, use_lineage_prefix=False, mode=mode, glm_probability=1.0,
        enable_reverse_augmentation=False, max_seq_length=256)
    sample = ds[0]
    assert (sample['labels'] != -100).any()
    assert tok.token_to_id('<eos_span>') in sample['labels'].tolist()


@pytest.mark.parametrize('kind', ['midtrain', 'finetune'])
def test_training_requires_checkpoint(kind):
    if kind == 'midtrain':
        from training.midtrain.train_midtrain import MidTrainingTrainer
        trainer = object.__new__(MidTrainingTrainer)
        trainer.config = {}
        loader = trainer._load_pretrain_checkpoint
    else:
        from finetune.train_finetune import FinetuneTrainer
        trainer = object.__new__(FinetuneTrainer)
        trainer.training_config = {}
        loader = trainer._load_pretrain_checkpoint
    with pytest.raises(ValueError, match='resume_from_pretrain'):
        loader()


def test_aptamer_cli_rejects_documented_old_invalid_flag():
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'finetune/aptamer/script/run_aptamer_finetuning.sh'
    result = subprocess.run(['bash', str(script), '--rna-type', 'aptamer'], capture_output=True, text=True)
    assert result.returncode == 2 and 'Unknown option' in result.stderr


def test_steering_hook_removed_on_failure():
    import torch
    from tools.sae_steering import forward_with_steer
    layer = torch.nn.Identity()
    class FailingModel:
        model = SimpleNamespace(layers=[layer])
        def __call__(self, **kwargs):
            raise RuntimeError('test forward failure')
    prompt = SimpleNamespace(input_ids=None, position_ids=None, sequence_ids=None)
    args = SimpleNamespace(layer=0, device='cpu')
    with pytest.raises(RuntimeError, match='test forward failure'):
        forward_with_steer(FailingModel(), prompt, None, args, [0], 1.0, {})
    assert len(layer._forward_hooks) == 0


def test_sae_loader_uses_saved_topk(tmp_path):
    import torch
    from tools.sae_steering import load_sae
    state = {'bias': torch.zeros(4), 'encoder.weight': torch.zeros(8, 4),
             'encoder.bias': torch.zeros(8), 'decoder.weight': torch.zeros(4, 8)}
    path = tmp_path / 'sae.pt'
    torch.save({'model_state_dict': state, 'cfg': {'mode': 'batch_topk', 'k': 2}}, path)
    assert load_sae(str(path), 'auto', 'cpu').k == 2
