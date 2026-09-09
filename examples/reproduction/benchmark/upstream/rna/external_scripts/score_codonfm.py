#!/usr/bin/env python3
"""
CodonFM 打分脚本 - 最小化版本

从 optimize_codonfm.py 中提取打分逻辑，对输入的 CDS 序列计算 CodonFM log-prob 得分。
打分方式：逐密码子前缀增量打分，累加每个密码子 token 的 log-prob。

用法（容器内）:
    python3 score_codonfm.py input.fasta -o scores.csv
    python3 score_codonfm.py input.fasta -o scores.csv --checkpoint /data/checkpoints/CodonFM-1B/NV-CodonFM-Encodon-1B-v1.safetensors
"""

import sys
import os
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import csv

# Monkey patch: transformers 5.x 移除了 modeling_utils.apply_chunking_to_forward,
# CodonFM 源码从旧路径 import，这里补回去避免修改源码
import transformers
import transformers.modeling_utils
if not hasattr(transformers.modeling_utils, 'apply_chunking_to_forward'):
    transformers.modeling_utils.apply_chunking_to_forward = transformers.apply_chunking_to_forward

import torch
import torch.nn.functional as F

# Monkey patch: 容器内未安装 xformers，用 PyTorch 原生 scaled_dot_product_attention 替代
import types
import importlib
xformers_mod = types.ModuleType('xformers')
xformers_ops = types.ModuleType('xformers.ops')

def _memory_efficient_attention(query, key, value, op=None, attn_bias=None, p=0.0):
    """用 PyTorch 原生 SDPA 替代 xformers.ops.memory_efficient_attention"""
    # query/key/value: (B, seq_len, n_heads, head_dim) -> (B, n_heads, seq_len, head_dim)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if attn_bias is not None:
        # attn_bias 形状已经是 (B, n_heads, seq_len, seq_len)
        # 将 0 的位置转为 -inf（被 mask 的位置）
        attn_bias = attn_bias.masked_fill(attn_bias == 0, float('-inf'))
        attn_bias = attn_bias.masked_fill(attn_bias == 1, 0.0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
    return out.transpose(1, 2)  # -> (B, seq_len, n_heads, head_dim)

xformers_ops.memory_efficient_attention = _memory_efficient_attention
xformers_mod.ops = xformers_ops
sys.modules['xformers'] = xformers_mod
sys.modules['xformers.ops'] = xformers_ops

# Monkey patch: 容器内未安装 lightning / torchmetrics / peft
# 打分时不使用这些库的功能，仅需满足 import 和类继承

# --- lightning mock ---
import inspect
lightning_mod = types.ModuleType('lightning')

class _AttrDict(dict):
    """支持属性访问的字典，模拟 Lightning 的 hparams 容器"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value

class _MockLightningModule(torch.nn.Module):
    """LightningModule 最小替身"""
    def save_hyperparameters(self, *args, ignore=None, frame=None, logger=True):
        caller_frame = frame or inspect.currentframe().f_back
        init_locals = {k: v for k, v in caller_frame.f_locals.items()
                       if k not in ('self', '__class__')}
        if args:
            init_locals = {k: v for k, v in init_locals.items() if k in args}
        if ignore:
            for key in ignore:
                init_locals.pop(key, None)
        if not hasattr(self, '_hparams'):
            self._hparams = _AttrDict()
        self._hparams.update(init_locals)

    @property
    def hparams(self):
        if not hasattr(self, '_hparams'):
            self._hparams = _AttrDict()
        return self._hparams

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            self._mock_device = torch.device(device) if isinstance(device, str) else device
        return super().to(*args, **kwargs)

    @property
    def device(self):
        if hasattr(self, '_mock_device'):
            return self._mock_device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def log(self, *args, **kwargs):
        pass

lightning_mod.LightningModule = _MockLightningModule
sys.modules['lightning'] = lightning_mod

# --- torchmetrics mock ---
torchmetrics_mod = types.ModuleType('torchmetrics')

class _MockMeanMetric:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass

torchmetrics_mod.MeanMetric = _MockMeanMetric
sys.modules['torchmetrics'] = torchmetrics_mod

# --- peft mock ---
peft_mod = types.ModuleType('peft')

class _MockLoraConfig:
    def __init__(self, **kwargs):
        pass

class _MockPeftType:
    LORA = 'LORA'

def _mock_get_peft_model(model, config):
    return model

peft_mod.LoraConfig = _MockLoraConfig
peft_mod.PeftType = _MockPeftType
peft_mod.get_peft_model = _mock_get_peft_model
sys.modules['peft'] = peft_mod

sys.path.insert(0, '/workspace')


# ==================== CodonFM 打分器 ====================

class CodonFMScorer:
    """CodonFM 打分器

    前缀增量打分：将完整 CDS 编码后输入模型，
    取每个密码子位置的 log-prob 并累加。
    """

    def __init__(self, checkpoint_path, device='cuda:0'):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        from src.tokenizer import Tokenizer
        from src.inference.encodon import EncodonInference
        from src.inference.task_types import TaskTypes

        print(f"Loading CodonFM from {self.checkpoint_path}...")

        self.tokenizer = Tokenizer(seq_type='dna')

        self.inference = EncodonInference(
            model_path=self.checkpoint_path,
            task_type=TaskTypes.FITNESS_PREDICTION
        )
        self.inference = self.inference.to(self.device)
        self.inference.configure_model()
        self.model = self.inference.model

        print(f"CodonFM loaded on {self.device}")

    def encode_sequence(self, dna_seq):
        """编码 DNA 序列为 token IDs（不含 special tokens）"""
        dna_seq = dna_seq.upper().replace('U', 'T')
        return self.tokenizer.encode(dna_seq)

    def encode_with_special(self, dna_seq):
        """编码 DNA 序列为 token IDs（含 CLS 和 SEP）"""
        token_ids = self.encode_sequence(dna_seq)
        return self.tokenizer.build_inputs_with_special_tokens(token_ids)

    def score_sequence(self, dna_seq):
        """对单条 CDS 序列打分

        逐密码子前缀增量打分（与 optimize_codonfm.py 中 beam search 的打分逻辑一致）：
        - 依次取 prefix = seq[:3], seq[:6], seq[:9], ...
        - 每次将 prefix 输入模型，取最后一个密码子 token 位置的 log-prob
        - 累加所有密码子的 log-prob 作为总分

        Returns:
            total_score: 累加 log-prob
            avg_score: 每密码子平均 log-prob
            n_codons: 密码子数量
        """
        dna_seq = dna_seq.upper().replace('U', 'T')

        # 验证序列长度是3的倍数
        if len(dna_seq) % 3 != 0:
            print(f"  Warning: sequence length {len(dna_seq)} not multiple of 3, truncating")
            dna_seq = dna_seq[:len(dna_seq) // 3 * 3]

        n_codons = len(dna_seq) // 3
        if n_codons == 0:
            return 0.0, 0.0, 0

        total_score = 0.0

        for i in range(n_codons):
            # prefix = 前 i+1 个密码子
            prefix_end = (i + 1) * 3
            prefix_dna = dna_seq[:prefix_end]

            # 编码
            full_ids = self.encode_with_special(prefix_dna)
            ctx_len = i  # 前 i 个密码子（不含 special tokens）

            # 新密码子在 full_ids 中的位置: ctx_len + 1 (CLS 在位置 0)
            new_codon_pos = ctx_len + 1
            new_codon_token_id = full_ids[new_codon_pos]

            # Padding 到 8 的倍数
            seq_len = len(full_ids)
            padded_len = ((seq_len + 7) // 8) * 8
            pad_len = padded_len - seq_len

            input_ids = torch.tensor(
                [full_ids + [self.tokenizer.pad_token_id] * pad_len],
                dtype=torch.long, device=self.device
            )
            attention_mask = torch.tensor(
                [[1] * seq_len + [0] * pad_len],
                dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                out = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
                logits = out.logits.float()
                log_probs = F.log_softmax(logits, dim=-1)

                codon_log_prob = log_probs[0, new_codon_pos, new_codon_token_id].item()
                total_score += codon_log_prob

            del input_ids, attention_mask, logits, log_probs

        avg_score = total_score / n_codons
        return total_score, avg_score, n_codons


# ==================== FASTA 读取 ====================

def read_fasta(fasta_path):
    """读取 FASTA 文件，返回 [(name, sequence), ...]"""
    sequences = []
    name = None
    seq_lines = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if name is not None:
                    sequences.append((name, ''.join(seq_lines)))
                name = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
    if name is not None:
        sequences.append((name, ''.join(seq_lines)))
    return sequences


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='CodonFM scoring for CDS sequences')
    parser.add_argument('input', help='Input FASTA file with CDS sequences (DNA or RNA)')
    parser.add_argument('-o', '--output', default='scores.csv', help='Output file (default: scores.csv)')
    parser.add_argument('--checkpoint', default='/data/checkpoints/CodonFM-600M/NV-CodonFM-Encodon-600M-v1.safetensors',
                        help='CodonFM checkpoint path')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--format', choices=['csv', 'json'], default=None,
                        help='Output format (default: auto-detect from file extension)')
    args = parser.parse_args()

    # 自动检测输出格式
    out_format = args.format
    if out_format is None:
        out_format = 'json' if args.output.endswith('.json') else 'csv'

    # 读取序列
    sequences = read_fasta(args.input)
    print(f"Read {len(sequences)} sequences from {args.input}", file=sys.stderr)

    if not sequences:
        print("No sequences found, exiting.", file=sys.stderr)
        sys.exit(1)

    # 加载模型
    scorer = CodonFMScorer(args.checkpoint, device=args.device)
    scorer.load()

    # 打分
    avg_scores = []
    for idx, (name, seq) in enumerate(sequences):
        print(f"Scoring [{idx+1}/{len(sequences)}] {name} ({len(seq)} nt)...", file=sys.stderr, flush=True)
        total_score, avg_score, n_codons = scorer.score_sequence(seq)
        avg_scores.append(avg_score)
        print(f"  total_logprob={total_score:.4f}  avg_logprob={avg_score:.4f}  n_codons={n_codons}", file=sys.stderr)

    # 写出结果
    if out_format == 'json':
        import json
        with open(args.output, 'w') as f:
            json.dump({'success': True, 'scores': avg_scores}, f)
    else:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'avg_logprob'])
            writer.writeheader()
            for (name, _), score in zip(sequences, avg_scores):
                writer.writerow({'name': name, 'avg_logprob': score})

    print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
