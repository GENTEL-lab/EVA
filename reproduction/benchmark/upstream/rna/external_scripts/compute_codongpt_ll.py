#!/usr/bin/python3
"""
CodonGPT Log-Likelihood计算脚本

计算RNA/DNA序列的log-likelihood，用于benchmark评估。
CodonGPT使用DNA密码子（3个碱基）作为token。

用法:
    python compute_codongpt_ll.py --fasta input.fasta --device cuda:0

输出:
    JSON格式的log-likelihood列表（输出到stdout）
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from itertools import product
from Bio import SeqIO
import json
from tqdm import tqdm
import sys


class CodonTokenizer(PreTrainedTokenizer):
    """
    自定义Codon Tokenizer

    将DNA/RNA序列分割成3碱基的密码子作为token
    词汇表: 67个tokens = 3个特殊tokens ([PAD], [BOS], [EOS]) + 64个DNA密码子
    """

    def __init__(self, **kwargs):
        # 创建密码子词汇表
        bases = ['A', 'T', 'G', 'C']
        codons = [''.join(p) for p in product(bases, repeat=3)]

        # 特殊tokens
        special_tokens = ['[PAD]', '[BOS]', '[EOS]']
        self.vocab_list = special_tokens + codons

        # 创建词汇表字典
        self.vocab = {token: idx for idx, token in enumerate(self.vocab_list)}
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}

        # 先调用父类初始化
        super().__init__(
            pad_token='[PAD]',
            bos_token='[BOS]',
            eos_token='[EOS]',
            **kwargs
        )

        # 设置特殊token ID
        self._pad_token_id = self.vocab['[PAD]']
        self._bos_token_id = self.vocab['[BOS]']
        self._eos_token_id = self.vocab['[EOS]']

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def bos_token_id(self):
        return self._bos_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, text):
        """将序列分割成密码子"""
        codons = []
        for i in range(0, len(text), 3):
            codon = text[i:i+3]
            if len(codon) == 3:
                codons.append(codon)
        return codons

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self._pad_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, '[PAD]')

    def convert_tokens_to_string(self, tokens):
        """将tokens转换回序列字符串"""
        codons = [t for t in tokens if t not in ['[PAD]', '[BOS]', '[EOS]']]
        return ''.join(codons)


def calculate_log_likelihood(model, tokenizer, sequence, device='cuda', normalize=False):
    """
    计算单条序列的autoregressive log-likelihood

    Args:
        model: CodonGPT模型
        tokenizer: CodonTokenizer
        sequence: RNA/DNA序列字符串
        device: 计算设备
        normalize: 是否归一化为PTLL（除以token数量）

    Returns:
        log_likelihood: 对数似然值（总LL或PTLL）
    """
    # RNA to DNA转换 (U -> T)
    sequence = sequence.upper().replace('U', 'T')

    # Tokenize: 将序列分割为密码子
    # 丢弃不足3碱基的尾部
    num_full_codons = len(sequence) // 3
    codons = [sequence[i*3:(i+1)*3] for i in range(num_full_codons)]

    # 构建token ID序列: [BOS] + codons
    token_ids = [tokenizer.bos_token_id]
    for codon in codons:
        token_id = tokenizer.vocab.get(codon, tokenizer.pad_token_id)
        token_ids.append(token_id)

    input_ids = torch.tensor([token_ids]).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # 计算log-likelihood
    # 使用 logits[:-1] 预测 tokens[1:]
    # 即: P(token_1|BOS), P(token_2|BOS,token_1), ...
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)  # (seq_len-1, vocab_size)
    target_ids = input_ids[0, 1:]  # (seq_len-1,)

    # 累加每个position的log probability
    log_likelihood = 0.0
    num_tokens = len(target_ids)
    for i, target_id in enumerate(target_ids):
        log_likelihood += log_probs[i, target_id].item()

    # 根据normalize参数决定返回总LL还是PTLL
    if normalize and num_tokens > 0:
        return log_likelihood / num_tokens  # PTLL
    return log_likelihood  # 总LL（默认）


def main():
    parser = argparse.ArgumentParser(
        description='计算CodonGPT的log-likelihood',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python compute_codongpt_ll.py --fasta sequences.fasta --device cuda:0
        """
    )
    parser.add_argument('--fasta', type=str, required=True,
                        help='输入FASTA文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备 (default: cuda:0)')
    parser.add_argument('--normalize-by-length', action='store_true',
                        help='使用PTLL（除以token数量）代替总log-likelihood')
    args = parser.parse_args()

    # 加载模型
    print(f'加载CodonGPT模型 (naniltx/codonGPT)...', file=sys.stderr)
    model = GPT2LMHeadModel.from_pretrained('naniltx/codonGPT')
    model.to(args.device)
    model.eval()
    print(f'模型加载完成，使用设备: {args.device}', file=sys.stderr)

    # 创建tokenizer
    tokenizer = CodonTokenizer()
    print(f'Tokenizer词汇表大小: {tokenizer.vocab_size}', file=sys.stderr)

    # 读取序列
    sequences = list(SeqIO.parse(args.fasta, 'fasta'))
    print(f'读取了 {len(sequences)} 条序列', file=sys.stderr)

    # 检查序列长度是否为3的倍数（只检查第一条序列，假设数据集内序列等长）
    if sequences:
        first_seq_len = len(str(sequences[0].seq))
        remainder = first_seq_len % 3
        if remainder != 0:
            print(f'注意: 序列长度 {first_seq_len} 不是3的倍数，将截断尾部 {remainder} 个碱基', file=sys.stderr)
            print(f'      有效密码子数: {first_seq_len // 3}', file=sys.stderr)

    # 计算log-likelihood
    normalize = getattr(args, 'normalize_by_length', False)
    mode_str = "PTLL" if normalize else "log-likelihood"
    log_likelihoods = []
    for record in tqdm(sequences, desc=f'计算{mode_str}', file=sys.stderr):
        seq_str = str(record.seq)
        ll = calculate_log_likelihood(model, tokenizer, seq_str, args.device, normalize=normalize)
        log_likelihoods.append(ll)

    # 输出JSON结果（到stdout）
    result = {
        'log_likelihoods': log_likelihoods,
        'num_sequences': len(log_likelihoods)
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
