#!/usr/bin/env python3
"""
RNA序列生成脚本 - 用于Finetune模型的序列生成
支持：
- 指定RNA类型（如sRNA用于aptamer）
- 指定lineage信息（如virus任务需要的谱系前缀）
- CLM条件生成

Prompt格式：
- 只有RNA类型: <bos>|<rna_sRNA>|5...
- 有lineage和RNA类型: <bos>|D__Viruses;P__...;<rna_viral_RNA>|5...
- 只有lineage: <bos>|D__Viruses;P__...|5...
"""
import os
import sys
import argparse
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
import shutil

MODEL_CODE_PATH = str(Path(__file__).resolve().parent.parent.parent / 'model')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_CHECKPOINT_PATH = './results/checkpoint'
DEFAULT_NUM_SEQS = 1000
MAX_LENGTH = 8192
OUTPUT_DIR = str(Path(__file__).resolve().parent / 'fasta')

# RNA类型到token的映射
RNA_TOKENS = {
    'mRNA': '<rna_mRNA>',
    'rRNA': '<rna_rRNA>',
    'tRNA': '<rna_tRNA>',
    'sRNA': '<rna_sRNA>',
    'lncRNA': '<rna_lncRNA>',
    'circRNA': '<rna_circRNA>',
    'viral_RNA': '<rna_viral_RNA>',
    'miRNA': '<rna_miRNA>',
    'snoRNA': '<rna_snoRNA>',
    'snRNA': '<rna_snRNA>',
    'piRNA': '<rna_piRNA>',
    'ribozyme': '<rna_ribozyme>',
    'scaRNA': '<rna_scaRNA>',
    'Y_RNA': '<rna_Y_RNA>',
    'vault_RNA': '<rna_vault_RNA>'
}


def build_clm_prompt(rna_type=None, lineage=None):
    """构建CLM生成prompt，支持RNA类型和lineage

    Args:
        rna_type: RNA类型，如 'sRNA', 'viral_RNA' 等
        lineage: 谱系字符串，如 'D__Viruses;P__Pisuviricota;...'

    Returns:
        prompt字符串
    """
    rna_token = RNA_TOKENS.get(rna_type) if rna_type else None

    # 将lineage转为小写（与训练时的normalize_lineage保持一致）
    if lineage:
        lineage = lineage.lower()

    # 构建前缀
    if lineage and rna_token:
        prefix = f"|{lineage};{rna_token}|"
    elif lineage:
        prefix = f"|{lineage}|"
    elif rna_token:
        prefix = f"|{rna_token}|"
    else:
        prefix = ""

    return f"<bos>{prefix}5"


def generate_batch_clm(model, tokenizer, device, prompt, max_length, batch_size, temperature=1.0, top_k=50,
                       save_probs=False, prob_log_file=None):
    """批量生成RNA序列（CLM模式）

    Args:
        save_probs: 是否保存每步的概率分布
        prob_log_file: 概率日志文件路径
    """
    eos_id = tokenizer.token_to_id('<eos>')
    three_id = tokenizer.token_to_id('3')  # 3'端标记
    nucleotides = {'A', 'U', 'G', 'C'}
    input_ids = tokenizer.encode(prompt)
    current_ids = torch.tensor([input_ids] * batch_size, dtype=torch.long, device=device)
    seqs = ['' for _ in range(batch_size)]
    finished = [False] * batch_size
    start_time = time.time()

    # 概率记录
    prob_log = [] if save_probs else None

    with torch.no_grad():
        for step in range(max_length + 100):
            if all(finished):
                break
            pos_ids = torch.arange(current_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            seq_ids = torch.zeros_like(current_ids)
            outputs = model(input_ids=current_ids, position_ids=pos_ids, sequence_ids=seq_ids, use_cache=False)
            logits = outputs.logits
            next_logits = logits[:, -1, :].float() / temperature
            top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
            threshold = top_k_values[:, -1:]
            next_logits = torch.where(next_logits < threshold, torch.tensor(float('-inf'), device=device), next_logits)
            probs = F.softmax(next_logits, dim=-1)

            # 记录概率分布
            if save_probs and not finished[0]:
                eos_prob = probs[0, eos_id].item()
                three_prob = probs[0, three_id].item() if three_id is not None else 0.0
                # 获取top-10 token及其概率
                top10_probs, top10_ids = torch.topk(probs[0], 10)
                top10_tokens = [tokenizer.id_to_token(tid.item()) for tid in top10_ids]
                prob_log.append({
                    'step': step,
                    'seq_len': len(seqs[0]),
                    'eos_prob': eos_prob,
                    'three_prob': three_prob,
                    'top10': list(zip(top10_tokens, top10_probs.tolist()))
                })

            # 调试：打印<eos>概率
            if step < 20 or step % 100 == 0:  # 前20步和每100步打印
                eos_prob = probs[0, eos_id].item()
                three_prob = probs[0, three_id].item() if three_id is not None else 0.0
                # 获取top-5 token及其概率
                top5_probs, top5_ids = torch.topk(probs[0], 5)
                top5_tokens = [tokenizer.id_to_token(tid.item()) for tid in top5_ids]
                top5_str = ", ".join([f"{t}:{p:.4f}" for t, p in zip(top5_tokens, top5_probs.tolist())])
                print(f"  Step {step}: seq_len={len(seqs[0])}, <eos>={eos_prob:.6f}, 3={three_prob:.6f}, top5=[{top5_str}]")

            next_tokens = torch.multinomial(probs, num_samples=1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                tok_id = next_tokens[i].item()
                tok_str = tokenizer.id_to_token(tok_id)
                if tok_id == eos_id or tok_str in ['<eos>', '3', '</s>']:
                    finished[i] = True
                    print(f"  [STOP] Step {step}: 序列{i}结束, tok_str={tok_str}, tok_id={tok_id}, seq_len={len(seqs[i])}")
                elif tok_str in nucleotides:
                    seqs[i] += tok_str
                    if len(seqs[i]) >= max_length:
                        finished[i] = True
            current_ids = torch.cat([current_ids, next_tokens], dim=1)

    elapsed = time.time() - start_time

    # 保存概率日志
    if save_probs and prob_log_file and prob_log:
        with open(prob_log_file, 'w') as f:
            json.dump(prob_log, f, indent=2)
        print(f"  概率日志已保存到: {prob_log_file}")

    return seqs, elapsed


def generate_sequences(model, tokenizer, device, prompt, num_seqs, batch_size, temperature,
                       output_file, seq_prefix, rna_type=None, lineage=None, save_probs=False, top_k=50):
    """生成指定数量的序列并写入文件"""
    all_seqs = []
    seq_idx = 0
    num_batches = (num_seqs + batch_size - 1) // batch_size
    start_all = time.time()

    # 构建类型标识
    if rna_type and lineage:
        type_str = f"{rna_type}_lineage"
    elif rna_type:
        type_str = rna_type
    elif lineage:
        type_str = "lineage"
    else:
        type_str = "unconditional"

    # 概率日志目录
    prob_log_dir = Path(output_file).parent / "prob_logs"
    if save_probs:
        prob_log_dir.mkdir(parents=True, exist_ok=True)

    # 打开文件，追加模式
    f = open(output_file, 'a')

    for batch_idx in range(num_batches):
        cur_batch = min(batch_size, num_seqs - len(all_seqs))
        print(f"[{type_str}] 批次 {batch_idx+1}/{num_batches}: 生成 {cur_batch} 条...", flush=True)

        # 概率日志文件
        prob_log_file = str(prob_log_dir / f"seq_{seq_idx}_probs.json") if save_probs else None

        seqs, elapsed = generate_batch_clm(model, tokenizer, device, prompt, MAX_LENGTH, cur_batch,
                                           temperature=temperature, top_k=top_k, save_probs=save_probs,
                                           prob_log_file=prob_log_file)

        for seq in seqs:
            if len(seq) >= 10 and len(all_seqs) < num_seqs:
                line = f">{type_str}_{seq_prefix}_seq{seq_idx}_len{len(seq)}\n{seq}\n"
                f.write(line)
                f.flush()
                all_seqs.append(seq)
                seq_idx += 1

        # 每100批次打印一次进度
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed_total = time.time() - start_all
            speed = len(all_seqs) / elapsed_total if elapsed_total > 0 else 0
            eta = (num_seqs - len(all_seqs)) / speed if speed > 0 else 0
            print(f"  [{type_str}] 进度: {len(all_seqs)}/{num_seqs} ({100*len(all_seqs)/num_seqs:.1f}%), "
                  f"速度: {speed:.2f}条/秒, ETA: {eta/3600:.1f}小时", flush=True)
        else:
            print(f"  [{type_str}] 完成! 已生成 {len(all_seqs)}/{num_seqs}, 本批耗时 {elapsed:.1f}s", flush=True)

        torch.cuda.empty_cache()

    f.close()
    return all_seqs


def main():
    parser = argparse.ArgumentParser(description='RNA序列生成 - Finetune模型')
    parser.add_argument('--gpu', type=int, required=True, help='GPU编号')
    parser.add_argument('--instance_id', type=int, default=0, help='同一GPU上的实例ID，默认0')
    parser.add_argument('--num_seqs', type=int, default=DEFAULT_NUM_SEQS, help=f'生成序列数量，默认{DEFAULT_NUM_SEQS}')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小，默认1')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成温度，默认1.0')
    parser.add_argument('--top_k', type=int, default=50, help='top-k采样，默认50')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH, help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')

    # RNA类型和lineage参数
    parser.add_argument('--rna_type', type=str, default=None,
                        help='RNA类型，如 sRNA, viral_RNA 等')
    parser.add_argument('--lineage', type=str, default=None,
                        help='谱系字符串，如 "D__Viruses;P__Pisuviricota;..."')
    parser.add_argument('--save_probs', action='store_true',
                        help='保存每步的概率分布到JSON文件')

    args = parser.parse_args()

    device = 'cuda:0'

    gpu_id = args.gpu
    instance_id = args.instance_id
    batch_size = args.batch_size
    temperature = args.temperature
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    rna_type = args.rna_type
    lineage = args.lineage
    num_seqs = args.num_seqs

    # 序列前缀：包含GPU和实例信息
    seq_prefix = f"gpu{gpu_id}" if instance_id == 0 else f"gpu{gpu_id}_inst{instance_id}"

    print(f"=" * 70)
    print(f"RNA序列生成 - GPU {gpu_id} 实例 {instance_id}")
    print(f"RNA类型: {rna_type if rna_type else '无'}")
    print(f"Lineage: {lineage if lineage else '无'}")
    print(f"生成数量: {num_seqs}")
    print(f"Batch: {batch_size}, MaxLen: {MAX_LENGTH}, Temperature: {temperature}, Top-k: {args.top_k}")
    print(f"模型权重: {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print(f"=" * 70, flush=True)

    # 加载模型
    print("加载模型...", flush=True)
    from model.config import RNAGenConfig
    from model.causal_lm import RNAGenForCausalLM
    from model.lineage_tokenizer import LineageRNATokenizer

    shutil.copy(Path(checkpoint_path) / 'tokenizer.json', Path(MODEL_CODE_PATH) / 'tokenizer.json')
    tokenizer = LineageRNATokenizer.from_pretrained(str(checkpoint_path))
    with open(Path(checkpoint_path) / 'config.json') as f:
        config_dict = json.load(f)
    model_config = RNAGenConfig(tokenizer=tokenizer, **config_dict)
    model_config.moe_world_size = 1

    # 查找权重文件：优先 model_weights.pt，否则查找 model_checkpoint_*.pt
    weight_file = Path(checkpoint_path) / 'model_weights.pt'
    if not weight_file.exists():
        weight_files = list(Path(checkpoint_path).glob('model_checkpoint_*.pt'))
        if weight_files:
            weight_file = weight_files[0]
            print(f"使用权重文件: {weight_file.name}")
        else:
            raise FileNotFoundError(f"在 {checkpoint_path} 中找不到权重文件")

    weights = torch.load(weight_file, map_location='cpu', weights_only=False)
    # 处理不同格式的checkpoint：完整checkpoint vs 纯权重
    if 'model' in weights and isinstance(weights['model'], dict):
        # 完整checkpoint格式 (model_checkpoint_*.pt)
        weights = weights['model']
    model_config.vocab_size = weights['model.embed_tokens.weight'].shape[0]
    model = RNAGenForCausalLM(model_config)
    model.load_state_dict(weights)
    model.to(device).bfloat16().eval()
    print("模型加载完成!", flush=True)

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # 构建prompt
    prompt = build_clm_prompt(rna_type, lineage)
    print(f"Prompt: {prompt}")

    # 构建输出文件名
    if rna_type and lineage:
        output_file = f'{output_dir}/{rna_type}_lineage_{num_seqs}_gpu{gpu_id}.fasta'
    elif rna_type:
        output_file = f'{output_dir}/{rna_type}_{num_seqs}_gpu{gpu_id}.fasta'
    elif lineage:
        output_file = f'{output_dir}/lineage_{num_seqs}_gpu{gpu_id}.fasta'
    else:
        output_file = f'{output_dir}/unconditional_{num_seqs}_gpu{gpu_id}.fasta'

    if instance_id > 0:
        output_file = output_file.replace('.fasta', f'_inst{instance_id}.fasta')

    print(f"输出文件: {output_file}", flush=True)

    seqs = generate_sequences(model, tokenizer, device, prompt, num_seqs, batch_size,
                              temperature, output_file, seq_prefix, rna_type, lineage,
                              save_probs=args.save_probs, top_k=args.top_k)

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"生成完成!")
    print(f"总计: {len(seqs)} 条序列")
    print(f"总耗时: {total_time/60:.2f}分钟")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
