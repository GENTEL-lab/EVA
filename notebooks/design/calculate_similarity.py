#!/usr/bin/env python3
"""
使用与filter_fasta_by_length_similarity.sh相同的方法计算序列相似度
"""

def calculate_similarity(seq1, seq2):
    """计算两条序列的相似度（基于Needleman-Wunsch全局比对或简单匹配）"""
    # 如果长度相同，直接逐位比较
    if len(seq1) == len(seq2):
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    else:
        # 使用简化的相似度计算：基于较短序列的滑动窗口最佳匹配
        if len(seq1) < len(seq2):
            short_seq, long_seq = seq1, seq2
        else:
            short_seq, long_seq = seq2, seq1
        
        best_similarity = 0
        best_position = 0
        for i in range(len(long_seq) - len(short_seq) + 1):
            window = long_seq[i:i+len(short_seq)]
            matches = sum(1 for a, b in zip(short_seq, window) if a == b)
            similarity = matches / len(short_seq)
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = i
        
        # 考虑长度差异的惩罚
        length_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        final_similarity = best_similarity * length_ratio
        
        return final_similarity, best_similarity, best_position, length_ratio

# 读取序列
def read_fasta(filename):
    """读取FASTA文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    seq = ''.join(line.strip() for line in lines[1:])
    return header, seq.upper().replace('T', 'U')

# 读取两个序列
header1, seq1 = read_fasta('/storage9920/home/yanjie.huang/RNAVerse/notebooks/design/data/gRNA/designed_wrna.fasta')
header2, seq2 = read_fasta('/storage9920/home/yanjie.huang/RNAVerse/notebooks/design/data/gRNA/m16_WT.fasta')

print("=" * 60)
print("序列相似度计算")
print("=" * 60)
print(f"\n序列1: {header1}")
print(f"长度: {len(seq1)} nt")
print(f"序列: {seq1}\n")

print(f"序列2: {header2}")
print(f"长度: {len(seq2)} nt")
print(f"序列: {seq2}\n")

# 计算相似度
if len(seq1) == len(seq2):
    similarity = calculate_similarity(seq1, seq2)
    matches = int(similarity * len(seq1))
    print(f"长度相同，直接逐位比较")
    print(f"匹配碱基数: {matches} / {len(seq1)}")
    print(f"相似度: {similarity:.4f} ({similarity*100:.2f}%)")
else:
    final_sim, best_sim, best_pos, length_ratio = calculate_similarity(seq1, seq2)
    
    # 显示详细信息
    if len(seq1) < len(seq2):
        short_seq, long_seq = seq1, seq2
        short_name, long_name = header1, header2
    else:
        short_seq, long_seq = seq2, seq1
        short_name, long_name = header2, header1
    
    window = long_seq[best_pos:best_pos+len(short_seq)]
    matches = sum(1 for a, b in zip(short_seq, window) if a == b)
    mismatches = len(short_seq) - matches
    
    print(f"长度不同，使用滑动窗口匹配")
    print(f"较短序列: {short_name} ({len(short_seq)} nt)")
    print(f"较长序列: {long_name} ({len(long_seq)} nt)")
    print(f"长度差异: {abs(len(seq1) - len(seq2))} nt\n")
    
    print(f"最佳匹配位置: 位置 {best_pos} (0-based)")
    print(f"匹配窗口内相似度: {best_sim:.4f} ({best_sim*100:.2f}%)")
    print(f"  - 匹配碱基数: {matches} / {len(short_seq)}")
    print(f"  - 不匹配碱基数: {mismatches}")
    
    print(f"\n长度比例惩罚: {length_ratio:.4f}")
    print(f"最终相似度: {final_sim:.4f} ({final_sim*100:.2f}%)")
    
    # 显示比对结果
    print("\n比对可视化 (最佳匹配位置):")
    print(f"参考序列: {long_seq[:best_pos]}[{window}]{long_seq[best_pos+len(short_seq):]}")
    print(f"查询序列: {' '*best_pos}[{short_seq}]")
    
    # 显示差异位置
    print("\n差异位置:")
    diff_positions = []
    for i, (a, b) in enumerate(zip(short_seq, window)):
        if a != b:
            diff_positions.append((i, a, b))
    
    if diff_positions:
        for pos, base1, base2 in diff_positions[:10]:  # 只显示前10个差异
            print(f"  位置 {pos}: {base1} -> {base2}")
        if len(diff_positions) > 10:
            print(f"  ... 还有 {len(diff_positions)-10} 个差异位置")
    else:
        print("  无差异（完全匹配）")

print("\n" + "=" * 60)
