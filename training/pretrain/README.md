# Lineage-based 2阶段训练系统

基于Greengenes谱系字符串的RNA序列生成和补全训练系统。

## 概述

本系统使用新的编码方式：
- **物种信息**: Greengenes风格谱系字符串（byte-level编码）
- **RNA类型**: 15个专用token（如`<rna_mRNA>`）
- **格式**: `|D__域;P__门;C__纲;O__目;F__科;G__属;S__种;<rna_类型>|[序列]`

### 与现有3阶段训练的区别

| 特性 | 现有3阶段训练 | 新2阶段训练 |
|------|--------------|------------|
| 物种编码 | 离散token（如`<species_homo_sapiens>`） | Greengenes谱系字符串（byte-level） |
| 条件格式 | `<task_clm><rna_mrna><species_panda>` | `\|D__..;P__..;...;<rna_mRNA>\|` |
| 阶段数 | 3阶段（分类→CLM→补全） | 2阶段（生成→补全） |
| 独立性 | 与旧系统共存 | 完全独立的目录结构 |

## 训练阶段

### Stage 1: 条件序列生成
**任务**: 基于谱系+RNA类型生成完整序列
**格式**: `|谱系信息;<rna_类型>|[完整序列]<eos>`
**Loss计算**: 只在序列部分计算loss，条件部分被mask

```
示例:
|d__bacteria;p__pseudomonadota;c__gammaproteobacteria;o__enterobacterales;f__enterobacteriaceae;g__escherichia;s__escherichia_coli;<rna_mRNA>|AUGGCUAGCUAGC...<eos>
                                                条件部分(mask)                                                            序列部分(计算loss)
```

### Stage 2: 条件序列补全
**任务**: 基于谱系+RNA类型补全序列中间片段
**格式**: `|谱系信息;<rna_类型>|[前缀]<span_0>[后缀]<eos><span_0>[片段]<eos_span>`
**Loss计算**: 只在span片段部分计算loss

```
示例:
|谱系;<rna_mRNA>|AUG<span_0>CUA<eos><span_0>GCUAGC<eos_span>
      条件(mask)  前缀(mask)      后缀(mask)    片段(loss)
```

## 文件结构

```
training/pretrain/
├── __init__.py
├── README.md                                    # 本文档
└── train_stage1_generation.py                   # Stage 1训练脚本

data/
└── lineage_dataset.py                           # 谱系数据集处理

configs/lineage_training/
└── lineage_stage1_generation_16gpu.yaml         # Stage 1配置
```

## 使用方法

### 前置条件
1. **环境准备**
   - 训练数据: `/rna-multiverse/data/training_datasets/pilot_300k/pilot_300k_train.fa`
   - 谱系映射: `/rna-multiverse/data/training_data/lineage_greengenes.tsv`

3. **FASTA header格式要求**
   ```
   >seq_id|taxid=12345|rna_type=mRNA|...
   或
   >seq_id|taxid:12345|rna_type:mrna|...
   ```

### 训练命令

#### 1. 分阶段训练

**Stage 1: 序列生成**
```bash
cd <project_root>
./scripts/lineage_training/run_lineage_stage1.sh
```

**Stage 2: 序列补全** (自动加载Stage 1 checkpoint)
```bash
./scripts/lineage_training/run_lineage_stage2.sh
```

#### 2. 连续训练（推荐）

```bash
./scripts/lineage_training/run_lineage_all_stages.sh
```

自动依次执行Stage 1和Stage 2，Stage 2自动加载Stage 1的最终模型。

### 单机8卡训练（调试用）

在容器内执行：
```bash
# Stage 1
cd /rna-multiverse
torchrun --nproc_per_node=8 \
    training/lineage_training/train_lineage_stage1_generation.py \
    --config configs/lineage_training/lineage_stage1_generation_16gpu.yaml

# Stage 2
torchrun --nproc_per_node=8 \
    training/lineage_training/train_lineage_stage2_completion.py \
    --config configs/lineage_training/lineage_stage2_completion_16gpu.yaml
```

## 输出目录

```
results/
├── lineage_stage1_output/
│   ├── checkpoint-XXXX/          # 中间checkpoint
│   ├── final/                    # 最终模型
│   └── logs/                     # 训练日志
└── lineage_stage2_output/
    ├── checkpoint-XXXX/
    ├── final/
    └── logs/
```

## 配置说明

### Stage 1配置要点
- `learning_rate: 2e-3` - 较大学习率，从头训练
- `max_epochs: 2` - 2个epoch充分学习生成能力
- `warmup_ratio: 0.1` - 10% warmup

### Stage 2配置要点
- `learning_rate: 1e-3` - 较小学习率，微调
- `max_epochs: 1` - 1个epoch学习补全能力
- `warmup_ratio: 0.2` - 20% warmup
- `load_checkpoint: null` - 自动加载Stage 1最新checkpoint

## 数据处理细节

### 谱系映射
从`lineage_greengenes.tsv`加载：
```
taxid    lineage_greengenes
12345    d__bacteria;p__pseudomonadota;c__gammaproteobacteria;...
```

### RNA类型映射
支持15种RNA类型：
- mRNA, rRNA, tRNA, sRNA, lncRNA
- circRNA, viral_RNA, miRNA, snoRNA, snRNA
- piRNA, ribozyme, scaRNA, Y_RNA, vault_RNA

不支持的类型（自动过滤）：
- pseudogene系列
- 前体RNA (pri_miRNA, pre_miRNA)
- 模糊分类 (ncRNA, misc_RNA)

### 序列预处理
- T→U自动转换
- 非标准碱基移除
- 最小长度: 20 nt
- 最大长度: 8096 nt（可配置）

## 监控和调试

### 查看训练日志
```bash
# Stage 1
tail -f <project_root>/results/lineage_stage1_output/logs/training.log

# Stage 2
tail -f <project_root>/results/lineage_stage2_output/logs/training.log
```

### 终止训练
```bash
# 查找并终止进程
pkill -f train_lineage
```

### WandB监控
在配置文件中启用：
```yaml
logging_config:
  enable_wandb: true
  wandb_project: "rna-lineage-training"
```

## 常见问题

### 1. 谱系映射文件不存在
**错误**: `lineage_greengenes.tsv not found`
**解决**: 确保文件位于 `/rna-multiverse/data/training_data/lineage_greengenes.tsv`

### 2. 大量样本被过滤
**原因**: taxid在谱系文件中不存在，或RNA类型不支持
**解决**:
- 检查FASTA header的taxid格式
- 查看日志中的过滤统计
- 确认RNA类型在支持列表中

### 3. Stage 2找不到Stage 1 checkpoint
**错误**: `Stage 1 checkpoint not found`
**解决**:
- 检查 `results/lineage_stage1_output/` 是否存在
- 或在配置中手动指定: `load_checkpoint: /path/to/checkpoint`

### 4. OOM错误
**解决**:
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing: true`

## 性能优化建议

1. **批次大小调优**
   - A100 80GB推荐: `batch_size=12, accumulation=4`
   - 内存不足时: 减小batch_size，增加accumulation保持有效批次大小

2. **数据加载优化**
   - 增加 `dataloader_num_workers` (推荐8)
   - 启用 `dataloader_pin_memory`

3. **精度设置**
   - 推荐使用 `bf16: true` (A100最优)
   - RTX 3090等使用 `fp16: true`

4. **MoE专家并行**
   - 16卡配置: `data_parallel_size=4, expert_parallel_size=4`
   - 8卡配置: `data_parallel_size=2, expert_parallel_size=4`

## 与现有系统对比

| 对比项 | 现有3阶段训练 | 新2阶段训练 |
|--------|--------------|------------|
| 代码隔离 | 共用部分代码 | 完全独立 |
| 配置文件 | `configs/stage*.yaml` | `configs/lineage_training/` |
| 脚本位置 | `scripts/run_stage*.sh` | `scripts/lineage_training/` |
| 输出目录 | `results/stage*_output/` | `results/lineage_stage*_output/` |
| 数据集类 | `MultiTaskRNADataset` | `LineageRNADataset` |
| 可以共存 | ✅ 是 | ✅ 是 |

## 贡献和维护

本系统完全独立于现有3阶段训练系统，可以安全地进行修改和扩展而不影响原有系统。

如需修改：
1. 数据处理逻辑: 修改 `data/lineage_dataset.py`
2. 训练流程: 修改 `training/lineage_training/train_*.py`
3. 配置参数: 修改 `configs/lineage_training/*.yaml`
4. 启动方式: 修改 `scripts/lineage_training/*.sh`
