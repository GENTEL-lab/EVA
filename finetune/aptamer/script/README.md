# Aptamer微调训练脚本使用指南

## 概述

`run_aptamer_finetuning.sh` 是一个用于从预训练模型进行全参数微调的脚本，支持自定义RNA类型token（如`<rna_aptamer>`）。

## 主要特点

1. **自定义RNA类型token**: 支持指定任意RNA类型（如aptamer、ribozyme等）
2. **可选物种前缀**: 可以选择是否使用物种层次分类学前缀
3. **EOS loss权重**: 增加EOS token的loss权重，帮助模型学会正确断句
4. **全参数微调**: 从预训练checkpoint加载模型权重，optimizer和scheduler重新初始化
5. **分布式训练**: 支持单机8卡或多节点分布式训练

## 快速开始

### 1. 准备数据

首先准备你的训练数据（FASTA格式）：

```bash
# 数据文件示例
./finetune/data/aptamer/pepper_active_sequences.fasta
```

### 2. 创建实验目录

```bash
# 创建实验目录
mkdir -p ./results/aptamer_finetuning/aptamer_v1

# 复制配置模板
cp ./finetune/aptamer/script/experiment_config_template.yaml \
   ./results/aptamer_finetuning/aptamer_v1/experiment_config.yaml
```

### 3. 修改配置文件

编辑 `experiment_config.yaml`，重点修改以下部分：

```yaml
data_config:
  # 【必须修改】训练数据文件路径
  train_file: ./finetune/data/aptamer/pepper_active_sequences.fasta

  # 【可选】是否使用物种前缀（aptamer通常设为false）
  use_lineage_prefix: false

  # 【推荐保持true】是否使用RNA类型前缀
  use_rna_type_prefix: true

training_config:
  # 【必须修改】实验名称
  run_name: aptamer_finetuning_v1

  # 【必须修改】输出目录
  output_dir: ./results/aptamer_finetuning/aptamer_finetuning_v1

  # 【必须修改】预训练checkpoint路径
  resume_from_pretrain: ./results/experiments/scaling_1.4B_v31_pretrain/checkpoint-22000

  # 【可选修改】学习率（微调通常使用较小的学习率）
  learning_rate: 5.0e-5

  # 【可选修改】训练轮数
  max_epochs: 10
```

### 4. 启动训练

```bash
# 基本用法
cd /path/to/project_root
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    aptamer_v1 \
    --rna-type aptamer

# 或使用完整路径
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    ./results/aptamer_finetuning/aptamer_v1 \
    --rna-type aptamer
```

### 5. 监控训练

```bash
# 查看日志
tail -f ./results/logs/aptamer_finetuning/aptamer_finetuning_*.log

# 检查GPU状态
docker exec eva nvidia-smi

# 检查训练进程
docker exec eva ps aux | grep train_aptamer_finetuning

# 停止训练
./kill_training.sh
```

## 命令行参数

### 必需参数

- `<实验目录路径>`: 实验目录路径，支持多种格式：
  - 宿主机路径: `./results/aptamer_finetuning/xxx`
  - 容器路径: `./results/aptamer_finetuning/xxx`
  - 实验名称: `aptamer_v1`（会自动补全路径）

- `--rna-type <RNA类型>`: 自定义RNA类型token
  - 示例: `--rna-type aptamer` → 生成 `<rna_aptamer>`
  - 示例: `--rna-type ribozyme` → 生成 `<rna_ribozyme>`

### 可选参数

- `--node-rank <0|1>`: 节点rank（多节点训练时使用，默认: 0）
- `--master-addr <地址>`: Master节点地址（默认: localhost）
- `--master-port <端口>`: Master端口（默认: 29502）
- `--nnodes <数量>`: 节点总数（默认: 1）
- `--nproc-per-node <数量>`: 每节点GPU数（默认: 8）
- `--container <名称>`: 容器名称（默认: eva）
- `--help`: 显示帮助信息

## 使用示例

### 示例1: 训练Aptamer模型

```bash
# 1. 创建实验目录
mkdir -p ./results/aptamer_finetuning/pepper_aptamer_v1

# 2. 复制并修改配置
cp finetune/aptamer/script/experiment_config_template.yaml \
   ./results/aptamer_finetuning/pepper_aptamer_v1/experiment_config.yaml

# 3. 启动训练
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    pepper_aptamer_v1 \
    --rna-type aptamer
```

### 示例2: 训练Ribozyme模型

```bash
# 1. 准备ribozyme数据
# 2. 创建实验目录和配置
mkdir -p ./results/aptamer_finetuning/ribozyme_v1

# 3. 启动训练
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    ribozyme_v1 \
    --rna-type ribozyme
```

### 示例3: 多节点训练

```bash
# 节点0（主节点）
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    aptamer_v1 \
    --rna-type aptamer \
    --node-rank 0 \
    --master-addr 10.1.18.27 \
    --nnodes 2

# 节点1（从节点）
./finetune/aptamer/script/run_aptamer_finetuning.sh \
    aptamer_v1 \
    --rna-type aptamer \
    --node-rank 1 \
    --master-addr 10.1.18.27 \
    --nnodes 2
```

## 配置文件说明

### 关键配置项

#### 1. 数据配置 (data_config)

```yaml
data_config:
  # 训练数据文件
  train_file: ./finetune/data/aptamer/pepper_active_sequences.fasta

  # 是否使用物种前缀（aptamer等非生物序列建议设为false）
  use_lineage_prefix: false

  # 是否使用RNA类型前缀（建议设为true）
  use_rna_type_prefix: true

  # Pretraining任务混合比例（0.0-1.0）
  # 0.0 = 纯微调（全部使用条件前缀）
  # 0.1 = 10%无前缀 + 90%带前缀（推荐）
  pretrain_ratio: 0.1
```

#### 2. 模型配置 (model_config)

```yaml
model_config:
  # 【注意】这些参数必须与预训练checkpoint匹配
  hidden_size: 1024
  num_hidden_layers: 26
  num_experts: 8

  # Dropout配置（微调时可以适当增加防止过拟合）
  resid_dropout: 0.1
  hidden_dropout: 0.1

  # EOS loss权重（增加权重帮助模型学会正确断句）
  eos_loss_weight: 2.0
```

#### 3. 训练配置 (training_config)

```yaml
training_config:
  # 预训练checkpoint路径
  resume_from_pretrain: ./results/experiments/scaling_1.4B_v31_pretrain/checkpoint-22000

  # Batch配置
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 16
  # 有效batch = 8 * 16 * 8 = 1024

  # 学习率（微调通常使用较小的学习率）
  learning_rate: 5.0e-5

  # 训练轮数
  max_epochs: 10

  # Warmup步数（通常为总步数的5-10%）
  warmup_steps: 500
```

## 注意事项

1. **容器环境**: 所有训练必须在`eva`容器中运行，确保容器已启动：
   ```bash
   docker ps | grep eva
   ```

2. **模型架构匹配**: 配置文件中的模型架构参数必须与预训练checkpoint完全匹配

3. **学习率调整**: 微调时建议使用较小的学习率（1e-5到1e-4），避免破坏预训练权重

4. **Dropout设置**: 微调时可以适当增加dropout（0.1-0.2）防止过拟合

5. **数据格式**: 训练数据必须是FASTA格式，每条序列包含序列名和RNA序列

6. **GPU内存**: 根据GPU内存调整batch size和gradient accumulation steps

7. **自定义RNA类型**:
   - 脚本会自动添加`<rna_`前缀和`>`后缀
   - 例如: `--rna-type aptamer` → `<rna_aptamer>`
   - 确保tokenizer词汇表中包含该token

## 输出文件

训练完成后，输出文件结构如下：

```
results/aptamer_finetuning/aptamer_v1/
├── experiment_config.yaml      # 训练配置
├── logs/                       # 训练日志
│   └── train_*.log
├── checkpoint-500/             # 中间checkpoint
│   ├── model.pt
│   └── optimizer.pt
├── checkpoint-1000/
├── ...
└── final/                      # 最终模型
    ├── model.pt
    └── config.json
```

## 故障排除

### 问题1: 容器未运行

```bash
# 错误信息: eva容器未运行
# 解决方法: 启动容器
./start_eva.sh
```

### 问题2: 配置文件不存在

```bash
# 错误信息: 配置文件不存在
# 解决方法: 确保实验目录下有experiment_config.yaml
ls ./results/aptamer_finetuning/aptamer_v1/experiment_config.yaml
```

### 问题3: CUDA OOM

```bash
# 错误信息: CUDA out of memory
# 解决方法: 减小batch size或增加gradient accumulation
# 在配置文件中修改:
per_device_train_batch_size: 4  # 从8减到4
gradient_accumulation_steps: 32  # 从16增到32
```

### 问题4: 训练进程卡住

```bash
# 检查NCCL通信
docker exec eva nvidia-smi

# 查看日志
tail -f ./results/logs/aptamer_finetuning/*.log

# 如果需要，停止训练
./kill_training.sh
```

## 相关文件

- 训练脚本: `finetune/aptamer/script/run_aptamer_finetuning.sh`
- 配置模板: `finetune/aptamer/script/experiment_config_template.yaml`
- 训练数据: `finetune/data/aptamer/pepper_active_sequences.fasta`
- Python训练代码: `finetune/train_finetune.py`（需要创建）

## 下一步

训练完成后，你可以：

1. 使用训练好的模型生成新的aptamer序列
2. 评估模型性能
3. 进行进一步的微调或优化

如有问题，请查看日志文件或联系开发团队。
