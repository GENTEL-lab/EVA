"""
Mid-Training 训练模块

Mid-training是在预训练基础上的继续训练，主要特点：
1. 开启物种层次分类学前缀 (use_lineage_prefix: true)
2. EOS token增加loss权重，帮助模型学会正确断句
3. 从预训练checkpoint继续训练
"""
