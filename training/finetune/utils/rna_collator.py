"""RNA数据整理器"""

import logging
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RNADataCollator:
    """RNA数据整理器 - 负责将序列批次转换为模型输入格式"""

    tokenizer: Any = None
    max_length: int = 2048
    pad_token_id: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.tokenizer is not None:
            self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        logger.info(f"RNADataCollator初始化: max_length={self.max_length}")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理数据批次"""
        try:
            # 确保所有张量在CPU上
            for example in examples:
                for key in ["input_ids", "labels", "position_ids", "sequence_ids"]:
                    if key in example and hasattr(example[key], 'device'):
                        if example[key].device.type != 'cpu':
                            example[key] = example[key].cpu()

            # 找出最大长度
            max_length = max(len(ex["input_ids"]) for ex in examples)
            max_length = min(max_length, self.max_length)

            # 填充
            padded_input_ids, padded_labels, padded_position_ids, padded_sequence_ids = [], [], [], []

            for example in examples:
                curr_len = len(example["input_ids"])
                if curr_len > self.max_length:
                    curr_len = self.max_length

                padding_len = max_length - curr_len

                # 处理input_ids
                input_tensor = example["input_ids"][:curr_len] if hasattr(example["input_ids"], '__getitem__') else torch.tensor(example["input_ids"][:curr_len], dtype=torch.long)
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.tensor(input_tensor, dtype=torch.long)
                padded_input_ids.append(torch.cat([input_tensor, torch.full((padding_len,), self.pad_token_id, dtype=torch.long)]))

                # 处理labels
                label_tensor = example["labels"][:curr_len] if hasattr(example["labels"], '__getitem__') else torch.tensor(example["labels"][:curr_len], dtype=torch.long)
                if not isinstance(label_tensor, torch.Tensor):
                    label_tensor = torch.tensor(label_tensor, dtype=torch.long)
                padded_labels.append(torch.cat([label_tensor, torch.full((padding_len,), -100, dtype=torch.long)]))

                # 处理position_ids
                if "position_ids" in example:
                    pos_tensor = example["position_ids"][:curr_len]
                    if not isinstance(pos_tensor, torch.Tensor):
                        pos_tensor = torch.tensor(pos_tensor, dtype=torch.long)
                else:
                    pos_tensor = torch.arange(curr_len, dtype=torch.long)
                padded_position_ids.append(torch.cat([pos_tensor, torch.zeros(padding_len, dtype=torch.long)]))

                # 处理sequence_ids
                if "sequence_ids" in example:
                    seq_tensor = example["sequence_ids"][:curr_len]
                    if not isinstance(seq_tensor, torch.Tensor):
                        seq_tensor = torch.tensor(seq_tensor, dtype=torch.long)
                else:
                    seq_tensor = torch.zeros(curr_len, dtype=torch.long)
                padded_sequence_ids.append(torch.cat([seq_tensor, torch.zeros(padding_len, dtype=torch.long)]))

            # 构建batch
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "labels": torch.stack(padded_labels),
                "position_ids": torch.stack(padded_position_ids),
                "sequence_ids": torch.stack(padded_sequence_ids),
            }
            batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()

            # 收集task_type
            if "task_type" in examples[0]:
                batch["task_types"] = [ex["task_type"] for ex in examples]

            return batch

        except Exception as e:
            logger.error(f"处理批次出错: {e}")
            raise


def create_rna_data_collator(
    tokenizer=None,
    max_length: int = 2048,
    device: str = "cpu",
    **kwargs
) -> RNADataCollator:
    """创建RNA数据整理器"""
    return RNADataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        device=device
    )
