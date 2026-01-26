"""Probe module for hallucination detection.
视觉-语言多模态幻觉检测探针模块，专门用于VLM（Vision Language Model）

主要特性：
1. 支持多种VLM模型（LLaVA, Qwen-VL, InternVL等）
2. 四分类幻觉检测：
   - 类别0：真实/支持的内容
   - 类别1：视觉-先验知识冲突
   - 类别2：文本-先验知识冲突  
   - 类别3：视觉-文本冲突
3. 图像和文本联合处理
4. LoRA高效微调
5. 多层次评估指标

与纯文本探针的区别：
- 支持图像输入
- 多分类而非二分类
- 使用Processor处理图像
"""

from .value_head_probe import ValueHeadProbe
from .config import ProbeConfig, TrainingConfig, EvaluationConfig
from .loss import (
    compute_probe_ce_loss,
    compute_probe_max_aggregation_loss,
    compute_sparsity_loss,
    compute_kl_divergence_loss,
    mask_high_loss_spans
)
from .data_types import (
    ProbingItem,
    AnnotatedSpan
)
from .dataset import (
    TokenizedProbingDataset,
    tokenized_probing_collate_fn,
    create_probing_dataset
)

__all__ = [
    "ValueHeadProbe",
    "ProbeConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "compute_probe_ce_loss",
    "compute_probe_max_aggregation_loss",
    "compute_sparsity_loss",
    "compute_kl_divergence_loss",
    "mask_high_loss_spans",
    "setup_probe",
    "ProbingItem",
    "AnnotatedSpan",
    "TokenizedProbingDataset",
    "tokenized_probing_collate_fn",
    "create_probing_dataset"
]