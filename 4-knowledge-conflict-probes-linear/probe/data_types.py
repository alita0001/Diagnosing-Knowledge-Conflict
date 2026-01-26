"""
未修改，多模态二分类已满足
1. 支持图像输入（image_path字段）
2. 四分类标签体系：
   - 0.0: 真实/支持的内容 (Supported)
   - 1.0: 视觉-先验知识冲突 (Vision-Prior Conflict)  0
   - 2.0: 文本-先验知识冲突 (Text-Prior Conflict)    2
   - 3.0: 视觉-文本冲突 (Vision-Text Conflict)       1
   - -100.0: 忽略 (Ignored)
"""


from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class AnnotatedSpan:
    """A text span with its hallucination label."""
    
    span: str  # The span text
    label: float  # 1.0 for hallucination, 0.0 for supported, -100.0 for ignored
    index: int  # Start index in the completion


@dataclass
class ProbingItem:
    """A single item containing prompt, completion and annotated spans."""

    prompt: str
    completion: str
    spans: List[AnnotatedSpan]
    # 视觉输入
    image: Optional[Image.Image] = None
    # 样本级别的冲突类型标签（1, 2, 3，只有正类）
    conflict_type: Optional[int] = None