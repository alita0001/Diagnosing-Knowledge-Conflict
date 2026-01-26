"""
已刪除无关函数、修改标签映射

将不同格式的HuggingFace数据集转换为探针格式的转换器
"""

from typing import Callable, Dict, List, Optional

from datasets import Dataset

from .data_types import AnnotatedSpan, ProbingItem


# 标签映射
_MAP_LABEL_TO_SCALAR = {
    'Supported': 0.0,
    'Vision-Prior Conflict': 1.0,
    'Text-Prior Conflict': 2.0,
    'Vision-Text Conflict': 3.0,
    'Insufficient Information': -100.0,  # 视为需要忽略的
}

# conflict_type映射：数据集标签 -> 代码标签
_MAP_CONFLICT_TYPE = {
    0: 1,  # image_vs_prior -> Vision-Prior Conflict (1)
    1: 3,  # image_vs_text -> Vision-Text Conflict (3)
    2: 2,  # text_vs_prior -> Text-Prior Conflict (2)
}

def prepare_multimodal_hallucination_dataset(dataset: Dataset) -> List[ProbingItem]:
    """
    准备多模态幻觉标注数据集。
    
    数据集格式：
    - question: 问题文本
    - model_output: 模型输出（包含<think>标签的推理过程）
    - ground_truth: 正确答案
    - image: PIL Image对象
    - hallucination_annotation: 标注列表
    - detailed_prompt: 详细的提示词
    """
    probing_items: List[ProbingItem] = []


    for hf_item in dataset:
        prompt = hf_item.get('detailed_prompt', hf_item.get('question', ''))

        # 把 model_output 作为 completion
        model_output = hf_item.get('model_output', '')
        completion = model_output

        # 获取图像
        image = hf_item.get('image', None)

        # 获取样本级别的冲突类型标签
        conflict_type = hf_item.get('conflict_type', None)

        if conflict_type is not None:
            # 确保是整数类型
            conflict_type = int(conflict_type)
            # 使用映射字典将数据集标签转换为代码标签
            conflict_type = _MAP_CONFLICT_TYPE.get(conflict_type, conflict_type)
            # print(f"DEBUG: Final conflict_type: {conflict_type} (mapped from original {conflict_type})")

        # 处理标注
        annotations = hf_item.get('hallucination_annotation', [])
        annotated_spans: List[AnnotatedSpan] = []

        for annotation in annotations:
            if annotation is None or 'span' not in annotation:
                continue
            
            span_text = annotation['span']
            label_str = annotation['label']
            index = annotation['index']

            # 映射标签
            label_value = _MAP_LABEL_TO_SCALAR.get(label_str, -100.0)

            if index is None:
                print(f"Entity {repr(span_text)}'s idx set to None, discarding entity")
                continue
            elif not span_text or span_text not in completion:
                print(f"Entity {repr(span_text)} not found in completion, discarding entity")
                continue
            
            annotated_spans.append(
                AnnotatedSpan(
                    span=span_text,
                    label=label_value,
                    index=index
                )
            )

        # 只添加有标注的项
        if annotated_spans:
            probing_items.append(
                ProbingItem(
                    prompt=prompt,
                    completion=completion,
                    spans=annotated_spans,
                    image=image,
                    conflict_type=conflict_type
                )
            )
    return probing_items
    


def get_prepare_function(
    hf_repo: str,
    subset: Optional[str] = None
) -> Callable[[Dataset], List[ProbingItem]]:
    return prepare_multimodal_hallucination_dataset


