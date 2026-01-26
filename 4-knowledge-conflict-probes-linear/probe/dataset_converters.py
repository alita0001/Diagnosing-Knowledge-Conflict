
from typing import Callable, Dict, List, Optional

from datasets import Dataset

from .data_types import AnnotatedSpan, ProbingItem



_MAP_LABEL_TO_SCALAR = {
    'Supported': 0.0,
    'Vision-Prior Conflict': 1.0,
    'Text-Prior Conflict': 2.0,
    'Vision-Text Conflict': 3.0,
    'Insufficient Information': -100.0,  
}


_MAP_CONFLICT_TYPE = {
    0: 1,  # image_vs_prior -> Vision-Prior Conflict (1)
    1: 3,  # image_vs_text -> Vision-Text Conflict (3)
    2: 2,  # text_vs_prior -> Text-Prior Conflict (2)
}

def prepare_multimodal_hallucination_dataset(dataset: Dataset) -> List[ProbingItem]:

    probing_items: List[ProbingItem] = []


    for hf_item in dataset:
        prompt = hf_item.get('detailed_prompt', hf_item.get('question', ''))

        model_output = hf_item.get('model_output', '')
        completion = model_output

        image = hf_item.get('image', None)

        conflict_type = hf_item.get('conflict_type', None)

        if conflict_type is not None:
            conflict_type = int(conflict_type)
            conflict_type = _MAP_CONFLICT_TYPE.get(conflict_type, conflict_type)

        annotations = hf_item.get('hallucination_annotation', [])
        annotated_spans: List[AnnotatedSpan] = []

        for annotation in annotations:
            if annotation is None or 'span' not in annotation:
                continue
            
            span_text = annotation['span']
            label_str = annotation['label']
            index = annotation['index']

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


