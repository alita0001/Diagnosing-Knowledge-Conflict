

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

from utils.tokenization import find_assistant_tokens_slice, find_string_in_tokens, slice_to_list
from .data_types import AnnotatedSpan, ProbingItem
from .dataset_converters import get_prepare_function

@dataclass
class TokenizedProbingDatasetConfig:

    
    dataset_id: str                         # 
    hf_repo: str                            # 
    subset: Optional[str] = None            # 
    split: str = "train"                    # 
    max_length: int = 2048                  # 
    ignore_buffer: int = 0                  # 
    default_ignore: bool = False            # 
    last_span_token: bool = False           # 
    vpc_weight: float = 1.0                 # 
    tpc_weight: float = 1.0                 # 
    vtc_weight: float = 1.0                 # 
    neg_weight: float = 1.0                 # 
    shuffle: bool = True                    # 
    seed: int = 42                          # 
    process_on_the_fly: bool = False        # 
    max_num_samples: Optional[int] = None   # 


class TokenizedProbingDataset(Dataset):
    
    def __init__(
        self,
        items: List[ProbingItem],
        config: TokenizedProbingDatasetConfig,
        tokenizer: Union[AutoProcessor, AutoTokenizer],
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.items = deepcopy(items)
        self.processed_items = [None] * len(items)
        self.debug_mode = False
        self.print_first_example = False
        
        self._num_skipped_spans: int = 0
        self._num_added_spans: int = 0
        
        if self.config.shuffle:
            self._shuffle_items()
        
        if self.config.max_num_samples:
            self.items = self.items[:self.config.max_num_samples]
            self.processed_items = self.processed_items[:self.config.max_num_samples]
        
        if not self.config.process_on_the_fly:
            self._process_items()
    
    def _process_items(self):
        for i, item in tqdm(enumerate(self.items), desc=f"Processing items ({self.config.dataset_id})", total=len(self.items)):
            if i == 0 and self.print_first_example:
                self.debug_mode = True
            else:
                self.debug_mode = False
            
            processed_item = self._process_item(item)
            if processed_item:
                self.processed_items[i] = processed_item
        
        print(f"Dataset {self.config.dataset_id} stats:")
        print(f"\t- Number of added spans: {self._num_added_spans}")
        print(f"\t- Number of skipped spans: {self._num_skipped_spans} / {self._num_added_spans + self._num_skipped_spans}")
        print(f"\t- Total number of items: {len(self.items)}")
    
    def _process_item(self, item: ProbingItem) -> Optional[Dict]:
        if item.image is not None:
            image = item.image
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item.prompt}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [
                        {"type": "text", "text": item.completion}
                    ]
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item.prompt}
                    ]
                },
                {
                    'role': 'assistant', 
                    'content': [
                        {"type": "text", "text": item.completion} 
                    ]
                }
            ]

        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        pixel_values = inputs.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values[0]

        labels, weights, vpc_spans, tpc_spans, vtc_spans, neg_spans = self._compute_positional_labels(
            input_ids=input_ids,
            item=item
        )
        
        input_str: str = self.tokenizer.decode(input_ids)
        assistant_tokens_slice = find_assistant_tokens_slice(input_ids, input_str, self.tokenizer)
        completion_start_idx = assistant_tokens_slice.stop
        
        lm_labels = input_ids.clone()
        lm_labels[:completion_start_idx] = -100
        lm_labels[attention_mask == 0] = -100
        
        result = {
            "inputs": inputs,
            "input_ids": input_ids,  # Int[Tensor, "seq_len"]
            "attention_mask": attention_mask,  # Int[Tensor, "seq_len"]
            "classification_labels": labels,  # Float[Tensor, "seq_len"]
            "classification_weights": weights,  # Float[Tensor, "seq_len"]
            "vpc_spans": vpc_spans,  # List[List[int]]
            "tpc_spans": tpc_spans,  # List[List[int]]
            "vtc_spans": vtc_spans,  # List[List[int]]
            "neg_spans": neg_spans,  # List[List[int]]
            "lm_labels": lm_labels,  # Int[Tensor, "seq_len"]
            "item":item,
            "conversation": conversation,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result
    
    def print_token_labels(
        self,
        input_ids: torch.Tensor,
        vpc_indices: List[int],
        tpc_indices: List[int],
        vtc_indices: List[int],
        negative_indices: List[int],
        ignore_indices: List[int],
        spans: List[AnnotatedSpan]
    ):


        tokens = [self.tokenizer.decode(tok) for tok in input_ids]

        print(f"================================================")
        print(f"Number of spans: {len(spans)}")
        print(f"Number of vpc (hallucinated) spans: {len([f for f in spans if f.label == 1.0])}")
        print(f"Number of tpc (hallucinated) spans: {len([f for f in spans if f.label == 2.0])}")
        print(f"Number of vtc (hallucinated) spans: {len([f for f in spans if f.label == 3.0])}")
        print(f"Number of N/A spans: {len([f for f in spans if f.label == -100])}")
        print(f"Number of factual spans: {len([f for f in spans if f.label == 0.0])}")
        print(f"Legend: red - positive, green - negative, blue - ignored")

        for i, token in enumerate(tokens):
            if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'eos_token'):
                eos_token = self.tokenizer.tokenizer.eos_token
            else:
                eos_token = getattr(self.tokenizer, 'eos_token', None)
            if token == eos_token:
                continue

            if i in vpc_indices:
                print(colored(token, 'red'), end='')
            elif i in tpc_indices:
                print(colored(token, 'yellow'), end='')
            elif i in vtc_indices:
                print(colored(token, 'pink'), end='')
            elif i in negative_indices:
                print(colored(token, 'green'), end='')
            elif i in ignore_indices:
                print(colored(token, 'blue'), end='')
            else:
                print(token, end='')

        print(f"================================================")
    
    def _compute_positional_labels(
        self,
        input_ids: torch.Tensor,
        item: ProbingItem
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]]:

        input_str: str = self.tokenizer.decode(input_ids)
        completion: str = item.completion
        
        negative_indices: List[int] = []    # 
        ignore_indices: List[int] = []      # 
        vpc_indices: List[int] = []         # 
        tpc_indices: List[int] = []         # 
        vtc_indices: List[int] = []         # 

        negative_spans: List[List[int]] = []
        vpc_spans: List[List[int]] = []
        tpc_spans: List[List[int]] = []
        vtc_spans: List[List[int]] = []
        
        def get_nearby_indices(span_indices: List[int]) -> List[int]:
            left_window = list(range(max(0, span_indices[0] - self.config.ignore_buffer), span_indices[0]))
            right_window = list(range(span_indices[-1] + 1, min(len(input_ids), span_indices[-1] + 1 + self.config.ignore_buffer)))
            return left_window + right_window
        
        assistant_tokens_slice = find_assistant_tokens_slice(
            input_ids,
            input_str,
            self.tokenizer
        )
        completion_start_idx = assistant_tokens_slice.stop
        cur_idx = assistant_tokens_slice.stop
        
        spans = sorted(item.spans, key=lambda x: x.index)
        
        for span in spans:
            if span.span not in input_str:
                self._num_skipped_spans += 1
                continue
            
            try:
                positions_slice = find_string_in_tokens(span.span, input_ids[cur_idx:], self.tokenizer)
                positions_slice = slice(positions_slice.start + cur_idx, positions_slice.stop + cur_idx)
            except (AssertionError, ValueError):
                try:
                    print(f"Repeating position_slice search on all tokens after failing to find span {repr(span.span)} in input_ids[cur_idx:]: {repr(self.tokenizer.decode(input_ids[cur_idx:]))[:50]}...")
                    positions_slice = find_string_in_tokens(span.span, input_ids, self.tokenizer)
                except (AssertionError, ValueError) as e:
                    print(f"Span {repr(span.span)} not found in input_ids, skipping entity")
                    self._num_skipped_spans += 1
                    continue
            
            if positions_slice is None:
                continue
            
            span_indices = slice_to_list(positions_slice, len(input_ids))
            if not span_indices:
                continue
            
            cur_idx = positions_slice.start
            
            if self.config.last_span_token:
                span_indices = [span_indices[-1]]
            
            nearby_indices = get_nearby_indices(span_indices)
            
            if span.label == 1.0:  
                vpc_indices.extend(span_indices)
                ignore_indices.extend(nearby_indices)
                span_to_append = [span_indices[0], span_indices[-1]]
                if len(span_to_append) != 2:
                    print(f"⚠️ [DATASET] VPC span 格式异常: span_indices={span_indices}, span_to_append={span_to_append}")
                vpc_spans.append(span_to_append)

            elif span.label == 2.0:
                tpc_indices.extend(span_indices)
                ignore_indices.extend(nearby_indices)
                span_to_append = [span_indices[0], span_indices[-1]]
                if len(span_to_append) != 2:
                    print(f"⚠️ [DATASET] TPC span 格式异常: span_indices={span_indices}, span_to_append={span_to_append}")
                tpc_spans.append(span_to_append)
            elif span.label == 3.0:  # 视觉-文本冲突索引 (Vision-Text Conflict)
                vtc_indices.extend(span_indices)
                ignore_indices.extend(nearby_indices)
                span_to_append = [span_indices[0], span_indices[-1]]
                if len(span_to_append) != 2:
                    print(f"⚠️ [DATASET] VTC span 格式异常: span_indices={span_indices}, span_to_append={span_to_append}")
                vtc_spans.append(span_to_append)
            elif span.label == 0.0:  # Supported  # 支持
                negative_indices.extend(span_indices)
                span_to_append = [span_indices[0], span_indices[-1]]
                if len(span_to_append) != 2:
                    print(f"⚠️ [DATASET] Negative span 格式异常: span_indices={span_indices}, span_to_append={span_to_append}")
                negative_spans.append(span_to_append)
            else:  # -100.0 (ignored)  # -100.0（忽略）
                ignore_indices.extend(span_indices)
            
            self._num_added_spans += 1
        
        # 去重并排序
        vpc_indices = sorted(list(set(vpc_indices)))
        tpc_indices = sorted(list(set(tpc_indices)))
        vtc_indices = sorted(list(set(vtc_indices)))
        negative_indices = sorted(list(set(negative_indices) - set(vpc_indices) - set(tpc_indices) - set(vtc_indices)))
        ignore_indices = sorted(list(set(ignore_indices) - set(vpc_indices) - set(tpc_indices) - set(vtc_indices) - set(negative_indices)))
        
        # 初始化标签和权重
        default_label = -100.0 if self.config.default_ignore else 0.0
        labels = torch.full((len(input_ids),), default_label, dtype=torch.float32)
        
        # 设置标签和权重
        # 处理tokenizer和processor的兼容性
        if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.tokenizer.pad_token_id
        else:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        labels[input_ids == pad_token_id] = -100.0
        labels[:completion_start_idx] = -100.0
        labels[ignore_indices] = -100.0
        labels[vpc_indices] = 1.0
        labels[tpc_indices] = 2.0
        labels[vtc_indices] = 3.0
        labels[negative_indices] = 0.0
        
        # 创建一个与序列长度相同的张量，初始值全为 1.0
        weights = torch.full((len(input_ids),), 1.0, dtype=torch.float32)
        weights[ignore_indices] = 0.0  # N/A权重
        weights[vpc_indices] = self.config.vpc_weight
        weights[tpc_indices] = self.config.tpc_weight
        weights[vtc_indices] = self.config.vtc_weight
        weights[negative_indices] = self.config.neg_weight
        
        if self.debug_mode:
            self.print_token_labels(input_ids, vpc_indices, tpc_indices, vtc_indices, negative_indices, ignore_indices, spans)
        
        return labels, weights, vpc_spans, tpc_spans, vtc_spans, negative_spans
    
    def _shuffle_items(self):
        """使用配置的随机种子打乱项目"""
        random.seed(self.config.seed)
        random.shuffle(self.items)
        random.seed(self.config.seed)
        random.shuffle(self.processed_items)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        if self.config.process_on_the_fly and self.processed_items[idx] is None:
            self.processed_items[idx] = self._process_item(self.items[idx])
        
        return self.processed_items[idx]
    
    def __add__(self, other):
        """
        连接两个TokenizedProbingDataset实例。
        
        参数:
            other: 要连接的另一个TokenizedProbingDataset实例
            
        返回:
            TokenizedProbingDataset: 包含两个数据集项目的新数据集
        """
        if not isinstance(other, TokenizedProbingDataset):
            raise TypeError(f"Can only concatenate with another TokenizedProbingDataset, got {type(other)}")
        
        if self.config.max_length != other.config.max_length:
            raise ValueError("Can't concatenate datasets of different token lengths")
        
        if self.config.shuffle != other.config.shuffle:
            raise ValueError("Can't concatenate datasets if one of them (but not the other) are shuffled")
        
        # 创建包含合并项目的新数据集
        combined_items = self.items + other.items
        combined_processed_items = self.processed_items + other.processed_items
        
        # 使用第一个数据集的配置
        new_dataset = TokenizedProbingDataset(
            items=[],  # 我们不想重新计算所有内容
            tokenizer=self.tokenizer,
            config=self.config,
        )
        
        new_dataset.items = combined_items
        new_dataset.processed_items = combined_processed_items
        
        if self.config.shuffle:
            new_dataset._shuffle_items()
        
        return new_dataset
    
    def __radd__(self, other):
        return self.__add__(other)


def tokenized_probing_collate_fn(batch: List[Dict[str, torch.Tensor]], processor: Optional[AutoProcessor] = None) -> Dict[str, torch.Tensor]:
    """
    Modified: subdivided into three types of hallucinations
    Collate function for DataLoader, handles variable-length tokenized sequences.
    参数:
        batch: tokenized数据集项目列表
    返回:
        包含填充序列的批处理字典
    """
    # 找到批次中的最大长度
    max_len = max(len(item["input_ids"]) for item in batch)
    
    # 初始化批处理张量
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    classification_labels = torch.full((batch_size, max_len), -100.0, dtype=torch.float32)
    classification_weights = torch.zeros((batch_size, max_len), dtype=torch.float32)
    lm_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    # span列表
    vpc_spans = []
    tpc_spans = []
    vtc_spans = []
    neg_spans = []

    # 样本级别的冲突类型标签
    conflict_types = []

    # 填充批次
    messages_batch = []
    for i, item in enumerate(batch):
        messages_batch.append(item["conversation"])

        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        classification_labels[i, :seq_len] = item["classification_labels"]
        classification_weights[i, :seq_len] = item["classification_weights"]
        lm_labels[i, :seq_len] = item["lm_labels"]
        vpc_spans.append(item["vpc_spans"])
        tpc_spans.append(item["tpc_spans"])
        vtc_spans.append(item["vtc_spans"])
        neg_spans.append(item["neg_spans"])
        conflict_types.append(item["item"].conflict_type)

    inputs = processor.apply_chat_template(
        messages_batch,
        add_generation_prompt=False, # Add generation prompt to tell the model how to generate text. Without: "User: What is this?"  ; With:    "User: What is this?\nAssistant:" <- Prompts model to start generation
        tokenize=True, # Convert text to token IDs
        return_dict=True, # Return results as a dictionary
        return_tensors="pt", # Return PyTorch tensors
        padding=True,
    )
    
    return {
        "inputs": inputs,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "classification_labels": classification_labels,
        "classification_weights": classification_weights,
        "lm_labels": lm_labels,
        "vpc_spans": vpc_spans,
        "tpc_spans": tpc_spans,
        "vtc_spans": vtc_spans,
        "neg_spans": neg_spans,
        "conflict_types": conflict_types,
        "messages": messages_batch,
    }


def create_probing_dataset(
    cfg: TokenizedProbingDatasetConfig,
    tokenizer: AutoTokenizer
) -> TokenizedProbingDataset:
    """
    Unmodified
    Create probing dataset from configuration.
    This loads the dataset from HuggingFace and processes it using the corresponding dataset-specific preparation function.
    """
    # Load dataset from HuggingFace
    if cfg.subset:
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, cfg.subset, split=cfg.split)
    else:
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, split=cfg.split)

    # Handle maximum number of samples
    if cfg.max_num_samples is not None:
        if cfg.shuffle:
            print(f"Shuffling and truncating dataset to {cfg.max_num_samples} / {len(raw_hf_dataset)} samples")
            assert cfg.seed is not None, "Seed must be provided if shuffle is True"
            raw_hf_dataset = raw_hf_dataset.shuffle(seed=cfg.seed)
        else:
            print(f"Truncating dataset to first {cfg.max_num_samples} / {len(raw_hf_dataset)} samples")
        raw_hf_dataset = raw_hf_dataset.select(range(min(cfg.max_num_samples, len(raw_hf_dataset))))

    print(f"Loading dataset: {cfg.hf_repo} | {cfg.subset} | {cfg.split}")

    # Get the corresponding preparation function
    prepare_function = get_prepare_function(cfg.hf_repo, cfg.subset)

    # Convert HF dataset to list of probing items
    probing_items: List[ProbingItem] = prepare_function(raw_hf_dataset)

    tokenized_probing_dataset = TokenizedProbingDataset(
        items=probing_items,
        config=cfg,
        tokenizer=tokenizer
    )

    return tokenized_probing_dataset