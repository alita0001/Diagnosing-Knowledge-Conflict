#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态幻觉标注流程主脚本

功能：
1. 使用 Claude-4.5 模型对视觉问答模型的输出进行幻觉检测和标注
2. 支持断点续传，避免重复处理
3. 实时保存本地结果（Arrow 格式，包含图片）
4. 定期上传到 HuggingFace Hub
5. 支持多个数据集类别的处理

使用方法：
    1. 设置环境变量：
       export API_KEY='your-api-key-here'
       export HF_WRITE_TOKEN='your-hf-token-here'
       export HF_TOKEN='your-hf-token-here'
    
    
    2. 修改配置或使用命令行参数：
       CUDA_VISIBLE_DEVICES=1 python -m annotation_pipeline.run --source_split_name text_vs_prior --push_intermediate_every 50
       CUDA_VISIBLE_DEVICES=4 python -m annotation_pipeline.run --source_split_name train -- --push_intermediate_every 50
    
    3. 运行脚本：
       python -m annotation_pipeline.run
"""



import os
import sys
import time
import hashlib
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import traceback # 用于打印异常堆栈
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset, Dataset, Image, Features, Value, List as HFList
import asyncio
from tqdm.asyncio import tqdm_asyncio
sys.path.append(str(Path(__file__).parent.parent))
from utils.file_utils import load_jsonl,save_jsonl
from utils.parsing import validate_dicts_to_pydantic # 数据验证
from .annotate import annotate_sample
from .data_models import MultimodalDatasetItem, MultimodalAnnotatedSpan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,  # Python 3.8+ 强制重新配置
    handlers=[
        logging.StreamHandler(sys.stdout)  # 明确指定输出到 stdout
    ]
)
logger = logging.getLogger(__name__)

# 测试日志
logger.info("✅ Logging 配置完成")

# combined_dataset: Dataset = None

# ============================================================================
# 配置类
# ============================================================================

@dataclass
class PipelineConfig:
    """多模态标注流程配置
    
    集中管理所有配置参数
    """
    
    # ===== API 配置 =====
    api_key: str = None  # Set via environment variable API_KEY
    base_url: str = "https://api.openai.com/v1"  # Or your preferred API endpoint
    annotator_model: str = "gpt-4"
    
    # ===== 源数据集配置 =====
    source_dataset_name: str = "anonymous/TriConflict-inference"  # Replace with your dataset
    source_subset_name: str = "model-subset"  # Subset
    source_split_name: str = "text_vs_prior"

    # ===== 目标数据集配置 =====
    target_dataset_name: str = "anonymous/TriConflict-annotations"  # Replace with your dataset
    target_subset_name: str = "model-subset"  # Subset
    target_split_name: str = "text_vs_prior"

    hf_token: str = None  # Set via environment variable HF_TOKEN
    output_dir: Path = Path("./hallucination_annotations")
    enable_hf_upload: bool = True
    

    verbose: bool = True
    parallel_mode: bool = False
    # ===== 处理配置 =====
    push_intermediate_every: int = 10
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量获取 API 密钥
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")
        
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
 
        # 确保路径为 Path 对象
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_local_dataset_dir(self) -> Path:
        """获取本地数据集目录（Arrow 格式，含图片）"""
        return Path(self.output_dir / self.target_dataset_name / self.target_subset_name / (self.target_split_name))
    
    def validate(self):
        """验证必要的配置是否已设置"""
        if not self.api_key:
            raise ValueError(
                "未设置 API_KEY 环境变量\n"
                "请运行: export API_KEY='sk-or-v1-...'"
            )
        
        if self.enable_hf_upload and not self.hf_token:
            logger.warning("启用了 HuggingFace 上传但未设置 HF_WRITE_TOKEN 或 HF_TOKEN")
            logger.warning("上传功能将被禁用")
            self.enable_hf_upload = False


# ============================================================================
# 工具函数
# ============================================================================

def get_item_key(item: MultimodalDatasetItem) -> str:
    """生成 item 的唯一 key（用于去重）
    
    使用 question_id 的 MD5 哈希作为唯一标识
    
    Args:
        item: 包含 question_id 的字典
        
    Returns:
        MD5 哈希字符串
    """
    key_str = item.question_id
    return hashlib.md5(key_str.encode()).hexdigest()

def is_item_processed(item: MultimodalDatasetItem, processed_keys: set) -> bool:
    """Check if an item has already been processed"""
    return get_item_key(item) in processed_keys

def load_processed_items_from_disk(local_dataset_dir: Path) -> List[MultimodalDatasetItem]:
    """从本地 Arrow 数据集加载已处理的结果"""
    if not local_dataset_dir.exists():
        return []
    
    try:
        dataset = Dataset.load_from_disk(local_dataset_dir)
    except Exception as e:
        logger.error(f"从本地数据集加载已处理的结果失败: {e}")
        return []

    # 转换成List[Dict]
    raw_items = list(dataset)
    #验证并转换为Pydantic对象
    validated_items = validate_dicts_to_pydantic(raw_items, MultimodalDatasetItem, skip_invalid=True)

    # 如果有些项验证失败，则记录日志
    if len(validated_items) < len(raw_items):
        logger.warning(f"Skipped {len(raw_items) - len(validated_items)} invalid items from {local_dataset_dir}")

    return validated_items

# ============================================================================
# HuggingFace 上传相关函数
# ============================================================================
# 在上传前统一数据格式
def prepare_item_for_upload(item: MultimodalDatasetItem, config: PipelineConfig) -> dict:
    """
    准备数据用于上传，确保字段与 Hub 上的格式一致
    处理 None 值和缺失字段，使其符合 HuggingFace 数据集的类型要求
    """
    item_dict = item.model_dump()
    
    # 使用 get() + or 替换 None 和缺失字段
    # 这样可以确保类型为 string 而不是 null
    item_dict['ground_truth'] = item_dict.get('ground_truth') or ""
    item_dict['image_name'] = item_dict.get('image_name') or ""
    item_dict['task'] = item_dict.get('task') or ""
    item_dict['detailed_prompt'] = item_dict.get('detailed_prompt') or item_dict.get('question') or ""
    item_dict['model_name'] = item_dict.get('model_name') or config.target_subset_name or ""
    # image 字段特殊处理（Image 类型允许 None）
    if 'image' not in item_dict:
        item_dict['image'] = None
    
    # 移除不兼容的字段（如果存在）
    item_dict.pop('subset', None)
    
    return item_dict

# 上传到 HuggingFace
def upload_to_huggingface(
    config: PipelineConfig
) -> None:
    # Always load all processed items from local file to ensure nothing is missed
    all_local_items = load_processed_items_from_disk(config.get_local_dataset_dir())
    
    if all_local_items:
        logger.info(f"Loaded {len(all_local_items)} items from local file {config.get_local_dataset_dir()}")
    
    if not all_local_items:
        logger.info("No items to push to HuggingFace")
        return
    
    try:
        # Load existing items from HuggingFace
        existing_items = {}
        try:
            logger.warning(f"Loading existing items from HuggingFace for deduplication")
            hf_dataset = load_dataset(
                config.target_dataset_name,
                config.target_subset_name,
                split=config.target_split_name
            )
            
            # Convert HF dataset to list of dicts and validate
            hf_items_raw = list(hf_dataset)
            hf_items_validated = validate_dicts_to_pydantic(hf_items_raw, MultimodalDatasetItem, skip_invalid=True)
            
            # 建立key -> item的映射
            for item in hf_items_validated:
                existing_items[get_item_key(item)] = item
            
            logger.info(f"Loaded {len(existing_items)} existing items from HuggingFace")
        except Exception as e:
            logger.error(f"Could not load pre-existing items from HuggingFace (dataset may not exist yet): {e}")
        
        # Combine local items with existing HF items, deduplicating 合并本地和HF的items，去重
        combined_items = dict(existing_items)

        for item in all_local_items:
            combined_items[get_item_key(item)] = item  # overwrite HF with local
        
        all_items = list(combined_items.values())
        
        logger.info(f"Pushing {len(all_items)} total items, {len(existing_items)} existing) to {config.target_dataset_name}/{config.target_subset_name}/{config.target_split_name}")
        
        # Convert Pydantic models to dictionaries for HF dataset - Pydantic → Dict (HF需要Dict格式)
        processed_dicts_for_hf = [prepare_item_for_upload(item, config) for item in all_items]

        # 显式定义 Features，确保 image 字段类型为 Image
        features = Features({
            'question_id': Value('string'),
            'question': Value('string'),
            'model_output': Value('string'),
            'ground_truth': Value('string'),
            'image': Image(),  # 显式指定为 Image 类型，即使值为 None
            'image_name': Value('string'),
            'task': Value('string'),
            'hallucination_annotation': HFList({
                'index': Value('int64'),
                'label': Value('string'),
                'span': Value('string'),
                'verification_note': Value('string')
            }),
            'annotator_model': Value('string'),
            'model_name': Value('string'),
            'detailed_prompt': Value('string')
        })

        hf_dataset = Dataset.from_list(processed_dicts_for_hf, features=features)
        hf_dataset.push_to_hub(
            config.target_dataset_name,
            config.target_subset_name,
            split=config.target_split_name,  
        )
        
        logger.info(f"Successfully pushed {len(all_items)} items to {config.target_dataset_name}/{config.target_subset_name}/{config.target_split_name}")
        
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
        logger.error(traceback.format_exc())


# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_processed_item_keys(
    config: PipelineConfig
) -> set:
    """加载已处理的结果（实现断点续传）
    
    从本地 Arrow 数据集和 HuggingFace Hub 两个来源加载已处理的 items
    
    Args:
        local_dataset_dir: 本地数据集目录（Arrow 格式）
        repo_id: HuggingFace 仓库 ID
        config_name: 配置名称（subset）
        split_name: 分割名称
        token: HuggingFace 访问令牌
        
    Returns:
        tuple: (已处理的 question_id 集合, 本地数据集, HF 上传结果列表)
    """
    processed_question_ids = set()
    
    # ============ 步骤1: 从本地 Arrow 数据集加载 ============
    local_items = load_processed_items_from_disk(config.get_local_dataset_dir())
    for item in local_items:
        processed_question_ids.add(item.question_id)

    if local_items:
        logger.info(f"本地数据集加载完成！其中已处理的 question_id: {len(processed_question_ids)} 个")

    # ============ 步骤2: 从 HuggingFace Hub 加载（关键！）============
    if config.hf_token:
        try:
            logger.info("正在从 HuggingFace 加载断点数据...")
            logger.info(f"仓库: {config.target_dataset_name}/{config.target_subset_name}/{config.target_split_name}")
            hf_dataset = load_dataset(
                config.target_dataset_name,
                config.target_subset_name,
                split=config.target_split_name,
            )
            # Convert HF dataset to list of dicts and validate
            hf_items_raw = list(hf_dataset)
            hf_items_validated = validate_dicts_to_pydantic(hf_items_raw, MultimodalDatasetItem, skip_invalid=True)
            
            # 遍历 HF 数据集，添加到 processed_question_ids
            for item in hf_items_validated:
                processed_question_ids.add(get_item_key(item))
            
            logger.info(f"HuggingFace 加载: {len(hf_items_validated)} 条记录")
            # logger.info(f"其中新增断点: {len(hf_items_validated) - len(processed_question_ids)} 个 (本地没有的)")
            
        except Exception as e:
            logger.info("无法从 HuggingFace 加载（数据集可能不存在或网络问题）")
            logger.info(f"错误: {str(e)[:100]}")
            logger.info("将仅使用本地断点数据")
    else:
        logger.info("未提供 HF_TOKEN，跳过从 HuggingFace 加载断点")
    
    logger.info(f"已处理的 question_id: {len(processed_question_ids)} 个")
    
    return processed_question_ids

def load_items_to_process(config: PipelineConfig) -> List[MultimodalDatasetItem]:
    dataset = load_dataset(
        path=config.source_dataset_name,
        name=config.source_subset_name,
        split=config.source_split_name,
        )

    required_columns = ["question_id", "question", "model_output"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset is missing required column: {col}. Available columns: {dataset.column_names}")

    dataset_items_raw = list(dataset)
    logger.info(f"加载数据集: {len(dataset_items_raw)} 条记录")
    dataset_items_validated = validate_dicts_to_pydantic(dataset_items_raw, MultimodalDatasetItem, skip_invalid=True)
    if len(dataset_items_validated) < len(dataset_items_raw):
        logger.warning(f"Skipped {len(dataset_items_raw) - len(dataset_items_validated)} invalid items from {config.source_dataset_name}")
    logger.info(f"加载数据集完成！其中需要处理的 question_id: {len(dataset_items_validated)} 个")

    processed_question_ids = load_processed_item_keys(config)
    items_to_process = [item for item in dataset_items_validated if not is_item_processed(item, processed_question_ids)]
    logger.info(f"需要处理的样本数量: {len(items_to_process)} 个")
    return items_to_process

# 保存样本（追加模式）
def save_item_to_disk(items: List[MultimodalDatasetItem], dataset_dir: Path):
    # 1. 加载现有数据集（如果存在）
    if dataset_dir.exists():
        try:
            existing_items = load_processed_items_from_disk(dataset_dir)
        except Exception as e:
            logger.warning(f"无法加载现有数据集: {e}，将创建新数据集")
            existing_items = None
    else:
        existing_items = None

    # items_list = [item.model_dump() for item in items]
    # 在第399行修改为：
    items_list = [item.model_dump() for item in items if item is not None]
    
    # 2. 创建新样本的数据集
    if existing_items is not None:
        existing_items_list = [item.model_dump() for item in existing_items]
        combined_items_list = existing_items_list + items_list
    else:
        combined_items_list = items_list
    combined_ds = Dataset.from_list(combined_items_list)
    combined_ds.save_to_disk(dataset_dir)
# ============================================================================
# 核心处理函数
# ============================================================================

async def process_sample(config: PipelineConfig, client: OpenAI | AsyncOpenAI, item: MultimodalDatasetItem) -> Optional[MultimodalDatasetItem]:
    if item is None:
        return None

    try:
        annotated_item = await annotate_sample(
            client=client,
            instruction=item.question,
            groundtruth=item.ground_truth,
            completion_text=item.model_output,
            image=item.image,
            model=config.annotator_model
        )
        
        # 更新item的hallucination_annotation字段
        if annotated_item is None:
            return None
        else:
            item.hallucination_annotation = annotated_item
            item.annotator_model = config.annotator_model

        # 保存中间结果，断点续传的关键，追加模式
        save_jsonl([item.model_dump(exclude={'image'})], str(config.get_local_dataset_dir())+'/hallucination_annotations.jsonl', append=True)

        if config.verbose:
            labels: List[str] = [e.label for e in item.hallucination_annotation or []]
            label_counts = {label: len([l for l in labels if l == label]) for label in set(labels)}
            logger.info(f"Successfully processed item with label counts: {label_counts}, total annotations: {len(labels)}")

        return item

    except Exception as e:
        logger.error(f"处理样本失败: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# 主函数
# ============================================================================

async def main(config: Optional[PipelineConfig] = None):
    """主函数
    
    Args:
        config: 配置对象（可选，如果为 None 则使用默认配置）
    """
    if config is None:
        config = PipelineConfig()
    
    # 验证配置
    config.validate()
    
    # 初始化 OpenAI 客户端
    if config.parallel_mode:
        client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
    else:
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
    
    # 加载数据集
    items_to_process = load_items_to_process(config)

    if getattr(config, "max_samples", 0) > 0:
        items_to_process = items_to_process[: config.max_samples]
    
    # 处理样本
    if config.parallel_mode:
        logger.info("并行模式")

        if config.push_intermediate_every > 0:
            logger.info(f"Will push intermediate results every {config.push_intermediate_every} items")
            processed_count = 0
            batch_size = config.push_intermediate_every
            for i in range(0, len(items_to_process), batch_size):
                batch_end = min(i + batch_size, len(items_to_process))
                batch = items_to_process[i:batch_end]
                logger.info(f"Processing batch {i//batch_size + 1}: items {i+1}-{batch_end}")
                # 并发执行这一批push_intermediate_every个样本的处理
                results = await tqdm_asyncio.gather(*[process_sample(config, client, item) for item in batch], desc=f"Processing batch {i//batch_size + 1}")
                
                # Count successful results in this batch
                batch_successful = sum(1 for r in results if r is not None)
                processed_count += batch_successful
                logger.info(f"Batch completed: {batch_successful}/{len(batch)} items successful")
                
                # Push intermediate results after this batch
                logger.info(f"Pushing intermediate results after {processed_count} total items...")

                save_item_to_disk(results, config.get_local_dataset_dir())
                upload_to_huggingface(config)
    else:
        logger.info("串行模式")
        count = 0
        for i, item in enumerate(items_to_process):
            logger.info(f"Processing item {i+1}/{len(items_to_process)}")
            processed_item = await process_sample(config, client, item)
            if processed_item:
                logger.info(f"Successfully processed item {i+1}")
                count += 1
                
                # Push intermediate results if enabled
                if (config.push_intermediate_every > 0 and 
                    count % config.push_intermediate_every == 0):
                    
                    logger.info(f"Pushing intermediate results after {count} items...")
                    
                    upload_to_huggingface(config)
                    
            else:
                logger.error(f"Error processing item {i+1}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态幻觉标注流程")
    parser.add_argument("--source_dataset_name", type=str, default="anonymous/TriConflict-inference",
                        help="源数据集名称")
    parser.add_argument("--source_subset_name", type=str, default="model-subset",
                        help="源数据集子集名称")
    parser.add_argument("--source_split_name", type=str, default="text_vs_prior",
                        help="源数据集类别，例如 image_vs_prior")
    parser.add_argument("--target_dataset_name", type=str, default="anonymous/TriConflict-annotations",
                        help="目标数据集名称")
    parser.add_argument("--target_subset_name", type=str, default="model-subset",
                        help="目标数据集子集名称")
    parser.add_argument("--target_split_name", type=str, default="text_vs_prior",
                        help="目标数据集类别，例如 image_vs_prior")
    parser.add_argument("--annotator_model", type=str,
                        # default="gemini-3-pro-preview",
                        default="gpt-5",
                        help="标注模型")
    parser.add_argument("--parallel_mode", type=bool, default=True,
                        help="是否并行模式")
    parser.add_argument("--push_intermediate_every", type=int, default=200,
                        help="每处理多少个样本上传一次")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="是否打印详细信息")
    parser.add_argument("--max_samples", type=int, default=0, 
                        help="只处理的样本数量（0 表示不限制）")
    args = parser.parse_args()
    
    config = PipelineConfig(
        source_dataset_name=args.source_dataset_name,
        source_subset_name=args.source_subset_name,
        source_split_name=args.source_split_name,
        target_dataset_name=args.target_dataset_name,
        target_subset_name=args.target_subset_name,
        target_split_name=args.target_split_name,
        annotator_model=args.annotator_model,
        parallel_mode=args.parallel_mode,
        push_intermediate_every=args.push_intermediate_every,
        verbose=args.verbose
    )
    config.max_samples = args.max_samples
    asyncio.run(main(config))

