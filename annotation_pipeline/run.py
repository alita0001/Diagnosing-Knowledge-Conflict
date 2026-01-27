#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Hallucination Annotation Pipeline Main Script

Features:
1. Use Claude-4.5 model to detect and annotate hallucinations in VQA model outputs
2. Support resume from breakpoint to avoid duplicate processing
3. Real-time local result saving (Arrow format, including images)
4. Periodic upload to HuggingFace Hub
5. Support processing of multiple dataset classes

Usage:
    1. Set environment variables:
       export API_KEY='your-api-key-here'
       export HF_WRITE_TOKEN='your-hf-token-here'
       export HF_TOKEN='your-hf-token-here'
    
    
    2. Modify config or use command line arguments:
       CUDA_VISIBLE_DEVICES=1 python -m annotation_pipeline.run --source_split_name text_vs_prior --push_intermediate_every 50
       CUDA_VISIBLE_DEVICES=4 python -m annotation_pipeline.run --source_split_name train -- --push_intermediate_every 50
    
    3. Run script:
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
import traceback # Used for printing exception stack traces
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset, Dataset, Image, Features, Value, List as HFList
import asyncio
from tqdm.asyncio import tqdm_asyncio
sys.path.append(str(Path(__file__).parent.parent))
from utils.file_utils import load_jsonl,save_jsonl
from utils.parsing import validate_dicts_to_pydantic # Data validation
from .annotate import annotate_sample
from .data_models import MultimodalDatasetItem, MultimodalAnnotatedSpan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,  # Python 3.8+ Force reconfiguration
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly specify output to stdout
    ]
)
logger = logging.getLogger(__name__)

# Test logging
logger.info("✅ Logging configuration complete")

# combined_dataset: Dataset = None

# ============================================================================
# Configuration class
# ============================================================================

@dataclass
class PipelineConfig:
    """Multimodal Annotation Pipeline Configuration
    
    Centralized management of all configuration parameters
    """
    
    # ===== API Configuration =====
    api_key: str = None  # Set via environment variable API_KEY
    base_url: str = "https://api.openai.com/v1"  # Or your preferred API endpoint
    annotator_model: str = "gpt-4"
    
    # ===== Source Dataset Configuration =====
    source_dataset_name: str = "anonymous/TriConflict-inference"  # Replace with your dataset
    source_subset_name: str = "model-subset"  # Subset
    source_split_name: str = "text_vs_prior"

    # ===== Target Dataset Configuration =====
    target_dataset_name: str = "anonymous/TriConflict-annotations"  # Replace with your dataset
    target_subset_name: str = "model-subset"  # Subset
    target_split_name: str = "text_vs_prior"

    hf_token: str = None  # Set via environment variable HF_TOKEN
    output_dir: Path = Path("./hallucination_annotations")
    enable_hf_upload: bool = True
    

    verbose: bool = True
    parallel_mode: bool = False
    # ===== Processing Configuration =====
    push_intermediate_every: int = 10
    def __post_init__(self):
        """Post-initialization processing"""
        # Get API key from environment variables
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")
        
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
 
        # Ensure path is a Path object
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_local_dataset_dir(self) -> Path:
        """Get local dataset directory (Arrow format, including images)"""
        return Path(self.output_dir / self.target_dataset_name / self.target_subset_name / (self.target_split_name))
    
    def validate(self):
        """Validate if necessary configurations are set"""
        if not self.api_key:
            raise ValueError(
                "API_KEY environment variable not set\n"
                "Please run: export API_KEY='sk-or-v1-...'"
            )
        
        if self.enable_hf_upload and not self.hf_token:
            logger.warning("HuggingFace upload enabled but HF_WRITE_TOKEN or HF_TOKEN not set")
            logger.warning("Upload function will be disabled")
            self.enable_hf_upload = False


# ============================================================================
# Utility functions
# ============================================================================

def get_item_key(item: MultimodalDatasetItem) -> str:
    """Generate unique key for item (for deduplication)
    
    Use MD5 hash of question_id as unique identifier
    
    Args:
        item: Dictionary containing question_id
        
    Returns:
        MD5 hash string
    """
    key_str = item.question_id
    return hashlib.md5(key_str.encode()).hexdigest()

def is_item_processed(item: MultimodalDatasetItem, processed_keys: set) -> bool:
    """Check if an item has already been processed"""
    return get_item_key(item) in processed_keys

def load_processed_items_from_disk(local_dataset_dir: Path) -> List[MultimodalDatasetItem]:
    """Load processed results from local Arrow dataset"""
    if not local_dataset_dir.exists():
        return []
    
    try:
        dataset = Dataset.load_from_disk(local_dataset_dir)
    except Exception as e:
        logger.error(f"Failed to load processed results from local dataset: {e}")
        return []

    # Convert to List[Dict]
    raw_items = list(dataset)
    # Validate and convert to Pydantic objects
    validated_items = validate_dicts_to_pydantic(raw_items, MultimodalDatasetItem, skip_invalid=True)

    # Log if some items failed validation
    if len(validated_items) < len(raw_items):
        logger.warning(f"Skipped {len(raw_items) - len(validated_items)} invalid items from {local_dataset_dir}")

    return validated_items

# ============================================================================
# HuggingFace upload related functions
# ============================================================================
# Unify data format before upload
def prepare_item_for_upload(item: MultimodalDatasetItem, config: PipelineConfig) -> dict:
    """
    Prepare data for upload, ensuring fields match the format on Hub
    Handle None values and missing fields to meet HuggingFace dataset type requirements
    """
    item_dict = item.model_dump()
    
    # Use get() + or to replace None and missing fields
    # This ensures type is string instead of null
    item_dict['ground_truth'] = item_dict.get('ground_truth') or ""
    item_dict['image_name'] = item_dict.get('image_name') or ""
    item_dict['task'] = item_dict.get('task') or ""
    item_dict['detailed_prompt'] = item_dict.get('detailed_prompt') or item_dict.get('question') or ""
    item_dict['model_name'] = item_dict.get('model_name') or config.target_subset_name or ""
    # Special handling for image field (Image type allows None)
    if 'image' not in item_dict:
        item_dict['image'] = None
    
    # Remove incompatible fields (if any)
    item_dict.pop('subset', None)
    
    return item_dict

# Upload to HuggingFace
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
            
            # Build key -> item mapping
            for item in hf_items_validated:
                existing_items[get_item_key(item)] = item
            
            logger.info(f"Loaded {len(existing_items)} existing items from HuggingFace")
        except Exception as e:
            logger.error(f"Could not load pre-existing items from HuggingFace (dataset may not exist yet): {e}")
        
        # Combine local items with existing HF items, deduplicating
        combined_items = dict(existing_items)

        for item in all_local_items:
            combined_items[get_item_key(item)] = item  # overwrite HF with local
        
        all_items = list(combined_items.values())
        
        logger.info(f"Pushing {len(all_items)} total items, {len(existing_items)} existing) to {config.target_dataset_name}/{config.target_subset_name}/{config.target_split_name}")
        
        # Convert Pydantic models to dictionaries for HF dataset - Pydantic -> Dict (HF requires Dict format)
        processed_dicts_for_hf = [prepare_item_for_upload(item, config) for item in all_items]

        # Explicitly define Features, ensuring image field is Image type
        features = Features({
            'question_id': Value('string'),
            'question': Value('string'),
            'model_output': Value('string'),
            'ground_truth': Value('string'),
            'image': Image(),  # Explicitly specify as Image type, even if value is None
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
# Data loading and preprocessing
# ============================================================================

def load_processed_item_keys(
    config: PipelineConfig
) -> set:
    """Load processed results (implement resume from breakpoint)
    
    Load processed items from both local Arrow dataset and HuggingFace Hub
    
    Args:
        local_dataset_dir: Local dataset directory (Arrow format)
        repo_id: HuggingFace repository ID
        config_name: Configuration name (subset)
        split_name: Split name
        token: HuggingFace access token
        
    Returns:
        tuple: (Set of processed question_ids, Local dataset, HF upload result list)
    """
    processed_question_ids = set()
    
    # ============ Step 1: Load from local Arrow dataset ============
    local_items = load_processed_items_from_disk(config.get_local_dataset_dir())
    for item in local_items:
        processed_question_ids.add(item.question_id)

    if local_items:
        logger.info(f"Local dataset loaded! Processed question_ids: {len(processed_question_ids)}")

    # ============ Step 2: Load from HuggingFace Hub (Critical!) ============
    if config.hf_token:
        try:
            logger.info("Loading breakpoint data from HuggingFace...")
            logger.info(f"Repository: {config.target_dataset_name}/{config.target_subset_name}/{config.target_split_name}")
            hf_dataset = load_dataset(
                config.target_dataset_name,
                config.target_subset_name,
                split=config.target_split_name,
            )
            # Convert HF dataset to list of dicts and validate
            hf_items_raw = list(hf_dataset)
            hf_items_validated = validate_dicts_to_pydantic(hf_items_raw, MultimodalDatasetItem, skip_invalid=True)
            
            # Iterate through HF dataset, add to processed_question_ids
            for item in hf_items_validated:
                processed_question_ids.add(get_item_key(item))
            
            logger.info(f"HuggingFace loaded: {len(hf_items_validated)} records")
            # logger.info(f"New breakpoints: {len(hf_items_validated) - len(processed_question_ids)} (not in local)")
            
        except Exception as e:
            logger.info("Cannot load from HuggingFace (dataset may not exist or network issue)")
            logger.info(f"Error: {str(e)[:100]}")
            logger.info("Will only use local breakpoint data")
    else:
        logger.info("HF_TOKEN not provided, skipping breakpoint loading from HuggingFace")
    
    logger.info(f"Processed question_ids: {len(processed_question_ids)}")
    
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
    logger.info(f"Loading dataset: {len(dataset_items_raw)} records")
    dataset_items_validated = validate_dicts_to_pydantic(dataset_items_raw, MultimodalDatasetItem, skip_invalid=True)
    if len(dataset_items_validated) < len(dataset_items_raw):
        logger.warning(f"Skipped {len(dataset_items_raw) - len(dataset_items_validated)} invalid items from {config.source_dataset_name}")
    logger.info(f"Dataset loading complete! Question_ids to process: {len(dataset_items_validated)}")

    processed_question_ids = load_processed_item_keys(config)
    items_to_process = [item for item in dataset_items_validated if not is_item_processed(item, processed_question_ids)]
    logger.info(f"Number of samples to process: {len(items_to_process)}")
    return items_to_process

# Save samples (append mode)
def save_item_to_disk(items: List[MultimodalDatasetItem], dataset_dir: Path):
    # 1. Load existing dataset (if exists)
    if dataset_dir.exists():
        try:
            existing_items = load_processed_items_from_disk(dataset_dir)
        except Exception as e:
            logger.warning(f"Cannot load existing dataset: {e}, creating new dataset")
            existing_items = None
    else:
        existing_items = None

    # items_list = [item.model_dump() for item in items]
    # Modified in line 399:
    items_list = [item.model_dump() for item in items if item is not None]
    
    # 2. Create dataset for new samples
    if existing_items is not None:
        existing_items_list = [item.model_dump() for item in existing_items]
        combined_items_list = existing_items_list + items_list
    else:
        combined_items_list = items_list
    combined_ds = Dataset.from_list(combined_items_list)
    combined_ds.save_to_disk(dataset_dir)
# ============================================================================
# Core Processing Function
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
        
        # Update item's hallucination_annotation field
        if annotated_item is None:
            return None
        else:
            item.hallucination_annotation = annotated_item
            item.annotator_model = config.annotator_model

        # Save intermediate results, key for resume from breakpoint, append mode
        save_jsonl([item.model_dump(exclude={'image'})], str(config.get_local_dataset_dir())+'/hallucination_annotations.jsonl', append=True)

        if config.verbose:
            labels: List[str] = [e.label for e in item.hallucination_annotation or []]
            label_counts = {label: len([l for l in labels if l == label]) for label in set(labels)}
            logger.info(f"Successfully processed item with label counts: {label_counts}, total annotations: {len(labels)}")

        return item

    except Exception as e:
        logger.error(f"Failed to process sample: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# Main function
# ============================================================================

async def main(config: Optional[PipelineConfig] = None):
    """Main function
    
    Args:
        config: Configuration object (optional, uses default configuration if None)
    """
    if config is None:
        config = PipelineConfig()
    
    # Verify configuration
    config.validate()
    
    # Initialize OpenAI Client
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
    
    # Load dataset
    items_to_process = load_items_to_process(config)

    if getattr(config, "max_samples", 0) > 0:
        items_to_process = items_to_process[: config.max_samples]
    
    # Process samples
    if config.parallel_mode:
        logger.info("Parallel mode")

        if config.push_intermediate_every > 0:
            logger.info(f"Will push intermediate results every {config.push_intermediate_every} items")
            processed_count = 0
            batch_size = config.push_intermediate_every
            for i in range(0, len(items_to_process), batch_size):
                batch_end = min(i + batch_size, len(items_to_process))
                batch = items_to_process[i:batch_end]
                logger.info(f"Processing batch {i//batch_size + 1}: items {i+1}-{batch_end}")
                # Concurrently execute processing for this batch
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
        logger.info("Serial mode")
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
    
    parser = argparse.ArgumentParser(description="Multimodal Hallucination Annotation Pipeline")
    parser.add_argument("--source_dataset_name", type=str, default="anonymous/TriConflict-inference",
                        help="Source dataset name")
    parser.add_argument("--source_subset_name", type=str, default="model-subset",
                        help="Source dataset subset name")
    parser.add_argument("--source_split_name", type=str, default="text_vs_prior",
                        help="Source dataset split, e.g. image_vs_prior")
    parser.add_argument("--target_dataset_name", type=str, default="anonymous/TriConflict-annotations",
                        help="Target dataset name")
    parser.add_argument("--target_subset_name", type=str, default="model-subset",
                        help="Target dataset subset name")
    parser.add_argument("--target_split_name", type=str, default="text_vs_prior",
                        help="Target dataset split, e.g. image_vs_prior")
    parser.add_argument("--annotator_model", type=str,
                        # default="gemini-3-pro-preview",
                        default="gpt-5",
                        help="Annotation model")
    parser.add_argument("--parallel_mode", type=bool, default=True,
                        help="Whether to use parallel mode")
    parser.add_argument("--push_intermediate_every", type=int, default=200,
                        help="Upload once every N samples processed")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Whether to print detailed information")
    parser.add_argument("--max_samples", type=int, default=0, 
                        help="Only process this many samples (0 means no limit)")
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

