#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import sys
import time
import hashlib
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import traceback 
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset, Dataset, concatenate_datasets, Image, Features, Value, List as HFList
import asyncio
from tqdm.asyncio import tqdm_asyncio
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))  
from utils.file_utils import load_jsonl,save_jsonl
from utils.parsing import validate_dicts_to_pydantic 
from prompt_writer import annotate_sample
from data_types import MultimodalDatasetItem
from model_inference import generate_model_output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,  
    handlers=[
        logging.StreamHandler(sys.stdout)  
    ]
)
logger = logging.getLogger(__name__)

logger.info("✅ Logging configured")

# combined_dataset: Dataset = None

# ============================================================================
# Config
# ============================================================================

@dataclass
class PipelineConfig:
    
    # ===== API Config =====
    openrouter_api_key: str = None  # Set via environment variable OPENROUTER_API_KEY
    openrouter_base_url: str = "https://api.openai.com/v1"  # Or your preferred API endpoint
    annotator_model: str = "gpt-4"
    
    # ===== HuggingFace Config =====
    hf_repo_id: str = "anonymous/prompt-writer"  # Replace with your HuggingFace repo
    hf_token: str = None  # Set via environment variable HF_TOKEN
    hf_config_name: str = "Ocean_R1_7B_Instruct"  # Subset Ocean_R1_7B_Instruct
    enable_hf_upload: bool = True
    upload_interval: int = 10  
    
    # ===== Dataset Config =====
    dataset_class: str = "phd_icc"  # Options: phd_ccs, phd_sec, phd_icc, phd_base
    source_split_name: str = "phd_icc"
    source_dataset_name: str = "anonymous/datasets"  # Replace with your dataset
    cache_dir: str = "./cache"  # Local cache directory
    model_output_dir: Path = Path("./results/rewrite_results")
    output_dir: Path = Path("./results/question_rewrite")
    save_path:Optional[Path] = None

    
    verbose: bool = True
    parallel_mode: bool = False
    # ===== Processing Config =====
    num_process: int =50  # Number of processes to use
    push_intermediate_every: int = 10
    # ===== Inference Config =====
    inference_model_id: str = "path/to/your/model"  # Model to generate new model_output
    inference_device: str = "cuda"  # Inference device
    regenerate_model_output: bool = True  # Whether to regenerate model_output
    inference_temperature: float = 0.9  # Temperature parameter
    inference_top_p: float = 0.95  # nucleus sampling
    inference_do_sample: bool = True  # Whether to enable sampling


    def __post_init__(self):
        """Initialize after processing"""
        # Get API key from environment variable
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
        
        if self.save_path is None:
            self.save_path = self.output_dir / f"{self.dataset_class}_question_rewrite.jsonl"
 
        # Ensure paths are Path objects
        if isinstance(self.model_output_dir, str):
            self.model_output_dir = Path(self.model_output_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_split_name(self) -> str:
        """Get the split name of the current dataset"""
        return self.dataset_class
    
    def get_local_dataset_dir(self) -> Path:
        """Get the local dataset directory (Arrow format, containing images)"""
        # dataset_dir = self.output_dir / f"{self.dataset_class}_dataset"
        # dataset_dir.mkdir(parents=True, exist_ok=True)
        return Path(self.output_dir / self.hf_repo_id / self.hf_config_name / (self.dataset_class + '_annotations'))
    
    def validate(self):
        """Validate necessary configuration"""
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set\n"
                "Please run: export OPENROUTER_API_KEY='sk-or-v1-...'"
            )
        
        if self.enable_hf_upload and not self.hf_token:
            logger.warning("HuggingFace upload enabled but HF_WRITE_TOKEN or HF_TOKEN not set")
            logger.warning("Upload functionality will be disabled")
            self.enable_hf_upload = False


# ============================================================================
# Tools
# ============================================================================

def get_item_key(item: MultimodalDatasetItem) -> str:
    """Generate a unique key for the item (for deduplication)
    
    Use the MD5 hash of question_id as the unique identifier
    
    Args:
        item: A dictionary containing question_id
        
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
    source_dict = item.model_dump()
    
    # Only keep fields defined in features and handle default values
    return {
        'question_id': source_dict.get('question_id'),
        'question': str(source_dict.get('question', '')),
        'model_output': source_dict.get('model_output', ''),
        'ground_truth': source_dict.get('ground_truth') or "",
        'image': source_dict.get('image'),
        'image_name': source_dict.get('image_name') or "",
        'task': source_dict.get('task') or ""
    }

# Upload to HuggingFace
def upload_to_huggingface(
    config: PipelineConfig
) -> None:
    # Always load all processed items from local file to ensure nothing is missed
    all_local_items = load_processed_items_from_disk(config.get_local_dataset_dir())
    
    if all_local_items:
        logger.info(f"Loaded {len(all_local_items)} items from local file {config.save_path}")
    
    if not all_local_items:
        logger.info("No items to push to HuggingFace")
        return
    
    try:
        # Load existing items from HuggingFace
        existing_items = {}
        try:
            logger.warning(f"Loading existing items from HuggingFace for deduplication")
            hf_dataset = load_dataset(
                config.hf_repo_id,
                config.hf_config_name,
                split=config.get_split_name()+'_annotations'
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
        
        logger.info(f"Pushing {len(all_items)} total items, {len(existing_items)} existing) to {config.hf_repo_id}/{config.hf_config_name}/{config.get_split_name()}")
        
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
        })

        hf_dataset = Dataset.from_list(processed_dicts_for_hf, features=features)
        hf_dataset.push_to_hub(
            config.hf_repo_id,
            config.hf_config_name,
            split=config.get_split_name()+'_annotations',  
        )
        
        logger.info(f"Successfully pushed {len(all_items)} items to {config.hf_repo_id}/{config.hf_config_name}/{config.get_split_name()}")
        
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
        logger.error(traceback.format_exc())


# ============================================================================
# Data Loading and Preprocessing
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
        cache_dir: Cache directory
        
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
            logger.info(f"Repository: {config.hf_repo_id}/{config.hf_config_name}/{config.get_split_name()}")
            hf_dataset = load_dataset(
                config.hf_repo_id,
                config.hf_config_name,
                cache_dir=config.cache_dir,
                split=config.get_split_name()+'_annotations',
            )
            logger.info(f"Loaded {len(hf_dataset)} records")
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
        name=config.hf_config_name,
        cache_dir=config.cache_dir,
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
        # print(f"1-Processing item: {item.question}")
        annotated_item = await annotate_sample(
            client=client,
            instruction=item.question,
            image=item.image,
            model=config.annotator_model
        )
        
        # Update item's hallucination_annotation field
        if annotated_item is None:
            return None
        else:
            item.question = annotated_item
            # item.annotator_model = config.annotator_model
        # print(f"---------------------------------Processing item: {item.question}")

        # Regenerate model_output (use new question and image) chj
        if config.regenerate_model_output and item.image is not None:
            try:
                # Extract question text (build complete question, including conflicting_context)
                question_text = None
                if isinstance(item.question, str):
                    question_text = item.question
                elif isinstance(item.question, list) and len(item.question) > 0:
                    # If MultimodalPromptSpan list, build complete question
                    first_span = item.question[0]
                    if hasattr(first_span, 'question') and hasattr(first_span, 'conflicting_context'):
                        # Build complete question: conflicting_context + question
                        question_text = f"{first_span.conflicting_context}\n\n{first_span.question}"
                    elif isinstance(first_span, dict):
                        conflicting_context = first_span.get('conflicting_context', '')
                        question = first_span.get('question', '')
                        question_text = f"{conflicting_context}\n\n{question}" if conflicting_context else question
                    else:
                        question_text = str(first_span)
                elif hasattr(item.question, 'question') and hasattr(item.question, 'conflicting_context'):
                    # If single MultimodalPromptSpan object
                    question_text = f"{item.question.conflicting_context}\n\n{item.question.question}"
                else:
                    question_text = str(item.question)
                
                if question_text:
                    print(f"---------------------------------Question text: {question_text}")
                    logger.info(f"🔄 Using inference model to generate new model_output...")
                    new_model_output = generate_model_output(
                        image=item.image,
                        question=question_text,
                        model_id=config.inference_model_id,
                        device=config.inference_device,
                        temperature=config.inference_temperature,
                        top_p=config.inference_top_p,
                        do_sample=config.inference_do_sample
                    )
                    item.model_output = new_model_output
                    print(f"---------------------------------New model_output: {new_model_output}")
                    logger.info(f"✅ New model_output generated successfully (Length: {len(new_model_output)} chars)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate new model_output, using original value: {e}")
                # If generation fails, keep original model_output

        # Save intermediate results, key for resume from breakpoint, append mode
        save_jsonl([item.model_dump(exclude={'image'})], str(config.get_local_dataset_dir())+'/question_rewrite.jsonl', append=True)

        # if config.verbose:
        #     labels: List[str] = [e.label for e in item.hallucination_annotation or []]
        #     label_counts = {label: len([l for l in labels if l == label]) for label in set(labels)}
        #     logger.info(f"Successfully processed item with label counts: {label_counts}, total annotations: {len(labels)}")

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
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
        )
    else:
        client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
        )
    
    # Load dataset
    items_to_process = load_items_to_process(config)
    
    # Limit processing count (if num_process is set)
    if config.num_process > 0:
        original_count = len(items_to_process)
        items_to_process = items_to_process[:config.num_process]
        logger.info(f"Limit processing count: {original_count} -> {len(items_to_process)} samples")
    
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
                results = await tqdm_asyncio.gather(*[process_sample(config, client, item) for item in batch], desc=f"Processing batch {i//batch_size + 1}")
                
                # Count successful results in this batch
                batch_successful = sum(1 for r in results if r is not None)
                processed_count += batch_successful
                logger.info(f"Batch completed: {batch_successful}/{len(batch)} items successful")
                
                # Push intermediate results after this batch
                logger.info(f"Pushing intermediate results after {processed_count} total items...")

                save_item_to_disk(results, config.get_local_dataset_dir())
                upload_to_huggingface(config)
        # results = await asyncio.gather(*[process_sample(config, client, item) for item in items_to_process])
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
    parser.add_argument("--dataset_class", type=str, default="phd_icc",
                        choices=["phd_ccs", "phd_sec", "phd_icc", "phd_base", "longfact"],
                        help="Dataset class")
    parser.add_argument("--num_process", type=int, default=3,
                        help="Number of samples to process")
    parser.add_argument("--upload_interval", type=int, default=10,
                        help="Upload once every N samples processed")
    parser.add_argument("--annotator_model", type=str,
                        # default="openai/gpt-5:online",
                        # default="anthropic/claude-sonnet-4.5",
                        default="gpt-5",
                        # default="claude-sonnet-4-5-20250929",
                        help="Annotation model")
    parser.add_argument("--openrouter_base_url", type=str, 
                        default="https://api.openai.com/v1",
                        help="OpenAI API Endpoint")
    parser.add_argument("--openrouter_api_key", type=str, 
                        default=None,
                        help="OpenAI API Key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--parallel_mode", type=bool, default=True,
                        help="Whether to use parallel mode")
    parser.add_argument("--push_intermediate_every", type=int, default=5,
                        help="Upload once every N samples processed")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Whether to print detailed information")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        dataset_class=args.dataset_class,
        num_process=args.num_process,
        upload_interval=args.upload_interval,
        annotator_model=args.annotator_model,
        parallel_mode=args.parallel_mode,
        push_intermediate_every=args.push_intermediate_every,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_api_key=args.openrouter_api_key,
        verbose=args.verbose
    )
    
    asyncio.run(main(config))

