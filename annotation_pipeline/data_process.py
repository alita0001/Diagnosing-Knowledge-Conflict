"""
Process LongFact-Inference dataset
- Add question_id field
- Rename user_question to question
- Upload to new HuggingFace repository
"""

from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import random

# ============================================================================
# Configuration
# ============================================================================

# Original dataset info
ORIGINAL_DATASET = "anonymous/LongFact-Inference"  # Replace with your dataset
ORIGINAL_SUBSET = "model-subset"

# New dataset info
NEW_DATASET_NAME = "anonymous/LongFact-Inference-Processed"  # Replace with your username
NEW_SUBSET_NAME = "model-subset"

# Whether to use private repository
PRIVATE_REPO = False

# ============================================================================
# Main processing functions
# ============================================================================

def add_question_id(example, idx, split_name, subset_name):
    """
    Add question_id for each sample.
    
    Format: {subset}_{split}_{index}
    Example: concepts_train_1, concepts_validation_1
    """
    question_id = f"{subset_name}_{split_name}_{idx}"
    return {
        'question_id': question_id,
        'question': example['user_question'],
        'model_output': example['model_output'],
        'model_name': example['model_name'],
        'subset': example['subset']
    }

def process_dataset():
    """Main function for processing the dataset"""
    
    print("=" * 60)
    print("Starting to process LongFact-Inference dataset")
    print("=" * 60)
    
    # Load original dataset
    print(f"\nStep 2: Loading original dataset {ORIGINAL_DATASET}")
    dataset = load_dataset(ORIGINAL_DATASET, ORIGINAL_SUBSET)
    
    # View dataset info
    print("\nOriginal dataset info:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    print(f"\nOriginal columns: {dataset['train'].column_names}")
    
    # Process each split
    print("\nStep 3: Processing dataset...")
    processed_dataset = DatasetDict()
    
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        
        # Add question_id and rename fields for each sample
        processed_split = split_data.map(
            lambda example, idx: add_question_id(
                example, 
                idx, 
                split_name, 
                example['subset']
            ),
            with_indices=True,
            remove_columns=['user_question']
        )
        
        processed_dataset[split_name] = processed_split
        print(f"  Done! Processed {len(processed_split)} samples")
    
    # Validate processing results
    print("\nStep 4: Validating processing results")
    print("\nNew dataset info:")
    for split_name, split_data in processed_dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    print(f"\nNew columns: {processed_dataset['train'].column_names}")
    
    # Show example
    print("\nExample data (first item of train split):")
    example = processed_dataset['train'][0]
    print(f"  question_id: {example['question_id']}")
    print(f"  question: {example['question'][:100]}...")
    print(f"  model_name: {example['model_name']}")
    print(f"  subset: {example['subset']}")
    print(f"  model_output length: {len(example['model_output'])} characters")
    
    # Upload to HuggingFace
    print(f"\nStep 5: Uploading to HuggingFace...")
    print(f"Target repository: {NEW_DATASET_NAME}")
    print(f"Config name: {NEW_SUBSET_NAME}")
    
    try:
        processed_dataset.push_to_hub(
            NEW_DATASET_NAME,
            config_name=NEW_SUBSET_NAME,
            private=PRIVATE_REPO
        )
        print("\nDataset uploaded successfully!")
        print(f"Access at: https://huggingface.co/datasets/{NEW_DATASET_NAME}")
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("Tip: Please ensure your HuggingFace Token has write permissions")
        return None
    
    return processed_dataset

# ============================================================================
# Extra feature: Save to local
# ============================================================================

def save_to_local(dataset, output_dir="./processed_longfact_dataset"):
    """
    Optional: Save the processed dataset to local
    """
    print(f"\nSaving dataset to local: {output_dir}")
    dataset.save_to_disk(output_dir)
    print(f"Saved to {output_dir}")
    
    # Also save as JSONL format
    for split_name, split_data in dataset.items():
        jsonl_path = f"{output_dir}/{split_name}.jsonl"
        split_data.to_json(jsonl_path)
        print(f"Saved {split_name} to {jsonl_path}")

# ============================================================================
# Extra feature: Dataset statistics
# ============================================================================

def print_statistics(dataset):
    """
    Print dataset statistics
    """
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  Total samples: {len(split_data)}")
        
        # Count samples per subset
        subsets = {}
        for example in split_data:
            subset = example['subset']
            subsets[subset] = subsets.get(subset, 0) + 1
        
        print(f"  Subset distribution:")
        for subset, count in subsets.items():
            print(f"    {subset}: {count} samples")
        
        # Question length statistics
        question_lengths = [len(example['question']) for example in split_data]
        avg_length = sum(question_lengths) / len(question_lengths)
        print(f"  Average question length: {avg_length:.0f} characters")
        print(f"  Shortest question: {min(question_lengths)} characters")
        print(f"  Longest question: {max(question_lengths)} characters")

# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    # Process dataset
    processed_dataset = process_dataset()
    
    if processed_dataset is not None:
        # Print statistics
        print_statistics(processed_dataset)
        
        # Optional: Save to local
        save_choice = input("\nSave dataset to local? (y/n): ")
        if save_choice.lower() == 'y':
            save_to_local(processed_dataset)
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"\nYou can load the new dataset with:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{NEW_DATASET_NAME}', '{NEW_SUBSET_NAME}')")
        print(f"```")
