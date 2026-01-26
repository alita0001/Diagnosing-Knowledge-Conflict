import argparse
import torch
import sys
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mitigation.mitigation1 import HallucinationMitigator

def main():
    parser = argparse.ArgumentParser(description="Run Batch Hallucination Mitigation")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="anonymous/TriConflict-annotations", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="model-subset", help="Dataset configuration/subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process")
    parser.add_argument("--output_repo", type=str, default="anonymous/TriConflict-Mitigation", help="Output HuggingFace repository")
    parser.add_argument("--output_split", type=str, default="probe", help="Output split name")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="path/to/your/model", help="Path or name of the base LLM")
    parser.add_argument("--probe_path", type=str, default="./value_head_probes/your-probe-id", help="Path to the trained ValueHeadProbe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Mitigation arguments
    parser.add_argument("--method", type=str, default="token_rejection", help="Mitigation method")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for Truthfulness")
    parser.add_argument("--num_candidates", type=int, default=3, help="Number of candidates")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    
    # Run arguments
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--push_to_hub", action="store_true", help="Push results to Hugging Face Hub")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config}) split='{args.split}'")
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.limit:
        print(f"Limiting to first {args.limit} samples.")
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    print(f"Initializing Mitigator on {args.device}...")
    mitigator = HallucinationMitigator(
        model_name=args.model_name,
        probe_path=args.probe_path,
        device=args.device
    )

    results = []
    
    print("Starting batch inference...")
    
    # Create output filename
    os.makedirs("results", exist_ok=True)
    output_jsonl = f"results/mitigation_results_{args.dataset_config}_{args.split}.jsonl"
    
    import json
    
    # if valid jsonl exists, read it to skip processed
    processed_count = 0
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                processed_count += 1
        print(f"Found {processed_count} existing results in {output_jsonl}. They will be preserved (append mode).")

    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if i < processed_count:
                continue

            prompt = sample.get("detailed_prompt", "")
            # print(f"prompt:{prompt}")
            # Try to find image column
            image = None
            if "image" in sample and sample["image"] is not None:
                image = sample["image"]
            
            try:
                mitigated_output = mitigator.generate(
                    prompt=prompt,
                    image=image,
                    method=args.method,
                    alpha=args.alpha,
                    num_candidates=args.num_candidates,
                    max_new_tokens=args.max_new_tokens
                )
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                mitigated_output = "<ERROR>"
            
            # Prepare result dict (exclude image)
            result_item = {k: v for k, v in sample.items() if k != "image"}
            result_item["mitigated_response"] = mitigated_output
            result_item["method"] = args.method
            result_item["alpha"] = args.alpha
            
            # Append to list for final Dataset creation (optional if memory is concern, but consistent with original plan)
            results.append(result_item)
            
            # Write to JSONL immediately
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()

    # Convert results to Dataset from the complete list (or reload from jsonl if we want to be safe)
    # Re-loading from JSONL is safer if we want to ensure consistency with what's on disk
    print("Creating output dataset from collected results...")


    # Convert results to Dataset
    # Note: If 'image' contains PIL objects, Dataset.from_list handles them if we specify features or let it infer.
    print("Creating output dataset...")
    # To ensure image columns are handled correctly if they exist:
    from datasets import Features, Value, Image as ImageFeature
    
    # Auto-infer features might fail if we have mixed types or PIL images
    # Check if we have images
    has_images = any("image" in r for r in results)
    
    try:
        output_dataset = Dataset.from_list(results)
    except Exception as e:
        print(f"Error creating dataset from list: {e}. Trying to force features...")
        # Fallback logic could be complex, hoping auto-inference works for now.
        raise e

    print(f"Processed {len(output_dataset)} samples.")

    if args.push_to_hub:
        print(f"Pushing to {args.output_repo}, split='{args.output_split}'...")
        output_dataset.push_to_hub(args.output_repo, config_name=args.dataset_config, split=args.output_split)
        print("Push complete!")
    else:
        print("Dry run complete. Use --push_to_hub to upload.")
        # Optionally save locally
        output_dataset.save_to_disk("mitigation_results_local")
        print("Saved locally to 'mitigation_results_local'")

if __name__ == "__main__":
    main()
