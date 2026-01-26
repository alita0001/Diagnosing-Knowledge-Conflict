import argparse
import torch
import sys
import os
import time
from datasets import load_dataset, Dataset

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .mitigation import HallucinationMitigator

def main():
    parser = argparse.ArgumentParser(description="Run Hallucination Mitigation Demo")
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="anonymous/TriConflict-annotations", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="model-subset", help="Dataset configuration/subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process")

    parser.add_argument("--model_name", type=str, default="path/to/your/model", help="Path or name of the base LLM")
    parser.add_argument("--probe_path", type=str, default="./value_head_probes/your-probe-id", help="Path to the trained ValueHeadProbe")
    parser.add_argument("--prompt", type=str, default="Your input prompt here", help="Input prompt")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image for VLM")
    parser.add_argument("--method", type=str, choices=["token_rejection", "sentence_rejection", "none"], default="token_rejection", help="Mitigation method")
    parser.add_argument("--alpha", type=float, default=0.0, help="Weight for Truthfulness in token_rejection (0.0 = LM only, 1.0 = Truth only)")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates/sequences")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    print(f"Initializing Mitigator on {args.device}...")
    mitigator = HallucinationMitigator(
        model_name=args.model_name,
        probe_path=args.probe_path,
        device=args.device
    )

    # Load Image if provided
    image = None
    if args.image_path:
        from PIL import Image
        print(f"Loading image from: {args.image_path}")
        image = Image.open(args.image_path)
    
    print(f"\nPrompt: {args.prompt}")
    if image:
        print(f"Image Path: {args.image_path}")
    print("-" * 50)
    
    # 1. Generate without mitigation (Baseline) - unless method is 'none' which means ONLY baseline logic if we implemented it that way, 
    # but here we'll just run the requested method vs baseline for comparison.
    
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    sample = dataset[0]
    prompt = sample["detailed_prompt"]
    print(f"Question:{prompt}")
    image = sample.get("image", None)
    # return
    print("Generating BASELINE (No Mitigation)...")
    start_time = time.time()
    baseline_output = mitigator.generate(
        prompt,
        image=image,
        method="none",
        max_new_tokens=1024
    )
    end_time = time.time()
    print(f"Baseline Generation Time: {end_time - start_time:.2f} seconds")
    print(f"Baseline Output:\n{baseline_output}")
    print("-" * 50)

    if args.method != "none":
        print(f"Generating with MITIGATION ({args.method})...")
        start_time = time.time()
        mitigated_output = mitigator.generate( 
            args.prompt,
            image=image,
            method=args.method,
            alpha=args.alpha,
            num_candidates=args.num_candidates,
            max_new_tokens=1024
        )
        end_time = time.time()
        print(f"Mitigation Time: {end_time - start_time:.2f} seconds")
        print(f"Mitigated Output:\n{mitigated_output}")
        print("-" * 50)
        
        # Optional: Automated comparison of hallucination scores
        # We can implement a quick score check here if desired.
        
if __name__ == "__main__":
    main()
