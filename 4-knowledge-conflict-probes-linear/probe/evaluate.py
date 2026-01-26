"""Evaluation script for hallucination detection probe."""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import argparse
from dotenv import load_dotenv

from utils.file_utils import save_jsonl, load_yaml
from utils.metrics import compute_clf_metrics, plot_roc_curves, print_eval_metrics
from utils.model_utils import load_model_and_tokenizer
    
from .dataset import (
    TokenizedProbingDataset,
    create_probing_dataset,
    tokenized_probing_collate_fn
)
from .config import EvaluationConfig
from .loss import compute_probe_ce_loss
from .value_head_probe import ValueHeadProbe, setup_probe


@torch.no_grad()
def evaluate_probe(
    probe: ValueHeadProbe,
    eval_dataloader: DataLoader,
    metric_key_prefix: Optional[str] = None,
    threshold: float = 0.7,
    verbose: bool = True,
    save_roc_curves: bool = True,
    save_dir: Optional[Path] = None,
    dump_raw_results: bool = False,
) -> Dict[str, float]:
    """
    Modified
    Evaluate probe performance on dataset (4-class version).
    Args:
        probe: The probe to evaluate
        eval_dataloader: DataLoader for evaluation data
        metric_key_prefix: Prefix for metric keys
        threshold: Threshold for negative class prediction
        verbose: Whether to print metrics
        save_roc_curves: Whether to save ROC curve plots
        save_dir: Directory to save results
        dump_raw_results: Whether to save raw prediction results
    Returns:
        Dictionary of evaluation metrics
    """
    # Force garbage collection before evaluation
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize metric collections for different aggregation levels
    all_probs = {'all': [], 'span': [], 'span_max': []}
    all_preds = {'all': [], 'span': [], 'span_max': []}
    all_labels = {'all': [], 'span': [], 'span_max': []}

    # Sample-level data collection
    sample_labels = []  # Sample ground truth labels (conflict_type)
    sample_preds = []   # Sample predicted labels (based on majority vote)

    total_lm_loss = 0
    total_probe_loss = 0
    total_sparsity = 0
    num_batches = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(probe.device)
        attention_mask = batch["attention_mask"].to(probe.device)
        classification_labels = batch["classification_labels"].to(probe.device)
        classification_weights = batch["classification_weights"].to(probe.device)
        lm_labels = batch["lm_labels"].to(probe.device)
        vpc_spans: List[List[List[int]]] = batch["vpc_spans"]
        tpc_spans: List[List[List[int]]] = batch["tpc_spans"]
        vtc_spans: List[List[List[int]]] = batch["vtc_spans"]
        neg_spans: List[List[List[int]]] = batch["neg_spans"]
        conflict_types: List[Optional[int]] = batch["conflict_types"]
        print(f"DEBUG: conflict_types: {conflict_types}")

        # Forward pass
        outputs = probe(
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            inputs=batch["inputs"].to(probe.device),
            labels=lm_labels,
        )

        probe_logits = outputs["probe_logits"]
        probe_probs = torch.softmax(probe_logits, dim=-1).float()

        # Conditional prediction using threshold
        # If negative class (class 0) probability > threshold, predict as negative class
        # Otherwise select the class with the highest probability among the three positive classes (1, 2, 3)
        neg_class_prob = probe_probs[:, :, 0]  # [batch_size, seq_len] negative class probability
        pos_class_probs = probe_probs[:, :, 1:]  # [batch_size, seq_len, 3] positive class probabilities

        # Threshold judgment
        use_neg_class = neg_class_prob > threshold

        # For positive classes, select the one with the highest probability among classes 1, 2, 3
        pos_preds = torch.argmax(pos_class_probs, dim=-1) + 1  # +1 because class index starts from 1

        # Select final prediction based on threshold condition
        probe_preds = torch.where(
            use_neg_class,
            torch.zeros_like(pos_preds),  # Predict as class 0 (negative)
            pos_preds  # Predict as the class with max probability among positive classes
        )  

        # Calculate 4-class cross-entropy loss (with sample weights)
        probe_loss = compute_probe_ce_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )

        # 1. All token-level metrics (excluding padding and ignored tokens)
        valid_mask = (attention_mask == 1) & (classification_labels != -100.0)
        # For 4-class classification, store complete 4D probability vector and predicted class index for each token
        all_probs['all'].extend(probe_probs[valid_mask].cpu().numpy())
        all_preds['all'].extend(probe_preds[valid_mask].cpu().numpy())
        # Scope: All valid tokens
        all_labels['all'].extend(classification_labels[valid_mask].cpu().numpy().astype(int))

        # 2. Span-level metrics: does not include valid tokens that are not spans
        # Create a boolean mask with same shape as input_ids, initialized to False
        annotated_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(len(input_ids)):
            # 4-class: collect all types of spans (VPC, TPC, VTC, negative samples)
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            for span_range in all_spans:
                start, end = span_range[0], span_range[1]
                assert start <= end, f"Invalid span range: {span_range}"
                annotated_tokens_mask[i, start:end+1] = True

        # Filter ignored tokens
        annotated_tokens_mask = annotated_tokens_mask & (classification_labels != -100.0)
        
        all_probs['span'].extend(probe_probs[annotated_tokens_mask].cpu().numpy())
        all_preds['span'].extend(probe_preds[annotated_tokens_mask].cpu().numpy())
        # Scope: Only tokens within annotated spans
        all_labels['span'].extend(classification_labels[annotated_tokens_mask].cpu().numpy().astype(int))

        # 3. Span-level metrics (using max aggregation)
        for i in range(len(input_ids)):
            # 4-class: collect all types of spans and assign correct class labels
            vpc_spans_batch = vpc_spans[i]
            tpc_spans_batch = tpc_spans[i]
            vtc_spans_batch = vtc_spans[i]
            neg_spans_batch = neg_spans[i]
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            if len(all_spans) == 0:
                continue

            # 4-class labels: Class 1 (VPC), Class 2 (TPC), Class 3 (VTC), Class 0 (Negative)
            span_labels = (
                [1] * len(vpc_spans_batch) + 
                [2] * len(tpc_spans_batch) + 
                [3] * len(vtc_spans_batch) + 
                [0] * len(neg_spans_batch)
            )

            for label, (start, end) in zip(span_labels, all_spans):
                # For each span, find the max probability of the corresponding class within the span
                span_probs = probe_probs[i, start:end+1]  # [span_len, 4]
                
                # Based on max aggregation: find the token that maximizes the probability of ANY class, use that token's full row probability argmax as span prediction
                token_max_vals, _ = torch.max(span_probs, dim=1)  # [span_len]   max along row
                best_token_idx = torch.argmax(token_max_vals)  # Index of max of max (token index)
                best_token_probs = span_probs[best_token_idx]  # Aligned with prediction decision: 4 logits of the token with max Logit, used for prediction
                max_pred = int(torch.argmax(best_token_probs).cpu().item())  # Get the class with max logit for that representative token
                
                # For multi-class metrics, need full 4D probability vector
                # Select full probability vector of the token with max probability for the corresponding class
                max_class_prob_idx = span_probs[:, label].argmax()  # Token index with max probability for that class
                max_prob_vector = span_probs[max_class_prob_idx].cpu().numpy()  # Aligned with ground truth: full 4D probability vector of that token, used for evaluation
                
                all_probs['span_max'].append(max_prob_vector)  # 保存完整的4维概率向量
                all_preds['span_max'].append(max_pred)
                all_labels['span_max'].append(label)

        # Calculate sample-level predictions (based on majority vote)
        conflict_type_counts = {}
        for i in range(len(input_ids)):
            conflict_type = conflict_types[i]
            if conflict_type is not None:
                conflict_type_counts[conflict_type] = conflict_type_counts.get(conflict_type, 0) + 1

        print(f"DEBUG: Conflict type distribution: {conflict_type_counts}")
        print(f"DEBUG: Total samples with conflict_type: {sum(conflict_type_counts.values())}")

        # 4. Sample-level statistics
        for i in range(len(input_ids)):
            conflict_type = conflict_types[i]
            if conflict_type is not None:  # Only process samples with conflict_type labels
                # Count number of tokens predicted as class 1, 2, 3 in this sample (exclude class 0)
                sample_preds_count = torch.zeros(4, dtype=torch.int)  # index 0, 1, 2, 3

                # Get valid token predictions for this sample (attention_mask == 1)
                sample_valid_preds = probe_preds[i][attention_mask[i] == 1]

                # Count prediction count for each class (including class 0)
                for pred in sample_valid_preds:
                    sample_preds_count[pred] += 1

                # Print predicted token counts for each class
                print(f"DEBUG: Sample {i} token counts - Class 0: {sample_preds_count[0]}, Class 1: {sample_preds_count[1]}, Class 2: {sample_preds_count[2]}, Class 3: {sample_preds_count[3]}")

                # Sample-level prediction logic: since each sample has a ground truth hallucination label, should choose from 1, 2, 3
                # If there are positive class predictions (1, 2, 3), choose the one with the highest count
                if sample_preds_count[1:].sum() > 0:  # Has positive class predictions
                    max_count = sample_preds_count[1:].max()
                    candidates = torch.where(sample_preds_count[1:] == max_count)[0] + 1
                    sample_pred = candidates.min().item()
                    print(f"DEBUG: Sample {i} prediction: {sample_pred} (max positive count: {max_count})")
                else:
                    # Skip this sample, do not include in sample-level evaluation
                    print(f"DEBUG: Sample {i} skipped - no positive predictions detected")
                    continue

                sample_labels.append(conflict_type)
                sample_preds.append(sample_pred)

        # Update running metrics
        total_lm_loss += outputs["lm_loss"].item()
        total_probe_loss += probe_loss.item()
        # 4-class: Calculate sum of probabilities for hallucination classes (1, 2, 3)
        valid_probs = probe_probs[attention_mask == 1]  # [num_valid_tokens, 4]
        hallucination_probs = valid_probs[:, 1:].sum(dim=-1)  # [num_valid_tokens] - Sum of probabilities for classes 1, 2, 3
        total_sparsity += hallucination_probs.mean().item()
        num_batches += 1

    # Calculate average metrics
    metrics = {
        "lm_loss": total_lm_loss / num_batches,
        "probe_loss": total_probe_loss / num_batches,
        "sparsity": total_sparsity / num_batches,
        "num_classes": 4,  # 4-class
    }

    # print(f"DEBUG: Final sample statistics:")
    # print(f"  Total samples with conflict_type: {sum(conflict_type_counts.values())}")
    # print(f"  Samples processed for evaluation: {len(sample_labels)}")

    # Convert sample-level data to numpy arrays
    sample_labels = np.array(sample_labels) if sample_labels else np.array([])
    sample_preds = np.array(sample_preds) if sample_preds else np.array([])
    sample_probs = np.array([])  # No probability matrix for sample level

    # Convert lists to numpy arrays
    all_probs = {k: np.array(v) for k, v in all_probs.items()}
    all_preds = {k: np.array(v) for k, v in all_preds.items()}
    all_labels = {k: np.array(v) for k, v in all_labels.items()}

    # Add sample-level data to all_labels (for printing)
    if len(sample_labels) > 0:
        all_labels['sample'] = sample_labels
        all_preds['sample'] = sample_preds

    # Calculate classification metrics for each aggregation level
    for agg_level in ['all', 'span', 'span_max']:
        if len(all_labels[agg_level]) == 0:
            continue

        clf_metrics = compute_clf_metrics(
            all_preds[agg_level],
            all_labels[agg_level],
            all_probs[agg_level]
        )

        for metric_name, metric_value in clf_metrics.items():
            metrics[f"{agg_level}_{metric_name}"] = metric_value

    # Calculate sample-level metrics
    if len(sample_labels) > 0 and len(sample_preds) > 0:
        # For sample level, we don't have probability matrix, so create a dummy one
        # Probability vector for each sample: 1.0 for predicted class, 0.0 for others
        sample_probs_matrix = np.zeros((len(sample_labels), 4))
        for i, pred in enumerate(sample_preds):
            sample_probs_matrix[i, pred] = 1.0

        sample_metrics = compute_clf_metrics(
            sample_preds,
            sample_labels,
            sample_probs_matrix
        )

        for metric_name, metric_value in sample_metrics.items():
            metrics[f"sample_{metric_name}"] = metric_value

    # If prefix is specified, add prefix: metric_key_prefix (e.g. dataset ID)
    if metric_key_prefix:
        metrics = {
            f"{metric_key_prefix}/{k}": v
            for k, v in metrics.items()
        }

    # If verbose is True, print metrics
    if verbose:
        print_eval_metrics(
            metrics,
            metric_key_prefix=metric_key_prefix,
            all_labels=all_labels,
            include_random_baseline=True,
        )

    # Save ROC curves
    if save_roc_curves and save_dir:
        plot_roc_curves(
            all_preds, 
            all_labels, 
            all_probs, 
            save_dir=save_dir, 
            prefix=metric_key_prefix
        )

    # If requested, save raw results
    if dump_raw_results and save_dir:
        results_dir = save_dir / "eval_results"
        if metric_key_prefix:
            results_dir = results_dir / metric_key_prefix
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save raw prediction results
        for agg_level in ['span_max']:
            if len(all_labels[agg_level]) > 0:
                np.save(results_dir / f'{agg_level}_probs.npy', all_probs[agg_level])
                np.save(results_dir / f'{agg_level}_preds.npy', all_preds[agg_level])
                np.save(results_dir / f'{agg_level}_labels.npy', all_labels[agg_level])

        # Save sample-level data
        if len(sample_labels) > 0:
            np.save(results_dir / 'sample_preds.npy', sample_preds)
            np.save(results_dir / 'sample_labels.npy', sample_labels)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def evaluate_on_multiple_datasets(
    probe: ValueHeadProbe,
    eval_config: EvaluationConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate probe on multiple datasets.
    Args:
        probe: The probe to evaluate
        eval_config: Evaluation configuration
        tokenizer: Model tokenizer
    Returns:
        Dictionary mapping dataset ID to its metrics
    """
    all_metrics = {}
    
    # Create output directory
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on each dataset
    for dataset_config in eval_config.dataset_configs:
        print(f"\nEvaluating on {dataset_config.dataset_id}...")
        
        # Create dataset
        dataset = create_probing_dataset(dataset_config, tokenizer)
        print(f"  Dataset size: {len(dataset)} samples")
        
        from functools import partial
        collate_fn = partial(tokenized_probing_collate_fn, processor=tokenizer)
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=eval_config.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        
        # Evaluate
        metrics = evaluate_probe(
            probe=probe,
            eval_dataloader=dataloader,
            metric_key_prefix=dataset_config.dataset_id,
            threshold=eval_config.probe_config.threshold,
            verbose=True,
            save_roc_curves=eval_config.save_roc_curves,
            save_dir=output_dir if eval_config.save_roc_curves else None,
            dump_raw_results=eval_config.save_raw_results,
        )
        
        # Save metrics
        metrics['dataset_id'] = dataset_config.dataset_id
        save_jsonl([metrics], output_dir / "eval_metrics.jsonl", append=True)

        # Prepare metrics for print_eval_metrics (remove prefix)
        display_metrics = {}
        prefix_to_remove = f"{dataset_config.dataset_id}/"
        print(f"DEBUG: dataset_id={dataset_config.dataset_id}, prefix_to_remove='{prefix_to_remove}'")

        for k, v in metrics.items():
            if k.startswith(prefix_to_remove):
                display_metrics[k[len(prefix_to_remove):]] = v
            else:
                display_metrics[k] = v

        print(f"DEBUG: Metrics keys with sample: {[k for k in display_metrics.keys() if 'sample' in k][:2]}")

        # Save formatted evaluation report
        print_eval_metrics(
            metrics=display_metrics,
            metric_key_prefix="",  # metrics has no prefix now
            save_to_file=True,
            output_file=str(output_dir / "eval_report.txt"),
            save_json=True,
            json_output_file=str(output_dir / "eval_report.json")
        )
        
        all_metrics[dataset_config.dataset_id] = metrics
    
    return all_metrics


def main(eval_config: EvaluationConfig):
    """Main evaluation function."""
    # ===== 1. Load Environment Variables =====
    # Load environment variables if .env file exists
    load_dotenv()
    
    # ===== 2. Print Evaluation Config =====
    print("Evaluation Configuration:")
    for key, value in eval_config.__dict__.items():
        print(f"  {key}: {value}")
    
    # ===== 3. Load Model and Processor/Tokenizer =====
    print(f"\nLoading model: {eval_config.probe_config.model_name}")
    # Check if it is a multimodal model
    model_name = eval_config.probe_config.model_name
    is_multimodal = ('vision' in model_name.lower() or
                     'onevision' in model_name.lower() or
                     'qwen2_5_vl' in model_name.lower() or
                     'qwen2.5_vl' in model_name.lower() or
                     'ocean_r1' in model_name.lower() or
                     'llama' in model_name.lower())
    
    if is_multimodal:
        # Use AutoProcessor instead of AutoTokenizer
        print(f"Loading multimodal model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # processor.tokenizer.padding_side = 'left'
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = processor  # Pass processor as tokenizer
    else:
        # Original text model loading logic
        model, tokenizer = load_model_and_tokenizer(model_name)
    
    # ===== 4. Load and Setup Probe according to config (ValueHeadProbe/LoRA etc.) =====
    print(f"\nLoading probe: {eval_config.probe_config.probe_id}")
    model, probe = setup_probe(model, eval_config.probe_config)
    
    print(f"Device: {next(probe.parameters()).device}")
    
    # ===== 5. Run Evaluation on each evaluation dataset =====
    print(f"\nEvaluating on {len(eval_config.dataset_configs)} datasets...")
    results = evaluate_on_multiple_datasets(probe, eval_config, tokenizer)
    
    # ===== 6. Print Key Metrics Summary =====
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for dataset_id, metrics in results.items():
        print(f"\n{dataset_id}:")
        key_metrics = [
            f"span_max_accuracy",
            f"span_max_f1",
            f"span_max_auc"
        ]
        for metric in key_metrics:
            full_key = f"{dataset_id}/{metric}"
            if full_key in metrics:
                print(f"  {metric}: {metrics[full_key]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a probe")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config from YAML
    config_dict = load_yaml(args.config)
    eval_config = EvaluationConfig(**config_dict)
    
    main(eval_config)