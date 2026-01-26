

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    verbose: bool = True,
    save_roc_curves: bool = True,
    save_dir: Optional[Path] = None,
    dump_raw_results: bool = False,
    tokenizer: Optional[Union[AutoTokenizer, AutoProcessor]] = None,
) -> Dict[str, float]:

    gc.collect()
    torch.cuda.empty_cache()


    all_probs = {'all': [], 'span': [], 'span_max': []}
    all_preds = {'all': [], 'span': [], 'span_max': []}
    all_preds = {'all': [], 'span': [], 'span_max': []}
    all_labels = {'all': [], 'span': [], 'span_max': []}
    

    detailed_results = []

    total_lm_loss = 0
    total_probe_loss = 0
    total_sparsity = 0
    num_batches = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids: Int[Tensor, "batch_size seq_len"] = batch["input_ids"].to(probe.device)
        attention_mask: Int[Tensor, "batch_size seq_len"] = batch["attention_mask"].to(probe.device)
        classification_labels: Float[Tensor, "batch_size seq_len"] = batch["classification_labels"].to(probe.device)
        classification_weights: Float[Tensor, "batch_size seq_len"] = batch["classification_weights"].to(probe.device)
        lm_labels: Int[Tensor, "batch_size seq_len"] = batch["lm_labels"].to(probe.device)
        vpc_spans: List[List[List[int]]] = batch["vpc_spans"]
        tpc_spans: List[List[List[int]]] = batch["tpc_spans"]
        vtc_spans: List[List[List[int]]] = batch["vtc_spans"]
        neg_spans: List[List[List[int]]] = batch["neg_spans"]

        outputs = probe(
            inputs=batch["inputs"].to(probe.device),
            labels=lm_labels,
        )

        probe_logits: Float[Tensor, "batch_size seq_len num_classes"] = outputs["probe_logits"]
        probe_probs: Float[Tensor, "batch_size seq_len num_classes"] = torch.softmax(probe_logits, dim=-1).float()
        probe_preds: Int[Tensor, "batch_size seq_len"] = torch.argmax(probe_logits, dim=-1).long()  

        probe_loss = compute_probe_ce_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )

        valid_mask = (attention_mask == 1) & (classification_labels != -100.0)
        all_probs['all'].extend(probe_probs[valid_mask].cpu().numpy())
        all_preds['all'].extend(probe_preds[valid_mask].cpu().numpy())
        all_labels['all'].extend(classification_labels[valid_mask].cpu().numpy().astype(int))

        annotated_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(len(input_ids)):
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            for span_range in all_spans:
                start, end = span_range[0], span_range[1]
                assert start <= end, f"Invalid span range: {span_range}"
                annotated_tokens_mask[i, start:end+1] = True

        annotated_tokens_mask = annotated_tokens_mask & (classification_labels != -100.0)
        
        all_probs['span'].extend(probe_probs[annotated_tokens_mask].cpu().numpy())
        all_preds['span'].extend(probe_preds[annotated_tokens_mask].cpu().numpy())
        all_labels['span'].extend(classification_labels[annotated_tokens_mask].cpu().numpy().astype(int))

        for i in range(len(input_ids)):
            vpc_spans_batch = vpc_spans[i]
            tpc_spans_batch = tpc_spans[i]
            vtc_spans_batch = vtc_spans[i]
            neg_spans_batch = neg_spans[i]
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            if len(all_spans) == 0:
                continue

            span_labels = (
                [1] * len(vpc_spans_batch) + 
                [2] * len(tpc_spans_batch) + 
                [3] * len(vtc_spans_batch) + 
                [0] * len(neg_spans_batch)
            )

            for label, (start, end) in zip(span_labels, all_spans):
                span_probs = probe_probs[i, start:end+1]  # [span_len, 4]
                
                token_max_vals, _ = torch.max(span_probs, dim=1)  
                best_token_idx = torch.argmax(token_max_vals)  
                best_token_probs = span_probs[best_token_idx]  
                max_pred = int(torch.argmax(best_token_probs).cpu().item())  
                
                max_class_prob_idx = span_probs[:, label].argmax()  
                max_prob_vector = span_probs[max_class_prob_idx].cpu().numpy()  
                
                all_probs['span_max'].append(max_prob_vector)  
                all_preds['span_max'].append(max_pred)
                all_labels['span_max'].append(label)

                if tokenizer is not None:
                    span_text = tokenizer.decode(input_ids[i, start:end+1])
                    
                    messages = batch["messages"][i]
                    
                    image_path = None
                    for msg in messages:
                        if isinstance(msg.get('content'), list):
                            for content in msg['content']:
                                if content.get('type') == 'image':
                                    image_path = "Image present" 
                                    if hasattr(content['image'], 'filename'):
                                        image_path = content['image'].filename
                    
                    detailed_results.append({
                        "span_text": span_text,
                        "label": label,
                        "prediction": max_pred,
                        "probs": max_prob_vector.tolist(),
                        "context": messages,
                        "image_info": image_path,
                        "span_indices": [start, end]
                    })
        
        total_lm_loss += outputs["lm_loss"].item()
        total_probe_loss += probe_loss.item()
        valid_probs = probe_probs[attention_mask == 1]  
        hallucination_probs = valid_probs[:, 1:].sum(dim=-1)  
        total_sparsity += hallucination_probs.mean().item()
        num_batches += 1

    metrics = {
        "lm_loss": total_lm_loss / num_batches,
        "probe_loss": total_probe_loss / num_batches,
        "sparsity": total_sparsity / num_batches,
        "num_classes": 4,
    }

    all_probs = {k: np.array(v) for k, v in all_probs.items()}
    all_preds = {k: np.array(v) for k, v in all_preds.items()}
    all_labels = {k: np.array(v) for k, v in all_labels.items()}
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

    if metric_key_prefix:
        metrics = {
            f"{metric_key_prefix}/{k}": v
            for k, v in metrics.items()
        }

    if verbose:
        print_eval_metrics(
            metrics,
            metric_key_prefix=metric_key_prefix,
            all_labels=all_labels,
            include_random_baseline=True,
        )

    if save_roc_curves and save_dir:
        plot_roc_curves(
            all_preds, 
            all_labels, 
            all_probs, 
            save_dir=save_dir, 
            prefix=metric_key_prefix
        )

    if dump_raw_results and save_dir:
        results_dir = save_dir / "eval_results"
        if metric_key_prefix:
            results_dir = results_dir / metric_key_prefix
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        for agg_level in ['span_max']:
            if len(all_labels[agg_level]) > 0:
                np.save(results_dir / f'{agg_level}_probs.npy', all_probs[agg_level])
                np.save(results_dir / f'{agg_level}_preds.npy', all_preds[agg_level])
                np.save(results_dir / f'{agg_level}_labels.npy', all_labels[agg_level])

        if tokenizer is not None and len(detailed_results) > 0:
            save_jsonl(detailed_results, results_dir / "detailed_results.jsonl")

    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def evaluate_on_multiple_datasets(
    probe: ValueHeadProbe,
    eval_config: EvaluationConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Dict[str, float]]:

    all_metrics = {}
    
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_config in eval_config.dataset_configs:
        print(f"\nEvaluating on {dataset_config.dataset_id}...")
        
        dataset = create_probing_dataset(dataset_config, tokenizer)
        print(f"  Dataset size: {len(dataset)} samples")
        
        from functools import partial
        collate_fn = partial(tokenized_probing_collate_fn, processor=tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=eval_config.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        
        metrics = evaluate_probe(
            probe=probe,
            eval_dataloader=dataloader,
            metric_key_prefix=dataset_config.dataset_id,
            verbose=True,
            save_roc_curves=eval_config.save_roc_curves,
            save_dir=output_dir if eval_config.save_roc_curves else None,
            dump_raw_results=eval_config.save_raw_results,
            tokenizer=tokenizer,
        )
        
        metrics['dataset_id'] = dataset_config.dataset_id
        save_jsonl([metrics], output_dir / "eval_metrics.jsonl", append=True)
        
        all_metrics[dataset_config.dataset_id] = metrics
    
    return all_metrics


def main(eval_config: EvaluationConfig):
    load_dotenv()
    
    print("Evaluation Configuration:")
    for key, value in eval_config.__dict__.items():
        print(f"  {key}: {value}")
    
    print(f"\nLoading model: {eval_config.probe_config.model_name}")
    model_name = eval_config.probe_config.model_name
    is_multimodal = 'vision' in model_name.lower() or 'onevision' in model_name.lower()
    
    if is_multimodal:
        print(f"Loading multimodal model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # processor.tokenizer.padding_side = 'left'
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = processor 
    else:
        model, tokenizer = load_model_and_tokenizer(model_name)
    
    print(f"\nLoading probe: {eval_config.probe_config.probe_id}")
    model, probe = setup_probe(model, eval_config.probe_config)
    
    print(f"Device: {next(probe.parameters()).device}")
    
    print(f"\nEvaluating on {len(eval_config.dataset_configs)} datasets...")
    results = evaluate_on_multiple_datasets(probe, eval_config, tokenizer)
    
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
    
    config_dict = load_yaml(args.config)
    eval_config = EvaluationConfig(**config_dict)
    
    main(eval_config)