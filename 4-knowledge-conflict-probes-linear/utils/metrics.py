

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)


def compute_clf_metrics(
    preds: np.ndarray,
    labels: np.ndarray, 
    probs: np.ndarray
) -> Dict[str, float]:
    preds = preds.astype(int) if preds.dtype != int else preds
    labels = labels.astype(int) if labels.dtype != int else labels
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    accuracy = accuracy_score(labels, preds)
    
    if probs.ndim == 1:
        raise ValueError(f"probs should be a 2D array [N, num_classes], but received 1D array shape={probs.shape}")
    
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
    
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
    
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Per-class metrics: returns per-class array when average=None
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    # Multi-class AUC and Recall@10%FPR (one-vs-rest strategy)
    auc_ovr_score = float('nan')
    auc_ovr_per_class = {}  # AUC for each class
    recall_at_10fpr_ovr_per_class = {}
    recall_at_10fpr_ovr_scores = []

    if probs.ndim == 2 and probs.shape[1] > 1:
        try:
            # Compute AUC and Recall@10%FPR using one-vs-rest strategy
            auc_ovr_scores = []
            for class_idx in range(num_classes):
                # Create binary labels: current class vs all other classes
                y_binary = (labels == class_idx).astype(int)
                if len(np.unique(y_binary)) == 2:  # Ensure there are two classes
                    probs_class = probs[:, class_idx]  # Probability for current class
                    try:
                        # Compute AUC
                        auc_class = roc_auc_score(y_binary, probs_class)
                        auc_ovr_scores.append(auc_class)
                        auc_ovr_per_class[int(unique_labels[class_idx])] = float(auc_class)

                        # Compute ROC curve and Recall@10%FPR
                        fpr, tpr, thresholds = roc_curve(y_binary, probs_class)
                        # Find the point closest to 10% FPR
                        target_fpr = 0.10
                        idx = np.argmin(np.abs(fpr - target_fpr))
                        recall_at_10fpr = tpr[idx]
                        recall_at_10fpr_ovr_per_class[int(unique_labels[class_idx])] = float(recall_at_10fpr)
                        recall_at_10fpr_ovr_scores.append(recall_at_10fpr)

                    except ValueError:
                        # Skip if cannot compute (e.g., all samples are the same class)
                        pass
            auc_ovr_score = np.mean(auc_ovr_scores) if len(auc_ovr_scores) > 0 else float('nan')
        except Exception:
            auc_ovr_score = float('nan')
            auc_ovr_per_class = {}
            recall_at_10fpr_ovr_per_class = {}
            recall_at_10fpr_ovr_scores = []
    
    # Confusion matrix: cm[i, j] represents the number of samples with true class i predicted as class j; diagonal cm[i,i] is the number of correct predictions
    cm = confusion_matrix(labels, preds)
    
    # Count samples for each class
    class_counts = {int(c): int(np.sum(labels == c)) for c in unique_labels}
    
    # Build return dictionary
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_micro": float(precision_micro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_micro": float(recall_micro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "auc_ovr": float(auc_ovr_score),
        "recall_at_10fpr_ovr": float(np.mean(recall_at_10fpr_ovr_scores)) if len(recall_at_10fpr_ovr_scores) > 0 else float('nan'),
        "num_classes": int(num_classes),
        "total_samples": int(len(labels)),
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }
    
    # Add per-class metrics (using class index as key)
    for idx, class_label in enumerate(sorted(unique_labels)):
        # Add standard One-vs-Rest metrics for each class
        metrics[f"precision_class_{class_label}"] = float(precision_per_class[idx])
        metrics[f"recall_class_{class_label}"] = float(recall_per_class[idx])
        metrics[f"f1_class_{class_label}"] = float(f1_per_class[idx])
        metrics[f"count_class_{class_label}"] = class_counts[class_label]
        # Add AUC for each class (One-vs-Rest)
        if int(class_label) in auc_ovr_per_class:
            metrics[f"auc_ovr_class_{class_label}"] = auc_ovr_per_class[int(class_label)]
        else:
            metrics[f"auc_ovr_class_{class_label}"] = float('nan')

        # Add recall@10%FPR for each class (One-vs-Rest)
        if int(class_label) in recall_at_10fpr_ovr_per_class:
            metrics[f"recall_at_10fpr_ovr_class_{class_label}"] = recall_at_10fpr_ovr_per_class[int(class_label)]
        else:
            metrics[f"recall_at_10fpr_ovr_class_{class_label}"] = float('nan')
    # For compatibility, add default precision, recall, f1 (using weighted)
    metrics["precision"] = float(precision_weighted)
    metrics["recall"] = float(recall_weighted)
    metrics["f1"] = float(f1_weighted)
    
    return metrics


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:

    if probabilities is None:
        probabilities = predictions
    
    return compute_clf_metrics(predictions, labels, probabilities)


def plot_roc_curves(
    all_preds: Dict[str, List[float]],
    all_labels: Dict[str, List[float]], 
    all_probs: Dict[str, List[float]],
    save_dir: str,
    prefix: Optional[str] = None
) -> None:
    """
    Modified: Changed from binary classification to multi-class ROC curve plotting (One-vs-Rest strategy).
    
    Plot multi-class ROC curves: draw One-vs-Rest ROC curve for each class.
    
    Args:
        all_preds: Prediction results for each aggregation level
        all_labels: Labels for each aggregation level
        all_probs: Probabilities for each aggregation level [N, num_classes]
        save_dir: Directory to save plots
        prefix: Optional filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Multi-class: plot One-vs-Rest ROC curves for each class
    plt.figure(figsize=(18, 6))
    
    for i, agg_level in enumerate(['all', 'span', 'span_max']):
        plt.subplot(1, 3, i+1)
        
        if agg_level not in all_labels or len(all_labels[agg_level]) == 0:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient Data")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        labels = np.array(all_labels[agg_level]).astype(int)
        probs_list = all_probs[agg_level]
        
        # Handle probs shape: should be list of lists (multi-class)
        if isinstance(probs_list[0], (list, np.ndarray)) and len(probs_list[0]) > 1:
            # Multi-class: each element is a probability vector
            probs = np.array(probs_list)  # [N, num_classes]
        else:
            # Skip if format is incorrect
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInvalid Probability Format")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        if probs.ndim != 2 or probs.shape[1] <= 1:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInvalid Probability Format")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        # Multi-class: probs is [N, num_classes]
        num_classes = probs.shape[1]
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient Label Diversity")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        # Plot ROC curves for each class (One-vs-Rest)
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        class_names = ['Class 0 (Negative)', 'Class 1 (VPC)', 'Class 2 (TPC)', 'Class 3 (VTC)']
        
        auc_scores = []
        for class_idx in range(num_classes):
            # Create binary labels: current class vs all other classes
            y_binary = (labels == class_idx).astype(int)
            if len(np.unique(y_binary)) == 2:  # Ensure there are two classes
                probs_class = probs[:, class_idx]  # Probability for current class
                try:
                    fpr, tpr, _ = roc_curve(y_binary, probs_class)
                    auc = roc_auc_score(y_binary, probs_class)
                    auc_scores.append(auc)
                    
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
                    plt.plot(fpr, tpr, lw=2, color=colors[class_idx], 
                           label=f'{class_name} (AUC = {auc:.2f})', alpha=0.8)
                except ValueError:
                    # Skip if cannot compute ROC curve
                    pass
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{agg_level.replace('_', ' ').title()}\nMulti-class ROC Curves (One-vs-Rest)")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    filename = f"{prefix}_roc_curves.png" if prefix else "roc_curves.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-class ROC curves saved to {os.path.join(save_dir, filename)}")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, save_path: str) -> None:

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_eval_metrics(
    metrics: dict,
    metric_key_prefix: str = "",
    all_labels: Optional[dict] = None,
    include_random_baseline: bool = True,
    seed: int = 42,
    save_to_file: bool = False,
    output_file: str = "evaluation_results.txt",
    save_json: bool = False,
    json_output_file: str = "evaluation_results.json"
) -> None:

    import io
    from contextlib import redirect_stdout
    from datetime import datetime

    # Create string buffer to capture output
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        if metric_key_prefix:
            print(f"{'='*70}")
            print(f"Multimodal Hallucination Detection Evaluation Report - {metric_key_prefix}")
            print(f"{'='*70}")
        else:
            print(f"{'='*70}")
            print("Multimodal Hallucination Detection Evaluation Report")
            print(f"{'='*70}")

        prefix = metric_key_prefix + "/" if metric_key_prefix else ""

        # Print loss metrics if available
        if f'{prefix}lm_loss' in metrics:
            print(f"\n{'─'*50}")
            print("Loss Metrics")
            print(f"{'─'*50}")
            print(f"LM Loss:          {metrics.get(f'{prefix}lm_loss', 0):.4f}")
            print(f"Probe Loss:       {metrics.get(f'{prefix}probe_loss', 0):.4f}")
            print(f"Sparsity:         {metrics.get(f'{prefix}sparsity', 0):.4f}")

        # Print classification metrics for different aggregation levels
        agg_levels = ['all', 'span', 'span_max', 'sample']
        for agg_level in agg_levels:
            if f'{prefix}{agg_level}_accuracy' in metrics:
                print(f"\n{'─'*50}")

                if agg_level == 'sample':
                    print("Sample-level Classification Metrics")
                else:
                    level_name = agg_level.replace('_', ' ').title()
                    print(f"{level_name}-level Classification Metrics")

                print(f"{'─'*50}")

                # Basic metrics
                print("Overall Performance:")
                print(f"  Accuracy:     {metrics[f'{prefix}{agg_level}_accuracy']:.4f}")
                print(f"  Total Samples:     {metrics.get(f'{prefix}{agg_level}_total_samples', 0)}")

                # Multi-class metrics table
                print(f"\nAggregated Metrics:")
                print(f"  {'Metric':<12} {'Macro':<8} {'Micro':<8} {'Weighted':<8}")
                print(f"  {'─'*42}")
                print(f"  {'Precision':<12} {metrics.get(f'{prefix}{agg_level}_precision_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_precision_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_precision_weighted', 0):<8.4f}")
                print(f"  {'Recall':<12} {metrics.get(f'{prefix}{agg_level}_recall_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_recall_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_recall_weighted', 0):<8.4f}")
                print(f"  {'F1 Score':<12} {metrics.get(f'{prefix}{agg_level}_f1_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_f1_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_f1_weighted', 0):<8.4f}")

                num_classes = int(metrics.get(f'{prefix}{agg_level}_num_classes', 4))
                print(f"\nPer-Class One-vs-Rest Detailed Metrics - {num_classes} classes:")
                print(f"  {'Class':<6} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8} {'R@10%FPR':<10}")
                print(f"  {'─'*58}")

                for class_idx in range(num_classes):

                    actual_class_idx = class_idx if agg_level != 'sample' else class_idx + 1

                    # Standard One-vs-Rest metrics
                    if f'{prefix}{agg_level}_precision_class_{actual_class_idx}' in metrics:
                        # Precision (One-vs-Rest)
                        precision_class = metrics[f'{prefix}{agg_level}_precision_class_{actual_class_idx}']
                        if not np.isnan(precision_class):
                            precision_str = f"{precision_class:.4f}"
                        else:
                            precision_str = "N/A"

                        # Recall (One-vs-Rest)
                        recall_class = metrics[f'{prefix}{agg_level}_recall_class_{actual_class_idx}']
                        if not np.isnan(recall_class):
                            recall_str = f"{recall_class:.4f}"
                        else:
                            recall_str = "N/A"

                        # F1 (One-vs-Rest)
                        f1_class = metrics[f'{prefix}{agg_level}_f1_class_{actual_class_idx}']
                        if not np.isnan(f1_class):
                            f1_str = f"{f1_class:.4f}"
                        else:
                            f1_str = "N/A"

                        # AUC (One-vs-Rest)
                        auc_class = metrics[f'{prefix}{agg_level}_auc_ovr_class_{actual_class_idx}']
                        if not np.isnan(auc_class):
                            auc_str = f"{auc_class:.4f}"
                        else:
                            auc_str = "N/A"

                        # Recall@10%FPR (One-vs-Rest)
                        if f'{prefix}{agg_level}_recall_at_10fpr_ovr_class_{actual_class_idx}' in metrics:
                            recall_at_10fpr_class = metrics[f'{prefix}{agg_level}_recall_at_10fpr_ovr_class_{actual_class_idx}']
                            if not np.isnan(recall_at_10fpr_class):
                                recall_at_10fpr_str = f"{recall_at_10fpr_class:.4f}"
                            else:
                                recall_at_10fpr_str = "N/A"
                        else:
                            recall_at_10fpr_str = "N/A"

                        print(f"  {actual_class_idx:<6} {precision_str:<10} {recall_str:<8} {f1_str:<8} {auc_str:<8} {recall_at_10fpr_str:<10}")

                has_roc_metrics = False
                roc_info = []
                if f'{prefix}{agg_level}_auc_ovr' in metrics:
                    auc_val = metrics[f'{prefix}{agg_level}_auc_ovr']
                    if not np.isnan(auc_val):
                        roc_info.append(f"AUC (OvR): {auc_val:.4f}")
                        has_roc_metrics = True

                if f'{prefix}{agg_level}_recall_at_10fpr_ovr' in metrics:
                    recall_at_10fpr_val = metrics[f'{prefix}{agg_level}_recall_at_10fpr_ovr']
                    if not np.isnan(recall_at_10fpr_val):
                        roc_info.append(f"Recall@10%FPR: {recall_at_10fpr_val:.4f}")
                        has_roc_metrics = True

                if has_roc_metrics:
                    print(f"\nROC Metrics:")
                    for info in roc_info:
                        print(f"  {info}")

                # Confusion matrix
                if f'{prefix}{agg_level}_confusion_matrix' in metrics:
                    cm = np.array(metrics[f'{prefix}{agg_level}_confusion_matrix'])
                    print(f"\nConfusion Matrix:")
                    print("  True →  Pred ↓")

                    if agg_level == 'sample':
                        header_labels = [str(i + 1) for i in range(cm.shape[1])]  # 1,2,3,...
                    else:
                        header_labels = [str(i) for i in range(cm.shape[1])]     # 0,1,2,3

                    header = "      " + " ".join([f"{label:>4}" for label in header_labels])
                    print(header)
                    print("      " + "─" * (5 * cm.shape[1]))


                    for i in range(cm.shape[0]):
                        row_str = " ".join([f"{int(cm[i,j]):>4}" for j in range(cm.shape[1])])
                        row_label = i + 1 if agg_level == 'sample' else i
                        print(f"  {row_label} │ {row_str}")

                # Baseline
                if all_labels and agg_level in all_labels:
                    labels = np.array(all_labels[agg_level])
                    if len(labels) > 0:
                        unique, counts = np.unique(labels, return_counts=True)
                        majority_class = unique[np.argmax(counts)]
                        majority_baseline = accuracy_score(labels, np.full_like(labels, majority_class))
                        print(f"\nBaseline:")
                        print(f"  Majority Class Accuracy: {majority_baseline:.4f} (Class {majority_class})")

        print(f"\n{'='*70}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Report Generated At: {current_time}")
        print(f"{'='*70}")

    output_content = output_buffer.getvalue()

    print(output_content)

    if save_to_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"\n✓ Evaluation results saved to file: {output_file}")


    if save_json:
        import json
        os.makedirs(os.path.dirname(json_output_file) if os.path.dirname(json_output_file) else ".", exist_ok=True)


        json_data = {
            "metadata": {
                "report_type": "multimodal_hallucination_detection_evaluation",
                "generated_at": current_time,
                "metric_prefix": metric_key_prefix if metric_key_prefix else None
            },
            "metrics": metrics
        }

        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
