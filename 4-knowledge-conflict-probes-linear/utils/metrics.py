

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
        raise ValueError(f"probs应该是2维数组 [N, num_classes]，但收到1维数组 shape={probs.shape}")
    
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
    
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
    
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # 每个类别的指标： average=None 时，才会返回按类别的数组
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    # 多分类AUC和Recall@10%FPR（one-vs-rest策略）
    auc_ovr_score = float('nan')
    auc_ovr_per_class = {}  # 每个类别的AUC
    recall_at_10fpr_ovr_per_class = {}
    recall_at_10fpr_ovr_scores = []

    if probs.ndim == 2 and probs.shape[1] > 1:
        try:
            # 使用one-vs-rest策略计算AUC和Recall@10%FPR
            auc_ovr_scores = []
            for class_idx in range(num_classes):
                # 创建二分类标签：当前类别 vs 其他类别
                y_binary = (labels == class_idx).astype(int)
                if len(np.unique(y_binary)) == 2:  # 确保有两个类别
                    probs_class = probs[:, class_idx]  # 当前类别的概率
                    try:
                        # 计算AUC
                        auc_class = roc_auc_score(y_binary, probs_class)
                        auc_ovr_scores.append(auc_class)
                        auc_ovr_per_class[int(unique_labels[class_idx])] = float(auc_class)

                        # 计算ROC曲线和Recall@10%FPR
                        fpr, tpr, thresholds = roc_curve(y_binary, probs_class)
                        # 找到FPR最接近10%的点
                        target_fpr = 0.10
                        idx = np.argmin(np.abs(fpr - target_fpr))
                        recall_at_10fpr = tpr[idx]
                        recall_at_10fpr_ovr_per_class[int(unique_labels[class_idx])] = float(recall_at_10fpr)
                        recall_at_10fpr_ovr_scores.append(recall_at_10fpr)

                    except ValueError:
                        # 如果无法计算（例如所有样本都是同一类别），跳过
                        pass
            auc_ovr_score = np.mean(auc_ovr_scores) if len(auc_ovr_scores) > 0 else float('nan')
        except Exception:
            auc_ovr_score = float('nan')
            auc_ovr_per_class = {}
            recall_at_10fpr_ovr_per_class = {}
            recall_at_10fpr_ovr_scores = []
    
    # 混淆矩阵：cm[i, j] 表示"真实为类 i，被预测为类 j"的样本数；对角线 cm[i,i] 是预测正确数
    cm = confusion_matrix(labels, preds)
    
    # 计算每个类别的样本数
    class_counts = {int(c): int(np.sum(labels == c)) for c in unique_labels}
    
    # 构建返回字典
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
        "confusion_matrix": cm.tolist()  # 转换为列表以便JSON序列化
    }
    
    # 添加每个类别的指标（使用类别索引作为键）
    for idx, class_label in enumerate(sorted(unique_labels)):
        # 添加每个类别的标准One-vs-Rest指标
        metrics[f"precision_class_{class_label}"] = float(precision_per_class[idx])
        metrics[f"recall_class_{class_label}"] = float(recall_per_class[idx])
        metrics[f"f1_class_{class_label}"] = float(f1_per_class[idx])
        metrics[f"count_class_{class_label}"] = class_counts[class_label]
        # 添加每个类别的AUC (One-vs-Rest)
        if int(class_label) in auc_ovr_per_class:
            metrics[f"auc_ovr_class_{class_label}"] = auc_ovr_per_class[int(class_label)]
        else:
            metrics[f"auc_ovr_class_{class_label}"] = float('nan')

        # 添加每个类别的recall@10%FPR (One-vs-Rest)
        if int(class_label) in recall_at_10fpr_ovr_per_class:
            metrics[f"recall_at_10fpr_ovr_class_{class_label}"] = recall_at_10fpr_ovr_per_class[int(class_label)]
        else:
            metrics[f"recall_at_10fpr_ovr_class_{class_label}"] = float('nan')
    # 为了兼容性，添加默认的precision, recall, f1（使用weighted）
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
    已修改：从二分类改为只支持多分类ROC曲线绘制（One-vs-Rest策略）。
    
    绘制多分类ROC曲线：为每个类别绘制One-vs-Rest ROC曲线。
    
    Args:
        all_preds: 每个聚合级别的预测结果
        all_labels: 每个聚合级别的标签
        all_probs: 每个聚合级别的概率 [N, num_classes]
        save_dir: 保存图表的目录
        prefix: 可选的文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 多分类：为每个类别绘制One-vs-Rest ROC曲线
    plt.figure(figsize=(18, 6))
    
    for i, agg_level in enumerate(['all', 'span', 'span_max']):
        plt.subplot(1, 3, i+1)
        
        if agg_level not in all_labels or len(all_labels[agg_level]) == 0:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient Data")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        labels = np.array(all_labels[agg_level]).astype(int)
        probs_list = all_probs[agg_level]
        
        # 处理probs的形状：应该是列表的列表（多分类）
        if isinstance(probs_list[0], (list, np.ndarray)) and len(probs_list[0]) > 1:
            # 多分类：每个元素是一个概率向量
            probs = np.array(probs_list)  # [N, num_classes]
        else:
            # 如果格式不对，跳过
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInvalid Probability Format")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        if probs.ndim != 2 or probs.shape[1] <= 1:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInvalid Probability Format")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        # 多分类：probs是[N, num_classes]
        num_classes = probs.shape[1]
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient Label Diversity")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        # 为每个类别绘制ROC曲线（One-vs-Rest）
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        class_names = ['Class 0 (Negative)', 'Class 1 (VPC)', 'Class 2 (TPC)', 'Class 3 (VTC)']
        
        auc_scores = []
        for class_idx in range(num_classes):
            # 创建二分类标签：当前类别 vs 其他类别
            y_binary = (labels == class_idx).astype(int)
            if len(np.unique(y_binary)) == 2:  # 确保有两个类别
                probs_class = probs[:, class_idx]  # 当前类别的概率
                try:
                    fpr, tpr, _ = roc_curve(y_binary, probs_class)
                    auc = roc_auc_score(y_binary, probs_class)
                    auc_scores.append(auc)
                    
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
                    plt.plot(fpr, tpr, lw=2, color=colors[class_idx], 
                           label=f'{class_name} (AUC = {auc:.2f})', alpha=0.8)
                except ValueError:
                    # 如果无法计算ROC曲线，跳过
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
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('接收者操作特征曲线 (Receiver Operating Characteristic)')
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

    # 创建字符串缓冲区来捕获输出
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        if metric_key_prefix:
            print(f"{'='*70}")
            print(f"多模态幻觉探测评估指标报告 - {metric_key_prefix}")
            print(f"{'='*70}")
        else:
            print(f"{'='*70}")
            print("多模态幻觉探测评估指标报告")
            print(f"{'='*70}")

        prefix = metric_key_prefix + "/" if metric_key_prefix else ""

        # 如果可用，打印损失指标
        if f'{prefix}lm_loss' in metrics:
            print(f"\n{'─'*50}")
            print("损失指标 (Loss Metrics)")
            print(f"{'─'*50}")
            print(f"LM Loss:          {metrics.get(f'{prefix}lm_loss', 0):.4f}")
            print(f"Probe Loss:       {metrics.get(f'{prefix}probe_loss', 0):.4f}")
            print(f"Sparsity:         {metrics.get(f'{prefix}sparsity', 0):.4f}")

        # 为不同聚合级别打印分类指标
        agg_levels = ['all', 'span', 'span_max', 'sample']
        for agg_level in agg_levels:
            if f'{prefix}{agg_level}_accuracy' in metrics:
                print(f"\n{'─'*50}")

                if agg_level == 'sample':
                    print("样本级别分类指标 (Sample-level Classification)")
                else:
                    level_name = agg_level.replace('_', ' ').title()
                    print(f"{level_name}级别分类指标 ({level_name}-level Classification)")

                print(f"{'─'*50}")

                # 基本指标
                print("整体性能指标 (Overall Performance):")
                print(f"  准确率 (Accuracy):     {metrics[f'{prefix}{agg_level}_accuracy']:.4f}")
                print(f"  总样本数 (Samples):     {metrics.get(f'{prefix}{agg_level}_total_samples', 0)}")

                # 多分类指标表格
                print(f"\n聚合指标 (Aggregated Metrics):")
                print(f"  {'指标':<12} {'Macro':<8} {'Micro':<8} {'Weighted':<8}")
                print(f"  {'─'*42}")
                print(f"  {'精确率':<12} {metrics.get(f'{prefix}{agg_level}_precision_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_precision_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_precision_weighted', 0):<8.4f}")
                print(f"  {'召回率':<12} {metrics.get(f'{prefix}{agg_level}_recall_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_recall_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_recall_weighted', 0):<8.4f}")
                print(f"  {'F1分数':<12} {metrics.get(f'{prefix}{agg_level}_f1_macro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_f1_micro', 0):<8.4f} {metrics.get(f'{prefix}{agg_level}_f1_weighted', 0):<8.4f}")

                num_classes = int(metrics.get(f'{prefix}{agg_level}_num_classes', 4))
                print(f"\n每个类别One-vs-Rest详细指标 (Per-Class One-vs-Rest Metrics) - 共{num_classes}个类别:")
                print(f"  {'类别':<6} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8} {'R@10%FPR':<10}")
                print(f"  {'─'*58}")

                for class_idx in range(num_classes):

                    actual_class_idx = class_idx if agg_level != 'sample' else class_idx + 1

                    # 标准One-vs-Rest指标
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
                    print(f"\nROC相关指标 (ROC Metrics):")
                    for info in roc_info:
                        print(f"  {info}")

                # 混淆矩阵
                if f'{prefix}{agg_level}_confusion_matrix' in metrics:
                    cm = np.array(metrics[f'{prefix}{agg_level}_confusion_matrix'])
                    print(f"\n混淆矩阵 (Confusion Matrix):")
                    print("  真实→  预测↓")

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
                        print(f"\n基准线 (Baseline):")
                        print(f"  多数类别准确率: {majority_baseline:.4f} (类别 {majority_class})")

        print(f"\n{'='*70}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"报告生成时间: {current_time}")
        print(f"{'='*70}")

    output_content = output_buffer.getvalue()

    print(output_content)

    if save_to_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"\n✓ 评估结果已保存到文件: {output_file}")


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
