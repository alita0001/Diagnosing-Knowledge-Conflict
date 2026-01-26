"""幻觉检测探针的评估脚本。"""

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
    已修改
    评估探针在数据集上的性能（四分类版本）。
    Args:
        probe: 要评估的探针
        eval_dataloader: 评估数据的DataLoader
        metric_key_prefix: 指标键的前缀
        threshold: 负类预测的阈值
        verbose: 是否打印指标
        save_roc_curves: 是否保存ROC曲线图
        save_dir: 保存结果的目录
        dump_raw_results: 是否保存原始预测结果
    Returns:
        评估指标字典
    """
    # 评估前强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()

    # 为不同聚合级别初始化指标集合
    all_probs = {'all': [], 'span': [], 'span_max': []}
    all_preds = {'all': [], 'span': [], 'span_max': []}
    all_labels = {'all': [], 'span': [], 'span_max': []}

    # 样本级别的数据收集
    sample_labels = []  # 样本的真实标签（conflict_type）
    sample_preds = []   # 样本的预测标签（基于多数投票）

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

        # 前向传播
        outputs = probe(
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            inputs=batch["inputs"].to(probe.device),
            labels=lm_labels,
        )

        probe_logits = outputs["probe_logits"]
        probe_probs = torch.softmax(probe_logits, dim=-1).float()

        # 使用阈值进行条件预测
        # 如果负类（类别0）概率 > threshold，预测为负类
        # 否则在三个正类（1,2,3）中选择概率最大的
        neg_class_prob = probe_probs[:, :, 0]  # [batch_size, seq_len] 负类概率
        pos_class_probs = probe_probs[:, :, 1:]  # [batch_size, seq_len, 3] 正类概率

        # 阈值判断
        use_neg_class = neg_class_prob > threshold

        # 对于正类，在类别1,2,3中选择概率最大的
        pos_preds = torch.argmax(pos_class_probs, dim=-1) + 1  # +1 因为是从1开始的类别索引

        # 根据阈值条件选择最终预测
        probe_preds = torch.where(
            use_neg_class,
            torch.zeros_like(pos_preds),  # 预测为类别0（负类）
            pos_preds  # 预测为正类中的最大概率类别
        )  

        # 计算四分类交叉熵损失（带样本权重）
        probe_loss = compute_probe_ce_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )

        # 1. 所有token级别的指标（排除padding和被忽略的tokens）
        valid_mask = (attention_mask == 1) & (classification_labels != -100.0)
        # 对于四分类，存储每个token的完整4维概率向量和预测类别索引
        all_probs['all'].extend(probe_probs[valid_mask].cpu().numpy())
        all_preds['all'].extend(probe_preds[valid_mask].cpu().numpy())
        # 范围	所有有效tokens
        all_labels['all'].extend(classification_labels[valid_mask].cpu().numpy().astype(int))

        # 2. Span级别的指标：不会包含非 span 的有效 token
        # 创建一个与 input_ids 同形状的布尔掩码，初始值全为 False
        annotated_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(len(input_ids)):
            # 四分类：收集所有类型的spans（VPC, TPC, VTC, 负样本）
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            for span_range in all_spans:
                start, end = span_range[0], span_range[1]
                assert start <= end, f"Invalid span range: {span_range}"
                annotated_tokens_mask[i, start:end+1] = True

        # 过滤被忽略的tokens
        annotated_tokens_mask = annotated_tokens_mask & (classification_labels != -100.0)
        
        all_probs['span'].extend(probe_probs[annotated_tokens_mask].cpu().numpy())
        all_preds['span'].extend(probe_preds[annotated_tokens_mask].cpu().numpy())
        # 范围	仅在已标注spans内的tokens
        all_labels['span'].extend(classification_labels[annotated_tokens_mask].cpu().numpy().astype(int))

        # 3. Span级别的指标（使用最大聚合）
        for i in range(len(input_ids)):
            # 四分类：收集所有类型的spans并分配正确的类别标签
            vpc_spans_batch = vpc_spans[i]
            tpc_spans_batch = tpc_spans[i]
            vtc_spans_batch = vtc_spans[i]
            neg_spans_batch = neg_spans[i]
            all_spans: List[List[int]] = vpc_spans[i] + tpc_spans[i] + vtc_spans[i] + neg_spans[i]
            if len(all_spans) == 0:
                continue

            # 四分类标签：类别1 (VPC), 类别2 (TPC), 类别3 (VTC), 类别0 (负样本)
            span_labels = (
                [1] * len(vpc_spans_batch) + 
                [2] * len(tpc_spans_batch) + 
                [3] * len(vtc_spans_batch) + 
                [0] * len(neg_spans_batch)
            )

            for label, (start, end) in zip(span_labels, all_spans):
                # 对于每个span，找到对应类别在span内的最大概率
                span_probs = probe_probs[i, start:end+1]  # [span_len, 4]
                
                # 基于max聚合：找出使任一类别概率最大的token，用该token的整行概率argmax作为span预测
                token_max_vals, _ = torch.max(span_probs, dim=1)  # [span_len]   按行取最大
                best_token_idx = torch.argmax(token_max_vals)  # 最大中的最大 的索引（token索引）
                best_token_probs = span_probs[best_token_idx]  # 对预测决策对齐：最大Logit所在token的四个logit，用来预测
                max_pred = int(torch.argmax(best_token_probs).cpu().item())  # 得到代表token之后选logit最大那个类别
                
                # 对于多分类指标，需要完整的4维概率向量
                # 选择对应类别概率最大的token的完整概率向量
                max_class_prob_idx = span_probs[:, label].argmax()  # 该类别的最大概率的token索引
                max_prob_vector = span_probs[max_class_prob_idx].cpu().numpy()  # 对真实类对齐：该token的完整4维概率向量，用来评估
                
                all_probs['span_max'].append(max_prob_vector)  # 保存完整的4维概率向量
                all_preds['span_max'].append(max_pred)
                all_labels['span_max'].append(label)

        # 计算样本级别的预测（基于多数投票）
        conflict_type_counts = {}
        for i in range(len(input_ids)):
            conflict_type = conflict_types[i]
            if conflict_type is not None:
                conflict_type_counts[conflict_type] = conflict_type_counts.get(conflict_type, 0) + 1

        print(f"DEBUG: Conflict type distribution: {conflict_type_counts}")
        print(f"DEBUG: Total samples with conflict_type: {sum(conflict_type_counts.values())}")

        # 4. 样本级别统计
        for i in range(len(input_ids)):
            conflict_type = conflict_types[i]
            if conflict_type is not None:  # 只处理有conflict_type标签的样本
                # 统计该样本中预测为1,2,3类的token数量（排除类别0）
                sample_preds_count = torch.zeros(4, dtype=torch.int)  # 索引0,1,2,3

                # 获取该样本的有效token预测（attention_mask == 1）
                sample_valid_preds = probe_preds[i][attention_mask[i] == 1]

                # 统计每个类别的预测数量（包括类别0）
                for pred in sample_valid_preds:
                    sample_preds_count[pred] += 1

                # 打印每个类别的预测token数量
                print(f"DEBUG: Sample {i} token counts - Class 0: {sample_preds_count[0]}, Class 1: {sample_preds_count[1]}, Class 2: {sample_preds_count[2]}, Class 3: {sample_preds_count[3]}")

                # 样本级别预测逻辑：既然每个样本都有真实幻觉标签，应该从1,2,3中选择
                # 如果有正类预测（1,2,3），选择数量最多的
                if sample_preds_count[1:].sum() > 0:  # 有正类预测
                    max_count = sample_preds_count[1:].max()
                    candidates = torch.where(sample_preds_count[1:] == max_count)[0] + 1
                    sample_pred = candidates.min().item()
                    print(f"DEBUG: Sample {i} prediction: {sample_pred} (max positive count: {max_count})")
                else:
                    # 跳过这个样本，不纳入样本级别评估
                    print(f"DEBUG: Sample {i} skipped - no positive predictions detected")
                    continue

                sample_labels.append(conflict_type)
                sample_preds.append(sample_pred)

        # 更新运行指标
        total_lm_loss += outputs["lm_loss"].item()
        total_probe_loss += probe_loss.item()
        # 四分类：计算幻觉类别（1,2,3）的平均概率总和
        valid_probs = probe_probs[attention_mask == 1]  # [num_valid_tokens, 4]
        hallucination_probs = valid_probs[:, 1:].sum(dim=-1)  # [num_valid_tokens] - 类别1,2,3的概率和
        total_sparsity += hallucination_probs.mean().item()
        num_batches += 1

    # 计算平均指标
    metrics = {
        "lm_loss": total_lm_loss / num_batches,
        "probe_loss": total_probe_loss / num_batches,
        "sparsity": total_sparsity / num_batches,
        "num_classes": 4,  # 四分类
    }

    # print(f"DEBUG: Final sample statistics:")
    # print(f"  Total samples with conflict_type: {sum(conflict_type_counts.values())}")
    # print(f"  Samples processed for evaluation: {len(sample_labels)}")

    # 样本级别的数据转换为numpy数组
    sample_labels = np.array(sample_labels) if sample_labels else np.array([])
    sample_preds = np.array(sample_preds) if sample_preds else np.array([])
    sample_probs = np.array([])  # 样本级别没有概率矩阵

    # 将列表转换为numpy数组
    all_probs = {k: np.array(v) for k, v in all_probs.items()}
    all_preds = {k: np.array(v) for k, v in all_preds.items()}
    all_labels = {k: np.array(v) for k, v in all_labels.items()}

    # 添加样本级别的数据到all_labels中（用于打印）
    if len(sample_labels) > 0:
        all_labels['sample'] = sample_labels
        all_preds['sample'] = sample_preds

    # 为每个聚合级别计算分类指标
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

    # 计算样本级别的指标
    if len(sample_labels) > 0 and len(sample_preds) > 0:
        # 对于样本级别，我们没有概率矩阵，所以创建一个虚拟的概率矩阵
        # 每个样本的概率向量：预测类别的概率为1，其他为0
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

    # 如果指定了前缀，则添加前缀： metric_key_prefix（比如数据集 ID）
    if metric_key_prefix:
        metrics = {
            f"{metric_key_prefix}/{k}": v
            for k, v in metrics.items()
        }

    # 如果verbose为True，则打印指标
    if verbose:
        print_eval_metrics(
            metrics,
            metric_key_prefix=metric_key_prefix,
            all_labels=all_labels,
            include_random_baseline=True,
        )

    # 保存ROC曲线
    if save_roc_curves and save_dir:
        plot_roc_curves(
            all_preds, 
            all_labels, 
            all_probs, 
            save_dir=save_dir, 
            prefix=metric_key_prefix
        )

    # 如果请求，保存原始结果
    if dump_raw_results and save_dir:
        results_dir = save_dir / "eval_results"
        if metric_key_prefix:
            results_dir = results_dir / metric_key_prefix
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 保存原始预测结果
        for agg_level in ['span_max']:
            if len(all_labels[agg_level]) > 0:
                np.save(results_dir / f'{agg_level}_probs.npy', all_probs[agg_level])
                np.save(results_dir / f'{agg_level}_preds.npy', all_preds[agg_level])
                np.save(results_dir / f'{agg_level}_labels.npy', all_labels[agg_level])

        # 保存样本级别的数据
        if len(sample_labels) > 0:
            np.save(results_dir / 'sample_preds.npy', sample_preds)
            np.save(results_dir / 'sample_labels.npy', sample_labels)

    # 清理
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def evaluate_on_multiple_datasets(
    probe: ValueHeadProbe,
    eval_config: EvaluationConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Dict[str, float]]:
    """
    在多个数据集上评估探针。
    Args:
        probe: 要评估的探针
        eval_config: 评估配置
        tokenizer: 模型的tokenizer
    Returns:
        映射数据集ID到其指标的字典
    """
    all_metrics = {}
    
    # 创建输出目录
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 在每个数据集上评估
    for dataset_config in eval_config.dataset_configs:
        print(f"\nEvaluating on {dataset_config.dataset_id}...")
        
        # 创建数据集
        dataset = create_probing_dataset(dataset_config, tokenizer)
        print(f"  Dataset size: {len(dataset)} samples")
        
        from functools import partial
        collate_fn = partial(tokenized_probing_collate_fn, processor=tokenizer)
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=eval_config.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        
        # 评估
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
        
        # 保存指标
        metrics['dataset_id'] = dataset_config.dataset_id
        save_jsonl([metrics], output_dir / "eval_metrics.jsonl", append=True)

        # 为print_eval_metrics准备metrics（移除前缀）
        display_metrics = {}
        prefix_to_remove = f"{dataset_config.dataset_id}/"
        print(f"DEBUG: dataset_id={dataset_config.dataset_id}, prefix_to_remove='{prefix_to_remove}'")

        for k, v in metrics.items():
            if k.startswith(prefix_to_remove):
                display_metrics[k[len(prefix_to_remove):]] = v
            else:
                display_metrics[k] = v

        print(f"DEBUG: Metrics keys with sample: {[k for k in display_metrics.keys() if 'sample' in k][:2]}")

        # 保存格式化的评估报告
        print_eval_metrics(
            metrics=display_metrics,
            metric_key_prefix="",  # 现在metrics没有前缀
            save_to_file=True,
            output_file=str(output_dir / "eval_report.txt"),
            save_json=True,
            json_output_file=str(output_dir / "eval_report.json")
        )
        
        all_metrics[dataset_config.dataset_id] = metrics
    
    return all_metrics


def main(eval_config: EvaluationConfig):
    """主评估函数。"""
    # ===== 1、加载环境变量 =====
    # 如果存在.env文件，加载环境变量
    load_dotenv()
    
    # ===== 2、打印评估配置 =====
    print("Evaluation Configuration:")
    for key, value in eval_config.__dict__.items():
        print(f"  {key}: {value}")
    
    # ===== 3、加载模型与处理器/分词器 =====
    print(f"\nLoading model: {eval_config.probe_config.model_name}")
    # 检查是否是多模态模型
    model_name = eval_config.probe_config.model_name
    is_multimodal = ('vision' in model_name.lower() or
                     'onevision' in model_name.lower() or
                     'qwen2_5_vl' in model_name.lower() or
                     'qwen2.5_vl' in model_name.lower() or
                     'ocean_r1' in model_name.lower() or
                     'llama' in model_name.lower())
    
    if is_multimodal:
        # 使用 AutoProcessor 代替 AutoTokenizer
        print(f"Loading multimodal model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # processor.tokenizer.padding_side = 'left'
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = processor  # 将 processor 作为 tokenizer 传递
    else:
        # 原有的文本模型加载逻辑
        model, tokenizer = load_model_and_tokenizer(model_name)
    
    # ===== 4、按照配置加载并装配探针（ValueHeadProbe/LoRA等） =====
    print(f"\nLoading probe: {eval_config.probe_config.probe_id}")
    model, probe = setup_probe(model, eval_config.probe_config)
    
    print(f"Device: {next(probe.parameters()).device}")
    
    # ===== 5、在各评估数据集上运行评估 =====
    print(f"\nEvaluating on {len(eval_config.dataset_configs)} datasets...")
    results = evaluate_on_multiple_datasets(probe, eval_config, tokenizer)
    
    # ===== 6、打印关键指标摘要 =====
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
    
    # 从YAML加载配置
    config_dict = load_yaml(args.config)
    eval_config = EvaluationConfig(**config_dict)
    
    main(eval_config)