"""用于幻觉检测探针的自定义训练器"""

import gc
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import Trainer, AutoTokenizer
from jaxtyping import Float, Int
from torch import Tensor
from peft import PeftModel

from utils.file_utils import save_jsonl
from utils.metrics import print_eval_metrics

from .dataset import TokenizedProbingDataset
from .config import TrainingConfig
from .loss import compute_probe_ce_loss, compute_kl_divergence_loss, compute_probe_max_aggregation_loss, mask_high_loss_spans
from .value_head_probe import ValueHeadProbe
from .evaluate import evaluate_probe


class ProbeTrainer(Trainer):
    """
    自定义训练器，将标准语言模型的下一个token预测损失（CE）
    与钩入内部层的"探针"的分类BCE损失合并。
    """
    def __init__(
        self,
        probe: ValueHeadProbe,
        eval_datasets: List[TokenizedProbingDataset],
        cfg: TrainingConfig,
        eval_steps: Optional[int] = None,
        tokenizer: AutoTokenizer = None,
        **kwargs
    ):
        super().__init__(model=probe, **kwargs)
        self.lambda_lm: float = cfg.lambda_lm
        self.lambda_kl: float = cfg.lambda_kl
        self.anneal_max_aggr: bool = cfg.anneal_max_aggr
        self.anneal_warmup: float = cfg.anneal_warmup
        self.eval_datasets: List[TokenizedProbingDataset] = eval_datasets
        self.threshold = cfg.probe_config.threshold
        self.ignore_label: float = -100.0
        self.gradient_accumulation_steps: int = cfg.gradient_accumulation_steps
        self.eval_steps: Optional[int] = eval_steps
        self.probe_dir: Path = cfg.probe_config.probe_path
        self.tokenizer: AutoTokenizer = tokenizer
        self.high_loss_threshold: Optional[float] = cfg.high_loss_threshold
        self.sparsity_penalty_weight: float = cfg.sparsity_penalty_weight
        self._last_eval_metrics: Optional[dict] = None

    def get_training_progress(self) -> float:
        """获取当前训练进度，返回0到1之间的浮点数"""
        if self.state.max_steps is None or self.state.max_steps == 0:
            return 1.0
        return min(1.0, self.state.global_step / self.state.max_steps)
    
    def compute_loss(
        self,
        model: ValueHeadProbe,
        batch: dict,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """
        已修改
        """
        # 如果使用DataParallel，从底层模型获取设备
        device = model.module.device if isinstance(model, nn.DataParallel) else model.device

        input_ids: torch.Tensor = batch["input_ids"].to(device)
        attention_mask: torch.Tensor = batch["attention_mask"].to(device)
        classification_labels: torch.Tensor = batch["classification_labels"].to(device)
        classification_weights: torch.Tensor = batch["classification_weights"].to(device)
        lm_labels: torch.Tensor = batch["lm_labels"].to(device)
        vpc_spans: List[List[Tuple[int, int]]] = batch["vpc_spans"]
        tpc_spans: List[List[Tuple[int, int]]] = batch["tpc_spans"]
        vtc_spans: List[List[Tuple[int, int]]] = batch["vtc_spans"]
        neg_spans: List[List[Tuple[int, int]]] = batch["neg_spans"]


        # 底层HF语言模型期望"labels"用于下一个token预测
        outputs = model(
            inputs = batch["inputs"],
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            labels=lm_labels,  # 这些是LM标签（不是探针标签！）
        )

        lm_logits = outputs["lm_logits"]
        # probe_logits_1 = outputs["probe_logits"].squeeze(-1)  # 二分类 shape [B, T]  # 形状 [B, T]
        probe_logits = outputs["probe_logits"]  # shape [B, T, 4] for 4-class classification
        lm_loss = outputs["lm_loss"]  # 标准下一个token的CE损失

        if torch.isnan(lm_loss):
            print(f"WARNING: NaN detected in lm_loss")
            lm_loss = torch.tensor(0.0, device=device)

        # 如果需要，计算KL散度 未修改
        kl_loss = torch.tensor(0., device=device)
        if self.lambda_kl > 0:
            kl_loss = compute_kl_divergence_loss(
                model=model,
                lm_logits=lm_logits,
                inputs=batch["inputs"],
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                lm_labels=lm_labels,
            )

        # 如果配置了，掩码高损失的spans  未修改
        if self.high_loss_threshold is not None:
            classification_labels = mask_high_loss_spans(
                lm_model=model.model,
                inputs=batch["inputs"],
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                classification_labels=classification_labels,  # [B, T] 和二分类一样
                spans=neg_spans,
                threshold=self.high_loss_threshold,
            )

        # 计算探针损失
        probe_loss = compute_probe_ce_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )

        # Span级别的最大聚合损失（如果启用）
        if self.anneal_max_aggr:
            max_aggr_probe_loss = compute_probe_max_aggregation_loss(
                probe_logits=probe_logits,
                classification_labels=classification_labels,
                classification_weights=classification_weights,
                vpc_spans=vpc_spans,
                tpc_spans=tpc_spans,
                vtc_spans=vtc_spans,
                negative_spans=neg_spans,
            )
            
            omega = min(1.0, self.get_training_progress() / self.anneal_warmup)
            probe_loss = (1 - omega) * probe_loss + omega * max_aggr_probe_loss
        else:
            omega = 0.0
            max_aggr_probe_loss = torch.tensor(0.0, device=device)

        # 合并损失
        loss = (
            self.lambda_lm * lm_loss +
            self.lambda_kl * kl_loss +
            (1 - self.lambda_lm - self.lambda_kl) * probe_loss
        )

        log_dict = {
            'loss': float(probe_loss.detach().float().item()),
            'lm_loss': float(lm_loss.detach().float().item()),
            'kl_loss': float(kl_loss.detach().float().item()),
            'lambda_lm': self.lambda_lm,
            'lambda_kl': self.lambda_kl,
            'omega': float(omega),
            'active_positions': int(torch.sum((classification_labels != self.ignore_label)).item()),
        }

        if self.anneal_max_aggr:
            log_dict['max_aggr_probe_loss'] = float(max_aggr_probe_loss.detach().float().item())

        self.log(log_dict)

        if return_outputs:
            outputs["loss"] = loss
            outputs["probe_loss"] = probe_loss
            outputs["lm_loss"] = lm_loss
            return (loss, outputs)
        
        # 如果不返回输出，清理内存
        del outputs, lm_logits, probe_logits
        gc.collect()
        torch.cuda.empty_cache()

        return loss
    
    def create_optimizer(self):
        """
        创建优化器，为探针头和LoRA适配器设置不同的学习率。
        """
        
        # 如果使用DataParallel，从底层模型获取设备
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # 将参数分为探针头和LoRA组
        probe_head_params = []
        lora_params = []
        other_params = []
        probe_head_added = False
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'value_head' in name:
                probe_head_params.append(param)
                probe_head_added = True
            elif 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)

        assert probe_head_added == True, f"Probe head not found when computing list of trainable parameters"
        
        # 创建具有不同学习率的参数组
        param_groups = []
        
        if probe_head_params:
            param_groups.append({
                'params': probe_head_params,
                'lr': self.args.probe_head_lr,
                'name': 'probe_head'
            })
            
        if lora_params:
            param_groups.append({
                'params': lora_params, 
                'lr': self.args.lora_lr,
                'name': 'lora'
            })
            
        if other_params:
            # 对于任何其他参数，回退到默认学习率
            param_groups.append({
                'params': other_params,
                'lr': self.args.learning_rate,
                'name': 'other'
            })
        
        # 打印参数组信息
        print("\n=== Optimizer Parameter Groups ===")
        for i, group in enumerate(param_groups):
            param_count = sum(p.numel() for p in group['params'])
            print(f"Group {i} ({group['name']}): {param_count:,} parameters, lr={group['lr']}")

        print(f"lora_lr: {self.args.lora_lr} (type={type(self.args.lora_lr)})")
        print(f"probe_head_lr: {self.args.probe_head_lr} (type={type(self.args.probe_head_lr)})")
        
        optimizer = AdamW(
            param_groups,
            eps=self.args.adam_epsilon if hasattr(self.args, 'adam_epsilon') else 1e-8
        )

        return optimizer
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        重写以确保在创建调度器之前创建我们的自定义优化器。
        """
        # 首先创建我们的自定义优化器
        self.optimizer = self.create_optimizer()
        
        # 然后使用父方法创建调度器
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=self.optimizer
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        save_roc_curves: bool = False,
        dump_raw_eval_results: bool = False,
        verbose: bool = False,
    ):
        all_eval_metrics = {}

        # 检查这是否是最终评估
        is_final_evaluation = self.get_training_progress() >= 1.0
        
        # 如果这是最终评估且已经完成，跳过
        if is_final_evaluation and self._last_eval_metrics is not None:
            if verbose:
                print("Final evaluation already completed, skipping duplicate evaluation.")
            return self._last_eval_metrics

        # 在每个数据集上评估
        for dataset in self.eval_datasets:
            eval_dataloader = self.get_eval_dataloader(dataset)

            model = self._wrap_model(
                self.model,
                training=False,
                dataloader=eval_dataloader
            ).eval()

            metrics = evaluate_probe(
                model,
                eval_dataloader,
                # threshold=self.threshold,
                metric_key_prefix=dataset.config.dataset_id,
                verbose=False,
                save_roc_curves=save_roc_curves,
                save_dir=self.probe_dir if save_roc_curves else None,
                dump_raw_results=dump_raw_eval_results,
            )

            if verbose:
                print_eval_metrics(metrics, metric_key_prefix=dataset.config.dataset_id)
            
            self.log(metrics)
            all_eval_metrics.update(metrics)

            # 将指标保存到JSONL文件
            metrics['global_step'] = self.state.global_step
            metrics['training_progress'] = self.get_training_progress()
            metrics['dataset_id'] = dataset.config.dataset_id
            save_jsonl([metrics], self.probe_dir / "eval_metrics.jsonl", append=True)

        # 存储指标以供后续检索
        self._last_eval_metrics = all_eval_metrics
        return all_eval_metrics