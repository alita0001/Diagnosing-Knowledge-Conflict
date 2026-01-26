"""
已修改
探针训练的损失函数
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from .value_head_probe import ValueHeadProbe

def compute_probe_ce_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len num_classes'],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    classification_weights: Float[Tensor, 'batch_size seq_len'],
    max_clipped_logits: float = 100.0,
    ignore_label: float = -100.0,
):
    """
    已修改：计算四分类交叉熵损失（带样本权重）。
    参数:
        probe_logits: [B, T, 4] - 四分类logits
        classification_labels: [B, T] - 类别标签 (0, 1, 2, 3, 或 -100表示忽略)
        classification_weights: [B, T] - 每个位置的 “样本权重”
        max_clipped_logits: logits裁剪范围
        ignore_label: 忽略标签值
    
    返回:
        标量损失值
    """
    # 裁剪logits以防止极端值
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )
    
    try:
        # 将 [B, T, 4] reshape 为 [B*T, 4] 用于 cross_entropy
        B, T, C = probe_logits_clipped.shape
        logits_flat = probe_logits_clipped.view(-1, C)  # [B*T, 4]
        labels_flat = classification_labels.view(-1).long()  # [B*T] - 转为long类型
        weights_flat = classification_weights.view(-1)  # [B*T]
        
        # 计算交叉熵损失（每个位置）
        ce_loss_flat = F.cross_entropy(
            logits_flat,  # [B*T, 4]
            labels_flat,  # [B*T]
            reduction='none'  # 返回每个样本的损失
        )  # [B*T]
        
        # 应用样本权重
        weighted_loss_flat = ce_loss_flat * weights_flat

        # 创建掩码：忽略标签为-100的位置
        mask = (labels_flat != ignore_label)
        
        # 只计算非忽略位置的平均损失
        if mask.any():
            ce_loss = weighted_loss_flat[mask].mean()
        else:
            ce_loss = torch.tensor(0.0, device=probe_logits.device)
            
        # 检查ce_loss中是否有NaN
        if torch.isnan(ce_loss):
            print(f"WARNING: NaN detected in ce_loss")
            ce_loss = torch.tensor(0.0, device=probe_logits.device)
        return ce_loss
    except Exception as e:
        print(f"Error in compute_probe_ce_loss: {e}")
        return torch.tensor(0.0, device=probe_logits.device)


def compute_probe_max_aggregation_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len num_classes'],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    classification_weights: Float[Tensor, 'batch_size seq_len'],
    vpc_spans: List[List[Tuple[int, int]]],
    tpc_spans: List[List[Tuple[int, int]]],
    vtc_spans: List[List[Tuple[int, int]]],
    negative_spans: List[List[Tuple[int, int]]],
    max_clipped_logits: float = 100.0,
    sparsity_penalty_weight: Optional[float] = None,
):
    """
    计算span级别的最大聚合损失（四分类版本）。
    对于每个span，找到对应类别在所有token位置的最大logit，然后计算交叉熵损失。
    参数:
        probe_logits: [B, T, 4] - 四分类logits
        classification_labels: [B, T] - 类别标签 (0,1,2,3,-100)
        vpc_spans: 视觉-先验知识冲突spans
        tpc_spans: 文本-先验知识冲突spans
        vtc_spans: 视觉-文本冲突spans
        negative_spans: 真实内容spans
    返回:
        标量平均损失
    """

    span_losses = []
    device = probe_logits.device
    dtype = probe_logits.dtype

    # 裁剪logits以防止极端值
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )

    for i in range(probe_logits_clipped.shape[0]):  # 遍历批次中的项目
        # 处理VPC spans（类别1）
        for start, end in vpc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue
            
            # 等价于 probe_logits_clipped[i, start:end+1, :]
            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # 找到类别1在span内的最大logit位置，并提取完整logit向量用于计算损失
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 1]), :]  # [4]
            
            target = torch.tensor(1, device=device, dtype=torch.long)  # 类别1
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # 处理TPC spans（类别2）
        for start, end in tpc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # 找到类别2在span内的最大logit位置，并提取完整logit向量用于计算损失
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 2]), :]  # [4]
            
            target = torch.tensor(2, device=device, dtype=torch.long)  # 类别2
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # 处理VTC spans（类别3）
        for start, end in vtc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # 找到类别3在span内的最大logit位置，并提取完整logit向量用于计算损失
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 3]), :]  # [4]
            
            target = torch.tensor(3, device=device, dtype=torch.long)  # 类别3
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # 处理负样本spans（类别0）
        for start, end in negative_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # 找到类别0在span内的最大logit位置，并提取完整logit向量用于计算损失
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 0]), :]  # [4]
            
            target = torch.tensor(0, device=device, dtype=torch.long)  # 类别0
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

    if not span_losses:
        # 批次中未找到有效spans，返回零损失
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 计算批次中所有spans的平均损失
    final_loss = torch.mean(torch.stack(span_losses))

    if sparsity_penalty_weight is not None:  # None
        sparsity_loss = compute_sparsity_loss(
            probe_logits=probe_logits,
            attention_mask=classification_labels != -100.0,  # 转换为attention_mask
        )

        final_loss = final_loss + sparsity_penalty_weight * sparsity_loss

    # 检查NaN
    if torch.isnan(final_loss):
        print(f"WARNING: NaN detected in compute_probe_max_aggregation_loss. Returning 0.0")
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return final_loss


def compute_sparsity_loss(
    probe_logits: Float[Tensor, "batch seq_len num_classes"],
    attention_mask: torch.Tensor,  # [batch, seq_len] 布尔或整型掩码
) -> Float[Tensor, ""]:
    """
    计算稀疏性损失以鼓励探针具有选择性（四分类版本）。
    这个损失鼓励探针具有较低的平均激活度，防止它将所有内容都标记为幻觉。
    
    对于四分类，我们计算幻觉类别（1,2,3）的概率总和，鼓励这个值较低。
    
    参数:
        probe_logits: [B, T, 4] - 探针输出logits
        attention_mask: [B, T] - 有效tokens的掩码（布尔或整型）
    返回:
        标量稀疏性损失
    """
    # 转换为布尔掩码
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    
    # 使用softmax获取概率（四分类）
    probe_probs = torch.softmax(probe_logits, dim=-1)  # [B, T, 4]
    
    # 计算幻觉类别（1,2,3）的概率总和
    hallucination_probs = probe_probs[:, :, 1:].sum(dim=-1)  # [B, T] - 类别1,2,3的概率和
    
    # 应用注意力掩码
    masked_probs = hallucination_probs * attention_mask.float()
    
    # 计算平均激活度
    num_valid_tokens = attention_mask.sum()
    if num_valid_tokens == 0:
        return torch.tensor(0.0, device=probe_logits.device)
    
    # 平均每个有效token的幻觉类别概率总和是 avg_activation (63.7%)
    avg_activation = masked_probs.sum() / num_valid_tokens
    
    # 稀疏性损失就是平均激活度
    return avg_activation


def mask_high_loss_spans(
    lm_model: Union[AutoModelForImageTextToText, PeftModel],
    # input_ids: Float[Tensor, 'batch_size seq_len'],
    # attention_mask: Float[Tensor, 'batch_size seq_len'],
    inputs: Dict[str, torch.Tensor],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    spans: List[List[Tuple[int, int]]],
    threshold: float = 1.0,  # 损失被视为高的阈值
):
    """
    对于语言模型损失较高的spans，将其标签设置为-100以忽略它们。
    通过掩码这些高损失的真实内容 spans，减少它们对训练的影响。
    """

    # 获取语言模型的 logits
    if isinstance(lm_model, PeftModel):
        with lm_model.disable_adapter():
            lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = lm_model(
                inputs=inputs,
                output_hidden_states=False,
            ).logits
    else:
        lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = lm_model(
            inputs=inputs,
            output_hidden_states=False,
        ).logits

    log_probs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
    log_probs: Float[Tensor, 'batch_size seq_len'] = log_probs.gather(-1, inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)

    for i in range(log_probs.shape[0]):  # 遍历批次中的项目
        # 仅对支持实体spans进行处理
        for start, end in spans[i]:
            if start > end: # 无效span，跳过
                continue

            span_neg_log_probs = -log_probs[i, start:end+1]
            max_neg_log_prob = span_neg_log_probs.max()

            if max_neg_log_prob > threshold:
                classification_labels[i, start:end+1] = -100.0

    return classification_labels


def compute_kl_divergence_loss(
    model: ValueHeadProbe,
    lm_logits: Float[Tensor, "batch seq_len vocab_size"],
    # input_ids: Int[Tensor, "batch seq_len"],
    # attention_mask: Int[Tensor, "batch seq_len"],
    inputs: Dict[str, torch.Tensor],
    lm_labels: Int[Tensor, "batch seq_len"],
) -> Float[Tensor, ""]:
    """
    计算带LoRA的模型和基础模型之间的KL散度。
    
    参数:
        model: ValueHeadProbe模型
        lm_logits: 来自带LoRA适配器模型的logits
        lm_labels: 语言建模标签
    返回:
        标量KL散度损失
    """
    
    # 检查模型是否有LoRA适配器
    if not isinstance(model.model, PeftModel) or not model.model.active_adapters:
        return torch.tensor(0.0, device=lm_logits.device)
    
    # 获取不带LoRA的基础模型输出
    with model.model.disable_adapter():
        base_outputs = model(
            inputs=inputs,
            labels=lm_labels,
        )
        base_logits = base_outputs["lm_logits"].detach()
    
    # 计算KL散度
    with torch.autocast(device_type=lm_logits.device.type, enabled=False):
        log_q = torch.log_softmax(lm_logits.float(), -1)  # model with LoRA  # 带LoRA的模型
        p_ref = torch.softmax(base_logits.float(), -1)    # reference model  # 参考模型
        kl = F.kl_div(log_q, p_ref, reduction='none', log_target=False).sum(-1)
        
        # 仅在有效tokens上计算损失
        active_mask = (lm_labels != -100)
        if not active_mask.any():
            return torch.tensor(0.0, device=lm_logits.device)
            
        kl_loss = kl[active_mask].mean()
    
    return kl_loss