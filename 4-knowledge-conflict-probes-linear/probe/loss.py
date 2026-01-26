"""
Modified
Loss functions for probe training
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
    Modified: Compute 4-class cross-entropy loss (with sample weights).
    Args:
        probe_logits: [B, T, 4] - 4-class logits
        classification_labels: [B, T] - Class labels (0, 1, 2, 3, or -100 for ignored)
        classification_weights: [B, T] - "Sample weights" for each position
        max_clipped_logits: Logits clipping range
        ignore_label: Ignored label value
    
    Returns:
        Scalar loss value
    """
    # Clip logits to prevent extreme values
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )
    
    try:
        # Reshape [B, T, 4] to [B*T, 4] for cross_entropy
        B, T, C = probe_logits_clipped.shape
        logits_flat = probe_logits_clipped.view(-1, C)  # [B*T, 4]
        labels_flat = classification_labels.view(-1).long()  # [B*T] - convert to long type
        weights_flat = classification_weights.view(-1)  # [B*T]
        
        # Compute cross-entropy loss (per position)
        ce_loss_flat = F.cross_entropy(
            logits_flat,  # [B*T, 4]
            labels_flat,  # [B*T]
            reduction='none'  # Return loss for each sample
        )  # [B*T]
        
        # Apply sample weights
        weighted_loss_flat = ce_loss_flat * weights_flat

        # Create mask: positions where label is -100 are ignored
        mask = (labels_flat != ignore_label)
        
        # Mean loss of non-ignored positions only
        if mask.any():
            ce_loss = weighted_loss_flat[mask].mean()
        else:
            ce_loss = torch.tensor(0.0, device=probe_logits.device)
            
        # Check for NaN in ce_loss
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
    Compute span-level max aggregation loss (4-class version).
    For each span, find the max logit of the corresponding class across all token positions, then compute cross-entropy loss.
    Args:
        probe_logits: [B, T, 4] - 4-class logits
        classification_labels: [B, T] - Class labels (0,1,2,3,-100)
        vpc_spans: Vision-Prior Conflict spans
        tpc_spans: Text-Prior Conflict spans
        vtc_spans: Vision-Text Conflict spans
        negative_spans: Truthful content spans
    Returns:
        Scalar mean loss
    """

    span_losses = []
    device = probe_logits.device
    dtype = probe_logits.dtype

    # Clip logits to prevent extreme values
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )

    for i in range(probe_logits_clipped.shape[0]):  # Iterate through items in batch
        # Process VPC spans (Class 1)
        for start, end in vpc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue
            
            # Equivalent to probe_logits_clipped[i, start:end+1, :]
            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # Find the position with max logit for class 1 within the span, and extract the full logit vector for loss computation
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 1]), :]  # [4]
            
            target = torch.tensor(1, device=device, dtype=torch.long)  # Class 1
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # Process TPC spans (Class 2)
        for start, end in tpc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # Find the position with max logit for class 2 within the span, and extract the full logit vector for loss computation
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 2]), :]  # [4]
            
            target = torch.tensor(2, device=device, dtype=torch.long)  # Class 2
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # Process VTC spans (Class 3)
        for start, end in vtc_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # Find the position with max logit for class 3 within the span, and extract the full logit vector for loss computation
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 3]), :]  # [4]
            
            target = torch.tensor(3, device=device, dtype=torch.long)  # Class 3
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

        # Process negative sample spans (Class 0)
        for start, end in negative_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]  # [span_len, 4]
            
            # Find the position with max logit for class 0 within the span, and extract the full logit vector for loss computation
            max_logits_vector = span_logits[torch.argmax(span_logits[:, 0]), :]  # [4]
            
            target = torch.tensor(0, device=device, dtype=torch.long)  # Class 0
            loss = F.cross_entropy(
                max_logits_vector.unsqueeze(0),  # [1, 4]
                target.unsqueeze(0),  # [1]
                reduction='none'
            )
            span_losses.append(loss)

    if not span_losses:
        # No valid spans found in batch, return zero loss
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Compute mean loss of all spans in the batch
    final_loss = torch.mean(torch.stack(span_losses))

    if sparsity_penalty_weight is not None:  # None
        sparsity_loss = compute_sparsity_loss(
            probe_logits=probe_logits,
            attention_mask=classification_labels != -100.0,  # Convert to attention_mask
        )

        final_loss = final_loss + sparsity_penalty_weight * sparsity_loss

    # Check for NaN
    if torch.isnan(final_loss):
        print(f"WARNING: NaN detected in compute_probe_max_aggregation_loss. Returning 0.0")
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return final_loss


def compute_sparsity_loss(
    probe_logits: Float[Tensor, "batch seq_len num_classes"],
    attention_mask: torch.Tensor,  # [batch, seq_len] 布尔或整型掩码
) -> Float[Tensor, ""]:
    """
    Compute sparsity loss to encourage selectivity in the probe (4-class version).
    This loss encourages the probe to have lower average activation, preventing it from marking everything as hallucinations.
    
    For 4-class, we calculate the sum of probabilities for hallucination classes (1, 2, 3), encouraging this value to be low.
    
    Args:
        probe_logits: [B, T, 4] - Probe output logits
        attention_mask: [B, T] - Mask for valid tokens (boolean or integer)
    Returns:
        Scalar sparsity loss
    """
    # Convert to boolean mask
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    
    # Get probabilities using softmax (4-class)
    probe_probs = torch.softmax(probe_logits, dim=-1)  # [B, T, 4]
    
    # Compute sum of probabilities for hallucination classes (1, 2, 3)
    hallucination_probs = probe_probs[:, :, 1:].sum(dim=-1)  # [B, T] - Sum of probabilities for classes 1, 2, 3
    
    # Apply attention mask
    masked_probs = hallucination_probs * attention_mask.float()
    
    # Compute average activation
    num_valid_tokens = attention_mask.sum()
    if num_valid_tokens == 0:
        return torch.tensor(0.0, device=probe_logits.device)
    
    # Average sum of hallucination class probabilities for each valid token is avg_activation (63.7%)
    avg_activation = masked_probs.sum() / num_valid_tokens
    
    # Sparsity loss is the average activation
    return avg_activation


def mask_high_loss_spans(
    lm_model: Union[AutoModelForImageTextToText, PeftModel],
    # input_ids: Float[Tensor, 'batch_size seq_len'],
    # attention_mask: Float[Tensor, 'batch_size seq_len'],
    inputs: Dict[str, torch.Tensor],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    spans: List[List[Tuple[int, int]]],
    threshold: float = 1.0,  # Threshold for considering loss as high
):
    """
    For spans with high language model loss, set their labels to -100 to ignore them.
    By masking these high-loss truthful content spans, reduce their impact on training.
    """

    # Get language model logits
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

    for i in range(log_probs.shape[0]):  # Iterate through items in batch
        # Only process supported entity spans
        for start, end in spans[i]:
            if start > end: # Invalid span, skip
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
    Compute KL divergence between model with LoRA and the base model.
    
    Args:
        model: ValueHeadProbe model
        lm_logits: Logits from model with LoRA adapter
        lm_labels: Language modeling labels
    Returns:
        Scalar KL divergence loss
    """
    
    # Check if model has LoRA adapter
    if not isinstance(model.model, PeftModel) or not model.model.active_adapters:
        return torch.tensor(0.0, device=lm_logits.device)
    
    # Get output from base model without LoRA
    with model.model.disable_adapter():
        base_outputs = model(
            inputs=inputs,
            labels=lm_labels,
        )
        base_logits = base_outputs["lm_logits"].detach()
    
    # Compute KL divergence
    with torch.autocast(device_type=lm_logits.device.type, enabled=False):
        log_q = torch.log_softmax(lm_logits.float(), -1)  # model with LoRA
        p_ref = torch.softmax(base_logits.float(), -1)    # reference model
        kl = F.kl_div(log_q, p_ref, reduction='none', log_target=False).sum(-1)
        
        # Compute loss only on valid tokens
        active_mask = (lm_labels != -100)
        if not active_mask.any():
            return torch.tensor(0.0, device=lm_logits.device)
            
        kl_loss = kl[active_mask].mean()
    
    return kl_loss