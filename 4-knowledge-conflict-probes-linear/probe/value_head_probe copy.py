"""
已修改：num_classes = 4，已将该参数添加到配置文件

ValueHeadProbe：附加到特定层并预测幻觉的探针
四分类：真实、视觉-先验知识冲突、文本-先验知识冲突、视觉-文本冲突
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from jaxtyping import Float, Int
from torch import Tensor

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from .config import ProbeConfig

from utils.hooks import add_hooks
from utils.model_utils import (
    get_model_layers,
    get_model_hidden_size,
    setup_lora_for_layers
)
from utils.probe_loader import download_probe_from_hf

class ValueHeadProbe(nn.Module):
    """
    一个探针，它钩入语言模型的特定层，并使用线性头来预测每个token的四分类。
    
    四分类类别：
    - 类别0：真实/支持的内容
    - 类别1：视觉-先验知识冲突
    - 类别2：文本-先验知识冲突
    - 类别3：视觉-文本冲突
    
    这个探针：
    1. 附加前向钩子以捕获指定层的隐藏状态
    2. 应用线性变换来预测四个类别的logits
    3. 可以与或不与LoRA适配器一起使用
    """
    
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, PeftModel],
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        num_classes: Optional[int] = 4,
    ):
        """
        已修改：num_classes = 4
        初始化ValueHeadProbe。
        参数:
            model: 语言模型（带或不带LoRA适配器）
            layer_idx: 要附加探针的层的可选索引（如果未指定，将从之前保存的探针配置文件中推断）
            path: 加载预训练探针权重的可选路径
            num_classes: 分类类别数量（如果未指定，将从保存的配置文件中推断，或默认为1以向后兼容）
        """
        super().__init__()

        assert layer_idx or path, "Either path or layer index must be provided, otherwise we can't infer the layer where we should hook the value head to."
        
        if path:
            saved_config = json.load(open(path / "probe_config.json"))

            if layer_idx is None:
                layer_idx = saved_config["layer_idx"]
            else:
                # 我们检查如果提供了layer_idx，它必须与之前保存的配置中的匹配
                assert layer_idx == saved_config["layer_idx"]


        hidden_size = get_model_hidden_size(model)
        print(f"---hidden_size:{hidden_size}")
        model_layers = get_model_layers(model)
        print(f"---model_layers:{model_layers}")

        self.model = model
        self.layer_idx = layer_idx
        self.num_classes = num_classes
        self.target_module = model_layers[layer_idx]
        self.target_layer_name = self.target_module.__class__.__name__
        
        if not isinstance(model, PeftModel):
            print("WARNING: Model is not a PeftModel. Remember to add LoRA adapters if needed.")
        
        # 初始化值头（线性层）
        if path:
            # 如果提供了路径，加载预训练权重
            self.value_head, _ = ValueHeadProbe.load_head(
                path,
                device=model.device,
                dtype=model.dtype
            )
            # 确保 num_classes 与实际加载的模型一致
            self.num_classes = self.value_head.out_features
        else:
            self.value_head = nn.Linear(
                hidden_size, num_classes,  # 输出4个类别
                device=model.device, 
                dtype=model.dtype
            )
            print(f"WARNING: Using seed=42 for the initialization of the probe")
            torch.manual_seed(42)
            self._initialize_weights()
        
        # 初始化钩子状态
        self._hooked_hidden_states: Optional[torch.Tensor] = None
        self._hook_fn = self._get_hook_fn()
    
    def _initialize_weights(self):
        """使用小的随机值初始化值头权重"""
        with torch.no_grad():
            self.value_head.weight.data.normal_(mean=0.0, std=0.01)
            if self.value_head.bias is not None:
                self.value_head.bias.data.zero_()
    
    def _get_hook_fn(self):
        """
        前向钩子，用于捕获目标层的隐藏状态。
        `module_output` 通常是 [batch_size, seq_len, hidden_dim]。
        我们不分离它，以便梯度可以反向传播。
        """
        def hook_fn(module, module_input, module_output):
            if isinstance(module_input, tuple) and module_output[0].ndim == 3:
                self._hooked_hidden_states = module_output[0]
            else:
                self._hooked_hidden_states = module_output
        return hook_fn
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        # input_ids: Int[Tensor, 'batch_size seq_len'],
        # attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        labels: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        # pixel_values: Optional[Tensor] = None,  # 图像输入
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        已修改：num_classes = 4
        通过附加了探针的模型进行前向传播。
        参数:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 语言建模损失的可选标签
            pixel_values: 图像特征（对于某些VLM）
            **kwargs: 传递给模型的额外参数
        返回:
            包含以下内容的字典:
                - lm_logits: 语言模型输出logits [batch_size, seq_len, vocab_size]
                - probe_logits: 探针输出logits [batch_size, seq_len, num_classes]
                - num_classes=4: 类别0=真实/支持，类别1=视觉-先验知识冲突，类别2=文本-先验知识冲突，类别3=视觉-文本冲突
                - lm_loss: 语言建模损失（如果提供了标签）
        """
        # 重置钩子捕获的隐藏状态
        self._hooked_hidden_states = None
        
        # 设置钩子
        fwd_hooks = [(self.target_module, self._hook_fn)]
        
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            # 通过模型进行前向传播
            outputs = self.model(**inputs, labels=labels, **kwargs)
        
        # 检查是否成功捕获了隐藏状态
        if self._hooked_hidden_states is None:
            raise RuntimeError("Failed to capture hidden states from target layer")

        probe_logits: Float[Tensor, 'batch_size seq_len num_classes'] = \
            self.value_head(
                self._hooked_hidden_states.to(
                    device=self.value_head.weight.device,
                    dtype=self.value_head.weight.dtype
                )
            )

        return {
            "lm_logits": outputs.logits,  # [B, T, vocab_size]
            "probe_logits": probe_logits,  # [B, T, num_classes]
            "lm_loss": outputs.loss if hasattr(outputs, 'loss') else None
        }
    
    def save(self, path: Union[str, Path]):
        """
        已修改：num_classes = 4
        将探针保存到磁盘。
        参数:
            path: 保存探针的目录
        """
        os.makedirs(path, exist_ok=True)
        
        # 如果存在，保存LoRA适配器
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        
        # 保存值头权重
        torch.save(
            self.value_head.state_dict(),
            path / "probe_head.bin"
        )
        
        # 保存配置
        probe_config = {
            "target_layer_name": self.target_module.__class__.__name__,
            "layer_idx": self.layer_idx,
            "hidden_size": self.value_head.in_features,
            "num_classes": self.value_head.out_features  # 保存类别数量
        }
        with open(path / "probe_config.json", 'w') as f:
            json.dump(probe_config, f, indent=4)
        
        print(f"Probe saved to {path}")
    
    @property
    def device(self):
        """Get the device of the model."""
        """获取模型的设备"""
        return self.model.device
    
    @classmethod
    def load_head(
        cls,
        path: Path,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[nn.Module, int]:
        """
        已修改：num_classes = 4
        从给定路径加载线性值头。
        参数:
            path: 探针目录路径
            device: 加载探针头的设备
            dtype: 探针头的数据类型 
        返回:
            (probe_head, layer_idx) 元组
        """
        with open(path / "probe_config.json") as f:
            probe_config = json.load(f)

        hidden_size = probe_config['hidden_size']
        probe_layer_idx = probe_config['layer_idx']
        # 从配置读取类别数量，向后兼容旧模型（默认为4）
        num_classes = probe_config.get('num_classes', 4)

        probe_head = nn.Linear(hidden_size, num_classes, device=device, dtype=dtype)
        
        state_dict = torch.load(
            path / "probe_head.bin",
            map_location=device,
            weights_only=True
        )
        probe_head.load_state_dict(state_dict)
        
        return probe_head, probe_layer_idx


def setup_probe(
    model: PreTrainedModel,
    probe_config: ProbeConfig
) -> Tuple[PreTrainedModel, ValueHeadProbe]:
    """
    未修改
    使用给定配置设置探针。
    这处理从HuggingFace下载（如果需要）、设置LoRA适配器以及创建ValueHeadProbe实例。
    参数:
        model: 基础语言模型
        probe_config: 探针配置
    返回:
        (带LoRA的模型（如果适用），探针实例) 元组 
    """
    
    # 冻结基础模型的所有参数，否则优化器将进行全量微调
    for _, param in model.named_parameters():
        param.requires_grad = False

    # 如果需要，从HuggingFace下载
    if probe_config.load_from == 'hf' and not probe_config.probe_path.exists():
        download_probe_from_hf(
            repo_id=probe_config.hf_repo_id,
            probe_id=probe_config.probe_id
        )

    if probe_config.load_from in ['hf', 'disk']:
        # 设置LoRA适配器（如果适用）
        if (probe_config.probe_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, probe_config.probe_path)

        # 加载现有探针（layer_idx将从之前保存的配置中推断）
        probe = ValueHeadProbe(model, path=probe_config.probe_path)

    else:
        # 从头初始化探针
        if probe_config.lora_layers: 
            print(f"Initializing new LoRA adapters for layers {probe_config.lora_layers}")
            model = setup_lora_for_layers(
                model,
                probe_config.lora_layers,
                lora_r=probe_config.lora_r,
                lora_alpha=probe_config.lora_alpha,
                lora_dropout=probe_config.lora_dropout,
            )
        # lora_layers为None/[]/"none"（在YAML中使用null），则不使用LoRA适配器
        else:  
            print("No LoRA adapters specified. Training probe head only (base model frozen).")
        
        # 使用指定的层初始化新探针，从配置中读取num_classes
        probe = ValueHeadProbe(model, layer_idx=probe_config.layer, num_classes=probe_config.num_classes)
    
    return model, probe
