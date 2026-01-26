

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
 
    
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, PeftModel],
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        num_classes: Optional[int] = 4,
        layers_to_hook: Optional[List[int]] = None,
    ):

        super().__init__()
        print(f"---layer_idx:{layer_idx}")
        print(f"---path:{path}")
        print(f"---num_classes:{num_classes}")
        print(f"---layers_to_hook:{layers_to_hook}")
        assert layer_idx is not None or path, "Either path or layer index must be provided, otherwise we can't infer the layer where we should hook the value head to."
        
        saved_config = None
        saved_layers_to_hook: Optional[List[int]] = None

        if path:
            saved_config = json.load(open(path / "probe_config.json"))

            if layer_idx is None:
                layer_idx = saved_config["layer_idx"]
            else:

                assert layer_idx == saved_config["layer_idx"]

            saved_layers_to_hook = saved_config.get("layers_to_hook")

            if saved_layers_to_hook is None:
                saved_layers_to_hook = [layer_idx]


        hidden_size = get_model_hidden_size(model)
        print(f"---hidden_size:{hidden_size}")
        model_layers = get_model_layers(model)
        print(f"---model_layers:{model_layers}")

        self.model = model
        self.model_layers = model_layers
        self.layer_idx = layer_idx
        self.num_classes = num_classes

        if layers_to_hook is None:
            if saved_layers_to_hook is not None:
                layers_to_hook = saved_layers_to_hook
            else:
                layers_to_hook = list(range(len(model_layers)))
        elif len(layers_to_hook) == 0:
            layers_to_hook = list(range(len(model_layers)))

        if saved_layers_to_hook is not None and layers_to_hook != saved_layers_to_hook:
            raise ValueError(
                f"Configured layers_to_hook {layers_to_hook} does not match saved probe {saved_layers_to_hook}"
            )
        self.layers_to_hook = sorted(set(int(idx) for idx in layers_to_hook))


        cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        fallback_layers = []
        for layer_idx in self.layers_to_hook:
            if layer_idx in cross_attention_layers and layer_idx > 0:
                fallback_idx = layer_idx - 1
                while fallback_idx >= 0 and fallback_idx in cross_attention_layers:
                    fallback_idx -= 1
                if fallback_idx >= 0:
                    fallback_layers.append(fallback_idx)

        if fallback_layers:
            self.layers_to_hook = sorted(set(self.layers_to_hook + fallback_layers))
            print(f"Added fallback layers: {fallback_layers}, total layers to hook: {self.layers_to_hook}")



        self.target_module = model_layers[self.layer_idx]
        self.target_layer_name = self.target_module.__class__.__name__
        
        if not isinstance(model, PeftModel):
            print("WARNING: Model is not a PeftModel. Remember to add LoRA adapters if needed.")
        
        self.concat_hidden_size = hidden_size * len(self.layers_to_hook)

        if path:
            self.value_head, _ = ValueHeadProbe.load_head(
                path,
                device=model.device,
                dtype=model.dtype
            )
            self.num_classes = self.value_head.out_features
        else:
            self.value_head = nn.Linear(
                self.concat_hidden_size, num_classes,  
                device=model.device, 
                dtype=model.dtype
            )
            print(f"WARNING: Using seed=42 for the initialization of the probe")
            torch.manual_seed(42)
            self._initialize_weights()
        
        self._hooked_hidden_states: Dict[int, Optional[torch.Tensor]] = {
            idx: None for idx in self.layers_to_hook
        }
        self._hook_fns = {
            idx: self._get_hook_fn(idx) for idx in self.layers_to_hook
        }
    
    def _initialize_weights(self):
        with torch.no_grad():
            self.value_head.weight.data.normal_(mean=0.0, std=0.01)
            if self.value_head.bias is not None:
                self.value_head.bias.data.zero_()
    
    def _get_hook_fn(self, layer_idx: int):
        def hook_fn(module, module_input, module_output):
            # print("----------------module_output-------------------")
            # print(module_output)
            if isinstance(module_output, tuple):
                output = module_output[0]
            else:
                output = module_output

            self._hooked_hidden_states[layer_idx] = output
        return hook_fn
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        # input_ids: Int[Tensor, 'batch_size seq_len'],
        # attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        labels: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        # pixel_values: Optional[Tensor] = None,  
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        for idx in self.layers_to_hook:
            self._hooked_hidden_states[idx] = None
        
        # Set up hooks
        fwd_hooks = [
            (self.model_layers[idx], self._hook_fns[idx])
            for idx in self.layers_to_hook
        ]
        
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            outputs = self.model(**inputs, labels=labels, **kwargs)
        
        concatenated_states: List[torch.Tensor] = []
        has_visual_input = 'pixel_values' in inputs and inputs['pixel_values'] is not None
        print("has_visual_input：",has_visual_input)
        for idx in self.layers_to_hook:
            hidden_state = self._hooked_hidden_states[idx]
            if hidden_state is None:
                cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
                if idx in cross_attention_layers and not has_visual_input:
                    fallback_found = False
                    for fallback_idx in range(idx - 1, -1, -1):  
                        if fallback_idx in self._hooked_hidden_states and self._hooked_hidden_states[fallback_idx] is not None:
                            print(f"Warning: Layer {idx} (CrossAttention) not executed due to no visual input. Using layer {fallback_idx} output instead.")
                            hidden_state = self._hooked_hidden_states[fallback_idx]
                            fallback_found = True
                            break

                    if not fallback_found:
                        print(f"Failed to capture hidden states from layer {idx} and no fallback available")
                        continue
                else:
                    raise RuntimeError(f"Failed to capture hidden states from layer {idx}")

            concatenated_states.append(hidden_state)

        concat_hidden = torch.cat(
            [
                hidden_state.to(
                    device=self.value_head.weight.device,
                    dtype=self.value_head.weight.dtype
                )
                for hidden_state in concatenated_states
            ],
            dim=-1
        )

        probe_logits: Float[Tensor, 'batch_size seq_len num_classes'] = self.value_head(concat_hidden)

        return {
            "lm_logits": outputs.logits,  # [B, T, vocab_size]
            "probe_logits": probe_logits,  # [B, T, num_classes]
            "lm_loss": outputs.loss if hasattr(outputs, 'loss') else None
        }
    
    def save(self, path: Union[str, Path]):
        os.makedirs(path, exist_ok=True)
        
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        
        torch.save(
            self.value_head.state_dict(),
            path / "probe_head.bin"
        )
        
        probe_config = {
            "target_layer_name": self.target_module.__class__.__name__,
            "layer_idx": self.layer_idx,
            "hidden_size": self.value_head.in_features,
            "num_classes": self.value_head.out_features,
            "layers_to_hook": self.layers_to_hook
        }
        with open(path / "probe_config.json", 'w') as f:
            json.dump(probe_config, f, indent=4)
        
        print(f"Probe saved to {path}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return self.model.device
    
    @classmethod
    def load_head(
        cls,
        path: Path,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[nn.Module, int]:
        """
        Load linear value head from the given path.
        Args:
            path: Probe directory path
            device: Device to load the probe head
            dtype: Data type of the probe head
        Returns:
            (probe_head, layer_idx) tuple
        """
        with open(path / "probe_config.json") as f:
            probe_config = json.load(f)

        hidden_size = probe_config['hidden_size']
        probe_layer_idx = probe_config['layer_idx']
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
    Set up the probe using the given configuration.
    This handles downloading from HuggingFace (if needed), setting up LoRA adapters, and creating a ValueHeadProbe instance.
    Args:
        model: Base language model
        probe_config: Probe configuration
    Returns:
        (Model with LoRA (if applicable), probe instance) tuple
    """
    
    # Freeze all parameters of the base model, otherwise the optimizer will perform full fine-tuning
    for _, param in model.named_parameters():
        param.requires_grad = False

    # If needed, download from HuggingFace
    if probe_config.load_from == 'hf' and not probe_config.probe_path.exists():
        download_probe_from_hf(
            repo_id=probe_config.hf_repo_id,
            probe_id=probe_config.probe_id
        )

    if probe_config.load_from in ['hf', 'disk']:
        # Set up LoRA adapters (if applicable)
        if (probe_config.probe_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, probe_config.probe_path)

        # Load existing probe (layer_idx will be inferred from the saved configuration)
        probe = ValueHeadProbe(model, path=probe_config.probe_path)

    else:
        # Initialize new probe from scratch
        if probe_config.lora_layers: 
            print(f"Initializing new LoRA adapters for layers {probe_config.lora_layers}")
            model = setup_lora_for_layers(
                model,
                probe_config.lora_layers,
                lora_r=probe_config.lora_r,
                lora_alpha=probe_config.lora_alpha,
                lora_dropout=probe_config.lora_dropout,
            )
        else:  
            print("No LoRA adapters specified. Training probe head only (base model frozen).")
        
        # Initialize new probe from scratch
        probe = ValueHeadProbe(
            model,
            layer_idx=probe_config.layer,
            num_classes=probe_config.num_classes,
            layers_to_hook=probe_config.layers_to_hook,
        )
    
    return model, probe
