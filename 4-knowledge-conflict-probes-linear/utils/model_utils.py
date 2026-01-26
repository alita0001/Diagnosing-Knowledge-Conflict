"""Model loading and setup utilities."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig, get_peft_model  # LoRA参数高效微调库
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, AutoProcessor, AutoModelForImageTextToText


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model_and_processor(
    model_name: str,
    device_map: Optional[Union[str, dict]] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    加载视觉语言模型（VLM），支持图像+文本输入  chj
    模型：AutoModelForImageTextToText（多模态模型）
    处理器：AutoProcessor（包含 tokenizer 和 image_processor）
    特点：
        processor.tokenizer.padding_side = 'left'（左侧填充）
        适用于编码器-解码器架构的多模态模型
    """
    # Set default dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True, # 允许执行远程代码
    )
    
    # Load tokenizer 分词器加载
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True, 
    )
    processor.tokenizer.padding_side = 'left'  # ？
    
    return model, processor


def load_model_and_tokenizer(
    model_name: str,
    device_map: Optional[Union[str, dict]] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.
    加载纯文本语言模型
    
    Args:
        model_name: Name or path of the model
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        device_map: Device mapping for model parallelism
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model, tokenizer)
        模型：AutoModelForCausalLM（因果语言模型）
        分词器：AutoTokenizer（仅文本分词器）
    """
    # Set default dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'  # 因果模型从左往右生成，右填充不会影响已生成的内容的注意力计算
    )
    
    # Set padding token if not set 如果未设置填充token，则使用结束token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_model_with_lora(
    model: AutoModelForCausalLM,
    lora_config: dict,
    lora_weights_path: Optional[str] = None,
) -> PeftModel:
    """
    LoRA适配器设置工具
    
    功能：
    1. 创建LoRA配置
    2. 应用到模型
    3. 支持预训练权重加载
    Setup a model with LoRA adapters.
    
    Args:
        model: Base model
        lora_config: LoRA configuration dictionary
        lora_weights_path: Optional path to pre-trained LoRA weights
        
    Returns:
        Model with LoRA adapters
    """
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),  # LoRA秩 - 控制适应能力
        lora_alpha=lora_config.get("alpha", 32),  # 缩放因子 - 控制适应速度 
        lora_dropout=lora_config.get("dropout", 0.05),  # 丢弃率 - 控制适应稳定性，防止过拟合
        bias="none",  # 偏置配置 - 控制适应效果
        task_type="CAUSAL_LM",  # 任务类型 - 因果语言模型
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),  # 目标模块 - 控制适应位置
    )
    
    # Apply LoRA to model
    if lora_weights_path:
        # Load pre-trained LoRA weights
        model = PeftModel.from_pretrained(model, lora_weights_path)
    else:
        # Initialize new LoRA adapters
        model = get_peft_model(model, peft_config)
    
    return model


def get_model_layers(model: PreTrainedModel) -> List[nn.Module]:
    """
    关键功能：统一不同模型架构的层访问接口 chj
    
    挑战：不同模型架构的层存储位置不同：
    - LLaMA: model.layers
    - GPT-2: transformer.h  
    - BERT: encoder.layer
    - GPT-NeoX: gpt_neox.layers
    Get the list of transformer layers from a model.
    
    Args:
        model: The transformer model
    
    Returns:
        List of layer modules
    """
    # Handle PeftModel by getting the base model
    if isinstance(model, PeftModel):   
        base_model = model.get_base_model()  # 获取被 PeftModel 包装的原始模型
    else:
        base_model = model  #  如果不是 PeftModel，直接使用
    
    # Common patterns for accessing layers in different model architectures
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        # LLaMA, Mistral, etc.
        return list(base_model.model.layers)
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model') and hasattr(base_model.model.language_model, 'layers'):
        # LLaMA, Mistral, etc.
        return list(base_model.model.language_model.layers)
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        # GPT-2, GPT-J, etc.
        return list(base_model.transformer.h)
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        # BERT, RoBERTa, etc.
        return list(base_model.encoder.layer)
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        # GPT-NeoX
        return list(base_model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")


def get_num_layers(model_or_name: Union[str, PreTrainedModel]) -> int:
    """
    获取模型层数 - 支持字符串名称和模型实例两种方式 chj
    
    设计优势：
    1. 字符串查表：无需加载模型，速度快
    2. 实例计算：支持自定义模型，通用性强
    Get the number of transformer layers in a model.
    
    Args:
        model_or_name: Either a model name string or a transformer model
    
    Returns:
        Number of layers
    """
    # If it's a string (model name), use the predefined mapping
    if isinstance(model_or_name, str):
        model_layers_map = {
            # LLaMA系列
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 80,
            "meta-llama/Meta-Llama-3.1-405B-Instruct": 126,
            "meta-llama/Llama-3.3-70B-Instruct": 80,
            "Llama-3.2V-11B-cot":40,
            # Gemini系列
            "google/gemma-2-2b-it": 26,
            "google/gemma-2-9b-it": 42,
            "google/gemma-2-27b-it": 46,
            # Qwen系列
            "Qwen/Qwen2.5-0.5B-Instruct": 24,
            "Qwen/Qwen2.5-1.5B-Instruct": 28,
            "Qwen/Qwen2.5-3B-Instruct": 36,
            "Qwen/Qwen2.5-7B-Instruct": 28,
            "Qwen/Qwen2.5-14B-Instruct": 48,
            "Qwen/Qwen2.5-32B-Instruct": 64,
            "Qwen/Qwen2.5-VL-7B": 28,
            "Fancy-MLLM/R1-Onevision-7B":28,
            "/new_disk/lhl1/models/R1-Onevision-7B":28,
            "/data2/lhl1/Models/R1-Onevision-7B":28,
            "/new_disk/cbl/Models/R1-Onevision-7B":28,
            "/data2/lhl1/Models/Ocean_R1_7B_Instruct":28,
            "/new_disk/cbl/Models/Ocean_R1_7B_Instruct":28,
            # Mistral系列
            "mistralai/Mistral-Small-24B-Instruct-2501": 40,
        }
        if model_or_name in model_layers_map:
            return model_layers_map[model_or_name]
        else:
            raise ValueError(f"Model {model_or_name} not supported. Please add it to the model_layers_map.")
    
    # If it's a model instance, count the layers
    return len(get_model_layers(model_or_name))


def get_model_layers_prefix(model: PreTrainedModel) -> str:
    """
    获取模型层的访问路径前缀
    
    用途：用于LoRA target_modules的路径构建
    例如："model.layers.0.self_attn.q_proj"
    Get the prefix path to the model layers.
    
    Args:
        model: The transformer model
    
    Returns:
        String prefix for accessing layers (e.g., "model.layers")
    """
    # Handle PeftModel
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    
    # Check for common model architectures
    # Most modern models (LLaMA, Qwen, etc.) use model.layers structure
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        return "model.layers"
    # Check for vision-language models (Qwen2.5-VL, Mllama, etc.) that use language_model.layers
    elif hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'layers'):
        return "language_model.layers"
    # Check for model.language_model.layers structure (nested)
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model') and hasattr(base_model.model.language_model, 'layers'):
        return "model.language_model.layers"
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        return "transformer.h"
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        return "encoder.layer"
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        return "gpt_neox.layers"
    else:
        # 提供更详细的错误信息，帮助调试
        # 检查是否有 language_model 或 model 属性
        debug_info = f"Model type: {type(base_model)}\n"
        debug_info += f"Has 'model' attribute: {hasattr(base_model, 'model')}\n"
        if hasattr(base_model, 'model'):
            debug_info += f"  'model' type: {type(base_model.model)}\n"
            debug_info += f"  'model' has 'layers': {hasattr(base_model.model, 'layers')}\n"
            debug_info += f"  'model' has 'language_model': {hasattr(base_model.model, 'language_model')}\n"
        debug_info += f"Has 'language_model' attribute: {hasattr(base_model, 'language_model')}\n"
        if hasattr(base_model, 'language_model'):
            debug_info += f"  'language_model' type: {type(base_model.language_model)}\n"
            debug_info += f"  'language_model' has 'layers': {hasattr(base_model.language_model, 'layers')}\n"
        raise ValueError(
            f"Unknown model architecture: {type(base_model)}\n"
            f"{debug_info}\n"
            f"All attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')]}\n"
            f"Please check if the model has 'model.layers', 'language_model.layers', 'transformer.h', 'encoder.layer', or 'gpt_neox.layers'"
        )


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """
    获取模型隐藏层维度 - 探测器初始化的关键参数
    
    重要性：探测器线性层需要知道输入维度
    例如：LLaMA-3.1-8B的hidden_size=4096
    Get the hidden size of a transformer model.
    
    Args:
        model: The transformer model
        
    Returns:
        Hidden size of the model
    """
    # Handle PeftModel
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    
    if hasattr(base_model, 'config'):
        config = base_model.config
        # Try common attribute names
        for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim']:
            if hasattr(config, attr):
                return getattr(config, attr)

        # Special handling for multimodal models like Mllama
        # Mllama has text_config and vision_config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            return config.text_config.hidden_size
    
    # If we can't find it in config, try to infer from the model structure
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        return base_model.model.embed_tokens.weight.shape[1]
    
    raise ValueError(f"Could not determine hidden size for model type {type(base_model)}")


def setup_lora_for_layers(
    model: PreTrainedModel,
    layer_indices: List[int],  # 目标层索引列表
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> Union[PeftModel, PreTrainedModel]:
    """
    Setup LoRA adapters for specific layers in a model.
    
    Args:
        model: Base model to apply LoRA to
        layer_indices: List of layer indices to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        bias: Bias configuration for LoRA
        
    Returns:
        Model with LoRA adapters applied (or original model if no layers specified)
    """
    if not layer_indices:
        print("No LoRA layers specified, returning base model")
        return model
    
    # Get the layer prefix for this model architecture
    layer_prefix = get_model_layers_prefix(model)
    
    # Build target modules list for the specified layers
    target_modules = []
    module_suffixes = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    ]
    
    for layer_idx in layer_indices:
        for module_suffix in module_suffixes:
            target_modules.append(f"{layer_prefix}.{layer_idx}.{module_suffix}")
    
    print(f"Creating LoRA adapters for layers {layer_indices}...")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    return get_peft_model(model, lora_config)


def print_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    训练参数分析工具 - 用于调试和监控
    功能：
    1. 列出所有可训练参数
    2. 计算参数总数和占比
    3. 显示参数分布情况
    Print information about trainable parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = 0
    total_params = 0
    
    print("Parameters that will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  - {name}: shape {param.shape}, device {param.device}")
        total_params += param.numel()
    
    trainable_params_percentage = 100 * trainable_params / total_params
    print(f"\nTotal trainable parameters: {trainable_params:,} ({trainable_params_percentage:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    return trainable_params, total_params