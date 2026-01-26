"""探针训练相关的配置类"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Literal

from utils.probe_loader import LOCAL_PROBES_DIR
from utils.model_utils import get_num_layers
from .dataset import TokenizedProbingDatasetConfig

@dataclass
class ProbeConfig:
    """Configuration for a probe model."""
    """探针模型配置类"""
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"  # 探针ID标识符
    
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 基础模型名称
    layer: Optional[int] = None  # Which layer to attach the probe to  # 探针附加到的层索引
    
    # LoRA configuration
    # LoRA配置参数
    lora_layers: Optional[Union[List[int], str]] = "all"  # Which layers to apply LoRA to  # 应用LoRA的层列表
    lora_r: int = 16  # LoRA rank  # LoRA秩
    lora_alpha: int = 32  # LoRA alpha scaling  # LoRA alpha缩放因子
    lora_dropout: float = 0.05  # LoRA dropout  # LoRA dropout率

    layers_to_hook: Optional[List[int]] = field(default_factory=list)  # 指定要截取隐藏状态的层（空列表=全部层）

    # Loading configuration
    # 加载配置
    load_from: Optional[Literal['disk', 'hf']] = None  # "disk", "hf", or None  # 从何处加载：本地磁盘或HuggingFace
    probe_path: Optional[Path] = None  # Local path for disk loading  # 本地磁盘加载路径
    hf_repo_id: Optional[str] = "andyrdt/hallucination-probes"  # HuggingFace repository ID  # HuggingFace仓库ID
    
    threshold: float = 0.5  # Classification threshold  # 分类阈值
    num_classes: int = 4  # Number of classes for classification  # 分类类别数量
    
    def __post_init__(self):
        """Validate configuration."""
        """验证并初始化配置参数"""
        self.probe_path = LOCAL_PROBES_DIR / self.probe_id

        if self.load_from == "hf" and not self.hf_repo_id:
            raise ValueError("hf_repo_id must be specified when load_from='hf'")

        if self.load_from in ['disk'] and not self.probe_path.exists():
            raise ValueError(f"Probe with ID {self.probe_id} not found in disk at path {self.probe_path}")

        if self.layer is None:
            # default to hooking the value head at the last layer of the underlying LM
            # 默认在基础语言模型的最后一层附加探针
            self.layer = get_num_layers(self.model_name) - 1

        if isinstance(self.lora_layers, str):
            if self.lora_layers == 'all':
                # default to training LoRA adaptors on all layers
                # (up to the layer where we hook the value head)
                # 默认在所有层（直到探针附加层）上训练LoRA适配器
                self.lora_layers = list(range(0, self.layer + 1))
            elif self.lora_layers.lower() == 'none':
                self.lora_layers = []
            else:
                # 解析字符串格式的层列表
                self.lora_layers = [int(layer) for layer in self.lora_layers.strip('[').strip(']').split(",")]
                assert len(self.lora_layers) > 0
        elif self.lora_layers is None:
            self.lora_layers = []
        assert all(isinstance(l, int) for l in self.lora_layers)

        if self.layers_to_hook is None:
            self.layers_to_hook = []
        else:
            self.layers_to_hook = [int(layer) for layer in self.layers_to_hook]


@dataclass
class TrainingConfig:
    """Configuration for probe training."""
    """探针训练配置类"""
    
    wandb_project: str = "hallucination-probes"  # Weights & Biases项目名称
    wandb_name: Optional[str] = None  # Weights & Biases运行名称
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)  # 探针配置
    
    upload_to_hf: bool = False  # 是否上传到HuggingFace
    save_evaluation_metrics: bool = True  # 是否保存评估指标
    save_roc_curves: bool = False  # 是否保存ROC曲线
    dump_raw_eval_results: bool = False  # 是否保存原始评估结果
    perform_final_eval: bool = True  # 是否在训练结束后进行最终评估
    
    # Training hyperparameters
    # 训练超参数
    per_device_train_batch_size: int = 4  # 每个设备的训练批次大小
    per_device_eval_batch_size: int = 4  # 每个设备的评估批次大小
    high_loss_threshold: Optional[float] = None  # Threshold for masking high-loss tokens  # 高损失token的掩码阈值
    lambda_lm: float = 0.0  # Weight for language modeling loss regularization  # 语言建模损失正则化权重
    lambda_kl: float = 0.0  # Weight for KL divergence regularization  # KL散度正则化权重
    anneal_max_aggr: bool = True  # Whether to anneal span-level max aggregation loss  # 是否对span级别最大聚合损失进行退火
    anneal_warmup: float = 1.0  # Fraction of training for span loss warmup  # span损失预热的训练比例
    learning_rate: float = 5e-5  # Overall learning rate (deprecated)  # 总体学习率（已弃用）
    probe_head_lr: Optional[float] = 5e-3 # Separate LR for probe head  # 探针头的独立学习率
    lora_lr: Optional[float] = 5e-5  # Separate LR for LoRA parameters  # LoRA参数的独立学习率
    sparsity_penalty_weight: Optional[float] = None  # 稀疏性惩罚权重
    num_train_samples: Optional[int] = None  # Limit training samples  # 限制训练样本数量
    max_steps: int = -1  # Override num_epochs if set  # 如果设置则覆盖训练轮数
    num_train_epochs: int = 1  # 训练轮数
    enable_gradient_checkpointing: bool = True  # 是否启用梯度检查点
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    max_grad_norm: float = 1.0  # 梯度裁剪的最大范数
    eval_steps: Optional[int] = -1  # Only manually evaluate at the end  # 仅在最后手动评估
    evaluation_strategy: str = "no"  # "steps", "epoch", or "no"  # 评估策略
    logging_steps: int = 10  # 日志记录步数间隔
    seed: int = 42  # 随机种子
    
    # Dataset configuration
    # 数据集配置
    train_datasets: List[dict] = field(default_factory=list)  # 训练数据集列表
    eval_datasets: List[dict] = field(default_factory=list)  # 评估数据集列表
    
    # These will be populated in __post_init__
    # 这些字段将在__post_init__中填充
    train_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)  # 训练数据集配置对象列表
    eval_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)  # 评估数据集配置对象列表
    
    def __post_init__(self):
        """Post-initialization processing."""
        """后初始化处理"""
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        # Handle special values
        # 处理特殊值
        if self.eval_steps == -1:
            self.eval_steps = None
        
        # Handle learning rates
        # 处理学习率
        if self.probe_head_lr is None:
            self.probe_head_lr = self.learning_rate
        if self.lora_lr is None:
            self.lora_lr = self.learning_rate
        
        # Parse dataset configurations
        # 解析数据集配置
        self.train_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.train_datasets
        ]
        
        self.eval_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.eval_datasets
        ]

        # Convert scientific notation strings to floats
        # otherwise yaml parses e.g. '1e-6' as a string instead of a float
        # 将科学计数法字符串转换为浮点数（YAML可能将'1e-6'解析为字符串而非浮点数）
        float_fields = [
            'learning_rate', 'probe_head_lr', 'lora_lr', 'max_grad_norm', 
            'anneal_warmup', 'lambda_lm', 'lambda_kl', 'sparsity_penalty_weight'
        ]
        for field_name in float_fields:
            value = getattr(self, field_name)
            if value is not None and isinstance(value, str):
                setattr(self, field_name, float(value))


@dataclass
class EvaluationConfig:
    """Configuration for probe evaluation."""
    """探针评估配置类"""
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)  # 探针配置
    
    datasets: List[dict] = field(default_factory=list)  # 评估数据集列表
    per_device_eval_batch_size: int = 8  # 每个设备的评估批次大小
    
    output_dir: Optional[Path] = None  # 输出目录
    save_predictions: bool = True  # 是否保存预测结果
    save_roc_curves: bool = True  # 是否保存ROC曲线
    save_raw_results: bool = False  # Save all predictions and labels  # 是否保存所有预测和标签的原始结果
    
    # This will be populated in __post_init__
    # 该字段将在__post_init__中填充
    dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)  # 数据集配置对象列表
    
    def __post_init__(self):
        """Post-initialization processing."""
        """后初始化处理"""
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        # 解析数据集配置
        self.dataset_configs = [
            TokenizedProbingDatasetConfig(**dataset_config)
            for dataset_config in self.datasets
        ]

        # 如果未指定输出目录，使用默认路径
        if self.output_dir is None:
            self.output_dir = self.probe_config.probe_path / "evaluation_results"

# %%