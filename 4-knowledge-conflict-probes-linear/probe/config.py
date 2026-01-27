

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Literal

from utils.probe_loader import LOCAL_PROBES_DIR
from utils.model_utils import get_num_layers
from .dataset import TokenizedProbingDatasetConfig

@dataclass
class ProbeConfig:
    """Configuration for a probe model."""
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"  
    
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
    layer: Optional[int] = None  
    
    # LoRA configuration
    lora_layers: Optional[Union[List[int], str]] = "all"  
    lora_r: int = 16  
    lora_alpha: int = 32  
    lora_dropout: float = 0.05

    layers_to_hook: Optional[List[int]] = field(default_factory=list)  

    load_from: Optional[Literal['disk', 'hf']] = None  
    probe_path: Optional[Path] = None  
    hf_repo_id: Optional[str] = "andyrdt/hallucination-probes"  
    
    threshold: float = 0.5  
    num_classes: int = 4  
    
    def __post_init__(self):
        self.probe_path = LOCAL_PROBES_DIR / self.probe_id

        if self.load_from == "hf" and not self.hf_repo_id:
            raise ValueError("hf_repo_id must be specified when load_from='hf'")

        if self.load_from in ['disk'] and not self.probe_path.exists():
            raise ValueError(f"Probe with ID {self.probe_id} not found in disk at path {self.probe_path}")

        if self.layer is None:
            self.layer = get_num_layers(self.model_name) - 1

        if isinstance(self.lora_layers, str):
            if self.lora_layers == 'all':
                self.lora_layers = list(range(0, self.layer + 1))
            elif self.lora_layers.lower() == 'none':
                self.lora_layers = []
            else:
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
    
    wandb_project: str = "hallucination-probes"  
    wandb_name: Optional[str] = None  
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)  
    
    upload_to_hf: bool = False  
    save_evaluation_metrics: bool = True  
    save_roc_curves: bool = False  
    dump_raw_eval_results: bool = False  
    perform_final_eval: bool = True  
    
    # Training hyperparameters
    per_device_train_batch_size: int = 4  
    per_device_eval_batch_size: int = 4  
    high_loss_threshold: Optional[float] = None  
    lambda_lm: float = 0.0  
    lambda_kl: float = 0.0  
    anneal_max_aggr: bool = True  
    anneal_warmup: float = 1.0 
    learning_rate: float = 5e-5 
    probe_head_lr: Optional[float] = 5e-3 
    lora_lr: Optional[float] = 5e-5 
    sparsity_penalty_weight: Optional[float] = None 
    num_train_samples: Optional[int] = None 
    max_steps: int = -1 
    num_train_epochs: int = 1 
    enable_gradient_checkpointing: bool = True 
    gradient_accumulation_steps: int = 1 
    max_grad_norm: float = 1.0 
    eval_steps: Optional[int] = -1 
    evaluation_strategy: str = "no" 
    logging_steps: int = 10 
    seed: int = 42 
    
    train_datasets: List[dict] = field(default_factory=list)
    eval_datasets: List[dict] = field(default_factory=list)
    
    # These will be populated in __post_init__
    train_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False) 
    eval_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False) 
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        if self.eval_steps == -1:
            self.eval_steps = None
        
        # Handle learning rates
        if self.probe_head_lr is None:
            self.probe_head_lr = self.learning_rate
        if self.lora_lr is None:
            self.lora_lr = self.learning_rate
        
        self.train_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.train_datasets
        ]
        
        self.eval_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.eval_datasets
        ]

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
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)  
    
    datasets: List[dict] = field(default_factory=list) 
    per_device_eval_batch_size: int = 8  
    
    output_dir: Optional[Path] = None  
    save_predictions: bool = True  
    save_roc_curves: bool = True  
    save_raw_results: bool = False 
    
    dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)  # List of dataset configuration objects
    
    def __post_init__(self):
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        self.dataset_configs = [
            TokenizedProbingDatasetConfig(**dataset_config)
            for dataset_config in self.datasets
        ]

        if self.output_dir is None:
            self.output_dir = self.probe_config.probe_path / "evaluation_results"

# %%