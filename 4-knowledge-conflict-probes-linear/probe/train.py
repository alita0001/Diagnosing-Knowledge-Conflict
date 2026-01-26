"""
Training script for hallucination detection probe
"""

import json
import atexit
from pathlib import Path
from typing import List
from dataclasses import asdict
import argparse
from functools import partial

import torch
import wandb
from torch.utils.data import Subset
from transformers import TrainingArguments, AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv

from utils.file_utils import save_jsonl, save_json, load_yaml
from utils.model_utils import load_model_and_tokenizer, print_trainable_parameters
from utils.probe_loader import upload_probe_to_hf

from .dataset import TokenizedProbingDataset, create_probing_dataset, tokenized_probing_collate_fn
from .config import TrainingConfig
from .value_head_probe import setup_probe
from .trainer import ProbeTrainer


def main(training_config: TrainingConfig):
    """Main training function"""

    # ===== 1. Load environment variables from .env if exists =====
    load_dotenv()

    if training_config.upload_to_hf:
        assert os.environ.get("HF_WRITE_TOKEN", None) is not None
    
    wandb.init(project=training_config.wandb_project, name=training_config.probe_config.probe_id)

    # ===== 2. Print training config =====
    print("Training config:")
    for key, value in asdict(training_config).items():
        print(f"\t{key}: {value}")


    # ===== 3. Load model and processor/tokenizer =====
    print(f"Loading model: {training_config.probe_config.model_name}")
    model_name = training_config.probe_config.model_name
    is_multimodal = 'vision' in model_name.lower() or 'onevision' in model_name.lower() or 'ocean_r1_7b_instruct' in model_name.lower() or 'llama-3.2v-11b-cot' in model_name.lower()
    
    if is_multimodal:
        # Use AutoProcessor instead of AutoTokenizer
        print(f"Loading multimodal model: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            trust_remote_code=True
        )
        tokenizer = processor
    else:
        # Original text model loading logic
        print("Loading original text model")    
        model, tokenizer = load_model_and_tokenizer(model_name)
        # processor = None  # Non-multimodal models do not need processor

    # ===== 4. VRAM Optimization =====
    if hasattr(model, 'config'):  # Model object attributes
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if training_config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            # Enable gradient checkpointing
            # Trade compute for VRAM during training: recompute intermediate activations during backward pass instead of saving them
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    
    # ===== 5. Load Probe =====
    print(f"Setting up probe: {training_config.probe_config.probe_id}")
    model, probe = setup_probe(model, training_config.probe_config)  # Args are base language model, return value is model with LoRA

    print_trainable_parameters(probe)

    # ===== 6. Load Datasets =====
    print("Loading datasets:")
    train_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.train_dataset_configs
    ]
    eval_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.eval_dataset_configs
    ]
    
    # Concatenate training datasets
    train_dataset = train_datasets[0]
    for dataset in train_datasets[1:]:
        train_dataset += dataset

    # If requested, shuffle and reduce training dataset to fixed number of samples
    if training_config.num_train_samples is not None:
        total = len(train_dataset)
        num = max(0, min(int(training_config.num_train_samples), total))
        if num < total:
            g = torch.Generator()  # Create PyTorch random number generator
            g.manual_seed(training_config.seed)  # Set random seed for reproducibility
            perm = torch.randperm(total, generator=g).tolist()
            selected_indices = perm[:num]  # Select first num samples
            train_dataset = Subset(train_dataset, selected_indices)
            print(f"Using a subset of the training dataset: {num}/{total} samples")

    # ===== 7. Load Training Arguments =====
    training_args = TrainingArguments(
        output_dir=str(training_config.probe_config.probe_path),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        max_steps=training_config.max_steps,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        remove_unused_columns=False,
        label_names=["classification_labels", "lm_labels"],
        report_to="wandb",
        run_name=training_config.probe_config.probe_id,
        eval_strategy="steps" if training_config.eval_steps else "no",
        logging_first_step=True,
        logging_strategy="steps",
        max_grad_norm=training_config.max_grad_norm,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
    )
    
    # Add separate learning rates to training_args
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr

    # Disable checkpoint saving (causes a weird bug when attempting to save during training)
    training_args.set_save(strategy="no")

    # ===== 8. Dataset Batching =====
    collate_fn = partial(tokenized_probing_collate_fn, processor=processor)


    # ===== 9. Build Trainer =====
    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=eval_datasets,
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # This is a dummy argument for HF base Trainer class
        data_collator=collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    # ===== 10. Emergency Save =====
    def save_model_callback():
        """Save probe weights, tokenizer and training config to disk"""
        probe.save(training_config.probe_config.probe_path)
        tokenizer.save_pretrained(training_config.probe_config.probe_path)
        save_json(
            training_config,
            training_config.probe_config.probe_path / "training_config.json"
        )

    # Register save callback for emergency exit
    atexit.register(save_model_callback)

    # ===== 11. Start Training =====
    print("Training...")
    trainer.train()

    # ===== 12. Save model after training =====
    print(f"Saving model to {training_config.probe_config.probe_path}")
    save_model_callback()

    # ===== 13. Final Evaluation =====
    if training_config.perform_final_eval:
        eval_metrics = trainer.evaluate(
            save_roc_curves=training_config.save_roc_curves,
            dump_raw_eval_results=training_config.dump_raw_eval_results,
            verbose=True,
        )

        if training_config.save_evaluation_metrics:
            save_json(
                eval_metrics,
                training_config.probe_config.probe_path / "evaluation_results.json"
            )
    else:
        print("Skipping final evaluation as requested.")
        eval_metrics = None

    wandb.finish()

    # ===== 14. Upload to HuggingFace =====
    if training_config.upload_to_hf:
        print(f"Uploading probe to HuggingFace Hub...")
        upload_probe_to_hf(
            repo_id=training_config.probe_config.hf_repo_id,
            probe_id=training_config.probe_config.probe_id,
            token=os.environ.get("HF_WRITE_TOKEN"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hallucination detection probe")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config from YAML
    training_config = TrainingConfig(**load_yaml(args.config))
    
    main(training_config)