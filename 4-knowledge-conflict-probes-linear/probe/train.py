"""
幻觉检测探针的训练脚本
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
    """主训练函数"""

    # ===== 1、如果存在，从.env加载环境变量 =====
    load_dotenv()

    if training_config.upload_to_hf:
        assert os.environ.get("HF_WRITE_TOKEN", None) is not None
    
    wandb.init(project=training_config.wandb_project, name=training_config.probe_config.probe_id)

    # ===== 2、打印训练配置 =====
    print("Training config:")
    for key, value in asdict(training_config).items():
        print(f"\t{key}: {value}")


    # ===== 3、加载模型和processor/tokenizer =====
    print(f"Loading model: {training_config.probe_config.model_name}")
    model_name = training_config.probe_config.model_name
    is_multimodal = 'vision' in model_name.lower() or 'onevision' in model_name.lower() or 'ocean_r1_7b_instruct' in model_name.lower() or 'llama-3.2v-11b-cot' in model_name.lower()
    
    if is_multimodal:
        # 使用 AutoProcessor 代替 AutoTokenizer
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
        # 原有的文本模型加载逻辑
        print("加载原来的文本模型")    
        model, tokenizer = load_model_and_tokenizer(model_name)
        # processor = None  # 非多模态模型不需要 processor

    # ===== 4、显存优化 =====
    if hasattr(model, 'config'):  # 模型对象属性
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if training_config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            # 启用梯度检查点
            # 在训练时用计算时间换显存：不保存所有中间激活，反向传播时重新计算，从而降低显存占用
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    
    # ===== 5、加载探针 =====
    print(f"Setting up probe: {training_config.probe_config.probe_id}")
    model, probe = setup_probe(model, training_config.probe_config)  # 参数是基础语言模型、返回值是带LoRA的模型

    print_trainable_parameters(probe)

    # ===== 6、加载数据集 =====
    print("Loading datasets:")
    train_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.train_dataset_configs
    ]
    eval_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.eval_dataset_configs
    ]
    
    # 连接训练数据集
    train_dataset = train_datasets[0]
    for dataset in train_datasets[1:]:
        train_dataset += dataset

    # 如果请求，打乱并将训练数据集缩减到固定数量的样本
    if training_config.num_train_samples is not None:
        total = len(train_dataset)
        num = max(0, min(int(training_config.num_train_samples), total))
        if num < total:
            g = torch.Generator()  # 创建 PyTorch 随机数生成器
            g.manual_seed(training_config.seed)  # 设置随机种子，保证结果可复现
            perm = torch.randperm(total, generator=g).tolist()
            selected_indices = perm[:num]  # 选择前 num 个样本
            train_dataset = Subset(train_dataset, selected_indices)
            print(f"Using a subset of the training dataset: {num}/{total} samples")

    # ===== 7、加载训练参数 =====
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
    
    # 向training_args添加独立的学习率
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr

    # 禁用检查点保存（在训练期间尝试保存时会出现一个奇怪的bug）
    training_args.set_save(strategy="no")

    # ===== 8、数据集批处理 =====
    collate_fn = partial(tokenized_probing_collate_fn, processor=processor)


    # ===== 9、构建训练器 =====
    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=eval_datasets,
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # 这是一个虚拟参数，用于HF基础Trainer类
        data_collator=collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    # ===== 10、意外保存 =====
    def save_model_callback():
        """将探针权重、tokenizer和训练配置保存到磁盘"""
        probe.save(training_config.probe_config.probe_path)
        tokenizer.save_pretrained(training_config.probe_config.probe_path)
        save_json(
            training_config,
            training_config.probe_config.probe_path / "training_config.json"
        )

    # 注册保存回调函数，用于意外退出时保存
    atexit.register(save_model_callback)

    # ===== 11、开始训练 =====
    print("Training...")
    trainer.train()

    # ===== 12、训练完毕保存模型 =====
    print(f"Saving model to {training_config.probe_config.probe_path}")
    save_model_callback()

    # ===== 13、最终评估 =====
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

    # ===== 14、上传到huggingface =====
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
    
    # 从YAML加载配置
    training_config = TrainingConfig(**load_yaml(args.config))
    
    main(training_config)