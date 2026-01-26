# Diagnosing Knowledge Conflict in Multimodal Large Language Models

This repository contains the official implementation for **"Diagnosing Knowledge Conflict in Multimodal Large Language Models"**.

We propose a **three-source knowledge framework** to systematically analyze hallucinations in Vision-Language Models (VLMs) caused by conflicts between:

1. **Visual Knowledge** — Objective facts visible in the image
2. **Textual Knowledge** — Information provided by the user in the prompt
3. **Parametric Prior Knowledge** — Knowledge encoded in the model's pretrained weights

## Overview

We identify three types of knowledge conflicts that can induce hallucinations in VLMs:

| Conflict Type | Abbreviation | Description |
|--------------|--------------|-------------|
| **Vision-Prior Conflict** | VPC | Conflicts between visual evidence and the model's parametric knowledge |
| **Text-Prior Conflict** | TPC | Conflicts between user-provided text and the model's prior knowledge |
| **Vision-Text Conflict** | VTC | Conflicts between objective visual facts and misleading textual descriptions |

Our framework includes:
- **TriConflict Benchmark**: A dataset for evaluating VLM behavior under knowledge conflicts
- **Conflict Detection Probes**: Linear probes trained to detect and classify conflict types from hidden states
- **Steering Vector Mitigation**: Methods to mitigate hallucinations using representation engineering

## Project Structure

```
Diagnosing-Knowledge-Conflict/
├── annotation_pipeline/              # Hallucination annotation pipeline
│   ├── annotate.py                  # Core annotation logic using LLM APIs
│   ├── data_models.py               # Pydantic data models for annotations
│   ├── run.py                       # Main annotation script
│   └── *.prompt                     # Prompt templates for annotation
├── prompt_writer_pipeline/           # Conflict data construction pipeline
│   ├── prompt_writer.py             # Prompt generation logic
│   ├── model_inference.py           # Model inference utilities
│   ├── data_types.py                # Data type definitions
│   └── *.prompt                     # Prompt templates for each conflict type
├── datasets_inference/               # Dataset inference scripts
│   ├── image_vs_text_inference.py   # VTC conflict inference
│   └── text_vs_prior_inference.py   # TPC conflict inference
├── utils/                            # Shared utility functions
│   ├── file_utils.py                # File I/O operations
│   ├── hooks.py                     # Model hooks for hidden state extraction
│   ├── metrics.py                   # Evaluation metrics
│   ├── model_utils.py               # Model loading utilities
│   ├── parsing.py                   # JSON parsing and validation
│   ├── string_utils.py              # String matching utilities
│   └── tokenization.py              # Tokenization helpers
├── run_steering_batch.py             # Steering vector intervention experiments
└── 4-knowledge-conflict-probes-linear/  # Probe training & mitigation (Modified from third-party)
    ├── probe/                       # Probe training and evaluation
    │   ├── train.py                 # Probe training script
    │   ├── evaluate.py              # Probe evaluation script
    │   ├── value_head_probe.py      # ValueHead probe implementation
    │   └── dataset.py               # Dataset processing
    ├── mitigation/                  # Hallucination mitigation methods
    │   ├── mitigation.py            # Token/sentence-level rejection sampling
    │   └── batch_run_mitigation.py  # Batch mitigation script
    ├── configs/                     # YAML configuration files
    └── utils/                       # Utility functions
```

## Installation

### Requirements

```bash
# Install dependencies
pip install torch>=2.0.0 transformers>=4.30.0 datasets>=2.0.0 peft>=0.5.0
pip install openai pydantic tqdm h5py scikit-learn matplotlib wandb
pip install accelerate vllm  # For local model inference
```

Or install from the probe module's requirements:
```bash
pip install -r 4-knowledge-conflict-probes-linear/requirements.txt
```

### Environment Variables

```bash
export HF_TOKEN='your_huggingface_token'
export API_KEY='your_openai_api_key'      # For annotation pipeline
export ANTHROPIC_API_KEY='your_key'       # Optional: for Claude-based annotation
```

## Usage

### 1. Conflict Data Construction

Generate conflict datasets using the prompt writer pipeline:

```bash
python -m prompt_writer_pipeline.run
```

This creates samples with:
- Vision facts extraction from images
- Conflicting context generation based on conflict type
- Complex reasoning questions requiring multi-modal understanding
- Gold reference answers for evaluation

### 2. Model Inference on Conflict Data

Run VLM inference on conflict datasets using vLLM:

```bash
# Start vLLM server
vllm serve /path/to/your/model --max-model-len 16384

# Run inference
python datasets_inference/image_vs_text_inference.py
```

### 3. Hallucination Annotation Pipeline

Use LLMs (e.g., GPT-4, Claude) to detect and annotate hallucinations in VLM outputs:

```bash
python -m annotation_pipeline.run \
    --source_dataset_name "your/source_dataset" \
    --source_subset_name "model_name" \
    --source_split_name "image_vs_text" \
    --target_dataset_name "your/target_dataset" \
    --annotator_model "gpt-4" \
    --parallel_mode True \
    --push_intermediate_every 50
```

**Features:**
- Checkpoint resumption support
- Parallel/sequential processing modes
- Automatic upload to HuggingFace Hub

### 4. Train Conflict Detection Probes

Train linear probes to detect conflict types from hidden states:

```bash
cd 4-knowledge-conflict-probes-linear
python -m probe.train --config configs/train_config_linear.yaml
```

### 5. Evaluate Probes

```bash
python -m probe.evaluate --config configs/eval_config_linear.yaml
```

### 6. Hallucination Mitigation

Apply probe-guided rejection sampling to mitigate hallucinations:

```bash
python -m mitigation.batch_run_mitigation \
    --dataset_name "your/dataset" \
    --model_name "/path/to/model" \
    --probe_path "./value_head_probes/your-probe-id" \
    --method "token_rejection" \
    --alpha 0.6 \
    --push_to_hub
```

### 7. Steering Vector Intervention

Apply steering vectors to modify model behavior:

```bash
python run_steering_batch.py \
    --model_path "/path/to/model" \
    --h5_path "/path/to/hidden_states.h5" \
    --layer_idx 22 \
    --coefficient -1.0 \
    --source_repo "your/source_dataset" \
    --target_repo "your/target_dataset"
```

## Annotation Labels

The annotation pipeline assigns the following labels:

| Label | Description |
|-------|-------------|
| `Supported` | Claim is supported by available evidence |
| `Vision-Prior Conflict` | Conflict between image and model's prior knowledge |
| `Text-Prior Conflict` | Conflict between text and model's prior knowledge |
| `Vision-Text Conflict` | Conflict between image and textual description |
| `Insufficient Information` | Cannot be verified with available information |

## Supported Models

The codebase supports various Vision-Language Models:
- LLaMA-3.2-Vision (11B)
- R1-Onevision-7B
- Ocean-R1-7B-Instruct
- Qwen2.5-VL series
- And other HuggingFace compatible VLMs

## Data Format

### Input Dataset Schema

```python
{
    "question_id": str,           # Unique identifier
    "question": str,              # User question/instruction
    "model_output": str,          # VLM generated response
    "ground_truth": str,          # Optional ground truth answer
    "image": PIL.Image,           # Input image (optional)
    "conflict_type": str,         # Type of knowledge conflict
    "detailed_prompt": str        # Full prompt with context
}
```

### Annotation Output Schema

```python
{
    "span": str,                  # Annotated text span
    "label": str,                 # Conflict label (VPC/TPC/VTC/Supported)
    "verification_note": str,     # Explanation of the annotation
    "index": int                  # Character position in text
}
```

## Evaluation Metrics

We provide comprehensive evaluation metrics:
- **Token-level**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Span-level**: Max-aggregated metrics over annotated spans
- **Sample-level**: Majority-vote based conflict type classification

## License

This project is released under the **MIT License**, with the following exceptions:

- The `4-knowledge-conflict-probes-linear/` directory is derived from modifications of third-party code released under the **Apache License 2.0**. See `4-knowledge-conflict-probes-linear/LICENSE` for details.

When using code from this repository, please ensure that all applicable licenses are complied with.

---
