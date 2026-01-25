# Diagnosing Knowledge Conflict in Multimodal Large Language Models

This repository contains the official implementation for diagnosing and mitigating knowledge conflicts in Vision-Language Models (VLMs). We adopt a **three-source knowledge perspective** to systematically analyze hallucinations caused by conflicts between:

1. **Visual Knowledge** - Objective facts visible in the image
2. **Textual Knowledge** - Information provided by the user in the prompt  
3. **Parametric Prior Knowledge** - Knowledge encoded in the model's pretrained weights

## Overview

We identify three types of knowledge conflicts that can induce hallucinations in VLMs:

| Conflict Type | Description |
|--------------|-------------|
| **Image vs Text** | Conflicts between objective visual facts and misleading textual descriptions |
| **Image vs Prior** | Conflicts between visual evidence and the model's parametric knowledge |
| **Text vs Prior** | Conflicts between user-provided text and the model's prior knowledge |

## Project Structure

```
Diagnosing-Knowledge-Conflict/
├── annotation_pipeline/          # Hallucination annotation pipeline
│   ├── annotate.py              # Core annotation logic using LLM APIs
│   ├── data_models.py           # Pydantic data models for annotations
│   ├── run.py                   # Main annotation script
│   └── *.prompt                 # Prompt templates for annotation
├── prompt_writer_pipeline/       # Conflict data construction pipeline
│   ├── prompt_writer.py         # Prompt generation logic
│   ├── model_inference.py       # Model inference utilities
│   ├── data_types.py            # Data type definitions
│   └── *.prompt                 # Prompt templates for each conflict type
├── datasets_inference/           # Dataset inference scripts
│   ├── image_vs_text_inference.py   # Image-Text conflict inference
│   └── text_vs_prior_inference.py   # Text-Prior conflict inference
├── utils/                        # Utility functions
│   ├── file_utils.py            # File I/O operations
│   ├── hooks.py                 # Model hooks for hidden state extraction
│   ├── metrics.py               # Evaluation metrics (Accuracy, F1, AUC, etc.)
│   ├── model_utils.py           # Model loading utilities
│   ├── parsing.py               # JSON parsing and validation
│   ├── probe_loader.py          # Probe model loading
│   ├── string_utils.py          # String matching utilities
│   └── tokenization.py          # Tokenization helpers
└── run_steering_batch.py         # Steering vector intervention experiments
```

## Installation

### Requirements

```bash
pip install torch transformers datasets huggingface_hub openai pydantic tqdm h5py scikit-learn matplotlib
```

For local model inference with vLLM:
```bash
pip install vllm
```

### Environment Variables

```bash
export HF_TOKEN='your_huggingface_token'
export API_KEY='your_openai_api_key'  # For annotation pipeline
```

## Usage

### 1. Hallucination Annotation Pipeline

The annotation pipeline uses LLMs (e.g., GPT-4, Claude) to detect and annotate hallucinations in VLM outputs.

```bash
python -m annotation_pipeline.run \
    --source_dataset_name "your/source_dataset" \
    --source_subset_name "model_name" \
    --source_split_name "image_vs_text" \
    --target_dataset_name "your/target_dataset" \
    --target_subset_name "model_name" \
    --target_split_name "image_vs_text" \
    --annotator_model "gpt-4" \
    --parallel_mode True \
    --push_intermediate_every 50
```

**Features:**
- Supports checkpoint resumption
- Parallel/sequential processing modes
- Automatic upload to HuggingFace Hub
- Multi-modal annotation with image support

### 2. Conflict Data Construction

Generate conflict datasets using the prompt writer pipeline:

```bash
python -m prompt_writer_pipeline.run
```

The pipeline creates samples with:
- **Vision facts extraction** from images
- **Conflicting context generation** based on conflict type
- **Complex reasoning questions** that require multi-modal understanding
- **Gold reference answers** for evaluation

### 3. Model Inference

Run VLM inference on conflict datasets using vLLM:

```bash
# Start vLLM server first
vllm serve /path/to/your/model --max-model-len 16384

# Run inference
python datasets_inference/image_vs_text_inference.py
```

### 4. Steering Vector Intervention

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

**Parameters:**
- `--layer_idx`: Target layer for steering intervention
- `--coefficient`: Steering strength (positive/negative)
- `--h5_path`: Path to pre-computed hidden states

## Annotation Labels

The annotation pipeline assigns the following labels to detected spans:

| Label | Description |
|-------|-------------|
| `Supported` | Claim is supported by available evidence |
| `Vision-Prior Conflict` | Conflict between image and model's prior knowledge |
| `Text-Prior Conflict` | Conflict between text and model's prior knowledge |
| `Vision-Text Conflict` | Conflict between image and textual description |
| `Insufficient Information` | Cannot be verified with available information |

## Evaluation Metrics

The `utils/metrics.py` module provides comprehensive evaluation:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Ranking Metrics**: AUC-ROC, Recall@0.1FPR
- **Threshold Optimization**: Automatic optimal threshold finding
- **Visualization**: ROC curves, threshold analysis plots

## Data Format

### Input Dataset Schema

```python
{
    "question_id": str,           # Unique identifier
    "question": str,              # User question/instruction
    "model_output": str,          # VLM generated response
    "ground_truth": str,          # Optional ground truth answer
    "image": PIL.Image,           # Input image (optional)
    "image_name": str,            # Image filename
    "detailed_prompt": str,       # Full prompt with context
    "conflict_type": str          # Type of knowledge conflict
}
```

### Annotation Output Schema

```python
{
    "span": str,                  # Annotated text span
    "label": str,                 # Conflict label
    "verification_note": str,     # Explanation of the annotation
    "index": int                  # Character position in text
}
```

## Supported Models

The codebase supports various Vision-Language Models:

- LLaMA-3.2-Vision (11B)
- R1-Onevision-7B
- Ocean-R1-7B-Instruct
- And other HuggingFace compatible VLMs

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{knowledge-conflict-vlm,
  title={Diagnosing Knowledge Conflict in Multimodal Large Language Models},
  author={Anonymous},
  booktitle={ICML},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

This work builds upon the HuggingFace Transformers library and leverages vLLM for efficient model inference.
