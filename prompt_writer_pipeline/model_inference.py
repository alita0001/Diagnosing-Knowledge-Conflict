#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Model Inference Module

Used for generating model_output with Qwen2.5-VL and similar models.
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import Optional
from PIL import Image


class MultimodalModelInference:
    """Multimodal model inference class"""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Initialize the model.
        
        Args:
            model_id: Model path or HuggingFace model ID
            device: Device ("cuda" or "cpu")
        """
        self.device = device
        print(f"Loading processor and model: {model_id}")
        
        # Check if it's a local path
        from pathlib import Path
        model_path = Path(model_id)
        is_local_path = model_path.exists() and model_path.is_dir()
        
        # Load processor (supports both local path and HuggingFace ID)
        self.processor = AutoProcessor.from_pretrained(
            str(model_id),
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_id),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=is_local_path
        ).to(device).eval()
        
        print("Model loaded successfully!")
    
    def generate_output(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 16384,
        temperature: float = 0.9,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """
        Generate model output.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text output
        """
        # Build messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Preprocess inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )
        
        # Check generated token count
        input_length = inputs.input_ids.shape[1]
        generated_length = generated_ids.shape[1] - input_length
        print(f"Input tokens: {input_length}, Generated tokens: {generated_length}, Total: {generated_ids.shape[1]}, max_new_tokens: {max_new_tokens}")
        
        # Warning if output may be truncated
        if generated_length >= max_new_tokens:
            print(f"Warning: Generated tokens ({generated_length}) reached max limit ({max_new_tokens}), output may be truncated!")
        
        # Extract only the generated part (remove input)
        generated_ids_only = generated_ids[0][input_length:]
        generated_text = self.processor.batch_decode(
            [generated_ids_only],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        print(f"Decoded output length: {len(generated_text)} characters (generated part only)")
        
        return generated_text


# Global model instance (lazy loading)
_model_instance: Optional[MultimodalModelInference] = None


def get_model_instance(model_id: str, device: str = "cuda") -> MultimodalModelInference:
    """Get model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MultimodalModelInference(model_id, device)
    return _model_instance


def generate_model_output(
    image: Image.Image,
    question: str,
    model_id: str = "path/to/your/model",
    device: str = "cuda",
    max_new_tokens: int = 16384,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True
) -> str:
    """
    Convenience function for generating model output.
    
    Args:
        image: PIL Image object
        question: Question text
        model_id: Model path
        device: Device
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text output
    """
    model = get_model_instance(model_id, device)
    return model.generate_output(
        image, 
        question, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
