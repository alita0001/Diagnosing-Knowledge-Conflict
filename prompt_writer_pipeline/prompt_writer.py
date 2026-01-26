#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Annotation Core Logic

Responsible for calling API to perform hallucination detection and annotation on multimodal content
"""

import os
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from openai import OpenAI, AsyncOpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))  # Add current directory to path to support direct import of modules in the same directory
from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from data_types import MultimodalPromptSpan

logger = logging.getLogger(__name__)

# Set the class for prompt generation
CLASS_OF_PROMPT = "image_vs_text"  # Options: image_vs_text, image_vs_prior, text_vs_prior

# Load prompt templates
if CLASS_OF_PROMPT == "image_vs_text":
    IMAGE_VS_TEXT_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "image_vs_text.prompt"
    PROMPT_TEMPLATE_PATH = IMAGE_VS_TEXT_PROMPT_TEMPLATE_PATH
elif CLASS_OF_PROMPT == "image_vs_prior":
    IMAGE_VS_PRIOR_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "image_vs_prior.prompt"
    PROMPT_TEMPLATE_PATH = IMAGE_VS_PRIOR_PROMPT_TEMPLATE_PATH
elif CLASS_OF_PROMPT == "text_vs_prior":
    TEXT_VS_PRIOR_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "text_vs_prior.prompt"
    PROMPT_TEMPLATE_PATH = TEXT_VS_PRIOR_PROMPT_TEMPLATE_PATH

PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text().strip()

# assert '{instruction}' in PROMPT_TEMPLATE and '{completion}' in PROMPT_TEMPLATE, \
#     "Prompt template must contain {instruction} and {completion} placeholders"


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 encoded string
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def format_prompt(instruction: Optional[str]) -> str:
    """Format prompt
    
    Args:
        instruction: User instruction
        groundtruth: Ground truth text
        completion: Model completed text
        
    Returns:
        Formatted prompt
    """
    if instruction is not None:
        return PROMPT_TEMPLATE.replace("{instruction}", instruction)
    else:
        return PROMPT_TEMPLATE


async def annotate_sample(
    client: OpenAI | AsyncOpenAI,
    instruction: Optional[str],
    image,
    model: str = "anthropic/claude-sonnet-4.5:online"
) -> List[MultimodalPromptSpan]:
    """Use API to perform hallucination annotation on a single sample
    
    Args:
        client: OpenAI client
        instruction: Instruction text
        image: PIL Image object
        model: Name of the model to use
        
    Returns:
        List[MultimodalPromptSpan] of annotation results
    """
    
    # Format prompt
    prompt = format_prompt(instruction)
    
    # Build message
    if image is not None:
        image_base64 = image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    
    # Call API
    if isinstance(client, AsyncOpenAI):
        response = await client.chat.completions.create(
        model=model,
        messages=messages,
        # extra_body={
        #     "provider": {
        #         "order": ["azure"],
        #         "allow_fallbacks": False  # Force use of specified provider
        #     }}
        )
    else:
        response = client.chat.completions.create(
        model=model,
        messages=messages,
        # extra_body={
        #     "provider": {
        #         "order": ["azure"],
        #         "allow_fallbacks": False  # Force use of specified provider
        #     }}
    )

    # Extract response text
    response_text = response.choices[0].message.content
    
    # Print raw response (for debugging)
    # logger.info(f" Raw API response (first 500 chars): {response_text[:500]}")
    # logger.info(f" Raw API response (full): {response_text}")

    # Parse JSON as List[MultimodalAnnotatedSpan]
    completion = parse_and_validate_json(
        response_text,
        MultimodalPromptSpan,
        allow_partial=True
    )
    
    return completion

