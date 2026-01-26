#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Logic for Multimodal Annotation

Responsible for calling API to detect and annotate hallucinations in multimodal content
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
from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from .data_models import MultimodalAnnotatedSpan

logger = logging.getLogger(__name__)


# Load prompt template
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "multimodal_annotation.prompt"
PROMPT_TEMPLATE_PATH_WO_GT = Path(__file__).parent / "multimodal_annotation_wo_gt.prompt"
PROMPT_TEMPLATE_PATH_WO_IMAGE = Path(__file__).parent / "multimodal_annotation_wo_image.prompt"
# PROMPT_TEMPLATE_PATH_WO_IMAGE = Path(__file__).parent / "multimodal_annotation.prompt"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text().strip()
PROMPT_TEMPLATE_WO_GT = PROMPT_TEMPLATE_PATH_WO_GT.read_text().strip()
PROMPT_TEMPLATE_WO_IMAGE = PROMPT_TEMPLATE_PATH_WO_IMAGE.read_text().strip()

assert '{instruction}' in PROMPT_TEMPLATE and '{completion}' in PROMPT_TEMPLATE, \
    "Prompt template must contain {instruction} and {completion} placeholders"
assert '{instruction}' in PROMPT_TEMPLATE_WO_GT and '{completion}' in PROMPT_TEMPLATE_WO_GT, \
    "Prompt template must contain {instruction} and {completion} placeholders"


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 encoded string
    
    Args:
        image: PIL Image object
        
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def format_prompt(instruction: str,groundtruth: Optional[str], completion: str, image) -> str:
    """Format prompt
    
    Args:
        instruction: User instruction
        groundtruth: Ground truth text
        completion: Model completion text
        
    Returns:
        Formatted prompt
    """
    if groundtruth is not None and image is not None:
        return PROMPT_TEMPLATE.replace("{instruction}", instruction).replace("{groundtruth}", groundtruth).replace("{completion}", completion)
    elif groundtruth is not None and image is None:
        return PROMPT_TEMPLATE_WO_IMAGE.replace("{instruction}", instruction).replace("{completion}", completion).replace("{groundtruth}", groundtruth)
    else:
        return PROMPT_TEMPLATE_WO_GT.replace("{instruction}", instruction).replace("{completion}", completion)

def assign_span_positions(
    spans: List[MultimodalAnnotatedSpan],
    text: str,
    min_similarity: float = 0.8
) -> List[MultimodalAnnotatedSpan]:
    """Assign positions for annotated spans in original text
    
    Core issue: Spans extracted by Claude might not perfectly match the original text, fuzzy matching is needed
    
    Args:
        spans: List of annotated spans
        text: Original text
        min_similarity: Minimum similarity threshold
        
    Returns:
        List of annotated spans with position information
    """
    results = []
    cur_idx = 0  # Current search start index
    used_positions = set()  # Set of used positions

    for span in spans:
        # Fuzzy match span position in text
        closest_match, matched_idx = try_matching_span_in_text(
            span.span,  # Text extracted by Claude
            text,  # Original completion
            cur_idx=cur_idx,  # Current search start index
            min_similarity=min_similarity  # Minimum similarity threshold
        )

        # Case 1: No match found
        if closest_match is None: # LLM extracted text does not exist in original; LLM rewrote or summarized; LLM hallucinated, extracted non-existent content
            logger.warning(
                f"Cannot locate span {repr(span.span)} in text (total spans: {len(spans)})."
                f"\nKeep the span but remove label and index."
            )
            span.label = None  # Fault tolerance, keep span but mark as invalid
            span.index = None
            results.append(span)
            continue
        
        # Case 2: Duplicate match found; LLM annotated same entity twice; Same word appears multiple times, LLM annotated them separately but fuzzy match points to first one
        if matched_idx is not None and all(
            pos in used_positions
            for pos in range(matched_idx, matched_idx + len(closest_match))
        ):
            logger.warning(
                f"Span {repr(span.span)} matched to same position as already matched span "
                f"{repr(text[matched_idx:matched_idx+len(closest_match)])}"
            )
            continue  # Skip duplicate match

        # Case 3: Match found; Match found and position not used
        if closest_match != span.span:
            logger.info(f"Span {repr(span.span)} matched to {repr(closest_match)}")

        # Update span position info
        span.index = matched_idx
        span.span = closest_match

        # Update used positions set
        results.append(span)
        used_positions.update(range(matched_idx, matched_idx + len(closest_match)))
        cur_idx = max(cur_idx, matched_idx + len(closest_match))
    
    return results


async def annotate_sample(
    client: OpenAI | AsyncOpenAI,
    instruction: str,
    groundtruth: str,
    completion_text: str,
    image,
    model: str = "anthropic/claude-sonnet-4.5:online"
) -> List[MultimodalAnnotatedSpan]:
    """Annotate single sample for hallucination using API
    
    Args:
        client: OpenAI Client
        instruction: Instruction text
        groundtruth: Ground truth text
        completion_text: Model completion text
        image: PIL Image object
        model: Model name to use
        
    Returns:
        List[MultimodalAnnotatedSpan] of annotation results
    """
    
    # Format prompt
    prompt = format_prompt(instruction, groundtruth, completion_text, image)
    
    # Build messages
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
        #         "allow_fallbacks": False  # Force specific provider
        #     }}
        )
    else:
        response = client.chat.completions.create(
        model=model,
        messages=messages,
        # extra_body={
        #     "provider": {
        #         "order": ["azure"],
        #         "allow_fallbacks": False  # Force specific provider
        #     }}
    )

    # Extract response text
    response_text = response.choices[0].message.content

    # Parse JSON to List[MultimodalAnnotatedSpan]
    annotations = parse_and_validate_json(
        response_text,
        List[MultimodalAnnotatedSpan],
        allow_partial=True
    )

    # Align span positions
    annotated_spans = assign_span_positions(annotations, completion_text)
    
    return annotated_spans

