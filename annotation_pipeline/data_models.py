#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Annotation Data Models

Define data structures for multimodal hallucination annotation
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator


class MultimodalAnnotatedSpan(BaseModel):
    """Multimodal Annotated Span
    
    Represents a verified entity or fact span, containing its verification status in multimodal context
    """
    span: str = Field(
        description="Minimized text span, containing only the entity itself (e.g. 'Sarah Chen', not 'Dr. Sarah Chen from MIT')"
    )
    label: Optional[Literal[
        "Supported",
        "Vision-Prior Conflict",
        "Text-Prior Conflict",
        "Vision-Text Conflict",
        "Insufficient Information"
    ]] = Field(
        description="Annotation label: whether entity/fact is supported, what conflict exists, or unverifiable"
    )
    verification_note: str = Field(
        description="Brief explanation of verification result, including conflict type (if applicable)"
    )
    index: Optional[int] = Field(
        default=None,
        description="Start position index of the span in completion text"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """Validate and normalize label field"""
        if isinstance(values, dict) and 'label' in values:
            valid_labels = [
                "Supported",
                "Vision-Prior Conflict",
                "Text-Prior Conflict",
                "Vision-Text Conflict",
                "Insufficient Information"
            ]
            if values['label'] not in valid_labels:
                # Set to default value instead of raising error
                values['label'] = None
        return values


class MultimodalDatasetItem(BaseModel):
    """Multimodal Dataset Item
    
    Contains complete information for multimodal annotation, including question, model output, image, etc.
    """
    model_config = {"extra": "allow"}  # Automatically preserve all extra fields in dataset
    
    # Required fields
    question_id: Optional[str] = Field(default=None, description="Unique identifier for question")
    question: str = Field(description="User question/instruction")
    model_output: str = Field(description="Model generated output text")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth answer")
    image: Optional[Any] = Field(default=None, description="PIL Image object or image data")
    
    # Optional fields
    image_name: Optional[str] = Field(default=None, description="Image filename")
    task: Optional[str] = Field(default=None, description="Task type")
    
    # Fields added by annotation pipeline
    hallucination_annotation: Optional[List[MultimodalAnnotatedSpan]] = Field(
        default=None,
        description="Hallucination annotation, each element is a MultimodalAnnotatedSpan object"
    )
    annotator_model: Optional[str] = Field(
        default=None,
        description="Name of the model performing annotation"
    )
