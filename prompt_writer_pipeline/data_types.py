#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator


class MultimodalPromptSpan(BaseModel):
    question: str = Field(
        description="Minimal text span, containing only the entity itself (e.g. 'Sarah Chen', not 'Dr. Sarah Chen from MIT')"
    )
    conflicting_context: str = Field(
        description="Context text span conflicting with image content"
    )
    conflict_type: Literal["image_vs_text", "image_vs_prior", "text_vs_prior"] = Field(
        description="Conflict type"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """Validate and normalize label fields"""
        if isinstance(values, dict) and 'conflict_type' in values:
            valid_labels = [
                "image_vs_text",
                "image_vs_prior",
                "text_vs_prior"
            ]
            if values['conflict_type'] not in valid_labels:
                # Set to default value instead of raising error
                values['conflict_type'] = None
        return values
    
class MultimodalDatasetItem(BaseModel):
    """Multimodal Dataset Item
    
    Contains complete information for multimodal annotation, including questions, model outputs, images, etc.
    """
    model_config = {"extra": "allow"}  # Automatically preserve all extra fields in the dataset
    
    # Required Fields
    question_id: Optional[str] = Field(default=None, description="Unique identifier for the question")
    question: Union[str, MultimodalPromptSpan, List[MultimodalPromptSpan]] = Field(description="User question/instruction")
    model_output: str = Field(description="Model generated output text")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth answer")
    image: Optional[Any] = Field(default=None, description="PIL Image object or image data")
    
    # Optional Fields
    image_name: Optional[str] = Field(default=None, description="Image filename")
    task: Optional[str] = Field(default=None, description="Task type")
