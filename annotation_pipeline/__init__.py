#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Hallucination Annotation Pipeline

This module provides tools for detecting and annotating hallucinations in vision-language model outputs.
Identifies whether entities and facts in model output are supported, what conflicts exist, or if they are unverifiable.
"""

from .annotate import annotate_sample, assign_span_positions
from .run import main, PipelineConfig
from .data_models import MultimodalAnnotatedSpan, MultimodalDatasetItem

__all__ = [
    "annotate_sample",
    "assign_span_positions",
    "main",
    "PipelineConfig",
    "MultimodalAnnotatedSpan",
    "MultimodalDatasetItem",
]

