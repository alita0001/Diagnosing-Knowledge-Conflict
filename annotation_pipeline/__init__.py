#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态幻觉标注流程

该模块提供了对视觉语言模型输出进行幻觉检测和标注的工具。
识别模型输出中的实体和事实是否得到支持、存在何种冲突，或无法验证。
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

