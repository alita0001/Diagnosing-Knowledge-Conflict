#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态标注数据模型

定义用于多模态幻觉标注的数据结构
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator


class MultimodalAnnotatedSpan(BaseModel):
    """多模态标注片段
    
    表示一个经过验证的实体或事实片段，包含其在多模态上下文中的验证状态
    """
    span: str = Field(
        description="最小化的文本片段，仅包含实体本身（例如 'Sarah Chen'，而不是 'Dr. Sarah Chen from MIT'）"
    )
    label: Optional[Literal[
        "Supported",
        "Vision-Prior Conflict",
        "Text-Prior Conflict",
        "Vision-Text Conflict",
        "Insufficient Information"
    ]] = Field(
        description="标注标签：实体/事实是否得到支持、存在何种冲突，或无法验证"
    )
    verification_note: str = Field(
        description="验证结果的简要说明，包括冲突类型（如适用）"
    )
    index: Optional[int] = Field(
        default=None,
        description="片段在完成文本中的起始位置索引"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """验证和规范化标签字段"""
        if isinstance(values, dict) and 'label' in values:
            valid_labels = [
                "Supported",
                "Vision-Prior Conflict",
                "Text-Prior Conflict",
                "Vision-Text Conflict",
                "Insufficient Information"
            ]
            if values['label'] not in valid_labels:
                # 设置为默认值而不是抛出错误
                values['label'] = None
        return values


class MultimodalDatasetItem(BaseModel):
    """多模态数据集项
    
    包含用于多模态标注的完整信息，包括问题、模型输出、图片等
    """
    model_config = {"extra": "allow"}  # 自动保留数据集中的所有额外字段
    
    # 必需字段
    question_id: Optional[str] = Field(default=None, description="问题的唯一标识符")
    question: str = Field(description="用户的问题/指令")
    model_output: str = Field(description="模型生成的输出文本")
    ground_truth: Optional[str] = Field(default=None, description="ground truth 答案")
    image: Optional[Any] = Field(default=None, description="PIL Image 对象或图片数据")
    
    # 可选字段
    image_name: Optional[str] = Field(default=None, description="图片文件名")
    task: Optional[str] = Field(default=None, description="任务类型")
    
    # 标注管道添加的字段
    hallucination_annotation: Optional[List[MultimodalAnnotatedSpan]] = Field(
        default=None,
        description="幻觉标注,每个元素为 MultimodalAnnotatedSpan 对象"
    )
    annotator_model: Optional[str] = Field(
        default=None,
        description="执行标注的模型名称"
    )
