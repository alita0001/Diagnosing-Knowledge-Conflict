#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator


class MultimodalPromptSpan(BaseModel):
    question: str = Field(
        description="最小化的文本片段，仅包含实体本身（例如 'Sarah Chen'，而不是 'Dr. Sarah Chen from MIT'）"
    )
    conflicting_context: str = Field(
        description="与图像内容冲突的上下文文本片段"
    )
    conflict_type: Literal["image_vs_text", "image_vs_prior", "text_vs_prior"] = Field(
        description="冲突类型"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """验证和规范化标签字段"""
        if isinstance(values, dict) and 'conflict_type' in values:
            valid_labels = [
                "image_vs_text",
                "image_vs_prior",
                "text_vs_prior"
            ]
            if values['conflict_type'] not in valid_labels:
                # 设置为默认值而不是抛出错误
                values['conflict_type'] = None
        return values
    
class MultimodalDatasetItem(BaseModel):
    """多模态数据集项
    
    包含用于多模态标注的完整信息，包括问题、模型输出、图片等
    """
    model_config = {"extra": "allow"}  # 自动保留数据集中的所有额外字段
    
    # 必需字段
    question_id: Optional[str] = Field(default=None, description="问题的唯一标识符")
    question: Union[str, MultimodalPromptSpan, List[MultimodalPromptSpan]] = Field(description="用户的问题/指令")
    model_output: str = Field(description="模型生成的输出文本")
    ground_truth: Optional[str] = Field(default=None, description="ground truth 答案")
    image: Optional[Any] = Field(default=None, description="PIL Image 对象或图片数据")
    
    # 可选字段
    image_name: Optional[str] = Field(default=None, description="图片文件名")
    task: Optional[str] = Field(default=None, description="任务类型")
