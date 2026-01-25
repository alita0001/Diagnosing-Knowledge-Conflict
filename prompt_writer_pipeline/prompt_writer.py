#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态标注核心逻辑

负责调用 API 对多模态内容进行幻觉检测和标注
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
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到路径，支持直接导入同目录模块
from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from data_types import MultimodalPromptSpan

logger = logging.getLogger(__name__)

# 设置要进行提示词生成的类别
CLASS_OF_PROMPT = "image_vs_text"  # 可选值：image_vs_text, image_vs_prior, text_vs_prior

# 加载提示词模板
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
#     "提示词模板必须包含 {instruction} 和 {completion} 占位符"


def image_to_base64(image) -> str:
    """将 PIL Image 转换为 base64 编码字符串
    
    Args:
        image: PIL Image 对象
        
    Returns:
        base64 编码的字符串
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def format_prompt(instruction: Optional[str]) -> str:
    """格式化提示词
    
    Args:
        instruction: 用户指令
        groundtruth: 真实文本
        completion: 模型完成的文本
        
    Returns:
        格式化后的提示词
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
    """使用 API 对单个样本进行幻觉标注
    
    Args:
        client: OpenAI 客户端
        instruction: 指令文本
        image: PIL Image 对象
        model: 使用的模型名称
        
    Returns:
        标注结果的 List[MultimodalPromptSpan]
    """
    
    # 格式化提示词
    prompt = format_prompt(instruction)
    
    # 构建消息
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
    
    # 调用 API
    if isinstance(client, AsyncOpenAI):
        response = await client.chat.completions.create(
        model=model,
        messages=messages,
        # extra_body={
        #     "provider": {
        #         "order": ["azure"],
        #         "allow_fallbacks": False  # 强制使用指定供应商
        #     }}
        )
    else:
        response = client.chat.completions.create(
        model=model,
        messages=messages,
        # extra_body={
        #     "provider": {
        #         "order": ["azure"],
        #         "allow_fallbacks": False  # 强制使用指定供应商
        #     }}
    )

    # 提取响应文本
    response_text = response.choices[0].message.content
    
    # 打印原始响应（用于调试）
    # logger.info(f" API 返回的原始响应（前500字符）: {response_text[:500]}")
    # logger.info(f" API 返回的原始响应（完整）: {response_text}")

    # 解析 JSON 为 List[MultimodalAnnotatedSpan]
    completion = parse_and_validate_json(
        response_text,
        MultimodalPromptSpan,
        allow_partial=True
    )
    
    return completion

