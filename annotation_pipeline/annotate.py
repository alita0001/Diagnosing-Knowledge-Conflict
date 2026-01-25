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
from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from .data_models import MultimodalAnnotatedSpan

logger = logging.getLogger(__name__)


# 加载提示词模板
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "multimodal_annotation.prompt"
PROMPT_TEMPLATE_PATH_WO_GT = Path(__file__).parent / "multimodal_annotation_wo_gt.prompt"
PROMPT_TEMPLATE_PATH_WO_IMAGE = Path(__file__).parent / "multimodal_annotation_wo_image.prompt"
# PROMPT_TEMPLATE_PATH_WO_IMAGE = Path(__file__).parent / "multimodal_annotation.prompt"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text().strip()
PROMPT_TEMPLATE_WO_GT = PROMPT_TEMPLATE_PATH_WO_GT.read_text().strip()
PROMPT_TEMPLATE_WO_IMAGE = PROMPT_TEMPLATE_PATH_WO_IMAGE.read_text().strip()

assert '{instruction}' in PROMPT_TEMPLATE and '{completion}' in PROMPT_TEMPLATE, \
    "提示词模板必须包含 {instruction} 和 {completion} 占位符"
assert '{instruction}' in PROMPT_TEMPLATE_WO_GT and '{completion}' in PROMPT_TEMPLATE_WO_GT, \
    "提示词模板必须包含 {instruction} 和 {completion} 占位符"


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


def format_prompt(instruction: str,groundtruth: Optional[str], completion: str, image) -> str:
    """格式化提示词
    
    Args:
        instruction: 用户指令
        groundtruth: 真实文本
        completion: 模型完成的文本
        
    Returns:
        格式化后的提示词
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
    """为标注片段分配在原文中的位置
    
    核心问题：Claude 提取的 span 可能与原文不完全匹配，需要进行模糊匹配
    
    Args:
        spans: 标注片段列表
        text: 原始文本
        min_similarity: 最小相似度阈值
        
    Returns:
        添加了位置信息的标注片段列表
    """
    results = []
    cur_idx = 0  # 当前搜索起点
    used_positions = set()  # 已使用的位置集合

    for span in spans:
        # 模糊匹配 span 在文本中的位置
        closest_match, matched_idx = try_matching_span_in_text(
            span.span,  # Claude 提取的文本
            text,  # 原始 completion
            cur_idx=cur_idx,  # 当前搜索起点
            min_similarity=min_similarity  # 最小相似度阈值
        )

        # 情况1：无法找到匹配
        if closest_match is None: # LLM 提取的文本在原文中根本不存在；LLM 对文本进行了改写或总结；LLM 产生了幻觉，提取了不存在的内容
            logger.warning(
                f"无法在文本中定位片段 {repr(span.span)} (总片段数: {len(spans)})。"
                f"\n保留该片段但移除标签和索引。"
            )
            span.label = None  # 容错处理，保留 span 但标记为无效
            span.index = None
            results.append(span)
            continue
        
        # 情况2：找到重复匹配;LLM 重复标注了同一个实体;同一个词在文本中出现多次，LLM 将它们分别标注，但模糊匹配都指向了第一个
        if matched_idx is not None and all(
            pos in used_positions
            for pos in range(matched_idx, matched_idx + len(closest_match))
        ):
            logger.warning(
                f"片段 {repr(span.span)} 匹配到与已匹配片段相同的位置 "
                f"{repr(text[matched_idx:matched_idx+len(closest_match)])}"
            )
            continue  # 跳过重复匹配

        # 情况3：成功匹配；找到匹配且位置未被使用
        if closest_match != span.span:
            logger.info(f"片段 {repr(span.span)} 匹配到 {repr(closest_match)}")

        # 更新 span 的位置信息
        span.index = matched_idx
        span.span = closest_match

        # 更新已使用的位置集合
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
    """使用 API 对单个样本进行幻觉标注
    
    Args:
        client: OpenAI 客户端
        instruction: 指令文本
        groundtruth: 真实文本
        completion_text: 模型完成的文本
        image: PIL Image 对象
        model: 使用的模型名称
        
    Returns:
        标注结果的 List[MultimodalAnnotatedSpan]
    """
    
    # 格式化提示词
    prompt = format_prompt(instruction, groundtruth, completion_text, image)
    
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

    # 解析 JSON 为 List[MultimodalAnnotatedSpan]
    annotations = parse_and_validate_json(
        response_text,
        List[MultimodalAnnotatedSpan],
        allow_partial=True
    )

    # 对齐 span 位置
    annotated_spans = assign_span_positions(annotations, completion_text)
    
    return annotated_spans

