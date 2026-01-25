#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态模型推理模块

用于使用 Qwen2.5-VL 等模型生成新的 model_output
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import Optional
from PIL import Image


class MultimodalModelInference:
    """多模态模型推理类"""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        """
        初始化模型
        
        Args:
            model_id: 模型路径或 HuggingFace 模型 ID
            device: 设备 ("cuda" 或 "cpu")
        """
        self.device = device
        print(f"🔄 正在加载处理器和模型: {model_id}")
        
        # 检查是否是本地路径
        from pathlib import Path
        model_path = Path(model_id)
        is_local_path = model_path.exists() and model_path.is_dir()
        
        # 加载处理器（本地路径和 HuggingFace ID 都支持）
        self.processor = AutoProcessor.from_pretrained(
            str(model_id),  # 确保转换为字符串
            trust_remote_code=True,
            local_files_only=is_local_path  # 如果是本地路径，强制只使用本地文件
        )
        
        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_id),  # 确保转换为字符串
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=is_local_path  # 如果是本地路径，强制只使用本地文件
        ).to(device).eval()
        
        print("✅ 模型加载完成！")
    
    def generate_output(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 16384,  # 增加到 16384，确保输出完整
        temperature: float = 0.9,  # 温度参数：越大越随机（0.1-2.0）
        top_p: float = 0.95,  # nucleus sampling：只从概率总和前95%的token中选择
        do_sample: bool = True  # 启用采样，增加随机性
    ) -> str:
        """
        生成模型输出
        
        Args:
            image: PIL Image 对象
            question: 问题文本
            max_new_tokens: 最大生成 token 数
            
        Returns:
            生成的文本输出
        """
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 预处理输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,  # 启用采样，增加随机性和幻觉概率
                temperature=temperature if do_sample else None,  # 温度参数：越大越随机
                top_p=top_p if do_sample else None,  # nucleus sampling
            )
        
        # 检查生成的 token 数量
        input_length = inputs.input_ids.shape[1]
        generated_length = generated_ids.shape[1] - input_length
        print(f" 输入 tokens: {input_length}, 生成 tokens: {generated_length}, 总计: {generated_ids.shape[1]}, max_new_tokens: {max_new_tokens}")
        
        # 如果生成的 token 数达到 max_new_tokens，可能被截断
        if generated_length >= max_new_tokens:
            print(f" 警告：生成的 token 数 ({generated_length}) 已达到最大限制 ({max_new_tokens})，输出可能被截断！请增加 max_new_tokens")
        
        # 只提取生成的部分（移除输入），只返回模型的思考和答案
        # 方法：直接从 token IDs 中提取生成的部分
        generated_ids_only = generated_ids[0][input_length:]
        generated_text = self.processor.batch_decode(
            [generated_ids_only],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        print(f" 解码后的输出长度: {len(generated_text)} 字符（仅生成部分，不含用户输入）")
        
        return generated_text  # 只返回模型生成的部分（思考过程和答案）


# 全局模型实例（懒加载）
_model_instance: Optional[MultimodalModelInference] = None


def get_model_instance(model_id: str, device: str = "cuda") -> MultimodalModelInference:
    """获取模型实例（单例模式）"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MultimodalModelInference(model_id, device)
    return _model_instance


def generate_model_output(
    image: Image.Image,
    question: str,
    model_id: str = "/new_disk/lhl1/models/R1-Onevision-7B",
    device: str = "cuda",
    max_new_tokens: int = 16384,  # 增加到 16384，确保输出完整
    temperature: float = 0.9,  # 温度参数：越大越随机，增加幻觉概率
    top_p: float = 0.95,  # nucleus sampling
    do_sample: bool = True  # 启用采样
) -> str:
    """
    生成模型输出的便捷函数
    
    Args:
        image: PIL Image 对象
        question: 问题文本
        model_id: 模型路径
        device: 设备
        max_new_tokens: 最大生成 token 数
        
    Returns:
        生成的文本输出
    """
    model = get_model_instance(model_id, device)
    return model.generate_output(
        image, 
        question, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )

