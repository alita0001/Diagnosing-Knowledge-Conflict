import asyncio
import base64
import io
import uuid
import ast
import pandas as pd
from datasets import load_dataset, Dataset, Features, Value, Image
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import login

# 运行前先启动模型，进行本地部署： vllm serve /new_disk/cbl/Models/R1-Onevision-7B/ --max-model-len 16384
# 在vllm或者hal-prob环境都可以
# vllm serve /new_disk/cbl/Models/Ocean_R1_7B_Instruct/ --max-model-len 16384
# vllm serve /new_disk/lhl1/models/Llama-3.2V-11B-cot/ --max-model-len 16384
# vllm serve /new_disk/lhl1/Models/Llama-3.2V-11B-cot/ --max-model-len 16384

# ================= Configuration =================
# 1. vLLM 服务配置
VLLM_API_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
# MODEL_NAME = "/new_disk/cbl/Models/R1-Onevision-7B/" # 确保与 vllm serve 的 model name 一致
MODEL_NAME = "/new_disk/cbl/Models/Llama-3.2V-11B-cot/"

# 2. 数据集配置
SOURCE_DATASET_ID = "1Jin1/image_vs_text_problems"
# TARGET_REPO_ID = "alita01/TriConflict-inference"  # 你的目标上传地址
TARGET_REPO_ID = "1Jin1/TriConflict-inference"  # 你的目标上传地址
# TARGET_SUBSET = "R1-Onevision-7B"
TARGET_SUBSET = "Llama-3.2V-11B-cot"
TARGET_SPLIT = "image_vs_text"
# HF_TOKEN = "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"
HF_TOKEN = "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"

# 3. 并发控制 (带图片的请求显存占用较大，建议比纯文本设低一点)
MAX_CONCURRENT_REQUESTS = 32
# =================================================

# 初始化客户端
client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)
login(token=HF_TOKEN)

def encode_image_to_base64(pil_image):
    """
    将 PIL Image 对象直接转换为 Base64 字符串 (PNG格式，无损，保留RGBA)
    """
    if pil_image is None:
        return None
        
    buffered = io.BytesIO()
    # 使用 PNG 保存：
    # 1. 它是无损的，不进行压缩处理
    # 2. 它支持 RGBA，所以不需要 convert("RGB")
    pil_image.save(buffered, format="PNG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

async def get_vlm_response(sem, prompt, pil_image, model_name):
    """
    发送多模态请求 (Text + Image) 到 vLLM
    """
    async with sem:
        try:
            # 1. 准备图片 Base64
            base64_image = encode_image_to_base64(pil_image)
            
            # 2. 构建消息体
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            },
                        }
                    ],
                }
            ]

            # 3. 发送请求
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
                max_tokens=4096, # 视觉模型通常输出较长，给够空间
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

async def main():
    print(f"正在加载源数据集: {SOURCE_DATASET_ID}...")
    ds = load_dataset(SOURCE_DATASET_ID, split="phd_icc_annotations")
    
    # 用于测试，只跑前 5 条 (正式运行时请注释掉这一行)
    # ds = ds.select(range(5))
    
    print(f"待处理数据量: {len(ds)} 条")
    
    # 准备数据容器
    processed_rows = []
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print("正在预处理数据与构建任务...")
    
    # 遍历数据集，解析文本并准备任务
    # 假设源数据集的结构：image 列是图片，text 列是那个字典字符串
    # 需要根据实际列名调整，这里假设文本列名叫 'text' 或 'problem_text'
    # 先打印一下列名确认
    print(f"数据集包含列: {ds.column_names}")
    
    # # 自动探测文本列 (通常除了 image 就是文本列)
    # text_col = [c for c in ds.column_names if c != 'image'][0]

    for row in ds:
        # 1. 解析字典字符串 (使用你刚才学的 ast)
        raw_text = row['question']
        try:
            data_dict = ast.literal_eval(raw_text)
            
            # 提取关键字段
            question = data_dict.get('question', '')
            conflicting_context = data_dict.get('conflicting_context', '')
            conflict_type = data_dict.get('conflict_type', '')
            
            # 2. 构建 detailed_prompt：conflicting_context + question（用换行符分隔）
            detailed_prompt = f"{conflicting_context}\n{question}" if conflicting_context else question
            
            # 3. 获取图片对象
            image_obj = row['image']

            # 4. 存储元数据 (用于最后构建数据集)
            # 注意：这里我们存 image_obj (PIL) 是为了上传到 HF 时能预览
            # 但发给 API 时会在任务里转成 Base64
            row_data = {
                "question_id": row['question_id'],
                "question": detailed_prompt,
                "ground_truth": None,
                "task": None,
                "image_name": row.get('image_name', ''),
                'model_name': 'R1-Onevision-7B',
                'detailed_prompt': detailed_prompt,  # 保存拼接后的完整 prompt
                "image": image_obj, # 保留原始 PIL 对象
            }
            processed_rows.append(row_data)

            # 5. 添加推理任务（使用 detailed_prompt 和 image）
            tasks.append(get_vlm_response(sem, detailed_prompt, image_obj, MODEL_NAME))
            
        except Exception as e:
            print(f"解析数据行出错: {e}")
            # 出错时添加空任务占位，保持索引对齐
            tasks.append(asyncio.sleep(0, result="Error: Parse Failed"))

    print("开始多模态批量推理 (Async VLM Inference)...")
    # 并发执行
    model_outputs = await tqdm_asyncio.gather(*tasks)

    # 将结果合并回数据
    print("合并推理结果...")
    final_data = []
    for i, row_data in enumerate(processed_rows):
        row_data['model_output'] = model_outputs[i]
        final_data.append(row_data)

    # 【修改点 1】：不要创建 DataFrame，直接检查 API 错误
    # 如果想看错误数，可以遍历 list 检查
    error_count = sum(1 for row in final_data if str(row.get('model_output', '')).startswith("Error:"))
    if error_count > 0:
        print(f"警告: {error_count} 条数据推理失败。")

    print("前 2 条结果预览:")
    # 简单的预览打印
    print(f"Sample 1 Question: {final_data[0]['question'][:50]}...")
    print(f"Sample 1 Output: {final_data[0]['model_output'][:50]}...")

    # === 创建 Hugging Face 数据集 ===
    print("正在构建 HF Dataset...")
    
    features = Features({
        "question_id": Value("string"),
        "question": Value("string"),
        "image": Image(),  # 这里定义了 Image 类型
        "ground_truth": Value("string"),
        "task": Value("string"),
        "image_name": Value("string"),
        "model_name": Value("string"),
        "detailed_prompt": Value("string"),
        "model_output": Value("string")
    })

    # 【修改点 2】：使用 from_list 替代 from_pandas
    # from_list 会自动识别 features 中的 Image() 类型，并把 list 中的 PIL 对象正确编码
    new_dataset = Dataset.from_list(final_data, features=features)

    # 上传
    print(f"正在上传至 {TARGET_REPO_ID} ...")
    new_dataset.push_to_hub(TARGET_REPO_ID, TARGET_SUBSET, split=TARGET_SPLIT, private=False)
    print("Done! 上传成功！")

if __name__ == "__main__":
    asyncio.run(main())