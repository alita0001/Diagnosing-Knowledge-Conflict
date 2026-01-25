import asyncio
import pandas as pd
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import login

# 运行前先启动模型，进行本地部署： vllm serve /new_disk/cbl/Models/R1-Onevision-7B/ --max-model-len 16384
# 在vllm或者hal-prob环境都可以

# ================= Configuration =================
# 1. vLLM 服务地址 (默认是 localhost:8000)
VLLM_API_URL = "http://localhost:8000/v1"
# 2. API Key (vLLM 默认通常是 "EMPTY"，除非你设置了)
VLLM_API_KEY = "EMPTY"
# 3. 模型名称 (必须与你 vllm serve 启动时的名称完全一致，或者是 vllm logs 里显示的 served model name)
# 你启动命令里的路径通常就是模型名
# MODEL_NAME = "/new_disk/cbl/Models/R1-Onevision-7B/"
MODEL_NAME = "/new_disk/lhl1/models/Llama-3.2V-11B-cot/"

# 4. 数据集配置
# 这里填写你上一步上传的数据集 ID，或者直接加载本地处理好的
SOURCE_DATASET_ID = "alita01/TruthfulQA-Refactored-Contextual" 
# 这里填写你想上传的新数据集 ID (包含推理结果)
# TARGET_REPO_ID = "alita01/TruthfulQA-Refactored-With-Output"
TARGET_REPO_ID = "1Jin1/TriConflict-inference"  # 你的目标上传地址
# HF_TOKEN = "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"
HF_TOKEN = "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"
TARGET_SUBSET = "Llama-3.2V-11B-cot"
TARGET_SPLIT = "text_vs_prior"
# 5. 并发控制 (根据显存大小调整，7B模型通常可以设为 50-100)
MAX_CONCURRENT_REQUESTS = 64
# =================================================

# 初始化异步客户端
client = AsyncOpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_URL,
)

# 登录 Hugging Face
login(token=HF_TOKEN)

async def get_model_response(sem, text, model_name):
    """
    单个请求的异步处理函数，使用信号量控制并发
    """
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    # 如果模型对 System Prompt 敏感，可以在这里添加
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ],
                temperature=0.6, # 根据需要调整参数
                max_tokens=4096,  # 控制最大输出长度
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

async def main():
    print(f"正在加载数据集: {SOURCE_DATASET_ID}...")
    # 加载数据集
    ds = load_dataset(SOURCE_DATASET_ID, split="train")
    # 如果想测试，可以先只取前10条: ds = ds.select(range(10))
    
    # 提取 questions
    questions = ds['detailed_prompt']
    print(f"待处理数据量: {len(questions)} 条")

    # 创建信号量以控制并发数，防止在此处就把 vLLM 压垮 (虽然 vLLM 很强，但客户端也有限制)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print("开始批量推理 (Async Batch Inference)...")
    
    # 创建所有任务
    tasks = [get_model_response(sem, q, MODEL_NAME) for q in questions]
    
    # 使用 tqdm_asyncio 运行任务并显示进度条
    model_outputs = await tqdm_asyncio.gather(*tasks)

    print("推理完成，正在合并数据...")
    
    # 将结果转换为 DataFrame 并合并
    df = ds.to_pandas()
    df['model_output'] = model_outputs

    # 简单的后处理：检查是否有报错的行
    error_count = df[df['model_output'].str.startswith("Error:")].shape[0]
    if error_count > 0:
        print(f"警告: 有 {error_count} 条数据推理失败。")

    print("前 2 条结果预览:")
    print(df[['question', 'model_output']].head(2))

    # 转换为 Dataset
    new_dataset = Dataset.from_pandas(df)

    # 上传到 Hugging Face
    print(f"正在上传至 {TARGET_REPO_ID} ...")
    new_dataset.push_to_hub(TARGET_REPO_ID, TARGET_SUBSET, split=TARGET_SPLIT, private=False)
    # new_dataset.push_to_hub(TARGET_REPO_ID, private=False)
    print("上传成功！")

if __name__ == "__main__":
    # 运行异步主程序
    asyncio.run(main())