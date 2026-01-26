import asyncio
import pandas as pd
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import login

# Before running, start the model with local deployment: vllm serve path/to/your/model --max-model-len 16384

# ================= Configuration =================
# 1. vLLM service address (default is localhost:8000)
VLLM_API_URL = "http://localhost:8000/v1"
# 2. API Key (vLLM default is usually "EMPTY" unless you set it)
VLLM_API_KEY = "EMPTY"
# 3. Model name (must match the name used in vllm serve)
MODEL_NAME = "path/to/your/model"

# 4. Dataset configuration
SOURCE_DATASET_ID = "anonymous/source-dataset"  # Replace with your dataset
TARGET_REPO_ID = "anonymous/TriConflict-inference"
HF_TOKEN = None  # Set via environment variable or replace with your token
TARGET_SUBSET = "model-subset"
TARGET_SPLIT = "text_vs_prior"
# 5. Concurrency control (adjust based on GPU memory, 7B model can usually be set to 50-100)
MAX_CONCURRENT_REQUESTS = 64
# =================================================

# Initialize async client
client = AsyncOpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_URL,
)

# Login to Hugging Face
if HF_TOKEN:
    login(token=HF_TOKEN)

async def get_model_response(sem, text, model_name):
    """
    Async function for handling a single request, using semaphore for concurrency control
    """
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": text}
                ],
                temperature=0.6,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

async def main():
    print(f"Loading dataset: {SOURCE_DATASET_ID}...")
    # Load dataset
    ds = load_dataset(SOURCE_DATASET_ID, split="train")
    # For testing, can select first 10: ds = ds.select(range(10))
    
    # Extract questions
    questions = ds['detailed_prompt']
    print(f"Total samples to process: {len(questions)}")

    # Create semaphore to control concurrency
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print("Starting batch inference (Async Batch Inference)...")
    
    # Create all tasks
    tasks = [get_model_response(sem, q, MODEL_NAME) for q in questions]
    
    # Run tasks with tqdm_asyncio progress bar
    model_outputs = await tqdm_asyncio.gather(*tasks)

    print("Inference complete, merging data...")
    
    # Convert results to DataFrame and merge
    df = ds.to_pandas()
    df['model_output'] = model_outputs

    # Simple post-processing: check for error rows
    error_count = df[df['model_output'].str.startswith("Error:")].shape[0]
    if error_count > 0:
        print(f"Warning: {error_count} samples failed inference.")

    print("Preview of first 2 results:")
    print(df[['question', 'model_output']].head(2))

    # Convert to Dataset
    new_dataset = Dataset.from_pandas(df)

    # Upload to Hugging Face
    print(f"Uploading to {TARGET_REPO_ID} ...")
    new_dataset.push_to_hub(TARGET_REPO_ID, TARGET_SUBSET, split=TARGET_SPLIT, private=False)
    print("Upload successful!")

if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
