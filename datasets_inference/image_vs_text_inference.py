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

# Run before execution to start the model locally: vllm serve path/to/your/model --max-model-len 16384

# ================= Configuration =================
# 1. vLLM Service Configuration
VLLM_API_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "path/to/your/model"  # Ensure consistency with vllm serve model name

# 2. Dataset Configuration
SOURCE_DATASET_ID = "anonymous/image_vs_text_problems"  # Replace with your dataset
TARGET_REPO_ID = "anonymous/TriConflict-inference"  # Your target upload address
TARGET_SUBSET = "model-subset"
TARGET_SPLIT = "image_vs_text"
HF_TOKEN = None  # Set via environment variable or replace with your token

# 3. Concurrency Control (Requests with images consume more VRAM, suggest lower setting than text-only)
MAX_CONCURRENT_REQUESTS = 32
# =================================================

# Initialize Client
client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_URL)
if HF_TOKEN:
    login(token=HF_TOKEN)

    """
    Convert PIL Image object directly to Base64 string (PNG format, lossless, retains RGBA)
    """
    if pil_image is None:
        return None
        
    buffered = io.BytesIO()
    # Use PNG for saving:
    # 1. It is lossless, no compression artifacts
    # 2. It supports RGBA, so no need for convert("RGB")
    pil_image.save(buffered, format="PNG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

async def get_vlm_response(sem, prompt, pil_image, model_name):
    """
    Send multimodal request (Text + Image) to vLLM
    """
    async with sem:
        try:
            # 1. Prepare Image Base64
            base64_image = encode_image_to_base64(pil_image)
            
            # 2. Construct Message Body
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

            # 3. Send Request
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
                max_tokens=4096, # Visual models typically output longer text, allocate enough space
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

async def main():
    print(f"Loading source dataset: {SOURCE_DATASET_ID}...")
    ds = load_dataset(SOURCE_DATASET_ID, split="phd_icc_annotations")
    
    # For testing, only run first 5 items (comment out this line for production run)
    # ds = ds.select(range(5))
    
    print(f"Pending data count: {len(ds)} items")
    
    # Prepare data container
    processed_rows = []
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print("Preprocessing data and building tasks...")
    
    # Iterate through dataset, parse text and prepare tasks
    # Assuming source dataset structure: image column is image, text column is the dictionary string
    # Adjust according to actual column names, assuming text column name is 'text' or 'problem_text'
    # Print column names first to confirm
    print(f"Dataset contains columns: {ds.column_names}")
    
    # # Auto-detect text column (usually the one that is not 'image')
    # text_col = [c for c in ds.column_names if c != 'image'][0]

    for row in ds:
        # 1. Parse dictionary string (using ast as learned)
        raw_text = row['question']
        try:
            data_dict = ast.literal_eval(raw_text)
            
            # Extract key fields
            question = data_dict.get('question', '')
            conflicting_context = data_dict.get('conflicting_context', '')
            conflict_type = data_dict.get('conflict_type', '')
            
            # 2. Construct detailed_prompt: conflicting_context + question (delimited by newline)
            detailed_prompt = f"{conflicting_context}\n{question}" if conflicting_context else question
            
            # 3. Get Image object
            image_obj = row['image']

            # 4. Store metadata (for building final dataset)
            # Note: We store image_obj (PIL) here for preview when uploading to HF
            # But it will be converted to Base64 in the task for API
            row_data = {
                "question_id": row['question_id'],
                "question": detailed_prompt,
                "ground_truth": None,
                "task": None,
                "image_name": row.get('image_name', ''),
                'model_name': 'R1-Onevision-7B',
                'detailed_prompt': detailed_prompt,  # Save concatenated full prompt
                "image": image_obj, # Retain original PIL object
            }
            processed_rows.append(row_data)

            # 5. Add inference task (using detailed_prompt and image)
            tasks.append(get_vlm_response(sem, detailed_prompt, image_obj, MODEL_NAME))
            
        except Exception as e:
            print(f"Error parsing data row: {e}")
            # Add empty task placeholder on error to keep index aligned
            tasks.append(asyncio.sleep(0, result="Error: Parse Failed"))

    print("Starting Multimodal Async VLM Inference...")
    # Concurrent execution
    model_outputs = await tqdm_asyncio.gather(*tasks)

    # Merge results back into data
    print("Merging inference results...")
    final_data = []
    for i, row_data in enumerate(processed_rows):
        row_data['model_output'] = model_outputs[i]
        final_data.append(row_data)

    # [Modification 1]: Do not create DataFrame, check API errors directly
    # If you want to count errors, iterate through list and check
    error_count = sum(1 for row in final_data if str(row.get('model_output', '')).startswith("Error:"))
    if error_count > 0:
        print(f"Warning: {error_count} items failed inference.")

    print("Preview of first 2 results:")
    # Simple preview print
    print(f"Sample 1 Question: {final_data[0]['question'][:50]}...")
    print(f"Sample 1 Output: {final_data[0]['model_output'][:50]}...")

    # === Create Hugging Face Dataset ===
    print("Building HF Dataset...")
    
    features = Features({
        "question_id": Value("string"),
        "question": Value("string"),
        "image": Image(),  # Image type defined here
        "ground_truth": Value("string"),
        "task": Value("string"),
        "image_name": Value("string"),
        "model_name": Value("string"),
        "detailed_prompt": Value("string"),
        "model_output": Value("string")
    })

    # [Modification 2]: Use from_list instead of from_pandas
    # from_list automatically recognizes Image() type in features and correctly encodes PIL objects from list
    new_dataset = Dataset.from_list(final_data, features=features)

    # Upload
    print(f"Uploading to {TARGET_REPO_ID} ...")
    new_dataset.push_to_hub(TARGET_REPO_ID, TARGET_SUBSET, split=TARGET_SPLIT, private=False)
    print("Done! Upload successful!")

if __name__ == "__main__":
    asyncio.run(main())