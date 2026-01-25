"""
整理 LongFact-Inference 数据集
- 添加 question_id 字段
- 将 user_question 重命名为 question
- 上传到新的 HuggingFace 仓库
"""

from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import random

# ============================================================================
# 配置部分
# ============================================================================

# 原始数据集信息
ORIGINAL_DATASET = "1Jin1/LongFact-Inference"
ORIGINAL_SUBSET = "R1-Onevision-7B"

# 新数据集信息
NEW_DATASET_NAME = "alita01/LongFact-Inference-Processed"  # 修改为你的用户名
NEW_SUBSET_NAME = "R1-Onevision-7B"

# 是否使用私有仓库
PRIVATE_REPO = False

# ============================================================================
# 主要处理函数
# ============================================================================

def add_question_id(example, idx, split_name, subset_name):
    """
    为每个样本添加 question_id
    
    格式: {subset}_{split}_{index}
    例如: concepts_train_1, concepts_validation_1
    """
    question_id = f"{subset_name}_{split_name}_{idx}"
    return {
        'question_id': question_id,
        'question': example['user_question'],  # 重命名字段
        'model_output': example['model_output'],
        'model_name': example['model_name'],
        'subset': example['subset']
    }

def process_dataset():
    """处理数据集的主函数"""
    
    print("=" * 60)
    print("开始处理 LongFact-Inference 数据集")
    print("=" * 60)
    
    # # 1. 登录 HuggingFace
    # print("\n步骤 1: 登录 HuggingFace")
    # print("请输入你的 HuggingFace Token:")
    # login()  # 会提示输入 token
    
    # 2. 加载原始数据集
    print(f"\n步骤 2: 加载原始数据集 {ORIGINAL_DATASET}")
    dataset = load_dataset(ORIGINAL_DATASET, ORIGINAL_SUBSET)
    
    # 查看数据集信息
    print("\n原始数据集信息:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} 个样本")
    
    print(f"\n原始字段: {dataset['train'].column_names}")
    
    # 3. 处理每个 split
    print("\n步骤 3: 处理数据集...")
    processed_dataset = DatasetDict()
    
    for split_name, split_data in dataset.items():
        print(f"\n处理 {split_name} split...")
        
        # 为每个样本添加 question_id 并重命名字段
        processed_split = split_data.map(
            lambda example, idx: add_question_id(
                example, 
                idx, 
                split_name, 
                example['subset']
            ),
            with_indices=True,
            remove_columns=['user_question']  # 移除原字段
        )
        
        processed_dataset[split_name] = processed_split
        print(f"  完成! 处理了 {len(processed_split)} 个样本")
    
    # 4. 验证处理结果
    print("\n步骤 4: 验证处理结果")
    print("\n新数据集信息:")
    for split_name, split_data in processed_dataset.items():
        print(f"  {split_name}: {len(split_data)} 个样本")
    
    print(f"\n新字段: {processed_dataset['train'].column_names}")
    
    # 显示示例
    print("\n示例数据 (train split 的第一条):")
    example = processed_dataset['train'][0]
    print(f"  question_id: {example['question_id']}")
    print(f"  question: {example['question'][:100]}...")
    print(f"  model_name: {example['model_name']}")
    print(f"  subset: {example['subset']}")
    print(f"  model_output length: {len(example['model_output'])} 字符")
    
    # 5. 上传到 HuggingFace
    print(f"\n步骤 5: 上传到 HuggingFace...")
    print(f"目标仓库: {NEW_DATASET_NAME}")
    print(f"配置名称: {NEW_SUBSET_NAME}")
    
    try:
        processed_dataset.push_to_hub(
            NEW_DATASET_NAME,
            config_name=NEW_SUBSET_NAME,
            private=PRIVATE_REPO
        )
        print("\n✅ 数据集上传成功!")
        print(f"访问地址: https://huggingface.co/datasets/{NEW_DATASET_NAME}")
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print("提示: 请确保你的 HuggingFace Token 有写入权限")
        return None
    
    return processed_dataset

# ============================================================================
# 额外功能: 保存到本地
# ============================================================================

def save_to_local(dataset, output_dir="./processed_longfact_dataset"):
    """
    可选: 将处理后的数据集保存到本地
    """
    print(f"\n保存数据集到本地: {output_dir}")
    dataset.save_to_disk(output_dir)
    print(f"✅ 已保存到 {output_dir}")
    
    # 也可以保存为 JSONL 格式
    for split_name, split_data in dataset.items():
        jsonl_path = f"{output_dir}/{split_name}.jsonl"
        split_data.to_json(jsonl_path)
        print(f"✅ 已保存 {split_name} 到 {jsonl_path}")

# ============================================================================
# 额外功能: 数据统计
# ============================================================================

def print_statistics(dataset):
    """
    打印数据集的统计信息
    """
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  总样本数: {len(split_data)}")
        
        # 统计每个 subset 的数量
        subsets = {}
        for example in split_data:
            subset = example['subset']
            subsets[subset] = subsets.get(subset, 0) + 1
        
        print(f"  Subset 分布:")
        for subset, count in subsets.items():
            print(f"    {subset}: {count} 个样本")
        
        # 统计问题长度
        question_lengths = [len(example['question']) for example in split_data]
        avg_length = sum(question_lengths) / len(question_lengths)
        print(f"  平均问题长度: {avg_length:.0f} 字符")
        print(f"  最短问题: {min(question_lengths)} 字符")
        print(f"  最长问题: {max(question_lengths)} 字符")

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 处理数据集
    processed_dataset = process_dataset()
    
    if processed_dataset is not None:
        # 打印统计信息
        print_statistics(processed_dataset)
        
        # 可选: 保存到本地
        save_choice = input("\n是否要将数据集保存到本地? (y/n): ")
        if save_choice.lower() == 'y':
            save_to_local(processed_dataset)
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"\n你可以使用以下代码加载新数据集:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{NEW_DATASET_NAME}', '{NEW_SUBSET_NAME}')")
        print(f"```")