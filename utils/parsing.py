"""Utilities for parsing and validating JSON from LLM responses."""

import json
import logging
import re
from typing import Any, List, Type, TypeVar, Union

from pydantic import BaseModel, parse_obj_as
from pydantic_core import from_json

from .string_utils import normalize_text

logger = logging.getLogger(__name__)


T = TypeVar('T', bound=BaseModel)


def fix_json_strings(json_str: str) -> str:
    """
    修复 JSON 字符串中的未转义引号
    
    方法：使用状态机找到字符串值内部未转义的引号并转义它们
    更智能的判断：检查引号后的内容，如果不符合 JSON 语法（不是 : 或 , 或 } 或 ]），则转义
    """
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(json_str):
        char = json_str[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            if not in_string:
                # 字符串开始（检查前面是否有键的开始标记）
                # 向前查找，跳过空白字符，检查前面是否有 : 或 { 或 , 或 [
                k = i - 1
                while k >= 0 and json_str[k] in ' \t\n\r':
                    k -= 1
                
                if k < 0 or json_str[k] in '{:,[':
                    in_string = True
                    result.append(char)
                else:
                    # 可能是字符串内部的引号
                    result.append('\\"')
            else:
                # 在字符串内部，检查是否是字符串结束
                # 查看接下来的字符，跳过空白字符
                j = i + 1
                while j < len(json_str) and json_str[j] in ' \t\n\r':
                    j += 1
                
                # 检查下一个非空白字符
                if j >= len(json_str):
                    # 字符串结束（到达文件末尾）
                    in_string = False
                    result.append(char)
                elif json_str[j] in ':},]':
                    # 如果下一个非空白字符是 JSON 结构字符，则是字符串结束
                    in_string = False
                    result.append(char)
                elif json_str[j].isalnum() or json_str[j] in "'-":
                    # 如果下一个字符是字母、数字、撇号或连字符，这个引号应该被转义
                    # 例如："vendor"s" -> "vendor\"s" 或 "it's" -> "it\"s"
                    result.append('\\"')
                else:
                    # 其他情况（标点符号等），更保守的处理：转义
                    result.append('\\"')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def parse_and_validate_json(
    llm_response: str, 
    schema: Union[Type[BaseModel], Any], 
    allow_partial: bool = False
) -> Any:
    """
    Parse and validate JSON from an LLM response.
    
    This function:
    1. Normalizes the text to handle encoding issues
    2. Removes markdown code fences
    3. Extracts the JSON structure
    4. Parses and validates against the provided schema
    
    Args:
        llm_response: Raw LLM response that may contain JSON with surrounding text
        schema: Pydantic model or type hint for validation (e.g., MyModel or List[MyModel])
        allow_partial: Whether to allow partial JSON parsing (useful for truncated responses)
        
    Returns:
        Parsed and validated object(s) according to the schema
        
    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Normalize text to handle control characters
    cleaned_response = normalize_text(llm_response)
    
    # Remove markdown code fences (```json or ```)
    cleaned_response = re.sub(
        r"```(?:json)?", "", 
        cleaned_response, 
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    # Find JSON structure (object or array)
    json_match = re.search(r'(\{.*\}|\[.*\])', cleaned_response, flags=re.DOTALL)
    if not json_match:
        raise ValueError(
            f"No valid JSON object or array found in response: {llm_response[:200]}..."
        )

    json_str = json_match.group(0).strip()
    
    # 打印提取的 JSON 字符串（用于调试）
    # logger.info(f" 提取的 JSON 字符串: {json_str}")
    # logger.info(f" JSON 字符串长度: {len(json_str)}")
    
    try:
        # 优先使用标准 json 库解析（更可靠，能处理完整的 JSON）
        try:
            parsed = json.loads(json_str)
            logger.info(f" 使用标准 json.loads 成功解析，包含字段: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
        except json.JSONDecodeError as e:
            logger.warning(f" 标准 json.loads 失败: {e}")
            logger.info(f" 错误位置: line {e.lineno}, column {e.colno}")
            
            # 尝试识别是哪个字段（变量）出错
            if hasattr(e, 'pos') and e.pos is not None:
                error_pos = e.pos
                logger.info(f" 错误字符位置: {error_pos}")
                
                # 向前查找最近的字段名（查找 "field_name": 模式）
                field_name = None
                search_start = max(0, error_pos - 500)  # 向前搜索500个字符
                before_error = json_str[search_start:error_pos]
                
                # 查找最后一个 "field_name": 模式
                field_match = re.search(r'"([^"]+)":\s*"', before_error)
                if field_match:
                    field_name = field_match.group(1)
                    logger.info(f" 🔍 错误发生在字段: '{field_name}'")
                else:
                    # 如果没找到，尝试查找其他可能的字段模式
                    field_match2 = re.search(r'"([^"]+)":\s*\{', before_error)
                    if field_match2:
                        field_name = field_match2.group(1)
                        logger.info(f" 🔍 错误发生在字段: '{field_name}' (对象类型)")
                
                # 显示错误位置前后的内容
                start = max(0, error_pos - 200)
                end = min(len(json_str), error_pos + 200)
                context = json_str[start:end]
                marker_pos = min(100, error_pos - start)  # 标记位置（最多100个空格）
                
                # 将上下文分成多行显示
                context_lines = context.split('\n')
                logger.info(f" 错误位置周围的文本（前200后200字符）:")
                for line in context_lines[:5]:  # 最多显示5行
                    logger.info(f"    {line}")
                logger.info(f"    {' ' * marker_pos}^ <-- 错误位置")
                logger.info(f" 错误位置的字符: '{json_str[error_pos] if error_pos < len(json_str) else 'EOF'}' (ASCII: {ord(json_str[error_pos]) if error_pos < len(json_str) else 'N/A'})")
                
                # 如果找到了字段名，显示该字段的完整值（如果可能）
                if field_name:
                    # 尝试提取该字段的值（使用转义的字段名）
                    escaped_field_name = re.escape(field_name)
                    field_pattern = rf'"{escaped_field_name}":\s*"([^"]*(?:"[^",}}\]]*)*)"'
                    field_value_match = re.search(field_pattern, json_str[:error_pos + 100], re.DOTALL)
                    if field_value_match:
                        field_value = field_value_match.group(1)
                        logger.info(f" 📝 字段 '{field_name}' 的值（截断）: {field_value[:200]}...")
                    else:
                        logger.info(f" ⚠️  无法完整提取字段 '{field_name}' 的值（可能包含未转义引号）")
                
                # 显示错误位置前后各50个字符的详细信息
                detailed_start = max(0, error_pos - 50)
                detailed_end = min(len(json_str), error_pos + 50)
                detailed_context = json_str[detailed_start:detailed_end]
                logger.info(f" 详细上下文（前50后50字符）:")
                logger.info(f"    {repr(detailed_context)}")
            
            # 尝试修复 JSON：处理字符串值中的未转义引号
            try:
                logger.info(f" 🔧 开始修复 JSON 字符串...")
                fixed_json_str = fix_json_strings(json_str)
                
                # 检查修复前后的差异
                if fixed_json_str != json_str:
                    # 找到第一个不同的位置
                    for idx in range(min(len(json_str), len(fixed_json_str))):
                        if json_str[idx] != fixed_json_str[idx]:
                            logger.info(f"  ✅ 检测到修复位置 {idx}: '{json_str[idx]}' -> '{fixed_json_str[idx:idx+2]}'")
                            # 显示修复位置周围的上下文
                            ctx_start = max(0, idx - 30)
                            ctx_end = min(len(fixed_json_str), idx + 30)
                            logger.info(f"  修复上下文: ...{fixed_json_str[ctx_start:ctx_end]}...")
                            break
                else:
                    logger.info(f"  ⚠️  修复后无变化，可能问题不在未转义引号")
                
                parsed = json.loads(fixed_json_str)
                logger.info(f"  ✅ 修复后使用 json.loads 成功解析")
            except Exception as e_fix:
                logger.warning(f" JSON 修复失败: {e_fix}，尝试使用 from_json")
                # 如果修复失败，尝试使用 from_json（可能更容错）
                try:
                    parsed = from_json(json_str, allow_partial=False)
                    logger.info(f" 使用 from_json 成功解析")
                except Exception as e2:
                    logger.error(f" from_json 也失败: {e2}")
                    # 最后尝试：使用 allow_partial（虽然会丢失字段，但至少能解析部分）
                    try:
                        parsed = from_json(json_str, allow_partial=True)
                        logger.warning(f" 使用 from_json with allow_partial=True 解析（可能丢失字段）")
                    except Exception as e3:
                        logger.error(f" 所有解析方法都失败")
                        raise e  # 抛出原始的 json.JSONDecodeError
        
        logger.info(f" 解析后的对象键: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
        if isinstance(parsed, dict):
            for key in parsed.keys():
                value = parsed[key]
                if isinstance(value, str):
                    logger.info(f"  {key}: 长度={len(value)}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Validate against schema
        validated_data = parse_obj_as(schema, parsed)
        
        return validated_data
        
    except Exception as e:
        raise ValueError(
            f"Error parsing/validating JSON: {e}\n"
            f"JSON string: {json_str[:200]}..."
        ) from e


def validate_dicts_to_pydantic(
    dicts: List[dict],
    model: Type[T],
    skip_invalid: bool = False
) -> List[T]:
    """
    Validate a list of dictionaries against a Pydantic model.
    
    Args:
        dicts: List of dictionaries to validate
        model: Pydantic model class to validate against
        skip_invalid: If True, skip invalid items instead of raising an error
        
    Returns:
        List of validated Pydantic model instances
        
    Raises:
        ValueError: If skip_invalid is False and validation fails for any item
    """
    validated = []
    
    for i, item_dict in enumerate(dicts):
        try:
            validated_item = model.model_validate(item_dict)
            validated.append(validated_item)
        except Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continue
            else:
                raise ValueError(
                    f"Validation failed for item {i}: {e}"
                ) from e
    
    return validated