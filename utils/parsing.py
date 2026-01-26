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
    Fix unescaped quotes in JSON strings.
    
    Method: Use a state machine to find unescaped quotes inside string values and escape them.
    Smarter judgment: Check content after the quote; if it doesn't fit JSON syntax (not :, ,, }, or ]), then escape it.
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
                # String start (check if there's a key start marker before)
                # Look backward, skip whitespace, check if preceded by :, {, ,, or [
                k = i - 1
                while k >= 0 and json_str[k] in ' \t\n\r':
                    k -= 1
                
                if k < 0 or json_str[k] in '{:,[':
                    in_string = True
                    result.append(char)
                else:
                    # Likely a quote inside a string
                    result.append('\\"')
            else:
                # Inside string, check if it is the end of the string
                # Check next characters, skipping whitespace
                j = i + 1
                while j < len(json_str) and json_str[j] in ' \t\n\r':
                    j += 1
                
                # Check next non-whitespace character
                if j >= len(json_str):
                    # End of string (reached end of file)
                    in_string = False
                    result.append(char)
                elif json_str[j] in ':},]':
                    # If next non-whitespace char is a JSON structure char, it matches a string end
                    in_string = False
                    result.append(char)
                elif json_str[j].isalnum() or json_str[j] in "'-":
                    # If next char is alphanumeric, apostrophe, or hyphen, this quote should be escaped
                    # e.g.: "vendor"s" -> "vendor\"s" or "it's" -> "it\"s"
                    result.append('\\"')
                else:
                    # Other cases (punctuation etc.), more conservative handling: escape
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
    
    json_str = json_match.group(0).strip()
    
    # Print extracted JSON string (for debugging)
    # logger.info(f" Extracted JSON string: {json_str}")
    # logger.info(f" JSON string length: {len(json_str)}")
    
    try:
        # Prioritize standard json library (more reliable, handles complete JSON)
        try:
            parsed = json.loads(json_str)
            logger.info(f" Successfully parsed using standard json.loads, contains keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
        except json.JSONDecodeError as e:
            logger.warning(f" Standard json.loads failed: {e}")
            logger.info(f" Error location: line {e.lineno}, column {e.colno}")
            
            # Attempt to identify which field (variable) caused the error
            if hasattr(e, 'pos') and e.pos is not None:
                error_pos = e.pos
                logger.info(f" Error character position: {error_pos}")
                
                # Look backward for nearest field name (looking for "field_name": pattern)
                field_name = None
                search_start = max(0, error_pos - 500)  # Search backward 500 chars
                before_error = json_str[search_start:error_pos]
                
                # Find last "field_name": pattern
                field_match = re.search(r'"([^"]+)":\s*"', before_error)
                if field_match:
                    field_name = field_match.group(1)
                    logger.info(f" 🔍 Error occurred in field: '{field_name}'")
                else:
                    # If not found, try finding other possible field patterns
                    field_match2 = re.search(r'"([^"]+)":\s*\{', before_error)
                    if field_match2:
                        field_name = field_match2.group(1)
                        logger.info(f" 🔍 Error occurred in field: '{field_name}' (object type)")
                
                # Show content around error position
                start = max(0, error_pos - 200)
                end = min(len(json_str), error_pos + 200)
                context = json_str[start:end]
                marker_pos = min(100, error_pos - start)  # Marker position (max 100 spaces)
                
                # Display context lines
                context_lines = context.split('\n')
                logger.info(f" Text around error position (prev 200 next 200 chars):")
                for line in context_lines[:5]:  # Show at most 5 lines
                    logger.info(f"    {line}")
                logger.info(f"    {' ' * marker_pos}^ <-- Error position")
                logger.info(f" Character at error position: '{json_str[error_pos] if error_pos < len(json_str) else 'EOF'}' (ASCII: {ord(json_str[error_pos]) if error_pos < len(json_str) else 'N/A'})")
                
                # If field name found, show full value of that field (if possible)
                if field_name:
                    # Attempt to extract that field value (using escaped field name)
                    escaped_field_name = re.escape(field_name)
                    field_pattern = rf'"{escaped_field_name}":\s*"([^"]*(?:"[^",}}\]]*)*)"'
                    field_value_match = re.search(field_pattern, json_str[:error_pos + 100], re.DOTALL)
                    if field_value_match:
                        field_value = field_value_match.group(1)
                        logger.info(f" 📝 Value of field '{field_name}' (truncated): {field_value[:200]}...")
                    else:
                        logger.info(f" ⚠️  Cannot fully extract value of field '{field_name}' (may contain unescaped quotes)")
                
                # Show detailed context 50 chars around error position
                detailed_start = max(0, error_pos - 50)
                detailed_end = min(len(json_str), error_pos + 50)
                detailed_context = json_str[detailed_start:detailed_end]
                logger.info(f" Detailed context (prev 50 next 50 chars):")
                logger.info(f"    {repr(detailed_context)}")
            
            # Attempt to fix JSON: handle unescaped quotes in string values
            try:
                logger.info(f" 🔧 Starting to fix JSON string...")
                fixed_json_str = fix_json_strings(json_str)
                
                # Check differences before and after fix
                if fixed_json_str != json_str:
                    # Find first different position
                    for idx in range(min(len(json_str), len(fixed_json_str))):
                        if json_str[idx] != fixed_json_str[idx]:
                            logger.info(f"  ✅ Fixed position detected {idx}: '{json_str[idx]}' -> '{fixed_json_str[idx:idx+2]}'")
                            # Show context around fixed position
                            ctx_start = max(0, idx - 30)
                            ctx_end = min(len(fixed_json_str), idx + 30)
                            logger.info(f"  Fix context: ...{fixed_json_str[ctx_start:ctx_end]}...")
                            break
                else:
                    logger.info(f"  ⚠️  No change after fix, problem might not be unescaped quotes")
                
                parsed = json.loads(fixed_json_str)
                logger.info(f"  ✅ Successfully parsed with json.loads after fix")
            except Exception as e_fix:
                logger.warning(f" JSON fix failed: {e_fix}, trying from_json")
                # If fix fails, try from_json (maybe more fault tolerant)
                try:
                    parsed = from_json(json_str, allow_partial=False)
                    logger.info(f" Successfully parsed with from_json")
                except Exception as e2:
                    logger.error(f" from_json also failed: {e2}")
                    # Last attempt: use allow_partial (although fields might be lost, at least can parse partially)
                    try:
                        parsed = from_json(json_str, allow_partial=True)
                        logger.warning(f" Parsed with from_json with allow_partial=True (fields might be lost)")
                    except Exception as e3:
                        logger.error(f" All parsing methods failed")
                        raise e  # Raise original json.JSONDecodeError
        
        logger.info(f" Parsed object keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
        if isinstance(parsed, dict):
            for key in parsed.keys():
                value = parsed[key]
                if isinstance(value, str):
                    logger.info(f"  {key}: length={len(value)}")
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