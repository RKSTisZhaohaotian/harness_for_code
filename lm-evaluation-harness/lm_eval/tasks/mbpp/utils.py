import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def extract_code_blocks(text: str) -> str:
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    
    # First try to match existing code blocks in the text
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0]
    
    # If no matches and text doesn't start with ```, try adding prefix
    if not text.strip().startswith("```"):
        matches = re.findall(pattern, r"```" + text, re.DOTALL)
        if matches:
            return matches[0]
    
    # Try removing language specifier
    text_without_lang = re.sub(r"```python", "```", text)
    matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if matches:
        return matches[0]
    
    # If still no matches, check if the text looks like pure Python code
    text_stripped = text.strip()
    if text_stripped and _looks_like_python_code(text_stripped):
        return text_stripped
    
    # If still no matches, return empty string
    return ""


def _looks_like_python_code(text: str) -> bool:
    """
    Check if text looks like valid Python code by examining common patterns.
    """
    text = text.strip()
    if not text:
        return False
    
    # Common Python keywords and patterns that indicate code
    python_indicators = [
        'def ', 'class ', 'import ', 'from ', 'if ', 'else:', 'elif ', 
        'for ', 'while ', 'try:', 'except:', 'return ', 'yield ', 
        'lambda ', 'with ', 'as ', 'assert ', 'raise ', 'del ',
    ]
    
    # Check if text starts with common Python patterns
    for indicator in python_indicators:
        if text.startswith(indicator):
            return True
    
    # Check for function definition pattern with parentheses and colon
    if re.match(r'^\s*def\s+\w+\s*\([^)]*\)\s*:', text, re.MULTILINE):
        return True
    
    # Check for class definition pattern
    if re.match(r'^\s*class\s+\w+.*:', text, re.MULTILINE):
        return True
    
    # Check for assignment patterns (variable = value)
    if re.match(r'^\s*\w+\s*=', text):
        return True
    
    # Check if it contains Python-like indentation and structure
    lines = text.split('\n')
    if len(lines) > 1:
        # Look for proper indentation patterns
        indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        if indented_lines:
            return True
    
    return False


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def build_predictions_simple(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """
    Simplified code extraction for responses that are expected to be pure code
    without markdown formatting.
    """
    def extract_simple_code(text: str) -> str:
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # If it starts with ```python, extract the code block
        if text.startswith("```python"):
            return extract_code_blocks(text)
        
        # If it contains code block markers, extract them
        if "```" in text:
            return extract_code_blocks(text)
        
        # Otherwise, assume the entire text is code
        return text
    
    return [[extract_simple_code(r) for r in resp] for resp in resps]

