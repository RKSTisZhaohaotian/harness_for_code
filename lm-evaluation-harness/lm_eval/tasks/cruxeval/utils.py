import tenacity

from typing import Dict, List
from loguru import logger
import logging
import evaluate as hf_evaluate
import subprocess
import sys
import tempfile
import os

GLOBAL_CNT = 0


def _input_prediction_postprocess_generation(generation):
    # 处理最后出现的 [ANSWER] 和 [/ANSWER] 标签
    if "[/ANSWER]" in generation:
        generation = generation.rsplit("[/ANSWER]", 1)[0].strip()
    if "[ANSWER]" in generation:
        generation = generation.rsplit("[ANSWER]", 1)[1].strip()
    
    # 处理代码块标记
    if generation.startswith("```python"):
        generation = generation[9:].strip()  # 移除 ```python
    if generation.endswith("```"):
        generation = generation[:-3].strip()  # 移除结尾的 ```
    
    if "==" in generation:
        generation = generation.split("==")[0].strip()
    if "assert f" in generation:
        generation = "f" + generation.split("assert f")[1].strip()
    
    # 处理外层括号问题：如果整个答案被括号包围，且内容是逗号分隔的参数，则移除外层括号
    generation = generation.strip()
    if generation.startswith('(') and generation.endswith(')'):
        # 检查是否是函数参数格式（不是单个元组）
        inner_content = generation[1:-1].strip()
        # 如果内容包含逗号且不是嵌套的复杂结构，可能是参数列表
        if ',' in inner_content:
            # 简单启发式：如果括号层级平衡且看起来像参数列表，则移除外层括号
            bracket_count = 0
            paren_count = 0
            for char in inner_content:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            # 如果括号平衡，说明可能是参数列表格式
            if bracket_count == 0 and paren_count == 0:
                generation = inner_content

    return generation.strip()


def _output_prediction_postprocess_generation(generation):
    # 处理最后出现的 [ANSWER] 和 [/ANSWER] 标签
    if "[/ANSWER]" in generation:
        generation = generation.rsplit("[/ANSWER]", 1)[0].strip()
    if "[ANSWER]" in generation:
        generation = generation.rsplit("[ANSWER]", 1)[1].strip()
    
    # 处理代码块标记
    if generation.startswith("```python"):
        generation = generation[9:].strip()  # 移除 ```python
    if generation.endswith("```"):
        generation = generation[:-3].strip()  # 移除结尾的 ```
    
    if "==" in generation:
        generation = generation.split("==")[1].strip()
    if "assert f" in generation:
        generation = "f" + generation.split("assert f")[1].strip()
    
    # 处理外层括号问题：如果整个答案被括号包围，且内容是逗号分隔的参数，则移除外层括号
    generation = generation.strip()
    if generation.startswith('(') and generation.endswith(')'):
        # 检查是否是函数参数格式（不是单个元组）
        inner_content = generation[1:-1].strip()
        # 如果内容包含逗号且不是嵌套的复杂结构，可能是参数列表
        if ',' in inner_content:
            # 简单启发式：如果括号层级平衡且看起来像参数列表，则移除外层括号
            bracket_count = 0
            paren_count = 0
            for char in inner_content:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            # 如果括号平衡，说明可能是参数列表格式
            if bracket_count == 0 and paren_count == 0:
                generation = inner_content
    
    # 对于输出预测，保留字符串字面量的引号
    # 因为输出可能就是一个字符串字面量表示
    
    return generation.strip()


def _execute_code(code, input_str, output_str):
    """
    Execute code locally using subprocess with timeout
    """
    if not input_str.startswith('f('):
        input_str = f'f({input_str})'
    full_code = f'{code}\nassert {input_str} == {output_str}'
    
    try:
        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        # Execute the code with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10  # 10 seconds timeout
        )
        
        # Clean up temporary file
        os.unlink(temp_file)
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        # Clean up temporary file if it exists
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        return {
            'returncode': 124,  # Timeout return code
            'stdout': '',
            'stderr': 'Execution timed out'
        }
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        return {
            'returncode': 1,
            'stdout': '',
            'stderr': str(e)
        }


def process_input_pred_results(doc: dict, results: List[str]) -> Dict[str, int]:
    global GLOBAL_CNT
    logger.warning(f'begin executing code locally to evaluate cruxeval: {GLOBAL_CNT}')
    logger.warning(results)
    assert len(results) == 1
    generation = results[0]
    GLOBAL_CNT += 1

    input_str = _input_prediction_postprocess_generation(generation)
    response = _execute_code(doc['code'], input_str, doc['output'])

    result = {
        "acc": response['returncode'] == 0
    }
    logger.warning(f'end executing code locally to evaluate cruxeval: {GLOBAL_CNT}')
    return result


def process_output_pred_results(doc: dict, results: List[str]) -> Dict[str, int]:
    global GLOBAL_CNT
    logger.warning(f'begin executing code locally to evaluate cruxeval: {GLOBAL_CNT}')
    logger.warning(results)
    assert len(results) == 1
    generation = results[0]
    GLOBAL_CNT += 1

    output_str = _output_prediction_postprocess_generation(generation)
    response = _execute_code(doc['code'], doc['input'], output_str)

    result = {
        "acc": response['returncode'] == 0
    }
    logger.warning(f'end executing code locally to evaluate cruxeval: {GLOBAL_CNT}')
    return result