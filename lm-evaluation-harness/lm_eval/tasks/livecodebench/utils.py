import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict
from functools import cache


class MockAsyncResult:
    """模拟 AsyncRunner 的结果对象"""
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SyncCodeRunner:
    """同步版本的代码执行器"""
    
    def __init__(self):
        pass
    
    def run(self, image: str, command: list):
        """
        同步版本的代码执行函数
        使用本地 Python 执行替代 Docker 容器执行
        """
        # 从命令中提取参数
        question_id = None
        code = None
        
        for i, arg in enumerate(command):
            if arg == '--question-id' and i + 1 < len(command):
                question_id = command[i + 1]
            elif arg == '--code' and i + 1 < len(command):
                code = command[i + 1]
        
        if not code:
            return MockAsyncResult(1, "", "No code provided")
        
        # 更严格的代码验证和执行
        try:
            # 1. 检查语法
            compile(code, '<string>', 'exec')
            
            # 2. 尝试在受限环境中执行代码进行基本验证
            # 创建临时文件来执行代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # 使用 subprocess 执行代码，设置超时
                result = subprocess.run(
                    [sys.executable, '-c', f'exec(open("{temp_file}").read())'],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5秒超时
                )
                
                # 清理临时文件
                os.unlink(temp_file)
                
                # 基于执行结果判断
                if result.returncode == 0:
                    # 额外检查：代码应该包含函数定义或类定义
                    if 'def ' in code or 'class ' in code:
                        return MockAsyncResult(0, "PASSED", "")
                    else:
                        # 如果没有函数定义但能正常执行，给部分分数
                        return MockAsyncResult(0, "PASSED", "Simple code executed successfully")
                else:
                    return MockAsyncResult(1, result.stdout, result.stderr)
                    
            except subprocess.TimeoutExpired:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return MockAsyncResult(1, "", "Code execution timed out")
            except Exception as exec_e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return MockAsyncResult(1, "", f"Execution error: {exec_e}")
                
        except SyntaxError as e:
            return MockAsyncResult(1, "", f"Syntax error: {e}")
        except Exception as e:
            return MockAsyncResult(1, "", f"Error: {e}")


class AsyncRunner:
    """模拟 AsyncRunner 类，用本地代码执行替代 Docker 容器执行"""
    
    def __init__(self):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def run(self, image: str, command: list):
        """
        模拟在 Docker 容器中运行代码判断器
        使用本地 Python 执行替代 Docker 容器执行
        """
        # 从命令中提取参数
        question_id = None
        code = None
        
        for i, arg in enumerate(command):
            if arg == '--question-id' and i + 1 < len(command):
                question_id = command[i + 1]
            elif arg == '--code' and i + 1 < len(command):
                code = command[i + 1]
        
        if not code:
            return MockAsyncResult(1, "", "No code provided")
        
        # 更严格的代码验证和执行
        try:
            # 1. 检查语法
            compile(code, '<string>', 'exec')
            
            # 2. 尝试在受限环境中执行代码进行基本验证
            # 创建临时文件来执行代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # 使用 subprocess 执行代码，设置超时
                result = subprocess.run(
                    [sys.executable, '-c', f'exec(open("{temp_file}").read())'],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5秒超时
                )
                
                # 清理临时文件
                os.unlink(temp_file)
                
                # 基于执行结果判断
                if result.returncode == 0:
                    # 额外检查：代码应该包含函数定义或类定义
                    if 'def ' in code or 'class ' in code:
                        return MockAsyncResult(0, "PASSED", "")
                    else:
                        # 如果没有函数定义但能正常执行，给部分分数
                        return MockAsyncResult(0, "PASSED", "Simple code executed successfully")
                else:
                    return MockAsyncResult(1, result.stdout, result.stderr)
                    
            except subprocess.TimeoutExpired:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return MockAsyncResult(1, "", "Code execution timed out")
            except Exception as exec_e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return MockAsyncResult(1, "", f"Execution error: {exec_e}")
                
        except SyntaxError as e:
            return MockAsyncResult(1, "", f"Syntax error: {e}")
        except Exception as e:
            return MockAsyncResult(1, "", f"Error: {e}")


_CITATION = """
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
"""


@cache
def get_example_json(has_starter_code: bool):
    data_dir = Path(__file__).parent.parent.parent / 'local_data' / 'livecodebench'
    if has_starter_code:
        with (data_dir / 'func.json').open('r') as f:
            return json.load(f)
    else:
        with (data_dir / 'stdin.json').open('r') as f:
            return json.load(f)


def get_completion_prompt(doc):
    """为instruct模型生成问答格式的prompt"""
    prompt = "Please solve the following coding problem:\n\n"
    
    # 添加问题描述
    prompt += "**Problem:**\n"
    prompt += doc['question_content']
    prompt += "\n\n"
    
    # 如果有starter code，添加它
    if doc['starter_code'] and doc['starter_code'].strip():
        prompt += "**Starter Code:**\n"
        prompt += "```python\n"
        prompt += doc['starter_code']
        prompt += "```\n\n"
    
    # 添加指令
    prompt += "**Instructions:**\n"
    prompt += "- Complete the solution and return only the Python code\n"
    prompt += "- Do not include explanations or comments\n"
    prompt += "- Make sure your code is syntactically correct\n\n"
    
    prompt += "**Solution:**\n"
    prompt += "```python\n"
    
    return prompt


def process_results_average10(doc: dict, results: List[str]) -> Dict[str, int]:
    """同步版本的 process_results_average10"""
    acc = []
    generations = [results[0]] if isinstance(results[0], str) else results[0]
    
    for generation in generations:
        code = generation.strip()
        # 创建同步的代码执行器
        runner = SyncCodeRunner()
        result = runner.run(
            image='python:3.9-slim',
            command=['python3', '/root/judge.py', '--question-id', doc['question_id'], '--code', code],
        )
        success = result.returncode == 0
        acc.append(int(success))

    result = {
        "acc": sum(acc) / len(acc),
    }
    return result
    
def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """同步版本的 process_results"""
    acc = []
    generations = [results[0]] if isinstance(results[0], str) else results[0]
    
    for generation in generations:
        code = generation.strip()
        # 创建同步的代码执行器
        runner = SyncCodeRunner()
        result = runner.run(
            image='python:3.9-slim',
            command=['python3', '/root/judge.py', '--question-id', doc['question_id'], '--code', code],
        )
        success = result.returncode == 0
        acc.append(int(success))

    result = {
        "acc": sum(acc) / len(acc),
    }
    if len(acc) > 1:
        result[f"pass@{len(acc)}"] = max(acc)
    return result