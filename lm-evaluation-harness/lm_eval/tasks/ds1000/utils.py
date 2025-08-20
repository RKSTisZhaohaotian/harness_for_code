import os
# 设置环境变量允许代码执行
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# disable tensorflow logging and no GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import evaluate as hf_evaluate
import re
import json
from lm_eval.tasks.ds1000.execution import check_correctness
from typing import Dict, List, Any, Optional
from collections import defaultdict
from lm_eval.api.registry import register_metric
import datetime
import random
import numpy as np


# ==================== Filter Model Responses & Extract Code ====================
def extract_code_blocks(text: str) -> str:
    """
    提取所有可能有效的代码片段

    输入:
        text: 包含代码块的文本

    输出:
        list: 提取出的所有有效代码片段列表
    """
    all_possible_code_blocks = []

    # 1. 提取最开始```前的代码
    code = text.split('```')[0]
    all_possible_code_blocks.append(code)

    # 2. 使用 open-compass 的后处理
    code = text.split("</code>")[0]
    if "```python" in code:
        code = code.split("```python")[1]
    code = code.split("```")[0]
    code = code.split("\nEND SOLUTION")[0]
    if "<code>" in code:
        code = code.split("<code>")[1]
    all_possible_code_blocks.append(code)

    return all_possible_code_blocks


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    predictions = []
    for resp in resps:
        for r in resp:
            predictions.append(extract_code_blocks(r))
    return predictions


# ==================== Post Process & Metric Functions ====================
def process_results(doc: Dict[str, Any], results: List[List[str]]) -> Dict[str, Any]:
    """
    后处理单条结果，每条单独计算得分

    输入:
        doc: 文档数据
        results: 模型生成的结果列表，可能是传入的时候，harness加了[]
    """

    res = 0
    for result in results[0]:
        test_program = (
            doc['code_context'] + '\n'
            + f'code = {repr(result)}\n'
            + 'test_execution(code)\n'
            + ('test_string(code)\n'  if 'test_string(' in doc['code_context']  else '\n')
        )

        result = check_correctness(test_program, timeout=120, completion_id=doc['metadata']['problem_id'])
        score = 1 if result['passed'] else 0
        if score == 1: 
            res = 1
            break

    return {"pass_at_1_custom": res}


def random_sample_docs(dataset, sample_size=100, seed=42):
    """
    从数据集中随机采样指定数量的样本
    
    Args:
        dataset: 原始数据集
        sample_size: 采样数量，默认100
        seed: 随机种子，保证结果可重现
    
    Returns:
        采样后的数据集
    """
    # 设置随机种子保证结果可重现
    random.seed(seed)
    np.random.seed(seed)
    
    # 获取数据集总长度
    total_size = len(dataset)
    
    if total_size <= sample_size:
        # 如果数据集本身就不够大，直接返回原数据集
        return dataset
    
    # 随机选择索引
    sampled_indices = random.sample(range(total_size), sample_size)
    sampled_indices.sort()  # 排序以保持某种顺序
    
    # 创建采样后的数据集
    sampled_dataset = dataset.select(sampled_indices)
    
    print(f"从 {total_size} 条数据中随机采样了 {len(sampled_dataset)} 条数据")
    return sampled_dataset