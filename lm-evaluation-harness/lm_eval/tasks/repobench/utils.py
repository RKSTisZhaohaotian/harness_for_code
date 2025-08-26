import os
import re
import json
from typing import Dict, List
from datasets import load_dataset
from fuzzywuzzy import fuzz
from codebleu import calc_codebleu

# Set API key environment variable for lm-evaluation-harness
if 'BAIDU_INT_API_KEY' in os.environ:
    os.environ['API_KEY'] = os.environ['BAIDU_INT_API_KEY']
    os.environ['OPENAI_API_KEY'] = os.environ['BAIDU_INT_API_KEY']
#if 'OPENAI_API_KEY' in os.environ:
#    os.environ['API_KEY'] = os.environ['OPENAI_API_KEY']

def setup_api_and_process_docs(dataset):
    """
    Setup API key and process documents.
    """
    # Ensure API key is set before any API calls
    if 'BAIDU_INT_API_KEY' in os.environ:
        os.environ['API_KEY'] = os.environ['BAIDU_INT_API_KEY']
        os.environ['OPENAI_API_KEY'] = os.environ['BAIDU_INT_API_KEY']
    return dataset

def get_first_line_not_comment(code: str, language: str = "python"):
    """
    This function gets the first line of code that is not a comment.
    
    Args:
    code: Str, the code
    language: Str, the programming language
    
    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """
    assert language in ["python", "java"], "language must be one of [python, java]"
    
    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')
    lines = code.split('\n')
    in_multiline_comment = False
    
    if language == "python":
        for line in lines:
            if not line.strip():
                continue
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_multiline_comment = True
                continue
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            if in_multiline_comment:
                continue
            if line.strip().startswith('#'):
                continue
            return line
            
    elif language == "java":
        for line in lines:
            if not line.strip():
                continue
            if not in_multiline_comment and line.strip().startswith('/*'):
                in_multiline_comment = True
                continue
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            if in_multiline_comment:
                continue
            if line.strip().startswith('//'):
                continue
            return line
    
    # if we cannot find a line that is not a comment, then return the first line
    return lines[0] if lines else ""

def construct_prompt(doc: Dict, language: str = "python", tokenizer=None, max_token_nums: int = 15800) -> str:
    """
    Construct the prompt for next line prediction using RepoBench format.
    
    Args:
    doc: Dict, data point from the dataset
    language: Str, the language of the code
    tokenizer: tokenizer of the evaluation model  
    max_token_nums: int, the maximum number of tokens constraint for the prompt
    
    Returns:
    Str, the constructed prompt
    """
    # comment symbol for different languages
    comment_symbol = "#" if language == "python" else "//"
    
    # construct the cross-file prompt and in-file prompt separately
    cross_file_prompt = f"{comment_symbol} Repo Name: {doc['repo_name']}\n"
    
    for snippet in doc['context']:
        cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
    
    # in-file prompt
    in_file_prompt = f"{comment_symbol} Path: {doc['file_path']}\n{doc['import_statement']}\n{doc['cropped_code'].rstrip()}\n"
    
    # if we assign the tokenizer and the max_token_nums, we will truncate the cross-file prompt to meet the constraint
    if tokenizer is not None and max_token_nums is not None:
        cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
        in_file_prompt_token_nums = len(tokenizer.encode(in_file_prompt))
        
        exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - max_token_nums
        
        if exceed_token_nums > 0:
            # split the cross-file prompt into lines
            cross_file_prompt_lines = cross_file_prompt.split("\n")
            # drop lines from end until the extra token number is less than 0
            for i in range(len(cross_file_prompt_lines)-1, -1, -1):
                exceed_token_nums -= len(tokenizer.encode(cross_file_prompt_lines[i]))
                if exceed_token_nums < 0:
                    break
            
            # join the lines back
            cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"
    
    # combine the cross-file prompt and in-file prompt
    prompt = cross_file_prompt + in_file_prompt
    
    # normalize some empty lines
    prompt = re.sub(r'\n{4,}', '\n\n', prompt)
    
    return prompt

def doc_to_text(doc: Dict) -> str:
    """
    Convert document to text prompt for generation.
    """
    return construct_prompt(doc, language="python")

def exact_match_score(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute the average exact match score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    exact_match = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.split() == gt.split():
            exact_match += 1
    
    return exact_match / len(predictions)

def edit_similarity_score(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute the average edit similarity score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    edit_sim = 0.0
    for pred, gt in zip(predictions, ground_truths):
        edit_sim += fuzz.ratio(pred, gt)
    
    return edit_sim / len(predictions) / 100.0  # Normalize to [0, 1]

def codebleu_score(predictions: List[str], ground_truths: List[str], language: str = "python", weight: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """
    Compute the average codebleu score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    # remove \r for both pred and gt
    predictions = [pred.replace("\r", "") for pred in predictions]
    ground_truths = [gt.replace("\r", "") for gt in ground_truths]
    
    res_list = calc_codebleu(
        ground_truths,
        predictions,
        language,
        weight,
        tokenizer=None
    )
    
    return res_list['codebleu']

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process the generation results for RepoBench evaluation.
    
    Args:
    doc: Dict, the document containing ground truth
    results: List[str], the generated predictions
    
    Returns:
    Dict[str, float], the evaluation metrics
    """
    if not results or not results[0]:
        return {
            "exact_match": 0.0,
            "edit_similarity": 0.0,
            "codebleu": 0.0
        }
    
    # Extract the first line from generated text (similar to RepoBench approach)
    prediction = results[0].strip()
    if prediction:
        prediction = get_first_line_not_comment(prediction, language="python")
    else:
        prediction = ""
    
    ground_truth = doc["next_line"]
    
    # Calculate metrics
    em_score = 1.0 if prediction.split() == ground_truth.split() else 0.0
    es_score = fuzz.ratio(prediction, ground_truth) / 100.0
    
    # For codebleu, we need lists
    try:
        cb_score = codebleu_score([prediction], [ground_truth], language="python")
    except:
        cb_score = 0.0
    
    return {
        "exact_match": em_score,
        "edit_similarity": es_score, 
        "codebleu": cb_score
    }