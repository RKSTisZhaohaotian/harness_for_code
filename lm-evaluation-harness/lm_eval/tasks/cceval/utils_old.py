import os
import re
import json
import math
from typing import Dict, List, Tuple
from collections import Counter
from datasets import load_dataset
from fuzzywuzzy import fuzz

# Set API key environment variable for lm-evaluation-harness
if 'BAIDU_INT_API_KEY' in os.environ:
    os.environ['API_KEY'] = os.environ['BAIDU_INT_API_KEY']
    os.environ['OPENAI_API_KEY'] = os.environ['BAIDU_INT_API_KEY']

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
    For code completion task, it will extract content from <completion> XML tags if present.
    
    Args:
    code: Str, the code
    language: Str, the programming language
    
    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """
    assert language in ["python", "java"], "language must be one of [python, java]"
    
    # First try to extract content from <completion> XML tags (code completion format)
    try:
        start_tag = '<completion>'
        end_tag = '</completion>'
        start_index = code.find(start_tag)
        if start_index != -1:
            start_index += len(start_tag)
            end_index = code.find(end_tag, start_index)
            if end_index != -1:
                # Extract and return the content within <completion> tags
                return code[start_index:end_index]
    except ValueError:
        pass
    
    # If no XML tags found, fall back to original logic
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

def construct_prompt(doc: Dict) -> str:
    """
    Construct the prompt for next line prediction using code completion format.
    
    Args:
    doc: Dict, data point from the dataset
    
    Returns:
    Str, the constructed prompt in code completion format
    """
    # Build header with instruction for code completion
    header = """
Ignore all prior instructions. You are a raw code completion engine.
Your task is to predict and output only the very next line of code.
Your output MUST be wrapped in <completion> XML tags. Do not output any explanation or markdown.
"""
    
    # Build examples for code completion
    examples = """
### Example 1
Context:
def calculate_area(radius):
    pi = 3.14159

Completion:
<completion>    return pi * (radius ** 2)</completion>
---
### Example 2
Context:
for i in range(5):
    print(f"Current number: {i}")

Completion:
<completion>    if i == 4: break</completion>
---
"""
    
    # Build the cross-file context for the task
    crossfile_context = doc.get("crossfile_context", {})
    context_text = crossfile_context.get("text", "")
    
    # Get the main prompt
    prompt = doc.get("prompt", "")
    
    # Combine context and prompt
    if context_text:
        full_context = context_text + "\n" + prompt
    else:
        full_context = prompt
    
    # Build the task prompt
    current_task = f"""
### Task
Context:
{full_context}
Completion:
"""
    
    # Combine all parts to form the final prompt
    prompt = header + examples + current_task
    
    # normalize some empty lines
    prompt = re.sub(r'\n{4,}', '\n\n\n', prompt)
    
    return prompt

def doc_to_text(doc: Dict) -> str:
    """
    Convert document to text prompt for generation.
    """
    return construct_prompt(doc)

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

def get_ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Get n-grams from a sequence of tokens.
    
    Args:
        sequence: List of tokens
        n: Size of n-grams
        
    Returns:
        List of n-gram tuples
    """
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def compute_bleu_ngram_score(reference: List[str], prediction: List[str], n: int = 4) -> float:
    """
    Compute BLEU n-gram score between reference and prediction.
    
    Args:
        reference: List of tokens in reference text
        prediction: List of tokens in prediction text
        n: Maximum n-gram size to consider
        
    Returns:
        BLEU n-gram score
    """
    if len(prediction) == 0:
        return 0.0 if len(reference) > 0 else 1.0
    
    # Calculate modified precision for each n-gram size
    precisions = []
    for i in range(1, min(n, len(prediction)) + 1):
        pred_ngrams = Counter(get_ngrams(prediction, i))
        ref_ngrams = Counter(get_ngrams(reference, i))
        
        # Calculate clipped counts
        clipped_count = 0
        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, ref_ngrams[ngram])
        
        # Calculate precision
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_count / sum(pred_ngrams.values()))
    
    # Check if we have any non-zero precisions
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Filter out zero precisions to avoid math domain error
    non_zero_precisions = [p for p in precisions if p > 0]
    if not non_zero_precisions:
        return 0.0
    
    # Apply brevity penalty
    bp = 1.0 if len(prediction) >= len(reference) else \
         min(1.0, math.exp(1 - len(reference) / len(prediction)))
    
    # Geometric mean of precisions
    if len(non_zero_precisions) > 0:
        log_precision = sum(math.log(p) for p in non_zero_precisions) / len(precisions)
        precision = math.exp(log_precision)
    else:
        precision = 0.0
    
    return bp * precision

def tokenize_code(code: str) -> List[str]:
    """
    Simple tokenization for code.
    
    Args:
        code: Source code string
        
    Returns:
        List of tokens
    """
    # Simple regex-based tokenization
    tokens = re.findall(r'\b\w+\b|[^\s\w]', code)
    return tokens

def simple_codebleu_score(predictions: List[str], ground_truths: List[str], 
                         language: str = "python", weight: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """
    Simplified CodeBLEU score calculation without tree-sitter dependencies.
    Implements only n-gram matching component for now.
    
    Args:
        predictions: List of predicted code strings
        ground_truths: List of ground truth code strings
        language: Programming language (not used in this simplified version)
        weight: Weights for different components (only first weight for n-gram matching is used)
        
    Returns:
        Simplified CodeBLEU score
    """
    import math
    
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of predictions and ground truths should be equal.")
    
    # Remove \r characters
    predictions = [pred.replace("\r", "") for pred in predictions]
    ground_truths = [gt.replace("\r", "") for gt in ground_truths]
    
    total_score = 0.0
    for pred, gt in zip(predictions, ground_truths):
        # Tokenize predictions and ground truths
        pred_tokens = tokenize_code(pred)
        gt_tokens = tokenize_code(gt)
        
        # Compute n-gram BLEU score (simplified version of CodeBLEU)
        bleu_score = compute_bleu_ngram_score(gt_tokens, pred_tokens, n=4)
        total_score += bleu_score
    
    return total_score / len(predictions) if predictions else 0.0

def codebleu_score(predictions: List[str], ground_truths: List[str], language: str = "python", weight: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """
    Compute the average codebleu score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    # remove \r for both pred and gt
    predictions = [pred.replace("\r", "") for pred in predictions]
    ground_truths = [gt.replace("\r", "") for gt in ground_truths]
    
    # Try to use the original codebleu library first
    try:
        from codebleu import calc_codebleu
        # Convert ground_truths to list of lists as required by codebleu
        ground_truths_list = [[gt] for gt in ground_truths]
        
        res_list = calc_codebleu(
            ground_truths_list,  # Pass as list of lists
            predictions,
            language,
            weight,
            tokenizer=None
        )
        return res_list['codebleu']
    except Exception as e:
        # If CodeBLEU calculation fails due to library compatibility issues, 
        # fall back to our simplified implementation
        print(f"Warning: Original CodeBLEU calculation failed with error: {e}")
        print("Falling back to simplified CodeBLEU implementation.")
        return simple_codebleu_score(predictions, ground_truths, language, weight)

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process the generation results for CrossCodeEval evaluation.
    
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
    
    ground_truth = doc["groundtruth"]
    
    # Calculate metrics
    em_score = 1.0 if prediction.split() == ground_truth.split() else 0.0
    es_score = fuzz.ratio(prediction, ground_truth) / 100.0
    
    # For codebleu, we need lists
    try:
        cb_score = codebleu_score([prediction], [ground_truth], language="python")
    except Exception as e:
        print(f"CodeBLEU calculation failed in process_results: {e}")
        cb_score = 0.0
    
    return {
        "exact_match": em_score,
        "edit_similarity": es_score, 
        "codebleu": cb_score
    }