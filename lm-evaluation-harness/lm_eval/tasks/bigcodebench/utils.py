import ast
import sys
import io
import subprocess
import tempfile
import os
import traceback
from typing import Dict, List, Optional, Set, Tuple, Generator
import logging

try:
    from tree_sitter import Node
    from tree_sitter_languages import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Node = None

eval_logger = logging.getLogger(__name__)

# Tree-sitter constants
CLASS_TYPE = 'class_definition'
FUNCTION_TYPE = 'function_definition'
IMPORT_TYPE = ['import_statement', 'import_from_statement']
IDENTIFIER_TYPE = 'identifier'
ATTRIBUTE_TYPE = 'attribute'
RETURN_TYPE = 'return_statement'
EXPRESSION_TYPE = 'expression_statement'
ASSIGNMENT_TYPE = 'assignment'


def syntax_check(code, verbose=False):
    """检查代码语法是否正确"""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    """提取文本中语法正确的最长代码段"""
    lines = text.split('\n')
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = '\n'.join(lines[i:j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i:j + 1]
                                     if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return '\n'.join(lines[longest_line_pair[0]:longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    """获取节点依赖关系"""
    if not TREE_SITTER_AVAILABLE:
        return {}
    
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode('utf8'))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    """获取函数依赖关系"""
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    """获取定义名称"""
    if not TREE_SITTER_AVAILABLE:
        return ""
    
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode('utf8')
    return ""


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """遍历语法树"""
    if not TREE_SITTER_AVAILABLE:
        return
    
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    """使用tree-sitter提取目标代码"""
    if not TREE_SITTER_AVAILABLE:
        return code_extract(code.strip())
    
    code = code_extract(code.strip())
    code_bytes = bytes(code, 'utf8')
    
    try:
        parser = get_parser('python')
        tree = parser.parse(code_bytes)
    except:
        return code_extract(code.strip())
    
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names
                    or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names
                    or name in class_names):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (child.type == EXPRESSION_TYPE
              and child.children[0].type == ASSIGNMENT_TYPE):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names
                    or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b''

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reachable):
            continue
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    sanitized_output = sanitized_output[:-1].decode('utf8')

    # ad-hoc approach to remove unnecessary lines, but it works
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(' '):
            break
        if not lines[i].startswith(' ') and entrypoint and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = '\n'.join(lines[:outer_lines[-1]])
    return sanitized_output

def execute_code_safely(code: str, timeout: int = 10) -> bool:
    """
    安全地执行Python代码并返回是否成功
    """
    temp_file = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 使用subprocess执行代码，设置超时
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # 如果返回码为0且没有错误输出，认为测试通过
            success = result.returncode == 0
            
            if not success:
                eval_logger.warning(f"Code execution failed with return code {result.returncode}")
                eval_logger.warning(f"stderr: {result.stderr}")
                eval_logger.warning(f"stdout: {result.stdout}")
            
            return success
            
        except subprocess.TimeoutExpired:
            eval_logger.warning("Code execution timed out")
            return False
        except Exception as e:
            eval_logger.warning(f"Code execution error: {e}")
            return False
                
    except Exception as e:
        eval_logger.warning(f"Error creating temporary file: {e}")
        return False
    finally:
        # 确保临时文件被删除
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                eval_logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                eval_logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

def extract_code_from_generation(generation: str, entrypoint: Optional[str] = None) -> str:
    """
    从模型生成的响应中提取Python代码
    优先提取markdown格式的```python代码块
    """
    all_possible_code_blocks = []
    
    # 1. 最优先：提取markdown格式的```python代码块
    if "```python" in generation:
        start_marker = "```python"
        start_pos = generation.find(start_marker)
        if start_pos != -1:
            start_pos += len(start_marker)
            # 跳过```python后的换行符
            if start_pos < len(generation) and generation[start_pos] == '\n':
                start_pos += 1
            end_pos = generation.find("```", start_pos)
            if end_pos != -1:
                code = generation[start_pos:end_pos].strip()
                # 如果提取到的markdown代码语法正确，直接返回
                if code and syntax_check(code):
                    return code
                all_possible_code_blocks.append(code)
    
    # 2. 如果没有```python，尝试普通的```代码块
    elif "```" in generation:
        parts = generation.split("```")
        if len(parts) >= 3:  # 至少有开始和结束标记
            # 通常第二个部分是代码（索引1）
            code = parts[1].strip()
            # 如果第二个部分是语言标识符，取第三个部分
            if code in ['python', 'py'] and len(parts) > 2:
                code = parts[2].strip()
            # 如果提取到的代码语法正确，直接返回
            if code and syntax_check(code):
                return code
            all_possible_code_blocks.append(code)
    
    # 3. 如果markdown方法没有成功，尝试高级的tree-sitter方法
    try:
        sanitized_code = extract_target_code_or_empty(generation, entrypoint).strip()
        if sanitized_code and syntax_check(sanitized_code):
            return sanitized_code
        if sanitized_code:
            all_possible_code_blocks.append(sanitized_code)
    except Exception as e:
        eval_logger.debug(f"Tree-sitter extraction failed: {e}")
    
    # 4. 尝试基础的代码语法提取
    try:
        code_from_syntax = code_extract(generation).strip()
        if code_from_syntax and syntax_check(code_from_syntax):
            return code_from_syntax
        if code_from_syntax:
            all_possible_code_blocks.append(code_from_syntax)
    except Exception as e:
        eval_logger.debug(f"Syntax-based extraction failed: {e}")
    
    # 5. 处理<code>标签格式
    code = generation.split("</code>")[0]
    if "```python" in code:
        code = code.split("```python")[1]
    code = code.split("```")[0]
    if "<code>" in code:
        code = code.split("<code>")[1]
    if code.strip():
        all_possible_code_blocks.append(code.strip())
    
    # 选择最佳的代码块（语法正确且最长的）
    best_code = ""
    for code_block in all_possible_code_blocks:
        code_block = code_block.strip()
        if code_block and syntax_check(code_block) and len(code_block) > len(best_code):
            best_code = code_block
    
    # 如果没有找到语法正确的代码，选择最长的非空代码块
    if not best_code:
        for code_block in all_possible_code_blocks:
            code_block = code_block.strip()
            if code_block and len(code_block) > len(best_code):
                best_code = code_block
    
    # 如果所有方法都没有提取到有效代码，返回原始内容
    if not best_code:
        best_code = generation.strip()
    
    return best_code

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """
    处理BigCodeBench的评测结果
    """
    eval_logger.info(f"Processing BigCodeBench evaluation")
    assert len(results) == 1
    
    generation = results[0]
    
    # 获取entrypoint（如果存在）
    entrypoint = doc.get('entrypoint', None)
    
    # 从生成的响应中提取代码
    extracted_code = extract_code_from_generation(generation, entrypoint)
    eval_logger.debug(f"Original generation:\n{generation}")
    eval_logger.debug(f"Extracted code:\n{extracted_code}")
    eval_logger.debug(f"Entrypoint: {entrypoint}")
    
    # 组装完整的可执行代码
    # yaml配置中已经组合了instruct_prompt和complete_prompt作为输入
    # 这里只需要添加生成的代码实现
    code = doc['complete_prompt'] + '\n' + extracted_code
    code = code + '\n\n' + doc['test'] + '\n\nif __name__ == "__main__":\n    unittest.main()'
    
    eval_logger.debug(f"Generated code:\n{code}")
    
    # 执行代码并获取结果
    success = execute_code_safely(code)
    
    result = {
        "acc": 1 if success else 0
    }
    
    eval_logger.info(f"Evaluation result: {result}")
    return result