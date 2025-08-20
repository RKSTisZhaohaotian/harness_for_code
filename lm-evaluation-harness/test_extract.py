#!/usr/bin/env python3
import re

def extract_code_blocks(text: str) -> str:
    # Pattern to match ```python...``` blocks
    pattern = r"```python\s*\n?(.*?)\n?```"
    
    # First try to match existing code blocks in the text
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no matches and text doesn't start with ```, try adding prefix
    if not text.strip().startswith("```"):
        matches = re.findall(pattern, "```python\n" + text + "\n```", re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Fallback: try generic code block pattern
    generic_pattern = r"```(?:\w+)?\s*\n?(.*?)\n?```"
    matches = re.findall(generic_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    return ""

# 测试用例
test_cases = [
    # 案例1：完整的python代码块
    """```python
def similar_elements(list1, list2):
    return tuple(set(list1) & set(list2))
```""",
    
    # 案例2：没有代码块标记的纯代码
    """def similar_elements(list1, list2):
    return tuple(set(list1) & set(list2))""",
    
    # 案例3：其他语言的代码块
    """```javascript
function test() {
    return 1;
}
```""",
    
    # 案例4：实际生成的格式（从日志中看到的）
    """```python
def similar_elements(list1, list2):
    return tuple(set(list1) & set(list2))
```"""
]

print("代码提取函数测试结果:")
print("=" * 50)

for i, test in enumerate(test_cases, 1):
    result = extract_code_blocks(test)
    print(f"测试案例 {i}:")
    print(f"输入: {repr(test[:50])}...")
    print(f"输出: {repr(result)}")
    print(f"成功: {'是' if result else '否'}")
    print("-" * 30) 