#!/bin/bash

# 定义支持的语言
LANGUAGES=("python" "java" "csharp" "typescript")

# 定义模型参数
MODEL_ARGS="model=qwen3-coder-30b-a3b-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False"

# 为每种语言运行评估
for language in "${LANGUAGES[@]}"; do
    echo "Running evaluation for $language..."
    
    # 生成任务列表
    TASKS=""
    for file in lm_eval/tasks/cceval/cceval_${language}_*.yaml; do
        if [ -f "$file" ]; then
            task_name=$(basename "$file" .yaml)
            if [ -z "$TASKS" ]; then
                TASKS="$task_name"
            else
                TASKS="$TASKS,$task_name"
            fi
        fi
    done
    
    # 如果找到了任务，则运行评估
    if [ -n "$TASKS" ]; then
        echo "Running tasks: $TASKS"
        
        # 创建语言特定的输出目录
        OUTPUT_PATH="../output/$language"
        mkdir -p "$OUTPUT_PATH"
        
        # 运行评估
        python -m lm_eval \
            --model local-chat-completions \
            --model_args $MODEL_ARGS \
            --tasks $TASKS \
            --apply_chat_template \
            --output_path $OUTPUT_PATH \
            --trust_remote_code \
            --log_samples \
            --confirm_run_unsafe_code \
            --limit 5 \
            || echo "Failed to run evaluation for $language"
    else
        echo "No tasks found for $language"
    fi
    
    echo "Completed evaluation for $language"
    echo "----------------------------------------"
done

echo "All evaluations completed!"