#!/bin/bash
# Navigate to the lm-evaluation-harness directory
# Run the cceval evaluation with more detailed output
python -m lm_eval --model local-chat-completions \
      --model_args model=qwen3-coder-30b-a3b-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False \
      --tasks cceval_python_rg1_unixcoder \
      --apply_chat_template \
      --output_path ../output \
      --trust_remote_code \
      --log_samples \
      --confirm_run_unsafe_code \
      --limit 5 \