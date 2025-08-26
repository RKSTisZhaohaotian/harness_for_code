python -m lm_eval --model local-chat-completions \
    --model_args model=qwen3-coder-30b-a3b-instruct,base_url=https://qianfan.baidubce.com/v2/chat/completions,num_concurrent=1,max_retries=10,tokenized_requests=False \
    --tasks repobench \
    --apply_chat_template \
    --output_path ../output \
    --trust_remote_code \
    --log_samples \
    --confirm_run_unsafe_code \
    --limit 5