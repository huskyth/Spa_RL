  python sft/chat.py \
    --base_model_name_or_path './models/Llama-3.2-3B-Instruct' \
    --lora_adapter_path './ckpt/llama3b_webshop_sft/checkpoint-700' \
    --model_max_length 2048 \
    --temperature 0.0 \
    --max_new_tokens 2048