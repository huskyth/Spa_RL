python sft/evaluate.py \
    --base_model_name_or_path './models/Llama-3.2-3B-Instruct' \
    --lora_adapter_path './ckpt/llama3b_webshop_sft/checkpoint-700' \
    --test_data_path 'data/webshop_sft_test.json' \
    --model_max_length 2048 \
    --batch_size 4 \
    --flash_attn  # 如果训练时使用了 FlashAttention，评估时建议保持一致