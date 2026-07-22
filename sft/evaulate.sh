python sft/evaluate.py \
    --base_model_name_or_path './models/Llama-3.2-3B-Instruct' \
    --lora_adapter_path './ckpt/llama3b_webshop_sft/checkpoint-700' \
    --test_data_path 'data/webshop_sft_test.json' \
    --model_max_length 2048 \
    --batch_size 4 \
    --flash_attn \
    --save_generations \
    --max_new_tokens 2048 \
    --generations_file my_generations.json