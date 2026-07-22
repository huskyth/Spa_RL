node_num=1  # number of GPUs
batch_size=1
micro_batch_size=1
# accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))
accumulation_step=1

# 采用0号来进行训练
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_path="./models/Llama-3.2-3B-Instruct"


python -m fastchat.train.train_lora_llama \
    --model_name_or_path ${model_path} \
    --data_path data/webshop_sft_train.json \
    --eval_data_path data/webshop_sft_test.json \
    --fp16 True \
    --output_dir ckpt/llama3b_webshop_sft \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps=100 \
    --load_best_model_at_end \
    --greater_is_better False \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --gradient_accumulation_steps 1 \
    --flash_attn True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --report_to "wandb"

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi
