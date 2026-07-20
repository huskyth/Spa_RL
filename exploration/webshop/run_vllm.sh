CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.vllm_worker \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --port 21012 \
    --worker-address http://127.0.0.1:21012 \
    --model-path ckpt/llama3b_webshop_sft_loramerged \
    --host 0.0.0.0 \
    --controller http://127.0.0.1:21001