# terminal 2
fs_worker_port=21012
CUDA_VISIBLE_DEVICES=0 \
python -u -m fastchat.serve.vllm_worker \
    --model-path ckpt/llama3b_webshop_rl_loramerged \
    --port ${fs_worker_port} \
     --host 0.0.0.0 \
    --worker-address http://0.0.0.0:${fs_worker_port} \
    --max-model-len 8192