# This is an example of PPO training for webshop environment
# You can change the data path according to different environments (e.g., alfworld, virtualhome)

export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo_example"
export CUDA_VISIBLE_DEVICES="0"

export MODEL_TYPE="llama3-1"
export MODEL_PATH="./ckpt/llama3b_webshop_sft_loramerged"


python ppo/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/StepTool_ppo.json \
    --data_file prm/sampled_data_rl_training_webshop_flatten.json \
    --epochs 5