# terminal 3
python -m eval_agent.main \
    --agent_config fastchat \
    --model_name llama3b_webshop_rl_loramerged \
    --exp_config webshop \
    --split test \
    --override \
    --output_path eval/webshop_eval