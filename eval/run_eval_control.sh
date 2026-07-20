# terminal 1
save_path=eval/webshop_eval/
logs_path=${save_path}logs

mkdir -p ${logs_path}

# 启动 controller
python -u -m fastchat.serve.controller --host 0.0.0.0