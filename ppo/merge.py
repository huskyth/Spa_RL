from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "ckpt/llama3b_webshop_sft_loramerged"
lora_model_path = "./ckpt/steptool_llama3-11759499040/epoch-4"
merged_model_path = "./ckpt/llama3b_webshop_rl_loramerged"

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 保存合并前的参数
params_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, lora_model_path)


# 合并 LoRA 到基础模型
model = model.merge_and_unload()

# 保存合并后的参数
params_after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# 4. 对比差异
changed_params = []
for name in params_before:
    if not torch.equal(params_before[name], params_after[name]):
        changed_params.append(name)

print(f"总参数数量: {len(params_before)}")
print(f"发生变化的参数数量: {len(changed_params)}")
print("变化的参数示例:", changed_params[:20])  # 只打印前20个

# # 保存为 HF 格式
model.save_pretrained(merged_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)
