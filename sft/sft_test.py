from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import Trainer, TrainingArguments
import torch

# 1. 加载你训练好的最优 LoRA 模型
base_model = AutoModelForCausalLM.from_pretrained("../models/Llama-3.2-3B-Instruct", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "../ckpt/llama3b_webshop_sft")  # 这里就是最优的

tokenizer = AutoTokenizer.from_pretrained("../models/Llama-3.2-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # Llama 通常设置 eos 为 pad

# 2. 加载你的真实测试集（假设是 json 格式，已预处理）
# 注意：测试集不需要梯度，batch_size 可以设大一点
test_dataset = load_dataset("json", data_files="data/webshop_sft_test.json")  # 替换路径

# 3. 设置评估参数（仅推理）
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=8,
    fp16=True,
    report_to="none",  # 不传 wandb
)

# 4. 创建 Trainer 并评估
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 5. 执行评估，打印 Loss
eval_results = trainer.evaluate()
print(f"Test Loss: {eval_results['eval_loss']:.4f}")
print(f"Test Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])).item():.2f}")