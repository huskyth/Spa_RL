
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import PeftModel
from datasets import Dataset

# ================== 配置 ==================
BASE_MODEL_PATH = "../models/Llama-3.2-3B-Instruct"
LORA_PATH = "../ckpt/llama3b_webshop_sft/checkpoint-1000"
TEST_DATA_PATH = "data/webshop_sft_test.json"
MAX_LENGTH = 2048
EVAL_BATCH_SIZE = 4
OUTPUT_DIR = "./eval_results"

# ================== 1. 加载分词器 ==================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================== 2. 加载 LoRA 模型 ==================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ================== 3. 加载并解析测试集 ==================
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

if isinstance(raw_data, list):
    samples = raw_data
elif isinstance(raw_data, dict) and 'data' in raw_data:
    samples = raw_data['data']
else:
    if len(raw_data) == 1 and isinstance(list(raw_data.values())[0], list):
        samples = list(raw_data.values())[0]
    else:
        raise ValueError("无法识别数据格式，请检查 JSON 结构")

texts = []
for s in samples:
    if 'text' in s:
        texts.append(s['text'])
    elif 'conversations' in s:
        conv_text = ""
        for turn in s['conversations']:
            role = turn.get('from', turn.get('role', 'unknown'))
            value = turn.get('value', turn.get('content', ''))
            conv_text += f"{role}: {value}\n"
        texts.append(conv_text.strip())
    elif 'instruction' in s and 'output' in s:
        texts.append(f"Instruction: {s['instruction']}\nOutput: {s['output']}")
    else:
        texts.append(json.dumps(s, ensure_ascii=False))

print(f"成功加载 {len(texts)} 条测试样本")

# ================== 4. 构建 Dataset ==================
dataset = Dataset.from_dict({"text": texts})

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    # 添加 labels（用于计算损失）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # 移除原始文本列
)

# ================== 5. 设置评估参数 ==================
eval_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    fp16=True,
    report_to="none",
    dataloader_drop_last=False,
    remove_unused_columns=True,  # 默认就是 True，但为了安全我们显式设置
)

# ================== 6. 创建 Trainer 并评估 ==================
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("开始评估测试集...")
eval_results = trainer.evaluate()

loss = eval_results["eval_loss"]
ppl = torch.exp(torch.tensor(loss)).item()

print("\n========== 最终测试结果 ==========")
print(f"测试集 Loss: {loss:.4f}")
print(f"测试集 Perplexity: {ppl:.2f}")