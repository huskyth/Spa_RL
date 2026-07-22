import sys

import torch
import json
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import PeftModel
from datasets import Dataset
from pathlib import Path
p = Path(__file__).parent.parent
if str(p) not in sys.path:
    sys.path.append(str(p))

from fastchat.model.model_adapter import get_model_adapter
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ================== 从训练脚本中复制的 preprocess 函数（略作调整） ==================
# 注意：需要依赖 fastchat 和 IGNORE_TOKEN_ID，确保已安装 fastchat

def preprocess(sources, tokenizer, model_path: str) -> Dict:
    """与训练代码完全一致的预处理函数"""
    conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # 以下是 Llama-3.2/3.1 特殊处理（与你训练代码保持一致）
    if 'Llama-3.2-3B-Instruct' in model_path or 'Llama-3.1-8B-Instruct' in model_path:
        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"
        targets = targets[:, 1:]
        input_ids = input_ids[:, 1:]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i % 2 == 0:
                    if i == 0:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                        cur_len += instruction_len
                    else:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len + 1] = IGNORE_TOKEN_ID
                        cur_len += instruction_len + 1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids[1:])
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids[1:])
                    target[cur_len: cur_len + 2] = IGNORE_TOKEN_ID
                    cur_len += turn_len + 1
            target[cur_len:] = IGNORE_TOKEN_ID
            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))

    # 其余分支（若模型不是 Llama-3.2，可省略，但保留以防万一）
    # ...（此处可略，因为你的模型是 Llama-3.2）
    # 但为保险，可完整复制训练脚本中后续所有 elif 和 else 分支。
    # 为了简洁，这里仅保留 Llama-3.2 分支，因为你的模型是这个。
    # 如果你未来换模型，请确保补全其他分支。

    # 对于非 Llama-3.2 模型，请从训练脚本中复制对应逻辑
    # 由于你明确使用 Llama-3.2-3B-Instruct，上述分支已覆盖。


# ================== 主评估脚本 ==================
BASE_MODEL_PATH = "../models/Llama-3.2-3B-Instruct"
LORA_PATH = "../ckpt/llama3b_webshop_sft/checkpoint-1000"  # 训练好的最优 LoRA
TEST_DATA_PATH = "data/webshop_sft_test.json"  # 你的测试集
MAX_LENGTH = 2048
EVAL_BATCH_SIZE = 4
OUTPUT_DIR = "./eval_results"

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|reserved_special_token_0|>'  # 与训练一致

# 2. 加载 LoRA 模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 3. 加载测试数据并预处理（使用训练同样的逻辑）
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 假设数据是列表，每个元素有 "conversations" 字段（与训练一致）
if isinstance(raw_data, list):
    samples = raw_data
elif isinstance(raw_data, dict) and 'data' in raw_data:
    samples = raw_data['data']
else:
    raise ValueError("数据格式不符合预期，需包含 'conversations' 字段的列表")

# 调用 preprocess 生成 input_ids, labels, attention_mask
# 注意：preprocess 返回的是 batch 形式（维度为 [batch_size, seq_len]）
# 我们需要将所有样本拼接成一个大的 tensor，但为了处理方便，我们逐个样本处理并合并
all_input_ids = []
all_labels = []
all_attention_masks = []

# 分批处理以防内存爆炸（但通常测试集不大，可一次性处理）
# 这里为了安全，循环处理每个样本
for i in range(0, len(samples), 32):  # 每次处理32条，避免超长
    batch_samples = samples[i:i + 32]
    # 提取 conversations 列表
    convs = [s["conversations"] for s in batch_samples]
    data_dict = preprocess(convs, tokenizer, BASE_MODEL_PATH)
    all_input_ids.append(data_dict["input_ids"])
    all_labels.append(data_dict["labels"])
    all_attention_masks.append(data_dict["attention_mask"])

# 合并所有 batch
input_ids = torch.cat(all_input_ids, dim=0)
labels = torch.cat(all_labels, dim=0)
attention_mask = torch.cat(all_attention_masks, dim=0)

# 构建 HuggingFace Dataset
dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": attention_mask,
})

# 4. 设置评估参数
eval_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    fp16=True,
    report_to="none",
    dataloader_drop_last=False,
    remove_unused_columns=False,  # 保留所有列
)

# 5. 创建 Trainer 并评估
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)

print("开始评估测试集...")
eval_results = trainer.evaluate()

loss = eval_results["eval_loss"]
ppl = torch.exp(torch.tensor(loss)).item()

print("\n========== 最终测试结果 ==========")
print(f"测试集 Loss: {loss:.4f}")
print(f"测试集 Perplexity: {ppl:.2f}")