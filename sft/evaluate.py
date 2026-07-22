import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

p = pathlib.Path(__file__).parent.parent
if str(p) not in sys.path:
    sys.path.append(str(p))

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

from peft import PeftModel, get_peft_model_state_dict

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ---------------------------- 数据预处理（与训练完全一致） ----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(default=False)
    padding_side: str = field(default="right")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    lazy_preprocess: bool = False


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        model_path: str,
) -> Dict:
    """完全复制训练脚本中的 preprocess 函数，保证预处理一致"""
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

    # Llama-3.2/3.1 特殊处理
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

    # 其他模型分隔符处理
    if conv.sep_style == SeparatorStyle.LLAMA3:
        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i % 2 == 0:
                    instruction_len = len(tokenizer(turn).input_ids)
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += instruction_len if i == 0 else instruction_len + 1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids)
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                    target[cur_len: cur_len + 2] = IGNORE_TOKEN_ID
                    cur_len += turn_len + 1
            target[cur_len:] = IGNORE_TOKEN_ID
            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))

    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    else:
        raise NotImplementedError

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and conv.roles[0] == 'USER':
                instruction_len -= 1
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            if conv.sep2 == '</s>':
                cur_len += turn_len + 1
            elif conv.sep2 == ' </s><s>':
                cur_len += turn_len + 3
            else:
                raise NotImplementedError
            if i != 0 and conv.roles[0] == 'USER':
                cur_len -= 1
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, tokenizer, model_path):
        super().__init__()
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )


def load_eval_dataset(data_path, tokenizer, model_path):
    """加载测试集并返回 Dataset 对象和原始数据列表"""
    raw_data = json.load(open(data_path, "r"))
    dataset = SupervisedDataset(raw_data, tokenizer, model_path)
    return dataset, raw_data


# ---------------------------- 主评估函数 ----------------------------
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model on test set")
    parser.add_argument("--base_model_name_or_path", type=str, required=True,
                        help="Path to the base model (same as used in training)")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to the saved LoRA adapter (output_dir from training)")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test data JSON file (same format as training data)")
    parser.add_argument("--model_max_length", type=int, default=512,
                        help="Maximum sequence length (should match training)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Evaluation batch size per device")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="Directory to save evaluation results (optional)")
    parser.add_argument("--flash_attn", action="store_true", default=False,
                        help="Use FlashAttention if available")
    # 新增生成相关参数
    parser.add_argument("--save_generations", action="store_true",
                        help="Save model generations for each test sample")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per sample")
    parser.add_argument("--generations_file", type=str, default="generations.json",
                        help="Filename for saving generations (under output_dir)")
    args = parser.parse_args()

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # 处理 pad_token（与训练代码一致）
    if tokenizer.pad_token is None:
        if 'Llama-3.2-3B-Instruct' in args.base_model_name_or_path or \
                'Llama-3.1-8B-Instruct' in args.base_model_name_or_path:
            tokenizer.pad_token = '<|reserved_special_token_0|>'
        else:
            tokenizer.pad_token = tokenizer.unk_token

    # 2. 加载基础模型
    config = transformers.AutoConfig.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = False  # 评估时也可关闭缓存
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        config=config,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        torch_dtype=torch.float16,  # 根据硬件情况可调整
        trust_remote_code=True,
    )

    # 3. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model.eval()
    print(f"Loaded LoRA adapter from {args.lora_adapter_path}")

    # 4. 准备测试数据集（同时保留原始数据用于记录）
    test_dataset, raw_data = load_eval_dataset(
        args.test_data_path, tokenizer, args.base_model_name_or_path
    )

    # 5. 设置 TrainingArguments 用于 Trainer
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        remove_unused_columns=False,   # 保留额外字段（虽然我们不添加）
        report_to=None,
        run_name="eval_run",
    )

    # 6. 创建 Trainer 并评估（获得整体损失）
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
    )
    eval_results = trainer.evaluate()
    test_loss = eval_results.get("eval_loss", None)
    if test_loss is not None:
        perplexity = math.exp(test_loss)
        print(f"\n========== Evaluation Results ==========")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"========================================")
    else:
        print("Error: eval_loss not found in evaluation results.")

    # 7. 如果指定了保存生成结果，则进行生成
    if args.save_generations:
        print("Generating responses for each test sample...")
        # 准备 DataLoader（shuffle=False）
        dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "indices": list(range(len(batch))),  # 记录原始索引
            }
        )

        generations = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                indices = batch["indices"]

                # 对每个样本单独生成（避免批次内 padding 干扰）
                for i, idx in enumerate(indices):
                    # 提取有效 token（去除 padding）
                    seq_len = attention_mask[i].sum().item()
                    prompt_ids = input_ids[i, :seq_len].unsqueeze(0)  # (1, seq_len)

                    # 生成
                    generated_ids = model.generate(
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,          # 贪心解码
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        repetition_penalty=1.0,
                    )
                    # 解码生成的 token（去除 prompt 部分）
                    response_ids = generated_ids[0, seq_len:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                    # 记录结果
                    generations.append({
                        "index": idx,
                        "conversations": raw_data[idx]["conversations"],  # 原始对话
                        "generated_response": response_text,
                    })
                # 可选打印进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")

        # 保存生成结果
        output_path = pathlib.Path(args.output_dir) / args.generations_file
        with open(output_path, "w") as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)
        print(f"Generations saved to {output_path}")

    # 额外保存一个包含整体损失和生成文件路径的摘要（可选）
    summary = {
        "test_loss": test_loss,
        "perplexity": perplexity if test_loss is not None else None,
        "generations_saved": args.save_generations,
    }
    if args.save_generations:
        summary["generations_file"] = str(pathlib.Path(args.output_dir) / args.generations_file)
    summary_path = pathlib.Path(args.output_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    evaluate()