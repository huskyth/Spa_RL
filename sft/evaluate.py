import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

from peft import PeftModel, get_peft_model_state_dict

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ---------------------------- 复制训练代码中的预处理函数和数据类 ----------------------------
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
    """加载测试集并返回 Dataset 对象"""
    raw_data = json.load(open(data_path, "r"))
    return SupervisedDataset(raw_data, tokenizer, model_path)


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

    # 4. 准备测试数据集
    test_dataset = load_eval_dataset(args.test_data_path, tokenizer, args.base_model_name_or_path)

    # 5. 设置 TrainingArguments 用于 Trainer
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=None,  # 不记录到 wandb/tensorboard
        run_name="eval_run",
    )

    # 6. 创建 Trainer 并评估
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
    )
    # 计算损失
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


if __name__ == "__main__":
    evaluate()