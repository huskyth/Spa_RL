# eval_prm.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

import sys
import pathlib
p = pathlib.Path(__file__).parent.parent
if str(p) not in sys.path:
    sys.path.append(str(p))

from fastchat.model.model_adapter import get_model_adapter
from fastchat.conversation import SeparatorStyle
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ---------- 模型定义（与训练代码一致） ----------
class prm_model(PreTrainedModel):
    def __init__(self, base_model, vocab_size=32000):
        super().__init__(base_model.config)
        self.backbone = base_model
        self.LN = nn.Linear(vocab_size, 1).to(torch.bfloat16)
        nn.init.zeros_(self.LN.weight)
        nn.init.zeros_(self.LN.bias)
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, gpt_unmask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
        batch_predictions = []
        for i in range(outputs.size(0)):
            sample_labels = gpt_unmask[i]
            valid_indices = torch.where(sample_labels != -100)[0]
            if len(valid_indices) == 0:
                zero_val = torch.zeros(1, device=outputs.device, dtype=outputs.dtype)
                zero_val = zero_val + outputs[i, 0, 0] * 0 + 1e-6
                batch_predictions.append(zero_val)
                continue
            turn_end_indices = []
            for j in range(1, len(valid_indices)):
                if valid_indices[j] - valid_indices[j - 1] > 1:
                    turn_end_indices.append(valid_indices[j - 1])
            turn_end_indices.append(valid_indices[-1])
            turn_logits = []
            for idx in turn_end_indices:
                turn_logits.append(outputs[i, idx, :])
            turn_logits = torch.stack(turn_logits)
            turn_values = self.LN(turn_logits)
            sample_prediction = turn_values.sum()
            batch_predictions.append(sample_prediction.unsqueeze(0))
        value_outputs = torch.cat(batch_predictions)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(value_outputs, labels)
        return {
            'loss': loss,
            'predictions': value_outputs
        }


# ---------- 数据预处理（仅保留 Llama-3.2 格式） ----------
def preprocess(
        sources,
        tokenizer,
        model_path: str,
) -> Dict:
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

    # 仅处理 Llama-3.2 格式（与训练代码一致）
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
            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        return dict(
            input_ids=input_ids,
            gpt_unmask=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    else:
        raise NotImplementedError("Only Llama-3.2-3B-Instruct is supported for evaluation script.")


# ---------- 评估数据集 ----------
class EvalDataset(Dataset):
    def __init__(self, raw_data, tokenizer, model_path):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.model_path = model_path
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)
        self.input_ids = data_dict["input_ids"]
        self.gpt_unmask = data_dict["gpt_unmask"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = torch.tensor([example['agent_final_reward'] for example in raw_data], dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'gpt_unmask': self.gpt_unmask[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# ---------- 评估函数 ----------
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gpt_unmask = batch['gpt_unmask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, gpt_unmask=gpt_unmask)
            preds = outputs['predictions'].cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    acc = np.mean(np.abs(all_preds - all_labels) <= 0.1)
    return {
        'mse': float(mse),
        'mae': float(mae),
        'accuracy': float(acc)
    }


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="Evaluate PRM model on test set")
    parser.add_argument("--model_dir", type=str, default="records/progress_model_webshop/",
                        help="Directory containing our_model_state.pt and our_base_model")
    parser.add_argument("--test_data", type=str, default="exploration/webshop/exploration_outputs/test_prm.json",
                        help="Path to test data JSON file (each line a JSON object)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 tokenizer 和 base model
    base_model_path = os.path.join(args.model_dir, "checkpoint-57465")
    print(f"Loading tokenizer and base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|reserved_special_token_0|>'
    tokenizer.model_max_length = args.max_length
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)

    # 创建 PRM 模型并加载训练好的权重
    vocab_size = base_model.config.vocab_size
    model = prm_model(base_model, vocab_size)
    state_dict_path = os.path.join(args.model_dir, "our_model_state.pt")
    print(f"Loading model state from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 加载测试数据
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    # 创建数据集和 DataLoader
    print("Preprocessing test data...")
    eval_dataset = EvalDataset(test_data, tokenizer, model_path="Llama-3.2-3B-Instruct")
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # 执行评估
    print("Starting evaluation...")
    metrics = evaluate(model, dataloader, device)
    print("\nEvaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()