from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
# -------------------------- 1. 新增 LoRA 依赖 --------------------------
from peft import PeftModel, LoraConfig, get_peft_model  # LoRA 核心库
from peft.utils import get_peft_model_state_dict  # LoRA 状态字典保存

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
import random
import os
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(default=False)
    padding_side: str = field(default="right", metadata={"help": "The padding side in tokenizer"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    # -------------------------- 2. 新增 LoRA 专属参数（命令行可配置） --------------------------
    lora_r: int = field(default=8, metadata={"help": "LoRA low-rank dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha (scaling factor)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate."})
    lora_target_modules: Sequence[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "gate_proj", "up_proj"],
        metadata={"help": "Target modules for LoRA (Llama 关键模块)."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias training mode (none/lora_only/all)."})
    flash_attn: bool = field(default=True, metadata={"help": "Enable FlashAttention for speed/显存优化."})


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    """适配 LoRA 模型保存，仅保存 LoRA 适配器（避免保存完整主模型）"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    # 若为 LoRA 模型，优先保存 LoRA 状态字典
    if hasattr(trainer.model, "peft_config"):
        peft_state_dict = get_peft_model_state_dict(trainer.model)
        trainer.model.save_pretrained(
            trainer.args.output_dir,
            state_dict=peft_state_dict,
            safe_serialization=True
        )
        rank0_print(f"LoRA adapter saved to {trainer.args.output_dir}")
        return

    # 原 FSDP 保存逻辑（兼容全量微调）
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    # 【保留原逻辑不变】数据预处理（对话模板、掩码用户输入）
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

    print("tokenizer.model_max_length:", tokenizer.model_max_length)
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    
    # Llama-3.2/3.1 特殊处理（保留原逻辑）
    if 'Llama-3.2-3B-Instruct' in model_path or 'Llama-3.1-8B-Instruct' in model_path:
        sep2 = "<|eot_id|>"
        sep = "<|end_header_id|>"
        targets = targets[:,1:]
        input_ids = input_ids[:,1:]
        
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if i%2 == 0:
                    if i == 0:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                        cur_len += instruction_len
                    else:
                        instruction_len = len(tokenizer(turn).input_ids[1:])
                        target[cur_len: cur_len + instruction_len+1] = IGNORE_TOKEN_ID
                        cur_len += instruction_len+1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids[1:])
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids[1:])
                    target[cur_len: cur_len +2] = IGNORE_TOKEN_ID
                    cur_len += turn_len+1
            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

        return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))
    
    # 其他模型分隔符处理（保留原逻辑）
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
                if i%2 == 0:
                    instruction_len = len(tokenizer(turn).input_ids)
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += instruction_len if i==0 else instruction_len+1
                else:
                    parts = turn.split(sep)
                    turn_len = len(tokenizer(turn).input_ids)
                    if len(parts) != 2:
                        break
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                    target[cur_len: cur_len +2] = IGNORE_TOKEN_ID
                    cur_len += turn_len+1
            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

        return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))

    # 其他分隔符风格（保留原逻辑）
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
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
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
            rank0_print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


# 保留原数据集类（SupervisedDataset/LazySupervisedDataset）和 make_supervised_data_module
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(SupervisedDataset, self).__init__()
        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])


class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(LazySupervisedDataset, self).__init__()
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.model_path = model_path

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.model_path)
        ret = dict(input_ids=ret["input_ids"][0], labels=ret["labels"][0], attention_mask=ret["attention_mask"][0])
        self.cached_data_dict[i] = ret
        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_path: str = None
) -> Dict:
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    rank0_print("Loading data...")
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_path=model_path)
    eval_dataset = dataset_cls(json.load(open(data_args.eval_data_path, "r")), tokenizer=tokenizer, model_path=model_path) if data_args.eval_data_path else None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    # 解析参数（包含新增的 LoRA 参数）
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # 1. 加载模型配置（保留原 RoPE 缩放逻辑）
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False  # 训练时禁用缓存

    # 2. 加载模型（兼容 FlashAttention）
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if training_args.flash_attn else "eager",
        torch_dtype=torch.float16,  # 强制 FP16 减少显存占用
    )

    # -------------------------- 3. 配置并包装 LoRA 模型 --------------------------
    lora_config = LoraConfig(
        r=training_args.lora_r,  # 低秩维度（命令行传入）
        lora_alpha=training_args.lora_alpha,  # 缩放因子
        target_modules=list(training_args.lora_target_modules),  # 目标模块（Llama 注意力+MLP）
        lora_dropout=training_args.lora_dropout,  # dropout 率
        bias=training_args.lora_bias,  # bias 训练模式
        task_type="CAUSAL_LM",  # 因果语言模型（SFT 任务）
        inference_mode=False,  # 训练模式
    )
    # 包装为 LoRA 模型
    model = get_peft_model(model_base, lora_config)
    # 打印可训练参数比例（验证 LoRA 配置生效）
    model.print_trainable_parameters()  # 输出示例：trainable params: 1.2M | all params: 3.0B | trainable%: 0.04%

    # 解决 LoRA 与梯度检查点的冲突
    model.enable_input_require_grads()
    
    # 正确配置梯度检查点，使用非重入式模式（适配 PyTorch 2.0+）
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # 额外保险：确保梯度检查点不忽略 LoRA 参数
        for name, module in model.named_modules():
            if "lora" in name or "Lora" in name:
                module._recompute_module = True  # 标记 LoRA 模块需要重计算
    

    # 4. 加载 Tokenizer（保留原逻辑）
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    # 处理 Llama 无 pad_token 问题
    if tokenizer.pad_token is None:
        if 'Llama-3.2-3B-Instruct' in model_args.model_name_or_path or 'Llama-3.1-8B-Instruct' in model_args.model_name_or_path:
            tokenizer.pad_token = '<|reserved_special_token_0|>'
        else:
            tokenizer.pad_token = tokenizer.unk_token

    # 5. 加载数据（保留原逻辑）
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_path=model_args.model_name_or_path)

    # 6. 初始化 Trainer（保留原逻辑）
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 保存模型
    trainer.save_state()
    # 若为 LoRA 模型，仅保存适配器权重
    if hasattr(trainer.model, "peft_config"):
        # 获取纯 LoRA 状态字典（排除主模型参数）
        from peft.utils import get_peft_model_state_dict
        peft_state_dict = get_peft_model_state_dict(trainer.model)
        # 检查几个 LoRA 权重的 sum 和是否为零
        for name, param in peft_state_dict.items():
            tensor_sum = param.sum().item()
            print(f"{name}: shape={param.shape}, sum={tensor_sum}")
            # 只打印前几项即可
            if list(peft_state_dict.keys()).index(name) >= 5:
                break
        # 保存 LoRA 适配器（仅生成 adapter_model.bin 和 adapter_config.json）
        trainer.model.save_pretrained(
            training_args.output_dir,
            # state_dict=peft_state_dict,
            safe_serialization=True  # 安全序列化，防止文件损坏
        )
        
        # 打印保存信息（验证保存路径）
        rank0_print(f"LoRA 适配器已保存至: {training_args.output_dir}")
        rank0_print(f"适配器文件包括: adapter_model.bin (权重) 和 adapter_config.json (配置)")
    else:
        # 兼容全量微调的保存逻辑（备用）
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)


    # 2. 加载 LoRA 适配器
    model_new = PeftModel.from_pretrained(model_base, training_args.output_dir)
    peft_state_dict = get_peft_model_state_dict(model_new)
    # 检查几个 LoRA 权重的 sum 和是否为零
    for name, param in peft_state_dict.items():
        tensor_sum = param.sum().item()
        print(f"{name}: shape={param.shape}, sum={tensor_sum}")
        # 只打印前几项即可
        if list(peft_state_dict.keys()).index(name) >= 5:
            break


if __name__ == "__main__":
    train()