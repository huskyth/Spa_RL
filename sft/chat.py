#!/usr/bin/env python3
import argparse
import sys
import pathlib
p = pathlib.Path(__file__).parent.parent
if str(p) not in sys.path:
    sys.path.append(str(p))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastchat.model.model_adapter import get_conversation_template

def main():
    parser = argparse.ArgumentParser(description="Chat with LoRA fine-tuned model")
    parser.add_argument("--base_model_name_or_path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to the saved LoRA adapter")
    parser.add_argument("--model_max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--flash_attn", action="store_true", default=False,
                        help="Use FlashAttention if available")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (1.0 = no change, 0.0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per turn")
    parser.add_argument("--no_stream", action="store_true",
                        help="Disable streaming output (print full response at once)")
    args = parser.parse_args()

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if 'Llama-3.2' in args.base_model_name_or_path or 'Llama-3.1' in args.base_model_name_or_path:
            tokenizer.pad_token = '<|reserved_special_token_0|>'
        else:
            tokenizer.pad_token = tokenizer.unk_token

    # 2. 加载基础模型
    config = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = True  # 推理时开启缓存
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        config=config,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # 3. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model.eval()
    print(f"Loaded LoRA adapter from {args.lora_adapter_path}")

    # 4. 准备对话模板（与训练时一致）
    conv = get_conversation_template(args.base_model_name_or_path)
    # 可选：设置系统提示（如果模型需要）
    # conv.system_message = "You are a helpful assistant."

    # 5. 交互循环
    print("\n=== 开始对话（输入 'exit' 或 'quit' 退出）===\n")
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        # 添加用户消息到对话历史
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)  # 占位，准备生成

        # 获取完整的对话 prompt
        prompt = conv.get_prompt()
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.model_max_length)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # 生成
        with torch.no_grad():
            if args.temperature == 0:
                # 贪心解码（确定性）
                do_sample = False
                temperature = None
            else:
                do_sample = True
                temperature = args.temperature

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.0,
                # 如果需要流式输出，可以使用 transformers 的 TextStreamer
                # 但这里为了兼容性，我们只做批量输出
            )

        # 解码生成的 tokens（只取新生成的）
        generated_tokens = generated_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 将回复添加到对话历史（实际内容）
        conv.messages[-1][1] = response  # 填充助手回复

        # 输出回复（支持流式显示，但目前一次性输出）
        if args.no_stream:
            print(f"助手: {response}\n")
        else:
            # 简单的逐字符流式打印（模拟）
            import sys
            print("助手: ", end="", flush=True)
            for char in response:
                print(char, end="", flush=True)
                # 可选：添加微小延迟，模拟打字效果
                # import time; time.sleep(0.02)
            print("\n")

if __name__ == "__main__":
    main()