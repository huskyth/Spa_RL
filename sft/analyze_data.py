import json
import os
from collections import Counter, defaultdict

# ===== 配置 =====
DATA_PATH = "data/webshop_sft_test.json"          # 数据集路径
SAMPLE_COUNT = 3                            # 打印样例个数
# =================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(data):
    print("="*60)
    print(f"数据集路径: {DATA_PATH}")
    print(f"总样本数: {len(data)}")
    print("="*60)

    # 1. 检查整体类型
    if isinstance(data, list):
        print("数据格式: 列表 (List)")
        sample = data[0] if data else None
    elif isinstance(data, dict):
        print("数据格式: 字典 (Dict)，可能包含 'data' 等键")
        # 常见格式：{"data": [...]} 或 {"train": [...]}
        if "data" in data:
            sample = data["data"][0] if data["data"] else None
        elif "train" in data:
            sample = data["train"][0] if data["train"] else None
        else:
            sample = next(iter(data.values()))[0] if data else None
    else:
        print("未知数据格式")
        return

    if not sample:
        print("数据集为空或无法解析")
        return

    print(f"\n第一条样本结构:")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "..." if len(json.dumps(sample)) > 500 else json.dumps(sample, indent=2, ensure_ascii=False))

    # 2. 提取所有样本（统一为列表）
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if "data" in data:
            samples = data["data"]
        elif "train" in data:
            samples = data["train"]
        else:
            # 尝试取第一个值
            samples = list(data.values())[0] if data else []
    else:
        samples = []

    # 3. 统计字段
    all_keys = set()
    for s in samples:
        all_keys.update(s.keys())
    print(f"\n所有样本共有的字段: {all_keys}")

    # 4. 分析对话结构（假设数据包含 'conversations' 或 'messages' 字段）
    # 常见字段名: conversations, messages, dialogue, turns
    possible_conv_keys = ['conversations', 'messages', 'dialogue', 'turns']
    conv_key = None
    for key in possible_conv_keys:
        if key in sample:
            conv_key = key
            break
    # 也可能直接是单轮，没有嵌套
    if not conv_key:
        # 检查是否所有样本都有 'instruction' 和 'output' 等
        if 'instruction' in sample and 'output' in sample:
            print("\n数据格式: 单轮问答 (instruction + output)")
            instr_len = [len(s.get('instruction', '')) for s in samples]
            out_len = [len(s.get('output', '')) for s in samples]
            print(f"Instruction 长度 (字符): 平均 {sum(instr_len)/len(instr_len):.1f}, 最小 {min(instr_len)}, 最大 {max(instr_len)}")
            print(f"Output 长度 (字符): 平均 {sum(out_len)/len(out_len):.1f}, 最小 {min(out_len)}, 最大 {max(out_len)}")
            # 打印样例
            print(f"\n前 {SAMPLE_COUNT} 个样例:")
            for i, s in enumerate(samples[:SAMPLE_COUNT]):
                print(f"\n--- 样本 {i+1} ---")
                print(f"指令: {s.get('instruction', '')[:200]}...")
                print(f"输出: {s.get('output', '')[:200]}...")
        else:
            print("\n未识别为常见对话格式，直接打印整体统计:")
            print(f"每个样本的键: {all_keys}")
            # 对每个键统计长度（如果值是字符串）
            for key in all_keys:
                if isinstance(sample.get(key), str):
                    lens = [len(s.get(key, '')) for s in samples]
                    print(f"  {key} 平均长度: {sum(lens)/len(lens):.1f}, 最小 {min(lens)}, 最大 {max(lens)}")
            print(f"\n前 {SAMPLE_COUNT} 个样例:")
            for i, s in enumerate(samples[:SAMPLE_COUNT]):
                print(f"\n--- 样本 {i+1} ---")
                print(json.dumps(s, indent=2, ensure_ascii=False)[:500])
        return

    # 5. 有对话字段（conversations/messages）
    print(f"\n检测到对话字段: '{conv_key}'")
    # 统计每个对话中的消息条数
    msg_counts = []
    role_counter = Counter()
    total_msg_len = 0
    total_msg_count = 0

    # 收集每条消息的长度（字符）
    for s in samples:
        conv = s.get(conv_key, [])
        if isinstance(conv, list):
            msg_counts.append(len(conv))
            for msg in conv:
                if isinstance(msg, dict):
                    role = msg.get('role', msg.get('from', ''))
                    content = msg.get('content', msg.get('text', msg.get('value', '')))
                    if role:
                        role_counter[role] += 1
                    if isinstance(content, str):
                        total_msg_len += len(content)
                        total_msg_count += 1
        elif isinstance(conv, dict):
            # 如果对话是dict，例如 {"system":..., "user":..., "assistant":...}
            for role, content in conv.items():
                if isinstance(content, str):
                    role_counter[role] += 1
                    total_msg_len += len(content)
                    total_msg_count += 1
            msg_counts.append(len(conv))

    if msg_counts:
        print(f"对话条数分布: 平均 {sum(msg_counts)/len(msg_counts):.1f}, 最小 {min(msg_counts)}, 最大 {max(msg_counts)}")
    if total_msg_count > 0:
        print(f"所有消息平均长度 (字符): {total_msg_len/total_msg_count:.1f}")
    print(f"角色分布: {dict(role_counter)}")

    # 6. 打印样例对话
    print(f"\n前 {SAMPLE_COUNT} 条对话样例:")
    for i, s in enumerate(samples[:SAMPLE_COUNT]):
        print(f"\n--- 样本 {i+1} ---")
        conv = s.get(conv_key, [])
        if isinstance(conv, list):
            for msg in conv:
                role = msg.get('role', msg.get('from', 'unknown'))
                content = msg.get('content', msg.get('text', msg.get('value', '')))
                print(f"{role}: {str(content)[:300]}...")
        elif isinstance(conv, dict):
            for role, content in conv.items():
                print(f"{role}: {str(content)[:300]}...")
        else:
            print(conv)

    # 7. 附加统计：平均总长度（字符）
    total_lens = []
    for s in samples:
        conv = s.get(conv_key, [])
        if isinstance(conv, list):
            total = sum(len(str(msg.get('content', msg.get('text', msg.get('value', ''))))) for msg in conv if isinstance(msg, dict))
        elif isinstance(conv, dict):
            total = sum(len(str(v)) for v in conv.values() if isinstance(v, str))
        else:
            total = 0
        total_lens.append(total)
    if total_lens:
        print(f"\n每条对话总字符数: 平均 {sum(total_lens)/len(total_lens):.1f}, 最小 {min(total_lens)}, 最大 {max(total_lens)}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"错误: 文件 {DATA_PATH} 不存在")
        exit(1)
    data = load_json(DATA_PATH)
    analyze_dataset(data)