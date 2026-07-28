import os
import json
import glob
import random
from pathlib import Path

cr = Path(__file__).parent.parent

# 目标目录
explore_dir = cr / "exploration/webshop/exploration_outputs/explore"
json_files = glob.glob(os.path.join(explore_dir, "*.json"))
print(f"Found {len(json_files)} json files")

# 读取所有json文件
all_data = []
for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
        print(f"Successfully read: {json_file}")
    except Exception as e:
        print(f"Error reading {json_file}: {str(e)}")

print(f"\nCollected data from {len(all_data)} json files")

# 截断对话（保留前2轮和后12轮之后的部分）
truncated_data = []
for data_list in all_data:
    truncated_list = []
    for item in data_list:
        new_item = item.copy()
        new_item['agent_conversations'] = (
            item['agent_conversations'][:2] +
            item['agent_conversations'][10:]
        )
        truncated_list.append(new_item)
    truncated_data.append(truncated_list)

# 展开所有数据
our_data = []
for data_list in truncated_data:
    our_data.extend(data_list)

# 格式转换：角色映射，并移除最后的用户消息（若有）
saved_data = []
for each_item in our_data:
    conversation = each_item['agent_conversations']
    new_conversation = []
    for i in range(len(conversation)):
        if conversation[i]['role'] == 'user':
            new_conversation.append({'from': 'human', 'value': conversation[i]['content'].strip()})
        else:
            new_conversation.append({'from': 'gpt', 'value': conversation[i]['content'].strip()})
    if new_conversation and new_conversation[-1]['from'] == 'human':
        new_conversation = new_conversation[:-1]
    saved_data.append({
        'conversations': new_conversation,
        'agent_final_reward': each_item['agent_final_reward'],
        'id': each_item['id'],          # 此id为整数，直接作为任务编号
        'iteration': each_item['iteration'],
        'success': each_item['success'],
    })

# ========== 按整数任务ID划分训练/测试集 ==========
# 收集所有唯一的任务ID（整数）
task_ids = set()
for item in saved_data:
    task_ids.add(item['id'])   # 直接使用整数id

task_ids = list(task_ids)
random.seed(42)   # 固定随机种子，可复现
random.shuffle(task_ids)

# 划分比例（测试集20%）
test_ratio = 0.2
split_idx = int(len(task_ids) * (1 - test_ratio))
train_task_ids = set(task_ids[:split_idx])
test_task_ids = set(task_ids[split_idx:])

print(f"Total tasks: {len(task_ids)}, Train tasks: {len(train_task_ids)}, Test tasks: {len(test_task_ids)}")

# 根据任务ID分配样本
train_data = []
test_data = []
for item in saved_data:
    if item['id'] in train_task_ids:
        train_data.append(item)
    elif item['id'] in test_task_ids:
        test_data.append(item)
    else:
        # 安全兜底
        print(f"Warning: id {item['id']} not in any set")

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# ========== 保存文件 ==========
# 全部数据（原样）
with open(cr / "exploration/webshop/exploration_outputs/exploration.json", "w") as f:
    json.dump(saved_data, f, indent=4)
print("All data saved to exploration.json")

# 前100条（tiny）
with open(cr / "exploration/webshop/exploration_outputs/exploration_tiny.json", "w") as f:
    json.dump(saved_data[:100], f, indent=4)
print("Tiny data saved to exploration_tiny.json")

# 训练集和测试集（按任务严格划分）
with open(cr / "exploration/webshop/exploration_outputs/train.json", "w") as f:
    json.dump(train_data, f, indent=4)
print(f"Train set saved to train.json ({len(train_data)} samples)")

with open(cr / "exploration/webshop/exploration_outputs/test.json", "w") as f:
    json.dump(test_data, f, indent=4)
print(f"Test set saved to test.json ({len(test_data)} samples)")