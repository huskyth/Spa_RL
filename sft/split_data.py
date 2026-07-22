import json
import random
import argparse
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_dataset(input_file, output_prefix, train_ratio=0.9, seed=42):
    """
    按比例切分数据集为训练集和测试集（无验证集）
    :param input_file: 输入 JSON 文件路径
    :param output_prefix: 输出文件前缀，例如 'data/webshop' 会生成 train.json 和 test.json
    :param train_ratio: 训练集比例 (0~1)
    :param seed: 随机种子
    """
    random.seed(seed)

    # 读取数据
    raw = load_data(input_file)
    is_dict = isinstance(raw, dict)
    samples = None
    container_key = None

    if is_dict:
        # 尝试识别常见容器键
        for key in ['data', 'train', 'samples']:
            if key in raw and isinstance(raw[key], list):
                container_key = key
                samples = raw[key]
                break
        if samples is None:
            # 如果字典中只有一个键且值为列表，则使用该键
            if len(raw) == 1:
                key = list(raw.keys())[0]
                if isinstance(raw[key], list):
                    container_key = key
                    samples = raw[key]
        if samples is None:
            raise ValueError("无法从字典中识别样本列表，请确保数据包含 'data' 或 'train' 键，或直接传递列表。")
    else:
        # 直接是列表
        samples = raw

    total = len(samples)
    if total == 0:
        raise ValueError("数据集为空")

    # 打乱
    indices = list(range(total))
    random.shuffle(indices)

    split_idx = int(total * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    # 构建输出结构
    if is_dict and container_key is not None:
        train_data = raw.copy()
        train_data[container_key] = train_samples
        test_data = raw.copy()
        test_data[container_key] = test_samples
    else:
        train_data = train_samples
        test_data = test_samples

    # 保存
    train_path = f"{output_prefix}_train.json"
    test_path = f"{output_prefix}_test.json"
    save_data(train_data, train_path)
    save_data(test_data, test_path)

    print(f"切分完成：训练集 {len(train_samples)} 条，测试集 {len(test_samples)} 条")
    print(f"训练集保存至: {train_path}")
    print(f"测试集保存至: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 JSON 数据集切分为训练集和测试集（无验证集）")
    parser.add_argument("--input", type=str, default="data/webshop_sft.json", help="输入 JSON 文件路径")
    parser.add_argument("--output_prefix", type=str, default="data/webshop_sft", help="输出文件前缀，将自动添加 _train.json 和 _test.json")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例，默认 0.9")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_dataset(args.input, args.output_prefix, args.train_ratio, args.seed)