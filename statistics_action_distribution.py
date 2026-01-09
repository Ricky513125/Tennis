#!/usr/bin/env python3
"""
统计 Tennis 数据集中不同动作的样本分布
"""
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def load_tennis_json(json_path):
    """加载 Tennis JSON 文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def analyze_action_distribution(json_path, output_file=None):
    """
    分析动作分布
    
    Args:
        json_path: JSON 文件路径
        output_file: 输出文件路径（可选）
    """
    print(f"正在加载数据: {json_path}")
    data = load_tennis_json(json_path)
    
    # 统计每个动作的样本数
    action_counter = Counter()
    action_to_label = {}
    
    # 从 events 中提取 action labels
    for video_dict in data:
        video_id = video_dict.get("video")
        if "events" not in video_dict:
            continue
            
        for event in video_dict["events"]:
            if "label" not in event:
                continue
                
            action_label = event["label"]
            action_counter[action_label] += 1
            # 记录 action label（所有相同的 label 都对应同一个 action）
            if action_label not in action_to_label:
                action_to_label[action_label] = action_label
    
    # 按样本数排序
    sorted_actions = sorted(action_counter.items(), key=lambda x: x[1], reverse=True)
    
    # 统计信息
    total_samples = sum(action_counter.values())
    num_actions = len(action_counter)
    
    # 统计不同样本数的动作数量
    samples_per_action = Counter(action_counter.values())
    
    # 输出结果
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("Tennis 数据集动作分布统计")
    output_lines.append("=" * 80)
    output_lines.append(f"数据文件: {json_path}")
    output_lines.append(f"总样本数: {total_samples}")
    output_lines.append(f"动作类别数: {num_actions}")
    output_lines.append(f"平均每个动作的样本数: {total_samples / num_actions:.2f}")
    output_lines.append("")
    
    # 样本数分布统计
    output_lines.append("样本数分布统计:")
    output_lines.append("-" * 80)
    for sample_count in sorted(samples_per_action.keys()):
        num_actions_with_count = samples_per_action[sample_count]
        output_lines.append(f"  {sample_count} 个样本的动作数: {num_actions_with_count}")
    output_lines.append("")
    
    # 详细的动作列表（按样本数排序）
    output_lines.append("详细动作列表（按样本数降序）:")
    output_lines.append("-" * 80)
    
    # 为每个动作分配索引（从 0 开始）
    action_to_idx = {}
    for idx, (action_label, count) in enumerate(sorted_actions):
        action_to_idx[action_label] = idx
        output_lines.append(f"  Action {idx} ({action_label}): {count} 个样本")
    
    output_lines.append("")
    
    # Few-shot 评估可行性分析
    output_lines.append("Few-shot 评估可行性分析:")
    output_lines.append("-" * 80)
    
    # 检查不同配置下的可行性
    configs = [
        (5, 5, 15),   # n_way=5, k_shot=5, q_sample=15
        (5, 3, 10),   # n_way=5, k_shot=3, q_sample=10
        (3, 5, 15),   # n_way=3, k_shot=5, q_sample=15
    ]
    
    for n_way, k_shot, q_sample in configs:
        samples_per_episode = n_way * (k_shot + q_sample)
        # 找出至少有 (k_shot + q_sample) 个样本的动作
        valid_actions = [label for label, count in action_counter.items() 
                        if count >= (k_shot + q_sample)]
        num_valid_actions = len(valid_actions)
        
        if num_valid_actions >= n_way:
            # 计算理论上可以创建的最大 episodes
            # 对于每个动作，可以创建 floor(count / (k_shot + q_sample)) 个 episodes
            max_episodes_per_action = {}
            for action_label in valid_actions:
                count = action_counter[action_label]
                max_episodes_per_action[action_label] = count // (k_shot + q_sample)
            
            # 理论上，如果每个 episode 选择不同的 n_way 个动作组合
            # 最大 episodes 受限于样本最少的动作
            min_episodes_per_action = min(max_episodes_per_action.values())
            theoretical_max_episodes = min_episodes_per_action * (num_valid_actions // n_way)
            
            output_lines.append(f"  配置: N-way={n_way}, K-shot={k_shot}, Q-sample={q_sample}")
            output_lines.append(f"    - 每个 episode 需要样本数: {samples_per_episode}")
            output_lines.append(f"    - 有效动作数（样本数 >= {k_shot + q_sample}）: {num_valid_actions}")
            output_lines.append(f"    - 理论上可创建的最大 episodes: ~{theoretical_max_episodes}")
        else:
            output_lines.append(f"  配置: N-way={n_way}, K-shot={k_shot}, Q-sample={q_sample}")
            output_lines.append(f"    - ⚠️  有效动作数 ({num_valid_actions}) < N-way ({n_way})，无法创建完整的 episode")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    
    # 输出到控制台
    result = "\n".join(output_lines)
    print(result)
    
    # 保存到文件（如果指定）
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n结果已保存到: {output_file}")
    
    return {
        'total_samples': total_samples,
        'num_actions': num_actions,
        'action_counter': action_counter,
        'action_to_idx': action_to_idx,
        'samples_per_action': samples_per_action,
    }


def main():
    parser = argparse.ArgumentParser(description='统计 Tennis 数据集中不同动作的样本分布')
    parser.add_argument(
        'json_path',
        type=str,
        help='Tennis JSON 文件路径（例如：train.json 或 test.json）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径（可选，默认不保存）'
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"错误: 文件不存在: {json_path}")
        return
    
    analyze_action_distribution(json_path, args.output)


if __name__ == "__main__":
    main()
