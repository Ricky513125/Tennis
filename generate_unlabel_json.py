"""
生成用于训练的 unlabel.json 文件
移除所有 label 和 outcome 信息，只保留视频元数据和帧号
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_unlabel_json(input_json_path, output_json_path, keep_events_frames=True):
    """
    生成 unlabel JSON 文件，移除所有标签信息
    
    Args:
        input_json_path: 输入的 JSON 文件路径（包含 label 的原始文件）
        output_json_path: 输出的 unlabel JSON 文件路径
        keep_events_frames: 是否保留 events 中的 frame 信息（只保留帧号，移除 label 和 outcome）
    """
    print(f"读取输入文件: {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"处理 {len(data)} 个视频...")
    
    unlabel_data = []
    for item in tqdm(data, desc="处理视频"):
        # 保留视频元数据
        unlabel_item = {
            "fps": item.get("fps"),
            "height": item.get("height"),
            "width": item.get("width"),
            "num_frames": item.get("num_frames"),
            "video": item.get("video"),
            "far_name": item.get("far_name"),
            "far_hand": item.get("far_hand"),
            "far_set": item.get("far_set"),
            "far_game": item.get("far_game"),
            "far_point": item.get("far_point"),
            "near_name": item.get("near_name"),
            "near_hand": item.get("near_hand"),
            "near_set": item.get("near_set"),
            "near_game": item.get("near_game"),
            "near_point": item.get("near_point"),
        }
        
        # 处理 events
        if keep_events_frames and "events" in item:
            # 只保留帧号，移除 label 和 outcome
            unlabel_item["events"] = [
                {"frame": event["frame"]} for event in item["events"]
            ]
        else:
            # 完全不保留 events
            unlabel_item["events"] = []
        
        unlabel_data.append(unlabel_item)
    
    # 保存输出文件
    print(f"保存到: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(unlabel_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 完成！生成了 {len(unlabel_data)} 个无标签视频条目")
    print(f"📊 统计信息:")
    print(f"   - 总视频数: {len(unlabel_data)}")
    if keep_events_frames:
        total_events = sum(len(item.get("events", [])) for item in unlabel_data)
        print(f"   - 总事件帧数: {total_events}")
    print(f"   - 输出文件: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description='生成用于训练的 unlabel.json 文件')
    parser.add_argument('--input',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/test.json',
                        help='输入的 JSON 文件路径（包含 label 的原始文件）')
    parser.add_argument('--output',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/unlabel.json',
                        help='输出的 unlabel JSON 文件路径')
    parser.add_argument('--remove-events',
                        action='store_true',
                        help='完全移除 events 信息（默认只移除 label 和 outcome，保留 frame）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ 错误：输入文件不存在: {input_path}")
        return
    
    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成 unlabel JSON
    generate_unlabel_json(
        input_path,
        output_path,
        keep_events_frames=not args.remove_events
    )
    
    print("\n" + "=" * 50)
    print("📝 使用说明:")
    print(f"   生成的 unlabel.json 已保存到: {output_path}")
    print(f"   请在配置文件中更新 unlabel_json_path 为: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
