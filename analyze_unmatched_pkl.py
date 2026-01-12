"""
分析不在 unlabel.json 中的 PKL 文件，查找匹配失败的原因
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re


def normalize_video_id(video_id):
    """标准化 video_id，用于匹配"""
    # 转换为小写
    normalized = video_id.lower()
    # 移除常见的分隔符差异
    normalized = normalized.replace('_', '-')
    normalized = normalized.replace(' ', '-')
    return normalized


def extract_video_id_patterns(pkl_name):
    """从 PKL 文件名中提取可能的 video_id 模式"""
    patterns = []
    
    # 原始名称
    patterns.append(pkl_name)
    
    # 小写版本
    patterns.append(pkl_name.lower())
    
    # 大写版本
    patterns.append(pkl_name.upper())
    
    # 替换下划线和连字符
    patterns.append(pkl_name.replace('_', '-'))
    patterns.append(pkl_name.replace('-', '_'))
    
    # 移除文件扩展名后的各种变体
    base_name = pkl_name
    patterns.append(base_name.replace('_', ''))
    patterns.append(base_name.replace('-', ''))
    
    return patterns


def analyze_unmatched_pkl(unlabel_json_path, skeleton_dir, output_report=None):
    """
    分析不在 unlabel.json 中的 PKL 文件
    
    Args:
        unlabel_json_path: unlabel.json 文件路径
        skeleton_dir: skeleton PKL 文件目录
        output_report: 可选，输出详细报告的文件路径
    """
    # 读取 unlabel.json
    print(f"📖 读取 unlabel.json: {unlabel_json_path}")
    with open(unlabel_json_path, 'r') as f:
        unlabel_data = json.load(f)
    
    print(f"📊 unlabel.json 中的视频数: {len(unlabel_data)}")
    
    # 建立 unlabel.json 中的 video_id 集合（包含各种变体）
    unlabel_video_ids = set()
    unlabel_video_ids_normalized = set()
    unlabel_video_ids_dict = {}  # 原始 -> 标准化映射
    
    for item in unlabel_data:
        video_id = item.get("video")
        if video_id:
            unlabel_video_ids.add(video_id)
            normalized = normalize_video_id(video_id)
            unlabel_video_ids_normalized.add(normalized)
            unlabel_video_ids_dict[video_id] = normalized
    
    print(f"📋 唯一 video_id 数: {len(unlabel_video_ids)}")
    
    # 获取 skeleton 目录中的所有 PKL 文件
    skeleton_path = Path(skeleton_dir)
    if not skeleton_path.exists():
        print(f"❌ Skeleton 目录不存在: {skeleton_dir}")
        return
    
    print(f"\n🔍 扫描 skeleton 目录: {skeleton_dir}")
    pkl_files = list(skeleton_path.glob("*.pkl"))
    print(f"📦 找到 {len(pkl_files)} 个 PKL 文件")
    
    # 分析每个 PKL 文件
    matched_pkl = []
    unmatched_pkl = []
    case_mismatch = []  # 大小写不匹配
    separator_mismatch = []  # 分隔符不匹配（_ vs -）
    partial_match = []  # 部分匹配
    no_match = []  # 完全无匹配
    
    print("\n🔎 分析 PKL 文件匹配情况...")
    for pkl_file in tqdm(pkl_files, desc="分析 PKL 文件"):
        pkl_name = pkl_file.stem  # 去掉 .pkl 后缀
        
        # 直接匹配
        if pkl_name in unlabel_video_ids:
            matched_pkl.append((pkl_name, "exact"))
            continue
        
        # 标准化后匹配
        pkl_normalized = normalize_video_id(pkl_name)
        if pkl_normalized in unlabel_video_ids_normalized:
            # 找到对应的原始 video_id
            matched_original = None
            for orig_id, norm_id in unlabel_video_ids_dict.items():
                if norm_id == pkl_normalized:
                    matched_original = orig_id
                    break
            
            if matched_original:
                if pkl_name.lower() != matched_original.lower():
                    case_mismatch.append((pkl_name, matched_original, "case"))
                elif pkl_name.replace('_', '-') != matched_original.replace('_', '-'):
                    separator_mismatch.append((pkl_name, matched_original, "separator"))
                else:
                    matched_pkl.append((pkl_name, "normalized"))
            continue
        
        # 部分匹配（检查是否包含或包含于）
        partial_found = False
        for video_id in unlabel_video_ids:
            # PKL 名称包含 video_id 或 video_id 包含 PKL 名称
            if pkl_name in video_id or video_id in pkl_name:
                partial_match.append((pkl_name, video_id, "partial"))
                partial_found = True
                break
        
        if not partial_found:
            no_match.append(pkl_name)
            unmatched_pkl.append(pkl_name)
    
    # 统计结果
    print("\n" + "="*80)
    print("📊 匹配分析结果")
    print("="*80)
    print(f"✅ 完全匹配: {len(matched_pkl)} 个")
    print(f"⚠️  大小写不匹配: {len(case_mismatch)} 个")
    print(f"⚠️  分隔符不匹配: {len(separator_mismatch)} 个")
    print(f"🔍 部分匹配: {len(partial_match)} 个")
    print(f"❌ 完全无匹配: {len(no_match)} 个")
    print(f"📦 总计未匹配: {len(unmatched_pkl)} 个")
    
    # 输出详细报告
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PKL 文件匹配分析报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"unlabel.json 中的视频数: {len(unlabel_data)}\n")
            f.write(f"PKL 文件总数: {len(pkl_files)}\n")
            f.write(f"完全匹配: {len(matched_pkl)}\n")
            f.write(f"大小写不匹配: {len(case_mismatch)}\n")
            f.write(f"分隔符不匹配: {len(separator_mismatch)}\n")
            f.write(f"部分匹配: {len(partial_match)}\n")
            f.write(f"完全无匹配: {len(no_match)}\n\n")
            
            if case_mismatch:
                f.write("\n" + "="*80 + "\n")
                f.write("大小写不匹配示例（前20个）:\n")
                f.write("="*80 + "\n")
                for pkl_name, video_id, _ in case_mismatch[:20]:
                    f.write(f"PKL: {pkl_name}\n")
                    f.write(f"JSON: {video_id}\n")
                    f.write(f"差异: 大小写不同\n\n")
            
            if separator_mismatch:
                f.write("\n" + "="*80 + "\n")
                f.write("分隔符不匹配示例（前20个）:\n")
                f.write("="*80 + "\n")
                for pkl_name, video_id, _ in separator_mismatch[:20]:
                    f.write(f"PKL: {pkl_name}\n")
                    f.write(f"JSON: {video_id}\n")
                    f.write(f"差异: 分隔符不同 (_ vs -)\n\n")
            
            if partial_match:
                f.write("\n" + "="*80 + "\n")
                f.write("部分匹配示例（前20个）:\n")
                f.write("="*80 + "\n")
                for pkl_name, video_id, _ in partial_match[:20]:
                    f.write(f"PKL: {pkl_name}\n")
                    f.write(f"JSON: {video_id}\n")
                    f.write(f"关系: 部分包含\n\n")
            
            if no_match:
                f.write("\n" + "="*80 + "\n")
                f.write("完全无匹配的 PKL 文件（前50个）:\n")
                f.write("="*80 + "\n")
                for pkl_name in no_match[:50]:
                    f.write(f"{pkl_name}\n")
        
        print(f"\n📄 详细报告已保存到: {output_report}")
    
    # 输出一些示例
    print("\n" + "="*80)
    print("📝 示例分析")
    print("="*80)
    
    if case_mismatch:
        print(f"\n⚠️  大小写不匹配示例（共 {len(case_mismatch)} 个）:")
        for pkl_name, video_id, _ in case_mismatch[:5]:
            print(f"  PKL: {pkl_name}")
            print(f"  JSON: {video_id}")
            print()
    
    if separator_mismatch:
        print(f"\n⚠️  分隔符不匹配示例（共 {len(separator_mismatch)} 个）:")
        for pkl_name, video_id, _ in separator_mismatch[:5]:
            print(f"  PKL: {pkl_name}")
            print(f"  JSON: {video_id}")
            print()
    
    if partial_match:
        print(f"\n🔍 部分匹配示例（共 {len(partial_match)} 个）:")
        for pkl_name, video_id, _ in partial_match[:5]:
            print(f"  PKL: {pkl_name}")
            print(f"  JSON: {video_id}")
            print()
    
    if no_match:
        print(f"\n❌ 完全无匹配示例（共 {len(no_match)} 个）:")
        for pkl_name in no_match[:10]:
            print(f"  {pkl_name}")
    
    # 分析命名模式
    print("\n" + "="*80)
    print("🔬 命名模式分析")
    print("="*80)
    
    # 分析 unlabel.json 中的 video_id 命名模式
    json_patterns = defaultdict(int)
    for video_id in unlabel_video_ids:
        # 统计下划线和连字符的使用
        if '_' in video_id and '-' in video_id:
            json_patterns['both'] += 1
        elif '_' in video_id:
            json_patterns['underscore'] += 1
        elif '-' in video_id:
            json_patterns['hyphen'] += 1
        else:
            json_patterns['none'] += 1
    
    print("\nunlabel.json 中的命名模式:")
    for pattern, count in sorted(json_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} 个")
    
    # 分析未匹配 PKL 的命名模式
    pkl_patterns = defaultdict(int)
    for pkl_name in unmatched_pkl:
        if '_' in pkl_name and '-' in pkl_name:
            pkl_patterns['both'] += 1
        elif '_' in pkl_name:
            pkl_patterns['underscore'] += 1
        elif '-' in pkl_name:
            pkl_patterns['hyphen'] += 1
        else:
            pkl_patterns['none'] += 1
    
    print("\n未匹配 PKL 文件的命名模式:")
    for pattern, count in sorted(pkl_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} 个")
    
    return {
        'matched': len(matched_pkl),
        'case_mismatch': len(case_mismatch),
        'separator_mismatch': len(separator_mismatch),
        'partial_match': len(partial_match),
        'no_match': len(no_match),
        'total_pkl': len(pkl_files),
        'total_unlabel': len(unlabel_data)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析不在 unlabel.json 中的 PKL 文件")
    parser.add_argument(
        "--unlabel_json",
        type=str,
        default="/mnt/ssd2/lingyu/Tennis/unlabel.json",
        help="unlabel.json 文件路径"
    )
    parser.add_argument(
        "--skeleton_dir",
        type=str,
        default="/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis",
        help="skeleton PKL 文件目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="unmatched_pkl_analysis.txt",
        help="输出报告文件路径"
    )
    
    args = parser.parse_args()
    
    results = analyze_unmatched_pkl(
        args.unlabel_json,
        args.skeleton_dir,
        args.output
    )
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
