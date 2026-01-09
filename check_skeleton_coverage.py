"""
统计有多少视频有对应的 skeleton PKL 文件
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def check_skeleton_coverage(unlabel_json_path, skeleton_dir, output_report=None):
    """
    检查 unlabel.json 中的视频有多少有对应的 skeleton PKL 文件
    
    Args:
        unlabel_json_path: unlabel.json 文件路径
        skeleton_dir: skeleton PKL 文件目录
        output_report: 可选，输出详细报告的文件路径
    """
    # 读取 unlabel.json
    print(f"读取 unlabel.json: {unlabel_json_path}")
    with open(unlabel_json_path, 'r') as f:
        unlabel_data = json.load(f)
    
    print(f"总视频数: {len(unlabel_data)}")
    
    # 获取 skeleton 目录中的所有 PKL 文件
    skeleton_path = Path(skeleton_dir)
    if not skeleton_path.exists():
        print(f"❌ Skeleton 目录不存在: {skeleton_dir}")
        return
    
    print(f"扫描 skeleton 目录: {skeleton_dir}")
    pkl_files = list(skeleton_path.glob("*.pkl"))
    print(f"找到 {len(pkl_files)} 个 PKL 文件")
    
    # 建立 PKL 文件映射（video_id -> pkl_path）
    pkl_cache = {}
    for pkl_file in pkl_files:
        video_id = pkl_file.stem  # 去掉 .pkl 后缀
        pkl_cache[video_id] = pkl_file
    
    # 检查每个视频是否有对应的 PKL 文件
    videos_with_skeleton = []
    videos_without_skeleton = []
    
    print("\n检查视频 coverage...")
    for item in tqdm(unlabel_data, desc="检查视频"):
        video_id = item.get("video")
        if not video_id:
            continue
        
        if video_id in pkl_cache:
            videos_with_skeleton.append(video_id)
        else:
            videos_without_skeleton.append(video_id)
    
    # 统计结果
    total_videos = len(unlabel_data)
    with_skeleton = len(videos_with_skeleton)
    without_skeleton = len(videos_without_skeleton)
    coverage_rate = (with_skeleton / total_videos * 100) if total_videos > 0 else 0
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("📊 Skeleton Coverage 统计结果")
    print("=" * 80)
    print(f"总视频数: {total_videos}")
    print(f"有 skeleton PKL 文件: {with_skeleton} ({coverage_rate:.2f}%)")
    print(f"没有 skeleton PKL 文件: {without_skeleton} ({100 - coverage_rate:.2f}%)")
    print(f"Skeleton 目录中的 PKL 文件总数: {len(pkl_cache)}")
    print("=" * 80)
    
    # 如果有缺失的视频，显示前20个
    if videos_without_skeleton:
        print(f"\n⚠️  缺失 skeleton 的视频（前 20 个）:")
        for i, video_id in enumerate(videos_without_skeleton[:20], 1):
            print(f"  {i}. {video_id}")
        if len(videos_without_skeleton) > 20:
            print(f"  ... 还有 {len(videos_without_skeleton) - 20} 个视频缺失 skeleton")
    
    # 检查是否有 PKL 文件但不在 unlabel.json 中
    unlabel_video_ids = {item.get("video") for item in unlabel_data if item.get("video")}
    pkl_only = [video_id for video_id in pkl_cache.keys() if video_id not in unlabel_video_ids]
    if pkl_only:
        print(f"\nℹ️  有 {len(pkl_only)} 个 PKL 文件不在 unlabel.json 中（这些文件不会被使用）")
    
    # 输出详细报告到文件（如果指定）
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Skeleton Coverage 详细报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"总视频数: {total_videos}\n")
            f.write(f"有 skeleton PKL 文件: {with_skeleton} ({coverage_rate:.2f}%)\n")
            f.write(f"没有 skeleton PKL 文件: {without_skeleton} ({100 - coverage_rate:.2f}%)\n")
            f.write(f"Skeleton 目录中的 PKL 文件总数: {len(pkl_cache)}\n\n")
            
            if videos_without_skeleton:
                f.write("=" * 80 + "\n")
                f.write(f"缺失 skeleton 的视频列表（共 {len(videos_without_skeleton)} 个）:\n")
                f.write("=" * 80 + "\n")
                for i, video_id in enumerate(videos_without_skeleton, 1):
                    f.write(f"{i}. {video_id}\n")
            
            if pkl_only:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"不在 unlabel.json 中的 PKL 文件（共 {len(pkl_only)} 个）:\n")
                f.write("=" * 80 + "\n")
                for i, video_id in enumerate(pkl_only[:100], 1):  # 只列出前100个
                    f.write(f"{i}. {video_id}\n")
                if len(pkl_only) > 100:
                    f.write(f"... 还有 {len(pkl_only) - 100} 个文件\n")
        
        print(f"\n✅ 详细报告已保存到: {report_path}")
    
    return {
        "total_videos": total_videos,
        "with_skeleton": with_skeleton,
        "without_skeleton": without_skeleton,
        "coverage_rate": coverage_rate,
        "pkl_files_total": len(pkl_cache),
    }


def main():
    parser = argparse.ArgumentParser(description='统计 skeleton PKL 文件的 coverage')
    parser.add_argument(
        '--unlabel-json',
        type=str,
        default='/mnt/ssd2/lingyu/Tennis/unlabel.json',
        help='unlabel.json 文件路径'
    )
    parser.add_argument(
        '--skeleton-dir',
        type=str,
        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis',
        help='skeleton PKL 文件目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='可选：输出详细报告的文件路径（如 skeleton_coverage_report.txt）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    unlabel_path = Path(args.unlabel_json)
    if not unlabel_path.exists():
        print(f"❌ unlabel.json 文件不存在: {args.unlabel_json}")
        return
    
    # 执行统计
    stats = check_skeleton_coverage(
        args.unlabel_json,
        args.skeleton_dir,
        args.output
    )
    
    if stats:
        print(f"\n✅ 统计完成！Coverage: {stats['coverage_rate']:.2f}%")


if __name__ == "__main__":
    main()
