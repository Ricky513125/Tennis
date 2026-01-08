"""
检查 flow .npy 文件的实际维度
"""
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def check_npy_file(npy_path):
    """检查单个 .npy 文件的维度"""
    try:
        data = np.load(str(npy_path))
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='检查 flow .npy 文件的维度')
    parser.add_argument('--input',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows',
                        help='flow 数据目录路径')
    parser.add_argument('--sample',
                        type=int,
                        default=10,
                        help='检查的样本文件数量')
    parser.add_argument('--video_id',
                        type=str,
                        default=None,
                        help='指定要检查的视频 ID（可选）')
    
    args = parser.parse_args()
    
    flow_dir = Path(args.input)
    
    if not flow_dir.exists():
        print(f"❌ 目录不存在: {flow_dir}")
        return
    
    # 收集所有 .npy 文件
    if args.video_id:
        # 检查指定视频
        video_dir = flow_dir / args.video_id
        if not video_dir.exists():
            print(f"❌ 视频目录不存在: {video_dir}")
            return
        npy_files = list(video_dir.glob("pair_*.npy"))
        print(f"📁 检查视频: {args.video_id}")
    else:
        # 检查所有视频
        npy_files = []
        video_dirs = [d for d in flow_dir.iterdir() if d.is_dir()]
        print(f"📁 找到 {len(video_dirs)} 个视频目录")
        
        for video_dir in video_dirs[:5]:  # 只检查前 5 个视频目录
            npy_files.extend(list(video_dir.glob("pair_*.npy"))[:args.sample])
    
    if len(npy_files) == 0:
        print("❌ 没有找到 .npy 文件")
        return
    
    print(f"\n🔍 检查 {len(npy_files)} 个 .npy 文件...\n")
    print("=" * 80)
    
    # 统计维度分布
    shape_counter = Counter()
    results = []
    
    for i, npy_file in enumerate(npy_files[:args.sample]):
        result = check_npy_file(npy_file)
        
        if 'error' in result:
            print(f"❌ {npy_file.name}: {result['error']}")
        else:
            shape = result['shape']
            shape_counter[shape] += 1
            
            print(f"📄 {npy_file.name}")
            print(f"   路径: {npy_file}")
            print(f"   形状: {shape}")
            print(f"   数据类型: {result['dtype']}")
            print(f"   数值范围: [{result['min']:.4f}, {result['max']:.4f}]")
            print(f"   均值: {result['mean']:.4f}, 标准差: {result['std']:.4f}")
            
            # 检查维度是否符合预期
            if len(shape) == 3:
                H, W, C = shape
                if H == 224 and W == 384 and C == 2:
                    print(f"   ✅ 符合预期: [224, 384, 2]")
                elif H == 224 and W == 398 and C == 2:
                    print(f"   ⚠️  需要裁剪: [224, 398, 2] -> [224, 384, 2]")
                else:
                    print(f"   ⚠️  意外维度: {shape}")
            else:
                print(f"   ⚠️  意外的维度数量: {len(shape)}")
            
            results.append((npy_file, result))
            print()
    
    # 统计总结
    print("=" * 80)
    print("\n📊 维度统计:")
    for shape, count in shape_counter.most_common():
        print(f"   {shape}: {count} 个文件")
    
    # 检查是否有不一致的维度
    if len(shape_counter) > 1:
        print("\n⚠️  警告: 发现多种不同的维度！")
        print("   这可能导致数据处理问题。")
    else:
        print("\n✅ 所有文件的维度一致")


if __name__ == "__main__":
    main()
