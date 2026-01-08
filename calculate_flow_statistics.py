"""
计算 flow 数据的均值和标准差，用于归一化
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def calculate_statistics(npy_files, sample_size=None):
    """
    计算 flow 数据的统计信息
    
    Args:
        npy_files: .npy 文件路径列表
        sample_size: 采样数量（如果为 None，使用所有文件）
    
    Returns:
        mean: 每个通道的均值 [mean_ch0, mean_ch1]
        std: 每个通道的标准差 [std_ch0, std_ch1]
    """
    if sample_size and sample_size < len(npy_files):
        import random
        npy_files = random.sample(npy_files, sample_size)
    
    print(f"📊 计算 {len(npy_files)} 个文件的统计信息...")
    
    # 用于累积统计
    sum_ch0 = 0.0
    sum_ch1 = 0.0
    sum_sq_ch0 = 0.0
    sum_sq_ch1 = 0.0
    total_pixels = 0
    
    # 用于计算全局 min/max
    min_ch0 = float('inf')
    max_ch0 = float('-inf')
    min_ch1 = float('inf')
    max_ch1 = float('-inf')
    
    for npy_file in tqdm(npy_files, desc="处理文件"):
        try:
            data = np.load(str(npy_file))
            
            # 检查维度
            if data.ndim != 3:
                print(f"⚠️  跳过 {npy_file.name}: 维度不是3维，实际维度 {data.shape}")
                continue
            
            # 假设维度是 [C, H, W] = [2, 224, 398] 或类似
            if data.shape[0] == 2:
                # [C, H, W] 格式
                ch0 = data[0, :, :].flatten()
                ch1 = data[1, :, :].flatten()
            elif data.shape[2] == 2:
                # [H, W, C] 格式
                ch0 = data[:, :, 0].flatten()
                ch1 = data[:, :, 1].flatten()
            else:
                print(f"⚠️  跳过 {npy_file.name}: 无法识别的维度 {data.shape}")
                continue
            
            # 累积统计
            n_pixels = len(ch0)
            sum_ch0 += ch0.sum()
            sum_ch1 += ch1.sum()
            sum_sq_ch0 += (ch0 ** 2).sum()
            sum_sq_ch1 += (ch1 ** 2).sum()
            total_pixels += n_pixels
            
            # 更新 min/max
            min_ch0 = min(min_ch0, ch0.min())
            max_ch0 = max(max_ch0, ch0.max())
            min_ch1 = min(min_ch1, ch1.min())
            max_ch1 = max(max_ch1, ch1.max())
            
        except Exception as e:
            print(f"❌ 处理 {npy_file.name} 时出错: {e}")
            continue
    
    if total_pixels == 0:
        print("❌ 没有成功处理任何文件")
        return None, None
    
    # 计算均值和标准差
    mean_ch0 = sum_ch0 / total_pixels
    mean_ch1 = sum_ch1 / total_pixels
    
    # 标准差公式: sqrt(E[X^2] - E[X]^2)
    std_ch0 = np.sqrt(sum_sq_ch0 / total_pixels - mean_ch0 ** 2)
    std_ch1 = np.sqrt(sum_sq_ch1 / total_pixels - mean_ch1 ** 2)
    
    mean = [float(mean_ch0), float(mean_ch1)]
    std = [float(std_ch0), float(std_ch1)]
    
    return mean, std, {
        'min': [float(min_ch0), float(min_ch1)],
        'max': [float(max_ch0), float(max_ch1)],
        'total_files': len(npy_files),
        'total_pixels': total_pixels,
    }


def main():
    parser = argparse.ArgumentParser(description='计算 flow 数据的归一化参数')
    parser.add_argument('--input',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows',
                        help='flow 数据目录路径')
    parser.add_argument('--sample',
                        type=int,
                        default=None,
                        help='采样文件数量（None 表示使用所有文件）')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='输出统计信息到 JSON 文件（可选）')
    
    args = parser.parse_args()
    
    flow_dir = Path(args.input)
    
    if not flow_dir.exists():
        print(f"❌ 目录不存在: {flow_dir}")
        return
    
    # 收集所有 .npy 文件
    print(f"🔍 搜索 .npy 文件...")
    npy_files = []
    video_dirs = [d for d in flow_dir.iterdir() if d.is_dir()]
    
    for video_dir in video_dirs:
        npy_files.extend(list(video_dir.glob("pair_*.npy")))
    
    if len(npy_files) == 0:
        print("❌ 没有找到 .npy 文件")
        return
    
    print(f"✅ 找到 {len(npy_files)} 个 .npy 文件")
    
    # 计算统计信息
    mean, std, stats = calculate_statistics(npy_files, sample_size=args.sample)
    
    if mean is None:
        return
    
    # 输出结果
    print("\n" + "=" * 80)
    print("📊 Flow 数据统计结果")
    print("=" * 80)
    print(f"\n📁 处理的文件数: {stats['total_files']}")
    print(f"📏 总像素数: {stats['total_pixels']:,}")
    print(f"\n📈 数值范围:")
    print(f"   通道 0 (x方向): [{stats['min'][0]:.6f}, {stats['max'][0]:.6f}]")
    print(f"   通道 1 (y方向): [{stats['min'][1]:.6f}, {stats['max'][1]:.6f}]")
    print(f"\n📊 归一化参数:")
    print(f"   Mean: [{mean[0]:.6f}, {mean[1]:.6f}]")
    print(f"   Std:  [{std[0]:.6f}, {std[1]:.6f}]")
    print("\n" + "=" * 80)
    
    # 输出配置文件格式
    print("\n💡 配置文件格式 (configs/data_module/modality/flow.yaml):")
    print("```yaml")
    print(f"mean: [{mean[0]:.6f}, {mean[1]:.6f}]")
    print(f"std: [{std[0]:.6f}, {std[1]:.6f}]")
    print("```")
    
    # 保存到文件（如果指定）
    if args.output:
        output_data = {
            'mean': mean,
            'std': std,
            'statistics': stats,
            'config_format': {
                'mean': mean,
                'std': std,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 统计信息已保存到: {args.output}")


if __name__ == "__main__":
    main()
