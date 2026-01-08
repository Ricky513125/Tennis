"""
计算 RGB 数据的均值和标准差，用于归一化
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import torch
from torchvision import transforms


def calculate_statistics(image_files, sample_size=None):
    """
    计算 RGB 图像的统计信息
    
    Args:
        image_files: 图像文件路径列表
        sample_size: 采样数量（如果为 None，使用所有文件）
    
    Returns:
        mean: 每个通道的均值 [mean_R, mean_G, mean_B]
        std: 每个通道的标准差 [std_R, std_G, std_B]
    """
    if sample_size and sample_size < len(image_files):
        import random
        image_files = random.sample(image_files, sample_size)
    
    print(f"📊 计算 {len(image_files)} 个图像的统计信息...")
    
    # 用于累积统计
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    sum_sq_r = 0.0
    sum_sq_g = 0.0
    sum_sq_b = 0.0
    total_pixels = 0
    
    # 用于计算全局 min/max
    min_r = float('inf')
    max_r = float('-inf')
    min_g = float('inf')
    max_g = float('-inf')
    min_b = float('inf')
    max_b = float('-inf')
    
    # 转换为 Tensor 以便计算（归一化到 [0, 1]）
    to_tensor = transforms.ToTensor()
    
    for img_file in tqdm(image_files, desc="处理图像"):
        try:
            # 加载图像
            img = Image.open(str(img_file))
            
            # 转换为 RGB（处理可能的 RGBA 或其他格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为 Tensor [C, H, W]，值在 [0, 1] 范围
            img_tensor = to_tensor(img)  # [3, H, W]
            
            # 展平每个通道
            r_channel = img_tensor[0, :, :].flatten().numpy()
            g_channel = img_tensor[1, :, :].flatten().numpy()
            b_channel = img_tensor[2, :, :].flatten().numpy()
            
            # 累积统计
            n_pixels = len(r_channel)
            sum_r += r_channel.sum()
            sum_g += g_channel.sum()
            sum_b += b_channel.sum()
            sum_sq_r += (r_channel ** 2).sum()
            sum_sq_g += (g_channel ** 2).sum()
            sum_sq_b += (b_channel ** 2).sum()
            total_pixels += n_pixels
            
            # 更新 min/max
            min_r = min(min_r, r_channel.min())
            max_r = max(max_r, r_channel.max())
            min_g = min(min_g, g_channel.min())
            max_g = max(max_g, g_channel.max())
            min_b = min(min_b, b_channel.min())
            max_b = max(max_b, b_channel.max())
            
        except Exception as e:
            print(f"❌ 处理 {img_file.name} 时出错: {e}")
            continue
    
    if total_pixels == 0:
        print("❌ 没有成功处理任何文件")
        return None, None, None
    
    # 计算均值和标准差
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels
    
    # 标准差公式: sqrt(E[X^2] - E[X]^2)
    std_r = np.sqrt(sum_sq_r / total_pixels - mean_r ** 2)
    std_g = np.sqrt(sum_sq_g / total_pixels - mean_g ** 2)
    std_b = np.sqrt(sum_sq_b / total_pixels - mean_b ** 2)
    
    mean = [float(mean_r), float(mean_g), float(mean_b)]
    std = [float(std_r), float(std_g), float(std_b)]
    
    return mean, std, {
        'min': [float(min_r), float(min_g), float(min_b)],
        'max': [float(max_r), float(max_g), float(max_b)],
        'total_files': len(image_files),
        'total_pixels': total_pixels,
    }


def main():
    parser = argparse.ArgumentParser(description='计算 RGB 图像的归一化参数')
    parser.add_argument('--input',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224',
                        help='RGB 图像数据目录路径')
    parser.add_argument('--sample',
                        type=int,
                        default=None,
                        help='采样文件数量（None 表示使用所有文件）')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='输出统计信息到 JSON 文件（可选）')
    
    args = parser.parse_args()
    
    img_dir = Path(args.input)
    
    if not img_dir.exists():
        print(f"❌ 目录不存在: {img_dir}")
        return
    
    # 收集所有图像文件
    print(f"🔍 搜索图像文件...")
    image_files = []
    
    # 搜索所有子目录中的 .jpg 文件
    for video_dir in img_dir.iterdir():
        if video_dir.is_dir():
            image_files.extend(list(video_dir.glob("*.jpg")))
            image_files.extend(list(video_dir.glob("*.png")))
    
    if len(image_files) == 0:
        print("❌ 没有找到图像文件")
        return
    
    print(f"✅ 找到 {len(image_files)} 个图像文件")
    
    # 计算统计信息
    mean, std, stats = calculate_statistics(image_files, sample_size=args.sample)
    
    if mean is None:
        return
    
    # 输出结果
    print("\n" + "=" * 80)
    print("📊 RGB 数据统计结果")
    print("=" * 80)
    print(f"\n📁 处理的文件数: {stats['total_files']}")
    print(f"📏 总像素数: {stats['total_pixels']:,}")
    print(f"\n📈 数值范围 (归一化到 [0, 1]):")
    print(f"   通道 R: [{stats['min'][0]:.6f}, {stats['max'][0]:.6f}]")
    print(f"   通道 G: [{stats['min'][1]:.6f}, {stats['max'][1]:.6f}]")
    print(f"   通道 B: [{stats['min'][2]:.6f}, {stats['max'][2]:.6f}]")
    print(f"\n📊 归一化参数:")
    print(f"   Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"   Std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("\n" + "=" * 80)
    
    # 输出配置文件格式
    print("\n💡 配置文件格式 (configs/data_module/modality/RGB.yaml):")
    print("```yaml")
    print(f"mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"std: [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("```")
    
    # 与 ImageNet 默认值对比
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    print(f"\n📊 与 ImageNet 默认值对比:")
    print(f"   ImageNet Mean: {imagenet_mean}")
    print(f"   你的数据 Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"   ImageNet Std:  {imagenet_std}")
    print(f"   你的数据 Std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    
    # 保存到文件（如果指定）
    if args.output:
        output_data = {
            'mean': mean,
            'std': std,
            'statistics': stats,
            'config_format': {
                'mean': mean,
                'std': std,
            },
            'imagenet_comparison': {
                'imagenet_mean': imagenet_mean,
                'imagenet_std': imagenet_std,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 统计信息已保存到: {args.output}")


if __name__ == "__main__":
    main()
