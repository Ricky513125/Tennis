"""
计算 skeleton 热图数据的均值和标准差，用于归一化
"""
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json
import random


def keypoints_to_heatmap(keypoints, H=56, W=98, sigma=2.0):
    """
    将关键点转换为热图
    keypoints: [K, 3] (x_norm, y_norm, confidence)，坐标已归一化到 [0, 1]
    返回: [K, H, W] 热图
    注意：使用宽屏格式 (H=56, W=98) 以保持宽高比，匹配最终尺寸 [224, 384]
    """
    K = keypoints.shape[0]
    heatmap = np.zeros((K, H, W), dtype=np.float32)
    
    # 坐标已经归一化到 [0, 1]
    x_coords = keypoints[:, 0]  # [0, 1]
    y_coords = keypoints[:, 1]  # [0, 1]
    confidences = keypoints[:, 2]
    
    # 转换为热图坐标
    x_centers = x_coords * W  # 转换为热图坐标
    y_centers = y_coords * H
    
    # 创建高斯热图
    for k in range(K):
        if confidences[k] > 0.1:  # 只处理置信度大于阈值的点
            x_center = x_centers[k]
            y_center = y_centers[k]
            
            # 创建高斯分布
            y_grid, x_grid = np.ogrid[:H, :W]
            gaussian = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))
            gaussian = gaussian * confidences[k]  # 乘以置信度
            
            heatmap[k] = np.maximum(heatmap[k], gaussian)
    
    return heatmap


def load_skeleton_from_pkl(pkl_path, frame_name):
    """
    从 PKL 文件加载指定帧的 skeleton 数据（与实际数据加载逻辑一致）
    
    Args:
        pkl_path: PKL 文件路径
        frame_name: 帧号（从1开始）
    
    Returns:
        keypoints: [K, 3] numpy array (x_norm, y_norm, confidence)，如果失败返回 None
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        keypoints = data['keypoint']  # [M, N, K, 2] 或 [N, K, 2]
        keypoint_scores = data.get('keypoint_score', None)  # [M, N, K] 或 [N, K]
        total_frames = data['total_frames']
        img_shape = data.get('img_shape', (720, 1280))  # (H, W)
        
        # 确保 frame_name 在有效范围内
        frame_idx = min(max(0, int(frame_name) - 1), total_frames - 1)
        
        # 处理多人情况：取第一个人（person_idx=0）
        if keypoints.ndim == 4:  # [M, N, K, 2] - 多人
            frame_kpts = keypoints[0, frame_idx]  # [K, 2]
            if keypoint_scores is not None and keypoint_scores.ndim == 3:
                frame_scores = keypoint_scores[0, frame_idx]  # [K]
            else:
                frame_scores = np.ones(frame_kpts.shape[0], dtype=np.float32)
        elif keypoints.ndim == 3:  # [N, K, 2] - 单人
            frame_kpts = keypoints[frame_idx]  # [K, 2]
            if keypoint_scores is not None and keypoint_scores.ndim == 2:
                frame_scores = keypoint_scores[frame_idx]  # [K]
            else:
                frame_scores = np.ones(frame_kpts.shape[0], dtype=np.float32)
        else:
            return None
        
        # 归一化坐标到 [0, 1] 范围（基于原始图像尺寸）
        H, W = img_shape
        frame_kpts_normalized = frame_kpts.copy().astype(np.float32)
        frame_kpts_normalized[:, 0] = frame_kpts_normalized[:, 0] / W  # x 坐标归一化
        frame_kpts_normalized[:, 1] = frame_kpts_normalized[:, 1] / H  # y 坐标归一化
        
        # 合并坐标和置信度：[K, 2] + [K] -> [K, 3]
        frame_scores = frame_scores.astype(np.float32).reshape(-1, 1)
        frame_kpts_with_score = np.concatenate([frame_kpts_normalized, frame_scores], axis=-1)
        
        return frame_kpts_with_score  # [K, 3] (x_norm, y_norm, confidence)
        
    except Exception as e:
        return None


def calculate_statistics(pkl_files, skeleton_dir, unlabel_json_path, sample_size=None, num_frames_per_video=16):
    """
    计算 skeleton 热图的统计信息
    
    Args:
        pkl_files: PKL 文件路径列表
        skeleton_dir: skeleton 目录
        unlabel_json_path: unlabel JSON 文件路径
        sample_size: 采样视频数量（如果为 None，使用所有文件）
        num_frames_per_video: 每个视频采样的帧数
    
    Returns:
        mean: 每个通道的均值 [mean_ch0, mean_ch1, ..., mean_ch16] (17个通道)
        std: 每个通道的标准差 [std_ch0, std_ch1, ..., std_ch16] (17个通道)
    """
    # 加载 unlabel JSON 以获取视频信息
    try:
        with open(unlabel_json_path, 'r') as f:
            unlabel_data = json.load(f)
        video_dict = {item['video']: item for item in unlabel_data}
    except Exception as e:
        print(f"⚠️  无法加载 unlabel JSON: {e}")
        video_dict = {}
    
    if sample_size and sample_size < len(pkl_files):
        pkl_files = random.sample(pkl_files, sample_size)
    
    print(f"📊 计算 {len(pkl_files)} 个 PKL 文件的统计信息...")
    
    # 用于累积统计（17个通道）
    K = 17  # 关键点数量
    sums = np.zeros(K, dtype=np.float64)
    sum_sqs = np.zeros(K, dtype=np.float64)
    total_pixels = 0
    
    # 用于计算全局 min/max
    mins = np.full(K, float('inf'))
    maxs = np.full(K, float('-inf'))
    
    processed_files = 0
    processed_frames = 0
    
    for pkl_file in tqdm(pkl_files, desc="处理文件"):
        try:
            # 尝试从文件名提取 video_id
            video_id = pkl_file.stem
            
            # 尝试从 unlabel JSON 获取视频信息
            video_info = video_dict.get(video_id, None)
            if video_info is None:
                # 尝试匹配部分文件名
                for vid, info in video_dict.items():
                    if vid in video_id or video_id in vid:
                        video_info = info
                        break
            
            # 确定要采样的帧数
            if video_info:
                num_frames = min(num_frames_per_video, video_info.get('num_frames', num_frames_per_video))
                # 均匀采样帧
                frame_indices = np.linspace(1, num_frames, num_frames_per_video, dtype=int)
            else:
                # 如果没有视频信息，使用默认采样
                frame_indices = np.linspace(1, 100, num_frames_per_video, dtype=int)
            
            # 处理每一帧
            for frame_idx in frame_indices:
                keypoints = load_skeleton_from_pkl(pkl_file, frame_idx)
                if keypoints is None:
                    continue
                
                # 转换为热图 [K, H, W]
                # 使用宽屏格式以保持宽高比：56×98 (宽高比 ≈ 1:1.75，匹配 224×384)
                heatmap = keypoints_to_heatmap(keypoints, H=56, W=98, sigma=2.0)
                
                # Resize 到目标尺寸 [224, 384]（使用双线性插值）以与 RGB/Flow 位置对应
                # 使用 PIL 或 numpy 进行 resize
                if heatmap.shape[1] != 224 or heatmap.shape[2] != 384:
                    # 使用 numpy 和简单的插值方法
                    # 或者使用 PIL（如果可用）
                    try:
                        from PIL import Image
                        # 对每个通道分别 resize
                        resized_heatmap = np.zeros((heatmap.shape[0], 224, 384), dtype=np.float32)
                        for k in range(heatmap.shape[0]):
                            img = Image.fromarray(heatmap[k])
                            img_resized = img.resize((384, 224), Image.BILINEAR)  # PIL resize 使用 (W, H)
                            resized_heatmap[k] = np.array(img_resized)
                        heatmap = resized_heatmap
                    except ImportError:
                        # 如果没有 PIL，使用简单的最近邻插值
                        # 计算缩放因子
                        scale_h = 224 / heatmap.shape[1]
                        scale_w = 384 / heatmap.shape[2]
                        # 创建新数组
                        resized_heatmap = np.zeros((heatmap.shape[0], 224, 384), dtype=np.float32)
                        for k in range(heatmap.shape[0]):
                            for i in range(224):
                                for j in range(384):
                                    src_i = int(i / scale_h)
                                    src_j = int(j / scale_w)
                                    resized_heatmap[k, i, j] = heatmap[k, src_i, src_j]
                        heatmap = resized_heatmap
                
                # 累积统计（对每个通道分别计算）
                for k in range(K):
                    channel_data = heatmap[k, :, :].flatten()
                    n_pixels = len(channel_data)
                    
                    sums[k] += channel_data.sum()
                    sum_sqs[k] += (channel_data ** 2).sum()
                    total_pixels += n_pixels
                    
                    # 更新 min/max
                    mins[k] = min(mins[k], channel_data.min())
                    maxs[k] = max(maxs[k], channel_data.max())
                
                processed_frames += 1
            
            processed_files += 1
            
        except Exception as e:
            print(f"❌ 处理 {pkl_file.name} 时出错: {e}")
            continue
    
    if total_pixels == 0:
        print("❌ 没有成功处理任何数据")
        return None, None, None
    
    # 计算均值和标准差（每个通道）
    mean_per_channel = sums / (total_pixels / K)  # 每个通道的总像素数
    std_per_channel = np.sqrt(sum_sqs / (total_pixels / K) - mean_per_channel ** 2)
    
    mean = [float(x) for x in mean_per_channel]
    std = [float(x) for x in std_per_channel]
    
    stats = {
        'min': [float(x) for x in mins],
        'max': [float(x) for x in maxs],
        'total_files': processed_files,
        'total_frames': processed_frames,
        'total_pixels': total_pixels,
    }
    
    return mean, std, stats


def main():
    parser = argparse.ArgumentParser(description='计算 skeleton 热图数据的归一化参数')
    parser.add_argument('--skeleton-dir',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis',
                        help='skeleton PKL 文件目录路径')
    parser.add_argument('--unlabel-json',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/unlabel.json',
                        help='unlabel JSON 文件路径')
    parser.add_argument('--sample',
                        type=int,
                        default=1000,
                        help='采样视频数量（默认 1000）')
    parser.add_argument('--frames-per-video',
                        type=int,
                        default=16,
                        help='每个视频采样的帧数（默认 16）')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='输出统计信息到 JSON 文件（可选）')
    
    args = parser.parse_args()
    
    skeleton_dir = Path(args.skeleton_dir)
    
    if not skeleton_dir.exists():
        print(f"❌ 目录不存在: {skeleton_dir}")
        return
    
    # 收集所有 .pkl 文件
    print(f"🔍 搜索 .pkl 文件...")
    pkl_files = list(skeleton_dir.glob("*.pkl"))
    
    if len(pkl_files) == 0:
        print("❌ 没有找到 .pkl 文件")
        return
    
    print(f"✅ 找到 {len(pkl_files)} 个 .pkl 文件")
    
    # 计算统计信息
    mean, std, stats = calculate_statistics(
        pkl_files, 
        skeleton_dir, 
        args.unlabel_json,
        sample_size=args.sample,
        num_frames_per_video=args.frames_per_video
    )
    
    if mean is None:
        return
    
    # 输出结果
    print("\n" + "=" * 80)
    print("📊 Skeleton 热图数据统计结果")
    print("=" * 80)
    print(f"\n📁 处理的文件数: {stats['total_files']}")
    print(f"🎬 处理的帧数: {stats['total_frames']}")
    print(f"📏 总像素数: {stats['total_pixels']:,}")
    
    print(f"\n📈 数值范围 (每个关键点通道):")
    for k in range(17):
        print(f"   关键点 {k:2d}: [{stats['min'][k]:.6f}, {stats['max'][k]:.6f}]")
    
    print(f"\n📊 归一化参数 (每个关键点通道):")
    print(f"   Mean: {mean}")
    print(f"   Std:  {std}")
    print("\n" + "=" * 80)
    
    # 输出配置文件格式
    print("\n💡 配置文件格式 (configs/data_module/modality/skeleton.yaml):")
    print("```yaml")
    print("mean: [")
    for i in range(0, 17, 7):
        end = min(i + 7, 17)
        values = ", ".join([f"{mean[j]:.6f}" for j in range(i, end)])
        if end < 17:
            print(f"    {values},")
        else:
            print(f"    {values}")
    print("]")
    print("std: [")
    for i in range(0, 17, 7):
        end = min(i + 7, 17)
        values = ", ".join([f"{std[j]:.6f}" for j in range(i, end)])
        if end < 17:
            print(f"    {values},")
        else:
            print(f"    {values}")
    print("]")
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
