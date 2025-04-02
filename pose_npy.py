import argparse
import pickle
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import traceback
from multiprocessing import Pool, cpu_count
import logging

# 配置日志记录
logging.basicConfig(
    filename='conversion_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validate_pkl_structure(data, filename):
    """验证实际PKL文件结构"""
    required_keys = ['keypoint', 'total_frames', 'frame_dir']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid PKL structure in {filename}: Missing key '{key}'")

    # 验证关键点维度 [N, K, 3]
    if data['keypoint'].ndim != 3 or data['keypoint'].shape[2] != 3:
        raise ValueError(
            f"Keypoint format error in {filename}: "
            f"Expected [N, K, 3], got {data['keypoint'].shape}"
        )

    # 验证帧数一致性
    if len(data['keypoint']) != data['total_frames']:
        raise ValueError(
            f"Frame count mismatch in {filename}: "
            f"Keypoints({len(data['keypoint'])}) vs total_frames({data['total_frames']})"
        )


def process_single_file(pkl_path, output_root):
    """处理单个PKL文件"""
    try:
        # 加载数据（兼容Python2/3）
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # 数据验证
        validate_pkl_structure(data, pkl_path.name)

        # 创建输出目录
        video_id = data['frame_dir']  # 使用frame_dir作为唯一标识
        output_dir = Path(output_root) / video_id / 'npy'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成元数据
        metadata = {
            "source_file": str(pkl_path),
            "video_info": {
                "original_resolution": data.get('original_shape', []).tolist(),
                "processed_resolution": data.get('img_shape', []).tolist(),
                "total_frames": data['total_frames'],
                "keypoints_per_frame": data['keypoint'].shape[1]
            },
            "data_attributes": {
                "coordinate_system": "pixel",
                "value_order": ["x", "y", "confidence"],
                "normalized": False
            }
        }

        # 保存元数据
        with (output_dir.parent / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 保存每帧关键点数据
        for frame_idx, keypoints in enumerate(data['keypoint'], start=1):
            # 转换为 float32 并保留三位小数
            processed_data = np.round(keypoints.astype(np.float32), 3)
            np.save(output_dir / f"{frame_idx:06d}.npy", processed_data)

        return True

    except Exception as e:
        error_msg = f"Error processing {pkl_path.name}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(f"\n[ERROR] {error_msg.splitlines()[0]}")
        return False


def batch_processor(args):
    """多进程包装函数"""
    return process_single_file(*args)


def main():
    parser = argparse.ArgumentParser(description='TENNIS 骨骼数据转换工具')
    parser.add_argument('--input',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis',
                        help='输入目录路径（包含.pkl文件）')
    parser.add_argument('--output',
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/pose',
                        help='输出根目录路径')
    parser.add_argument('--workers',
                        type=int,
                        default=max(1, cpu_count() // 2),
                        help='并行工作进程数（默认：CPU核心数50%）')
    parser.add_argument('--skip_errors',
                        action='store_true',
                        help='跳过错误文件继续处理')
    args = parser.parse_args()

    # 准备文件列表
    input_dir = Path(args.input)
    pkl_files = list(input_dir.glob('*.pkl'))
    print(f"▶ 发现 {len(pkl_files)} 个PKL文件待处理")

    # 创建进程池
    task_args = [(f, args.output) for f in pkl_files]
    success_count = 0

    with Pool(processes=args.workers) as pool:
        results = []
        with tqdm(total=len(pkl_files),
                  desc='转换进度',
                  unit='file',
                  bar_format='{l_bar}{bar:30}{r_bar}',
                  dynamic_ncols=True) as pbar:

            for result in pool.imap_unordered(batch_processor, task_args):
                results.append(result)
                if result:
                    success_count += 1
                else:
                    if not args.skip_errors:
                        print("\n❌ 检测到错误且未启用--skip_errors，终止处理...")
                        pool.terminate()
                        break
                pbar.update()
                pbar.set_postfix({
                    '成功率': f"{success_count / len(results) * 100:.1f}%",
                    '已处理': len(results)
                })

    # 生成报告
    print("\n" + "=" * 50)
    print(f"转换完成！成功: {success_count}, 失败: {len(pkl_files) - success_count}")
    print(f" 输出目录结构示例:")
    print(f"  {args.output}/")
    print(f"  └── [video_id]/")
    print(f"      ├── metadata.json")
    print(f"      └── npy/")
    print(f"          ├── 000001.npy (shape: [关键点数量, 3])")
    print(f"          └── ...")
    print("=" * 50)


if __name__ == "__main__":
    main()