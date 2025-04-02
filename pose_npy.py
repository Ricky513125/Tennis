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

import traceback




def validate_pkl_structure(data, filename):
    """验证PKL文件结构完整性"""
    required_keys = ['joints', 'frame_count']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid PKL structure in {filename}: Missing key '{key}'")

    if len(data['joints']) != data['frame_count']:
        raise ValueError(
            f"Data length mismatch in {filename}: "
            f"Joints({len(data['joints'])}) vs FrameCount({data['frame_count']})"
        )


def process_single_file(pkl_path, output_root):
    """处理单个PKL文件"""
    try:
        # 加载数据
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 数据验证
        validate_pkl_structure(data, pkl_path.name)

        # 创建输出目录
        video_id = pkl_path.stem
        output_dir = Path(output_root) / video_id / 'npy'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成元数据
        metadata = {
            'original_path': str(pkl_path),
            'resolution': data.get('resolution', 'unknown'),
            'fps': data.get('fps', 30),
            'joints_shape': data['joints'][0].shape,
            'total_frames': len(data['joints'])
        }

        # 保存元数据
        with (output_dir.parent / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2)

        # 保存每帧数据
        for idx, frame in enumerate(data['joints'], start=1):
            frame_path = output_dir / f"{idx:06d}.npy"
            np.save(frame_path, frame.astype(np.float32))

        return True

    except Exception as e:
        logging.error(f"Error processing {pkl_path.name}: {str(e)}\n{traceback.format_exc()}")
        return False


def batch_processor(args):
    """包装函数用于多进程"""
    return process_single_file(*args)


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Tennis PKL to NPY Converter')
    parser.add_argument('--input', type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis',
                        help='输入目录路径')
    parser.add_argument('--output', type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/pose',
                        help='输出目录路径')
    parser.add_argument('--workers', type=int,
                        default=cpu_count() // 2,
                        help='并行工作进程数 (默认: 50% CPU核心数)')
    args = parser.parse_args()

    # 准备文件列表
    input_dir = Path(args.input)
    pkl_files = list(input_dir.glob('*.pkl'))
    print(f"发现 {len(pkl_files)} 个PKL文件待处理")
    success_count = 0
    fail_count = 0

    for pkl_file in pkl_files:
        try:
            # 你的处理逻辑
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)  # 如果报错，可能是编码问题，尝试 encoding='latin1'
            # 转换和保存逻辑
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"处理失败: {pkl_file}")
            print(traceback.format_exc())  # 打印详细错误信息

    # 创建进程池
    task_args = [(f, args.output) for f in pkl_files]
    success_count = 0

    with Pool(processes=args.workers) as pool:
        results = []
        with tqdm(total=len(pkl_files), desc='转换进度', unit='file') as pbar:
            for result in pool.imap_unordered(batch_processor, task_args):
                results.append(result)
                success_count += 1 if result else 0
                pbar.update()
                pbar.set_postfix({'成功率': f"{success_count / len(results) * 100:.1f}%"})

    # 生成总结报告
    print(f"\n转换完成！成功: {success_count}, 失败: {len(pkl_files) - success_count}")
    print(f"输出目录结构示例:")
    print(f"  {args.output}/")
    print(f"  └── [video_id]/")
    print(f"      ├── metadata.json")
    print(f"      └── npy/")
    print(f"          ├── 000001.npy")
    print(f"          └── ...")


if __name__ == "__main__":
    main()