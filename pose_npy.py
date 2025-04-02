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
    """支持多人物和多种坐标格式的验证"""
    required_keys = ['keypoint', 'total_frames', 'frame_dir']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid PKL structure in {filename}: Missing key '{key}'")

    keypoints = data['keypoint']

    # 允许的维度格式:
    # - 单人: [N, K, 2] 或 [N, K, 3]
    # - 多人: [M, N, K, 2] 或 [M, N, K, 3]
    if keypoints.ndim not in (3, 4):
        raise ValueError(
            f"Keypoint format error in {filename}: "
            f"Expected 3 or 4 dimensions, got {keypoints.ndim}"
        )

    # 验证帧数一致性
    expected_frames = data['total_frames']
    actual_frames = keypoints.shape[-3] if keypoints.ndim == 3 else keypoints.shape[1]
    if actual_frames != expected_frames:
        raise ValueError(
            f"Frame count mismatch in {filename}: "
            f"Keypoints({actual_frames}) vs total_frames({expected_frames})"
        )


def process_single_file(pkl_path, output_root, fill_confidence=1.0, max_people=2):
    """处理包含多人物数据的PKL文件"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        validate_pkl_structure(data, pkl_path.name)

        # 获取关键点数据维度
        keypoints = data['keypoint']
        is_multi_person = keypoints.ndim == 4
        num_people = keypoints.shape[0] if is_multi_person else 1
        num_frames = data['total_frames']
        num_kpts = keypoints.shape[-2]
        coord_dim = keypoints.shape[-1]

        # 创建输出目录
        video_id = data['frame_dir']
        output_dir = Path(output_root) / video_id / 'npy'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理形状数据（兼容tuple和numpy array）
        def shape_to_list(shape_data):
            if isinstance(shape_data, (tuple, list)):
                return list(shape_data)
            elif isinstance(shape_data, np.ndarray):
                return shape_data.tolist()
            else:
                return []

        # 生成增强版元数据
        metadata = {
            "source_file": str(pkl_path),
            "structure": {
                "multi_person": is_multi_person,
                "num_people": num_people,
                "num_frames": num_frames,
                "num_keypoints": num_kpts,
                "coordinate_dim": coord_dim,
                "coordinate_meaning": ["x", "y", "confidence"][:coord_dim],
                "confidence_filled": fill_confidence if coord_dim == 2 else None
            },
            "processing_info": {
                "original_resolution": shape_to_list(data.get('original_shape', ())),
                "resized_resolution": shape_to_list(data.get('img_shape', ())),
                "normalized": False,
                "max_people_processed": min(num_people, max_people)
            }
        }

        # 保存元数据
        with (output_dir.parent / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 处理多人物数据（最多处理max_people个）
        for person_idx in range(min(num_people, max_people)):
            person_dir = output_dir / f"person_{person_idx + 1}"
            person_dir.mkdir(exist_ok=True)

            # 提取单人数据
            if is_multi_person:
                person_kpts = keypoints[person_idx]  # [N, K, 2/3]
            else:
                person_kpts = keypoints

            # 补充缺失的置信度
            if coord_dim == 2:
                padded_kpts = np.concatenate([
                    person_kpts,
                    np.full((*person_kpts.shape[:-1], 1), fill_confidence)
                ], axis=-1)
            else:
                padded_kpts = person_kpts

            # 保存每帧数据
            for frame_idx, frame_data in enumerate(padded_kpts, start=1):
                np.save(
                    person_dir / f"{frame_idx:06d}.npy",
                    np.round(frame_data.astype(np.float32), 3)
                )

        return True

    except Exception as e:
        error_msg = f"Error processing {pkl_path.name}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(f"\n[ERROR] {error_msg.splitlines()[0]}")
        return False


def batch_processor(args):
    """多进程参数解包"""
    return process_single_file(*args)


def main():
    parser = argparse.ArgumentParser(description='TENNIS 多人物骨骼数据转换工具')
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
                        help='遇到错误继续处理后续文件')
    parser.add_argument('--fill_confidence',
                        type=float,
                        default=1.0,
                        help='为2D坐标补充的置信度值（默认：1.0）')
    parser.add_argument('--max_people',
                        type=int,
                        default=2,
                        help='最大处理人物数（默认：2）')

    args = parser.parse_args()

    # 准备文件列表
    input_dir = Path(args.input)
    pkl_files = list(input_dir.glob('*.pkl'))
    print(f"▶ 发现 {len(pkl_files)} 个PKL文件待处理")

    # 创建进程池
    task_args = [(f, args.output, args.fill_confidence, args.max_people) for f in pkl_files]
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
    print(f"✅ 转换完成！成功: {success_count}, 失败: {len(pkl_files) - success_count}")
    print(f"📁 输出目录结构示例:")
    print(f"  {args.output}/")
    print(f"  └── [video_id]/")
    print(f"      ├── metadata.json")
    print(f"      └── npy/")
    print(f"          ├── person_1/")
    print(f"          │   ├── 000001.npy (shape: [17, 3])")
    print(f"          │   └── ...")
    print(f"          └── person_2/")
    print(f"              └── ...")
    print("=" * 50)


if __name__ == "__main__":
    main()