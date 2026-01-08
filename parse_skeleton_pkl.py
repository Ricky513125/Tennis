"""
解析 skeleton PKL 文件，查看其内部结构
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
import json


def parse_pkl_file(pkl_path):
    """解析单个 PKL 文件，输出其结构信息"""
    print(f"\n{'='*80}")
    print(f"Parsing PKL file: {pkl_path}")
    print(f"{'='*80}")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print(f"\n1. Top-level keys in PKL file:")
        print(f"   {list(data.keys())}")
        
        # 检查每个键的内容
        for key in data.keys():
            value = data[key]
            print(f"\n2. Key '{key}':")
            if isinstance(value, np.ndarray):
                print(f"   Type: numpy.ndarray")
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Min: {value.min()}, Max: {value.max()}")
                if value.size < 100:  # 如果数据量不大，打印一些样本
                    print(f"   Sample data (first few elements):")
                    print(f"   {value.flat[:min(10, value.size)]}")
            elif isinstance(value, (int, float, str)):
                print(f"   Type: {type(value).__name__}")
                print(f"   Value: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"   Type: {type(value).__name__}")
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First element type: {type(value[0]).__name__}")
                    if len(value) <= 5:
                        print(f"   Content: {value}")
            elif isinstance(value, dict):
                print(f"   Type: dict")
                print(f"   Keys: {list(value.keys())}")
            else:
                print(f"   Type: {type(value).__name__}")
                print(f"   Value: {value}")
        
        # 特别关注 keypoint 数据
        if 'keypoint' in data:
            keypoints = data['keypoint']
            print(f"\n3. Detailed keypoint analysis:")
            print(f"   Shape: {keypoints.shape}")
            print(f"   Dtype: {keypoints.dtype}")
            
            if keypoints.ndim == 3:  # [N, K, 2/3]
                num_frames, num_kpts, coord_dim = keypoints.shape
                print(f"   Format: Single person")
                print(f"   Number of frames: {num_frames}")
                print(f"   Number of keypoints per frame: {num_kpts}")
                print(f"   Coordinate dimensions: {coord_dim} (x, y, [confidence])")
                
                # 显示第一帧的关键点
                print(f"\n   First frame keypoints (frame 0):")
                frame_0 = keypoints[0]
                print(f"   Shape: {frame_0.shape}")
                print(f"   Sample (first 5 keypoints):")
                for i in range(min(5, num_kpts)):
                    print(f"     Keypoint {i}: {frame_0[i]}")
                
            elif keypoints.ndim == 4:  # [M, N, K, 2/3]
                num_people, num_frames, num_kpts, coord_dim = keypoints.shape
                print(f"   Format: Multiple people")
                print(f"   Number of people: {num_people}")
                print(f"   Number of frames: {num_frames}")
                print(f"   Number of keypoints per frame: {num_kpts}")
                print(f"   Coordinate dimensions: {coord_dim} (x, y, [confidence])")
                
                # 显示第一个人的第一帧
                print(f"\n   First person, first frame keypoints:")
                person_0_frame_0 = keypoints[0, 0]
                print(f"   Shape: {person_0_frame_0.shape}")
                print(f"   Sample (first 5 keypoints):")
                for i in range(min(5, num_kpts)):
                    print(f"     Keypoint {i}: {person_0_frame_0[i]}")
        
        # 检查 frame_dir
        if 'frame_dir' in data:
            print(f"\n4. Video ID (frame_dir): {data['frame_dir']}")
        
        # 检查 total_frames
        if 'total_frames' in data:
            print(f"5. Total frames: {data['total_frames']}")
            if 'keypoint' in data:
                keypoints = data['keypoint']
                if keypoints.ndim == 3:
                    actual_frames = keypoints.shape[0]
                elif keypoints.ndim == 4:
                    actual_frames = keypoints.shape[1]
                else:
                    actual_frames = None
                if actual_frames:
                    print(f"   Actual frames in keypoint data: {actual_frames}")
                    if actual_frames != data['total_frames']:
                        print(f"   ⚠️  WARNING: Frame count mismatch!")
        
        # 检查其他可能的字段
        other_keys = [k for k in data.keys() if k not in ['keypoint', 'frame_dir', 'total_frames']]
        if other_keys:
            print(f"\n6. Other fields:")
            for key in other_keys:
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"   {key}: numpy.ndarray, shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"   {key}: {type(value).__name__} = {value}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error parsing PKL file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Parse skeleton PKL files to understand their structure')
    parser.add_argument('--input', 
                        type=str,
                        default='/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis',
                        help='PKL file path or directory containing PKL files')
    parser.add_argument('--sample', 
                        type=int,
                        default=3,
                        help='Number of sample files to parse (if input is a directory)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == '.pkl':
        # 单个文件
        parse_pkl_file(input_path)
    elif input_path.is_dir():
        # 目录，解析几个样本文件
        pkl_files = list(input_path.glob('*.pkl'))
        print(f"Found {len(pkl_files)} PKL files in directory")
        print(f"Will parse {min(args.sample, len(pkl_files))} sample files\n")
        
        for i, pkl_file in enumerate(pkl_files[:args.sample]):
            parse_pkl_file(pkl_file)
            if i < min(args.sample, len(pkl_files)) - 1:
                print("\n" + "-"*80 + "\n")
    else:
        print(f"Error: {input_path} is not a valid PKL file or directory")


if __name__ == "__main__":
    main()
