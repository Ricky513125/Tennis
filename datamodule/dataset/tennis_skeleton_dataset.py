"""
直接从 PKL 文件加载 skeleton 数据的数据集
用于在无标签目标域上进行 skeleton 数据的微调
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from netscripts.get_unlabel_loader import get_unlabel_loader

logger = logging.getLogger(__name__)


class TennisSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, skeleton_dir):
        super(TennisSkeletonDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.skeleton_dir = Path(skeleton_dir)
        self.num_frames = cfg.num_frames
        self._construct_unlabel_loader(cfg)
        self.patch_size = 16
        self.tubelet_size = 2
        
        # 缓存 pkl 文件路径映射
        self._pkl_cache = {}
        # 记录已警告的缺失文件，避免重复输出
        self._missing_files_warned = set()
        self._load_pkl_mapping()

    def _construct_unlabel_loader(self, cfg):
        """只加载 unlabel 数据"""
        self.unlabel_loader = get_unlabel_loader(cfg.dataset)

    def _load_pkl_mapping(self):
        """建立 video_id 到 pkl 文件的映射"""
        if not self.skeleton_dir.exists():
            logger.error(f"Skeleton directory does not exist: {self.skeleton_dir}")
            return
        
        pkl_files = list(self.skeleton_dir.glob("*.pkl"))
        if len(pkl_files) == 0:
            logger.warning(f"No PKL files found in skeleton directory: {self.skeleton_dir}")
            return
            
        for pkl_file in pkl_files:
            # pkl 文件名格式: 20210220-W-Australian_Open-F-Naomi_Osaka-Jennifer_Brady_70873_71167.pkl
            # 提取 video_id（去掉 .pkl 后缀）
            video_id = pkl_file.stem
            self._pkl_cache[video_id] = pkl_file
        logger.info(f"Loaded {len(self._pkl_cache)} skeleton PKL files from {self.skeleton_dir}")

    def _get_pkl_path(self, video_id):
        """根据 video_id 获取对应的 pkl 文件路径"""
        if video_id in self._pkl_cache:
            return self._pkl_cache[video_id]
        # 如果缓存中没有，尝试直接查找
        pkl_path = self.skeleton_dir / f"{video_id}.pkl"
        if pkl_path.exists():
            self._pkl_cache[video_id] = pkl_path
            return pkl_path
        # 如果文件不存在，输出提示信息（只警告一次）
        if video_id not in self._missing_files_warned:
            self._missing_files_warned.add(video_id)
            logger.warning(
                f"[SKELETON] PKL file not found for video_id: {video_id}\n"
                f"  Expected path: {pkl_path}\n"
                f"  Skeleton directory: {self.skeleton_dir}\n"
                f"  Total available PKL files: {len(self._pkl_cache)} files\n"
                f"  Will use zero skeleton as fallback for this video."
            )
        return None

    def _load_skeleton_from_pkl(self, pkl_path, frame_name):
        """从 pkl 文件中加载指定帧的 skeleton 数据"""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            keypoints = data['keypoint']  # [N, K, 2/3] 或 [M, N, K, 2/3]
            total_frames = data['total_frames']
            
            # 确保 frame_name 在有效范围内
            frame_idx = min(max(0, int(frame_name) - 1), total_frames - 1)
            
            # 处理多人情况：取第一个人（person_idx=0）
            if keypoints.ndim == 4:  # [M, N, K, 2/3]
                frame_kpts = keypoints[0, frame_idx]  # [K, 2/3]
            elif keypoints.ndim == 3:  # [N, K, 2/3]
                frame_kpts = keypoints[frame_idx]  # [K, 2/3]
            else:
                raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")
            
            # 如果只有 2D 坐标，补充置信度
            if frame_kpts.shape[-1] == 2:
                confidence = np.ones((frame_kpts.shape[0], 1), dtype=np.float32)
                frame_kpts = np.concatenate([frame_kpts, confidence], axis=-1)
            
            return frame_kpts.astype(np.float32)  # [K, 3]
            
        except Exception as e:
            logger.warning(f"Error loading skeleton from {pkl_path} at frame {frame_name}: {e}")
            # 返回默认值：假设 17 个关键点（COCO 格式）
            return np.zeros((17, 3), dtype=np.float32)

    def _generate_mask(self, H, W, T):
        num_spatial_patches = (H // self.patch_size) * (W // self.patch_size)
        num_temporal_blocks = T // self.tubelet_size
        seq_length = num_spatial_patches * num_temporal_blocks
        mask = torch.rand(seq_length) < 0.75  # mask_ratio=0.75
        return mask

    def _keypoints_to_heatmap(self, keypoints, H=56, W=56):
        """
        将关键点转换为热图
        keypoints: [K, 3] (x, y, confidence)
        返回: [K, H, W] 热图
        """
        K = keypoints.shape[0]
        heatmap = np.zeros((K, H, W), dtype=np.float32)
        
        # 归一化坐标到 [0, 1]
        x_coords = keypoints[:, 0] / 224.0  # 假设原始图像尺寸为 224
        y_coords = keypoints[:, 1] / 224.0
        confidences = keypoints[:, 2]
        
        # 转换为热图坐标
        x_indices = (x_coords * W).astype(int)
        y_indices = (y_coords * H).astype(int)
        
        # 创建高斯热图
        for k in range(K):
            if confidences[k] > 0:
                x_idx = np.clip(x_indices[k], 0, W - 1)
                y_idx = np.clip(y_indices[k], 0, H - 1)
                
                # 简单的点热图（可以改进为高斯分布）
                heatmap[k, y_idx, x_idx] = confidences[k]
        
        return heatmap

    def _get_input(self, video_id, unlabel_clip_start_frame):
        """加载 skeleton 数据"""
        skeleton_frames = []
        
        # 获取 pkl 文件路径
        pkl_path = self._get_pkl_path(video_id)
        if pkl_path is None:
            # _get_pkl_path 已经输出了详细的警告信息
            # 返回默认的零骨架
            for _ in range(self.num_frames):
                skeleton_frames.append(np.zeros((17, 3), dtype=np.float32))
        else:
            # 生成帧名列表
            frame_names = [
                max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
                for i in range(self.num_frames)
            ]
            
            # 从 pkl 加载每帧的 skeleton
            for frame_name in frame_names:
                skeleton_frame = self._load_skeleton_from_pkl(pkl_path, frame_name)
                skeleton_frames.append(skeleton_frame)
        
        # 转换为热图格式 [T, K, H, W]
        # 这里我们转换为热图以便与 VideoMAE 兼容
        heatmap_frames = []
        for skeleton_frame in skeleton_frames:
            heatmap = self._keypoints_to_heatmap(skeleton_frame, H=56, W=56)
            heatmap_frames.append(heatmap)
        
        # 转换为张量 [T, K, H, W]
        skeleton_tensor = torch.from_numpy(np.stack(heatmap_frames, axis=0)).float()
        
        # 应用变换
        # skeleton_tensor: [T, K, H, W] -> [T, H, W, K] (为了应用 transform)
        skeleton_tensor = skeleton_tensor.permute(0, 2, 3, 1)  # [T, H, W, K]
        skeleton_tensor = self.transform.weak_aug(skeleton_tensor)
        
        # 生成 mask
        T, H, W, K = skeleton_tensor.shape
        mask = self._generate_mask(H, W, T)
        
        return skeleton_tensor, mask

    def __getitem__(self, index):
        input = {}
        
        # 只使用 unlabel 数据
        unlabel_index = index % len(self.unlabel_loader)
        video_id = self.unlabel_loader._video_id[unlabel_index]
        unlabel_clip_start_frame = self.unlabel_loader._start_frame[unlabel_index]
        
        skeleton_frames, mask = self._get_input(
            video_id,
            unlabel_clip_start_frame,
        )
        
        # 组装数据
        input["unlabel_frames"] = skeleton_frames
        input["mask"] = mask
        
        return input

    def __len__(self):
        return len(self.unlabel_loader)
