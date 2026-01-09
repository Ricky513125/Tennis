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
                raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")
            
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
            logger.warning(f"Error loading skeleton from {pkl_path} at frame {frame_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # 返回默认值：假设 17 个关键点（COCO 格式）
            return np.zeros((17, 3), dtype=np.float32)

    def _generate_mask(self, H, W, T):
        num_spatial_patches = (H // self.patch_size) * (W // self.patch_size)
        num_temporal_blocks = T // self.tubelet_size
        seq_length = num_spatial_patches * num_temporal_blocks
        mask = torch.rand(seq_length) < 0.75  # mask_ratio=0.75
        return mask

    def _keypoints_to_heatmap(self, keypoints, H=56, W=56, sigma=2.0):
        """
        将关键点转换为热图
        keypoints: [K, 3] (x_norm, y_norm, confidence)，坐标已归一化到 [0, 1]
        返回: [K, H, W] 热图
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
        
        # weak_aug 输出是 [T, C, H, W]，需要转换回 [T, H, W, C] 用于后续处理
        skeleton_tensor = skeleton_tensor.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
        
        # 生成 mask
        T, H, W, C = skeleton_tensor.shape
        mask = self._generate_mask(H, W, T)
        # 在 return skeleton_tensor, mask 之前添加
        logger.debug(f"[SKELETON DATASET] After transform - skeleton_tensor shape: {skeleton_tensor.shape}, mask shape: {mask.shape}")
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
