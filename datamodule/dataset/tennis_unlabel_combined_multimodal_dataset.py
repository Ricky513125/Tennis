"""
Tennis 数据集的多模态数据加载（RGB、Flow、Skeleton）
用于多模态蒸馏训练
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from netscripts.get_unlabel_loader import get_unlabel_loader

logger = logging.getLogger(__name__)


class TennisUnlabelCombinedMMDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen):
        super(TennisUnlabelCombinedMMDataset, self).__init__()
        self.cfg = cfg
        self.transform_rgb = transform[0]
        self.transform_flow = transform[1]
        self.transform_skeleton = transform[2]
        self.mask_gen = mask_gen
        self.num_frames = cfg.num_frames
        self.patch_size = 16
        self.tubelet_size = 2
        
        # Skeleton 相关
        self.skeleton_dir = Path(cfg.skeleton_dir)
        self._pkl_cache = {}
        self._missing_files_warned = set()
        self._load_pkl_mapping()
        
        self._construct_unlabel_loader(cfg)

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
            video_id = pkl_file.stem
            self._pkl_cache[video_id] = pkl_file
        logger.info(f"Loaded {len(self._pkl_cache)} skeleton PKL files from {self.skeleton_dir}")

    def _get_pkl_path(self, video_id):
        """获取指定 video_id 的 pkl 文件路径"""
        if video_id in self._pkl_cache:
            return self._pkl_cache[video_id]
        pkl_path = self.skeleton_dir / f"{video_id}.pkl"
        if pkl_path.exists():
            self._pkl_cache[video_id] = pkl_path
            return pkl_path
        if video_id not in self._missing_files_warned:
            self._missing_files_warned.add(video_id)
            logger.warning(
                f"[SKELETON] PKL file not found for video_id: {video_id}\n"
                f"  Expected path: {pkl_path}\n"
                f"  Will use zero skeleton as fallback for this video."
            )
        return None

    def _load_skeleton_from_pkl(self, pkl_path, frame_name):
        """从 pkl 文件中加载指定帧的 skeleton 数据"""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            keypoints = data['keypoint']
            keypoint_scores = data.get('keypoint_score', None)
            total_frames = data['total_frames']
            img_shape = data.get('img_shape', (720, 1280))
            
            frame_idx = min(max(0, int(frame_name) - 1), total_frames - 1)
            
            if keypoints.ndim == 4:
                frame_kpts = keypoints[0, frame_idx]
                if keypoint_scores is not None and keypoint_scores.ndim == 3:
                    frame_scores = keypoint_scores[0, frame_idx]
                else:
                    frame_scores = np.ones(frame_kpts.shape[0], dtype=np.float32)
            elif keypoints.ndim == 3:
                frame_kpts = keypoints[frame_idx]
                if keypoint_scores is not None and keypoint_scores.ndim == 2:
                    frame_scores = keypoint_scores[frame_idx]
                else:
                    frame_scores = np.ones(frame_kpts.shape[0], dtype=np.float32)
            else:
                raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")
            
            H, W = img_shape
            frame_kpts_normalized = frame_kpts.copy().astype(np.float32)
            frame_kpts_normalized[:, 0] = frame_kpts_normalized[:, 0] / W
            frame_kpts_normalized[:, 1] = frame_kpts_normalized[:, 1] / H
            
            frame_scores = frame_scores.astype(np.float32).reshape(-1, 1)
            frame_kpts_with_score = np.concatenate([frame_kpts_normalized, frame_scores], axis=-1)
            
            return frame_kpts_with_score
            
        except Exception as e:
            logger.warning(f"Error loading skeleton from {pkl_path} at frame {frame_name}: {e}")
            return np.zeros((17, 3), dtype=np.float32)

    def _keypoints_to_heatmap(self, keypoints, H=56, W=98, sigma=2.0):
        """将关键点转换为热图"""
        K = keypoints.shape[0]
        heatmap = np.zeros((K, H, W), dtype=np.float32)
        
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        confidences = keypoints[:, 2]
        
        x_centers = x_coords * W
        y_centers = y_coords * H
        
        for k in range(K):
            if confidences[k] > 0.1:
                x_center = x_centers[k]
                y_center = y_centers[k]
                
                y_grid, x_grid = np.ogrid[:H, :W]
                gaussian = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))
                gaussian = gaussian * confidences[k]
                
                heatmap[k] = np.maximum(heatmap[k], gaussian)
        
        return heatmap

    def _get_frame_unlabel(self, dir_to_img_frame, frame_name, mode, frames):
        """加载 unlabel 帧数据"""
        if mode == "RGB":
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1] if frames else Image.new('RGB', (224, 224), color='black')
        elif mode == "flow":
            video_id = dir_to_img_frame.name
            dir_to_flow_frame = Path("/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows") / video_id
            path = dir_to_flow_frame / f"pair_{str(frame_name).zfill(5)}.npy"
            
            if path.exists():
                frame = np.load(str(path))
                C, H, original_width = frame.shape
                target_width = 384
                if original_width != target_width:
                    start_x = (original_width - target_width) // 2
                    frame = frame[:, :, start_x: start_x + target_width]
            else:
                frame = np.zeros((2, 224, 384), dtype=np.float32)
            
            frame = torch.from_numpy(frame).float()
        elif mode == "skeleton":
            video_id = dir_to_img_frame.name
            pkl_path = self._get_pkl_path(video_id)
            if pkl_path is None:
                frame = np.zeros((17, 3), dtype=np.float32)
            else:
                frame = self._load_skeleton_from_pkl(pkl_path, frame_name)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        return frame

    def _generate_mask(self, H, W, T):
        num_spatial_patches = (H // self.patch_size) * (W // self.patch_size)
        num_temporal_blocks = T // self.tubelet_size
        seq_length = num_spatial_patches * num_temporal_blocks
        mask = torch.rand(seq_length) < 0.75
        return mask

    def _get_input(self, unlabel_dir_to_img_frame, unlabel_clip_start_frame):
        """加载多模态数据"""
        unlabel_frames_rgb = []
        unlabel_frames_flow = []
        unlabel_frames_skeleton = []
        
        frame_names = [
            max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
            for i in range(self.num_frames)
        ]
        
        for frame_name in frame_names:
            # RGB
            rgb_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "RGB", unlabel_frames_rgb
            )
            unlabel_frames_rgb.append(rgb_frame)
            
            # Flow
            flow_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "flow", unlabel_frames_flow
            )
            unlabel_frames_flow.append(flow_frame)
            
            # Skeleton
            skeleton_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "skeleton", unlabel_frames_skeleton
            )
            unlabel_frames_skeleton.append(skeleton_frame)
        
        # 处理 RGB: PIL Image 列表 -> Tensor
        unlabel_frames_rgb = self.transform_rgb.weak_aug(unlabel_frames_rgb)
        # weak_aug 输出是 [T, C, H, W]，需要转换为 [C, T, H, W] 以匹配模型期望
        unlabel_frames_rgb = unlabel_frames_rgb.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # 处理 Flow: 已经是 Tensor 列表
        flow_tensor_list = []
        for flow_frame in unlabel_frames_flow:
            if isinstance(flow_frame, torch.Tensor):
                flow_tensor_list.append(flow_frame)
            else:
                flow_tensor_list.append(torch.from_numpy(flow_frame).float())
        
        unlabel_frames_flow = torch.stack(flow_tensor_list, dim=0)  # [T, C, H, W]
        unlabel_frames_flow = unlabel_frames_flow.permute(0, 2, 3, 1)  # [T, H, W, C]
        unlabel_frames_flow = self.transform_flow.weak_aug(unlabel_frames_flow)
        # weak_aug 输出是 [T, C, H, W]，需要转换为 [C, T, H, W]
        unlabel_frames_flow = unlabel_frames_flow.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # 处理 Skeleton: 转换为热图
        heatmap_frames = []
        for skeleton_frame in unlabel_frames_skeleton:
            heatmap = self._keypoints_to_heatmap(skeleton_frame, H=56, W=98)
            heatmap_frames.append(heatmap)
        
        skeleton_tensor = torch.from_numpy(np.stack(heatmap_frames, axis=0)).float()  # [T, K, H, W]
        skeleton_tensor = skeleton_tensor.permute(0, 2, 3, 1)  # [T, H, W, K]
        skeleton_tensor = self.transform_skeleton.weak_aug(skeleton_tensor)
        # weak_aug 输出是 [T, K, H, W]，需要转换为 [K, T, H, W]
        skeleton_tensor = skeleton_tensor.permute(1, 0, 2, 3)  # [K, T, H, W]
        
        # 生成 mask
        # 使用 RGB 的尺寸来生成 mask（所有模态应该使用相同的 mask）
        _, T, H, W = unlabel_frames_rgb.shape
        mask = self._generate_mask(H, W, T)
        
        return unlabel_frames_rgb, unlabel_frames_flow, skeleton_tensor, mask

    def __getitem__(self, index):
        input = {}
        
        unlabel_index = index % len(self.unlabel_loader)
        unlabel_dir_to_img_frame = self.unlabel_loader._dir_to_img_frame[unlabel_index]
        unlabel_clip_start_frame = self.unlabel_loader._start_frame[unlabel_index]
        
        (
            unlabel_frames_rgb,
            unlabel_frames_flow,
            unlabel_frames_skeleton,
            mask,
        ) = self._get_input(
            unlabel_dir_to_img_frame,
            unlabel_clip_start_frame,
        )
        
        input["unlabel_frames_rgb"] = unlabel_frames_rgb
        input["unlabel_frames_flow"] = unlabel_frames_flow
        input["unlabel_frames_skeleton"] = unlabel_frames_skeleton
        input["mask"] = mask
        
        return input

    def __len__(self):
        return len(self.unlabel_loader)
