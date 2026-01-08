"""
只使用 Tennis unlabel 数据的数据集
用于在无标签目标域上进行微调，不使用源域数据
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from netscripts.get_unlabel_loader import get_unlabel_loader

logger = logging.getLogger(__name__)


class TennisUnlabelOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, mode="RGB"):
        super(TennisUnlabelOnlyDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.mode = mode
        self.num_frames = cfg.num_frames
        self._construct_unlabel_loader(cfg)
        self.patch_size = 16
        self.tubelet_size = 2

    def _construct_unlabel_loader(self, cfg):
        """只加载 unlabel 数据"""
        self.unlabel_loader = get_unlabel_loader(cfg.dataset)

    def _generate_mask(self, H, W, T):
        num_spatial_patches = (H // self.patch_size) * (W // self.patch_size)
        num_temporal_blocks = T // self.tubelet_size
        seq_length = num_spatial_patches * num_temporal_blocks
        mask = torch.rand(seq_length) < 0.75  # mask_ratio=0.75
        return mask

    def _get_frame_unlabel(self, dir_to_img_frame, frame_name, mode, frames):
        """加载 unlabel 帧数据"""
        if mode == "RGB":
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1] if frames else Image.new('RGB', (224, 224), color='black')
        elif mode == "flow":
            # 基于 video_id 构造光流路径
            video_id = dir_to_img_frame.name
            dir_to_flow_frame = Path("/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows") / video_id
            path = dir_to_flow_frame / f"pair_{str(frame_name).zfill(5)}.npy"
            
            if path.exists():
                frame = np.load(str(path))
            else:
                # 生成默认光流张量（全零），维度应该是 [C, H, W]
                frame = np.zeros((2, 224, 384), dtype=np.float32)  # [C, H, W]

            # 实际维度是 [C, H, W] = [2, 224, 398]
            # 居中裁剪宽度至 384（如果需要）
            C, H, original_width = frame.shape
            target_width = 384
            if original_width != target_width:
                start_x = (original_width - target_width) // 2
                # 裁剪宽度维度：frame[C, H, W] -> frame[C, H, 384]
                frame = frame[:, :, start_x: start_x + target_width]

            # 转换为张量（已经是 [C, H, W] 格式，不需要 permute）
            frame = torch.from_numpy(frame).float()  # [C=2, H=224, W=384]
        elif mode == "pose":
            dir_to_pose_frame = str(dir_to_img_frame).replace(
                "vid_frames_224", "hand-pose/heatmap"
            )
            path = Path(dir_to_pose_frame, f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1] if frames else np.zeros((17, 56, 56), dtype=np.float32)
        return frame

    def _get_input(self, unlabel_dir_to_img_frame, unlabel_clip_start_frame):
        """只加载 unlabel 数据"""
        unlabel_frames = []

        # 生成帧名列表
        unlabel_frame_names = [
            max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
            for i in range(self.num_frames)
        ]

        # 加载 unlabel 帧
        for frame_name in unlabel_frame_names:
            unlabel_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, self.mode, unlabel_frames
            )
            unlabel_frames.append(unlabel_frame)

        # 断言列表非空
        assert len(unlabel_frames) > 0, "unlabel_frames 为空，请检查数据加载逻辑"

        # 根据模式分别处理
        if self.mode == "RGB":
            # RGB 模式：PIL Image 列表
            # weak_aug 期望 PIL Image 列表，返回 Tensor [T, C, H, W]
            # 注意：ToTensor 内部会处理列表并 stack
            unlabel_frames = self.transform.weak_aug(unlabel_frames)
            
            # weak_aug 输出是 [T, C, H, W]，转换为 [T, H, W, C] 用于后续处理
            unlabel_frames = unlabel_frames.permute(0, 2, 3, 1)
        else:
            # Flow/Pose 模式：已经是 Tensor 或 numpy array
            # 转换为 Tensor
            tensor_frames = []
            for frame in unlabel_frames:
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame).float()
                elif not isinstance(frame, torch.Tensor):
                    raise TypeError(f"Unexpected frame type: {type(frame)}")
                tensor_frames.append(frame)
            
            # Stack 所有帧 [T, C, H, W]
            unlabel_frames = torch.stack(tensor_frames, dim=0)
            
            # 转换为 [T, H, W, C] 格式输入 weak_aug
            unlabel_frames = unlabel_frames.permute(0, 2, 3, 1)  # [T, H, W, C]
            
            # 应用 weak_aug（内部会转换为 [T, C, H, W] 并归一化）
            unlabel_frames = self.transform.weak_aug(unlabel_frames)
            
            # weak_aug 输出是 [T, C, H, W]，转换回 [T, H, W, C] 用于后续处理
            unlabel_frames = unlabel_frames.permute(0, 2, 3, 1)  # [T, H, W, C]

        # 生成 mask（基于 unlabel_frames 的形状）
        T, H, W, C = unlabel_frames.shape
        mask = self._generate_mask(H, W, T)

        return unlabel_frames, mask

    def __getitem__(self, index):
        input = {}

        # 只使用 unlabel 数据
        unlabel_index = index % len(self.unlabel_loader)
        unlabel_dir_to_img_frame = self.unlabel_loader._dir_to_img_frame[unlabel_index]
        unlabel_clip_start_frame = self.unlabel_loader._start_frame[unlabel_index]

        unlabel_frames, mask = self._get_input(
            unlabel_dir_to_img_frame,
            unlabel_clip_start_frame,
        )

        # 组装数据（只包含 unlabel_frames，不包含 source_frames）
        input["unlabel_frames"] = unlabel_frames
        input["mask"] = mask

        return input

    def __len__(self):
        return len(self.unlabel_loader)
