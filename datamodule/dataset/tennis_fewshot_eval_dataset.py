"""
Tennis 数据集的 Few-shot 评估数据集
用于多模态蒸馏的评估阶段
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TennisFewshotEvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, num_frames, mode="RGB"):
        super(TennisFewshotEvalDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.num_frames = num_frames
        self.mode = mode
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # initialization
        self._dir_to_img_frame = []
        self._video_id = []
        self._start_frame = []
        self._action_label = []
        self._action_idx = []

        # read annotation json file
        eval_json_path = getattr(cfg, 'fewshot_eval_json_path', None) or getattr(cfg, 'target_json_path', None)
        if eval_json_path is None:
            raise ValueError("Either 'fewshot_eval_json_path' or 'target_json_path' must be specified in config")
        logger.info(f"Loading evaluation data from: {eval_json_path}")
        with open(eval_json_path) as f:
            data = json.load(f)

        # 从 events 中提取 action labels
        # Tennis JSON 格式: { "video": "...", "events": [{"frame": ..., "label": "..."}, ...] }
        action_list = []
        action_to_idx = {}
        
        for video_dict in data:
            video_id = video_dict.get("video")
            if "events" not in video_dict:
                continue
                
            for event in video_dict["events"]:
                if "label" not in event:
                    continue
                    
                action_label = event["label"]
                if action_label not in action_to_idx:
                    action_to_idx[action_label] = len(action_list)
                    action_list.append(action_label)
                
                frame = event.get("frame", 1)
                
                # 构建图像路径
                dir_to_img_frame = Path(
                    cfg.target_data_dir,
                    "vid_frames_224",
                    video_id,
                )
                
                self._dir_to_img_frame.append(dir_to_img_frame)
                self._video_id.append(video_id)
                self._start_frame.append(frame)
                self._action_label.append(action_label)
                self._action_idx.append(action_to_idx[action_label])

        self.num_actions = len(action_list)
        self.action_list = action_list
        
        logger.info(f"Constructing Tennis few-shot eval dataloader (size: {len(self._start_frame)})")
        logger.info(f"Number of action classes: {self.num_actions}")

    def _get_frame(self, dir_to_img_frame, frame_name, mode, frames):
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
            # 评估阶段通常只使用 RGB，skeleton 可以返回零张量
            frame = np.zeros((17, 3), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        return frame

    def _get_input(self, dir_to_img_frame, clip_start_frame):
        frame_names = [
            max(1, clip_start_frame + self.cfg.target_sampling_rate * i)
            for i in range(self.num_frames)
        ]
        frames = []
        for frame_name in frame_names:
            frame = self._get_frame(dir_to_img_frame, frame_name, self.mode, frames)
            frames.append(frame)

        # 处理不同模态的数据
        if self.mode == "RGB":
            # RGB: PIL Image 列表 -> Tensor
            # transform 期望 (frames, None) 格式
            frames, _ = self.transform((frames, None))
            # transform 输出是 [T*C, H, W]，需要 reshape 和 permute
            frames = frames.view((self.num_frames, 3) + frames.size()[-2:]).transpose(0, 1)
            # 最终形状: [C, T, H, W]
        elif self.mode == "flow":
            # Flow: 已经是 Tensor 列表
            if isinstance(frames[0], torch.Tensor):
                frames = torch.stack(frames, dim=0)  # [T, C, H, W]
            else:
                frames = torch.stack([torch.from_numpy(f).float() for f in frames], dim=0)
            
            # 转换为 [T, H, W, C] 用于 transform
            frames = frames.permute(0, 2, 3, 1)  # [T, H, W, C]
            frames = self.transform.weak_aug(frames)
            # weak_aug 输出是 [T, C, H, W]，转换为 [C, T, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
        elif self.mode == "skeleton":
            # Skeleton: 评估阶段通常不使用，返回零张量
            frames = torch.zeros((17, self.num_frames, 224, 384), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # mask generation
        mask = self.mask_gen()

        return frames, mask

    def __getitem__(self, index):
        input = {}

        dir_to_img_frame = self._dir_to_img_frame[index]
        clip_start_frame = self._start_frame[index]
        action_label = self._action_label[index]
        action_idx = self._action_idx[index]

        # load frames
        frames, mask = self._get_input(dir_to_img_frame, clip_start_frame)

        input["frames"] = frames
        input["mask"] = mask
        input["action_label"] = action_label
        input["action_idx"] = action_idx

        return input, index

    def __len__(self):
        return len(self._start_frame)
