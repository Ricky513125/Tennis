import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from netscripts.get_unlabel_loader import get_unlabel_loader

logger = logging.getLogger(__name__)


class TennisDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, mode="RGB"):
        super(TennisDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.mode = mode
        self.num_frames = cfg.num_frames  # 固定采样的帧数
        self._construct_target_loader(cfg)
        self._construct_unlabel_loader(cfg)

    def _construct_unlabel_loader(self, cfg):
        self.unlabel_loader = get_unlabel_loader(cfg.dataset)

    def _construct_target_loader(self, cfg):
        """加载 Tennis target dataset，并存储远近端球员的信息"""
        self._clip_uid = []
        self._dir_to_img_frame = []
        self._clip_frame = []
        self._action_label = []
        self._action_list = []
        self._verb_list = []
        self._noun_list = []
        self._action_label_internal = []
        self._verb_label_internal = []
        self._noun_label_internal = []

        # 额外存储远端 & 近端信息
        self._far_name = []
        self._near_name = []
        self._far_hand = []
        self._near_hand = []
        self._far_set = []
        self._near_set = []
        self._far_game = []
        self._near_game = []
        self._far_point = []
        self._near_point = []

        # 读取 target dataset JSON
        with open(cfg.target_json_path) as f:
            data = json.load(f)

        # 遍历每个视频片段
        for clip_dict in data:
            video_uid = clip_dict["video"]
            clip_uid = video_uid
            clip_frame = [event["frame"] for event in clip_dict["events"]]

            if len(clip_frame) > self.num_frames:
                indices = np.linspace(0, len(clip_frame) - 1, self.num_frames, dtype=int)
                clip_frame = [clip_frame[i] for i in indices]
            elif len(clip_frame) < self.num_frames:
                pad_size = self.num_frames - len(clip_frame)
                clip_frame += [clip_frame[-1]] * pad_size  # 重复最后一帧填充

            verb_label = clip_dict["far_name"]  # 假设球员名字作为动词
            noun_label = clip_dict["near_name"]  # 对手名字作为名词
            action_label = (verb_label, noun_label)

            # 过滤无效视频
            if video_uid in cfg.delete:
                print(f"{video_uid} 是无效视频，跳过")
                continue

            # 处理动作类别
            if action_label not in self._action_list:
                self._action_list.append(action_label)
            if verb_label not in self._verb_list:
                self._verb_list.append(verb_label)
            if noun_label not in self._noun_list:
                self._noun_list.append(noun_label)

            # 计算内部索引
            action_label_internal = self._action_list.index(action_label)
            verb_label_internal = self._verb_list.index(verb_label)
            noun_label_internal = self._noun_list.index(noun_label)

            # 图片路径
            dir_to_img_frame = Path(cfg.target_data_dir, "vid_frames_224", clip_uid)

            # 存储数据
            self._clip_uid.append(clip_uid)
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._clip_frame.append(clip_frame)
            self._action_label.append(action_label)
            self._action_label_internal.append(action_label_internal)
            self._verb_label_internal.append(verb_label_internal)
            self._noun_label_internal.append(noun_label_internal)

            # 存储远近端球员信息
            self._far_name.append(clip_dict["far_name"])
            self._near_name.append(clip_dict["near_name"])
            self._far_hand.append(clip_dict["far_hand"])
            self._near_hand.append(clip_dict["near_hand"])
            self._far_set.append(clip_dict["far_set"])
            self._near_set.append(clip_dict["near_set"])
            self._far_game.append(clip_dict["far_game"])
            self._near_game.append(clip_dict["near_game"])
            self._far_point.append(clip_dict["far_point"])
            self._near_point.append(clip_dict["near_point"])

        logger.info(f"构建 Tennis 数据集 (size: {len(self._clip_frame)})")
        logger.info(f"动作类别数: {len(self._action_list)}")



    def _get_frame_source(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            print('_source_rgb_path:---', path)
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace(
                "image_frame", "optical_flow"
            )
            path = Path(dir_to_flow_frame, "npy", f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_pose_frame = str(dir_to_img_frame).replace(
                "image_frame", "hand-pose/heatmap"
            )
            path = Path(dir_to_pose_frame, f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                # frame = get_pose(str(path))
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        return frame

    def _get_frame_unlabel(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            # path = dir_to_img_frame / Path(
            #     self.unlabel_loader.get_frame_str(frame_name)
            # )
            # add
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            print('_unlabel_frame_path', path)
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace("RGB", "flow")
            path = Path(
                dir_to_flow_frame,
                self.unlabel_loader.get_frame_str(frame_name).replace("jpg", "npy"),
            )
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_keypoint_frame = str(dir_to_img_frame).replace(
                "RGB_frames", "hand-pose/heatmap"
            )
            path = Path(
                dir_to_keypoint_frame,
                self.unlabel_loader.get_frame_str(frame_name).replace("jpg", "npy"),
            )
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        return frame
    def _get_input(
        self,
        source_dir_to_img_frame,
        source_clip_start_frame,
        unlabel_dir_to_img_frame,
        unlabel_clip_start_frame,
    ):
        # initialization
        source_frames = []
        unlabel_frames = []

        # add
        # print(type(self.cfg.source_sampling_rate))  # 打印类型
        # print('---num_frames', self.cfg.num_frames)
        # print('---clip_start_frame', source_clip_start_frame)
        # source_frame_names = [
        #     max(1, source_clip_start_frame + self.cfg.source_sampling_rate * i)
        #     for i in range(self.cfg.num_frames)
        # ]  # original one
        # unlabel_frame_names = [
        #     max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
        #     for i in range(self.cfg.num_frames)
        # ]
        source_frame_names = [
            max(1, source_clip_start_frame[i] + self.cfg.source_sampling_rate * i)
            for i in range(self.cfg.num_frames)
        ]

        # print(type(self.cfg.dataset.target_sampling_rate))  # 打印类型
        # print('---num_frames', self.cfg.num_frames)
        # print('---clip_start_frame', unlabel_clip_start_frame)

        # TODO 这个unlabel的start_frame为int，只有一个，而source的有16个这个不要紧吗？
        unlabel_frame_names = [
            max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
            for i in range(self.cfg.num_frames)
        ]

        for frame_name in source_frame_names:
            source_frame = self._get_frame_source(
                source_dir_to_img_frame, frame_name, self.mode, source_frames
            )
            source_frames.append(source_frame)

        print('---unlabel_frames---', unlabel_frames)
        for frame_name in unlabel_frame_names:
            unlabel_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, self.mode, unlabel_frames
            )
            unlabel_frames.append(unlabel_frame)

        # [T, H, W, C] -> [T*C, H, W] -> [C, T, H, W]
        print('-----transform------', self.transform)
        print('-----------')
        print(dir(self.transform))  # 打印所有可用的属性和方法
        print('-----------')
        source_frames = self.transform.weak_aug(source_frames)
        unlabel_frames = self.transform.weak_aug(unlabel_frames)
        source_frames = source_frames.permute(1, 0, 2, 3)
        unlabel_frames = unlabel_frames.permute(1, 0, 2, 3)

        # mask generation
        mask = self.mask_gen()

        return source_frames, unlabel_frames, mask

    # def _get_input(self, source_dir, source_frames, unlabel_dir, unlabel_frames):
    #     """加载 source 和 unlabel 数据，并生成 mask"""
    #     # print(f"unlabel_frames type: {type(unlabel_frames)}, value: {unlabel_frames}")
    #
    #     source_images = []
    #     for frame in source_frames:
    #         # print("source_dir : ", source_dir)
    #         img_path = Path(f"{source_dir}/{frame:06d}.jpg")
    #         # print("img_path: ", img_path)
    #         if img_path.exists():
    #             img = Image.open(img_path).convert(self.mode)
    #             source_images.append(self.transform(img))
    #         else:
    #             logger.warning(f"缺失图像 source: {img_path}")
    #             source_images.append(torch.zeros(3, 224, 224))  # 用空白图填充
    #
    #     if isinstance(unlabel_frames, int):
    #         unlabel_frames = [unlabel_frames]  # 转换为列表
    #
    #     unlabel_images = []
    #     for frame in unlabel_frames:
    #         img_path = Path(f"{unlabel_dir}/{frame:06d}.jpg")
    #         if img_path.exists():
    #             img = Image.open(img_path).convert(self.mode)
    #             unlabel_images.append(self.transform(img))
    #         else:
    #             logger.warning(f"缺失图像 unlabel: {img_path}")
    #             unlabel_images.append(torch.zeros(3, 224, 224))
    #
    #     mask = self.mask_gen()  # 生成掩码
    #
    #     return torch.stack(source_images), torch.stack(unlabel_images), mask

    def __getitem__(self, index):
        input = {}

        # source
        source_dir_to_img_frame = self._dir_to_img_frame[index]
        source_clip_start_frame = self._clip_frame[index]

        # unlabel
        unlabel_index = index % len(self.unlabel_loader)
        unlabel_dir_to_img_frame = self.unlabel_loader._dir_to_img_frame[unlabel_index]
        unlabel_clip_start_frame = self.unlabel_loader._start_frame[unlabel_index]

        source_frames, unlabel_frames, mask = self._get_input(
            source_dir_to_img_frame,
            source_clip_start_frame,
            unlabel_dir_to_img_frame,
            unlabel_clip_start_frame,
        )

        # label
        verb_label, noun_label = self._action_label[index]
        action_label_internal = self._action_label_internal[index]
        verb_label_internal = self._verb_label_internal[index]
        noun_label_internal = self._noun_label_internal[index]

        assert self._action_list[action_label_internal] == (verb_label, noun_label)
        assert self._verb_list[verb_label_internal] == verb_label
        assert self._noun_list[noun_label_internal] == noun_label

        # 组装数据
        input["source_frames"] = source_frames
        input["unlabel_frames"] = unlabel_frames
        input["mask"] = mask
        input["action_label"] = action_label_internal
        input["verb_label"] = verb_label_internal
        input["noun_label"] = noun_label_internal

        # 远近端球员信息
        input["far_name"] = self._far_name[index]
        input["near_name"] = self._near_name[index]
        input["far_hand"] = self._far_hand[index]
        input["near_hand"] = self._near_hand[index]
        input["far_set"] = self._far_set[index]
        input["near_set"] = self._near_set[index]
        input["far_game"] = self._far_game[index]
        input["near_game"] = self._near_game[index]
        input["far_point"] = self._far_point[index]
        input["near_point"] = self._near_point[index]

        return input

    def __len__(self):
        return len(self._clip_frame)