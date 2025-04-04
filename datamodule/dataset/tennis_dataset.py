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
        self.patch_size = 16  # 从配置读取
        self.tubelet_size = 2  # 时间分块大小

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

    def _generate_mask(self, H, W, T):
        num_spatial_patches = (H // self.patch_size) * (W // self.patch_size)
        num_temporal_blocks = T // self.tubelet_size
        seq_length = num_spatial_patches * num_temporal_blocks
        mask = torch.rand(seq_length) < 0.75  # mask_ratio=0.75
        return mask

    def _get_frame_source(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            # print('_source_rgb_path:---', path)
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            # dir_to_flow_frame = str(dir_to_img_frame).replace(
            #     "image_frame", "optical_flow"
            # )
            # print("-----dir_to_flow_frame-----", dir_to_flow_frame)
            #
            # path = Path(dir_to_flow_frame, "npy", f"{str(frame_name).zfill(6)}.npy")
            # print("-----path-----", path)

            # # 直接基于视频ID构建路径，而不是替换字符串
            # video_id = dir_to_img_frame.split("/")[-1]  # 例如 20190712-M-Wimbledon-SF-...
            # dir_to_flow_frame = f"/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows/{video_id}"
            #
            # # 构建NPY文件路径（注意文件名格式匹配）
            # path = Path(dir_to_flow_frame, f"pair_{str(frame_name).zfill(5)}.npy")  # 00214 -> 5位补零
            #
            # print("-----Final flow path-----", path)

            # 或者更高效的方式：直接使用 Path 的 .name 属性
            video_id = dir_to_img_frame.name  # 直接获取目录名

            dir_to_flow_frame = Path("/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows") / video_id

            # 构建光流文件路径
            path = dir_to_flow_frame / f"pair_{str(frame_name).zfill(5)}.npy"
            # print("-----Final flow path-----", path)

            if path.exists():
                frame = np.load(str(path))
            else:

                # 生成默认光流张量（全零）
                H, W = 224, 384  # 根据配置设置
                frame = np.zeros((H, W, 2), dtype=np.float32)
                print(f"⚠️ 光流文件缺失: {path}, 使用零张量替代")

                # frame = frames[-1]

            # 转换为张量并调整维度
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # [C, H, W]
            print("-----转换为张量并调整维度-----", frame)
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
            # print('_unlabel_frame_path', path)
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            # dir_to_flow_frame = str(dir_to_img_frame).replace("RGB", "flow")
            # path = Path(
            #     dir_to_flow_frame,
            #     self.unlabel_loader.get_frame_str(frame_name).replace("jpg", "npy"),
            # )
            # if path.exists():
            #     frame = np.load(str(path))
            # else:
            #     frame = frames[-1]

            # 关键修改：直接基于video_id构造光流路径
            # 假设dir_to_img_frame结构为：/path/to/RGB_frames/{video_id}
            video_id = dir_to_img_frame.name  # 从RGB路径提取video_id（例如 "20230129-M-Australian_Open-F-Novak_Djokovic-Stefanos_Tsitsipas_96814_97030"）

            # 构建光流文件路径
            flow_base_dir = Path("/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows")
            flow_video_dir = flow_base_dir / video_id

            # 将frame_name转换为整数，并格式化为5位数字
            # 假设frame_name为字符串"41"或整数41，需要统一处理
            # frame_id = int(frame_name.split(".")[0])  # 如果frame_name是"000041.jpg"，得到41
            frame_id = int(frame_name)
            flow_filename = f"pair_{frame_id:05d}.npy"  # 生成 pair_00041.npy

            path = flow_video_dir / flow_filename

            # 调试输出
            # print(f"Loading flow: {path}")

            if path.exists():
                frame = np.load(str(path))
            else:
                H, W = 224, 384
                frame = np.zeros((H, W, 2), dtype=np.float32)
                raise FileNotFoundError(f"光流文件缺失: {path}")  # 严格报错，避免静默失败

            print(f"_get_frame_unlabel 光流数据形状: {frame.shape}") # (2, 224, 398)
            # 转换为张量并调整维度
            flow_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # [C=2, H, W]
            print(f"tennis_dataset _get_frame_unlabel -> flow_tensor: {flow_tensor.shape}")

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

        # # 假设 self.modality 是 "RGB" 或 "flow"
        # if self.mode == "RGB":
        #     transform = self.transform_rgb  # 使用RGB的预处理（3通道）
        # elif self.mode == "flow":
        #     transform = self.transform_flow  # 使用光流的预处理（2通道）
        #
        #
        source_frame_names = [
            max(1, source_clip_start_frame[i] + self.cfg.source_sampling_rate * i)
            for i in range(self.cfg.num_frames)
        ]
        #
        # # print(type(self.cfg.dataset.target_sampling_rate))  # 打印类型
        # # print('---num_frames', self.cfg.num_frames)
        # # print('---clip_start_frame', unlabel_clip_start_frame)
        #
        # # TODO 这个unlabel的start_frame为int，只有一个，而source的有16个这个不要紧吗？
        unlabel_frame_names = [
            max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
            for i in range(self.cfg.num_frames)
        ]
        #
        for frame_name in source_frame_names:
            source_frame = self._get_frame_source(
                source_dir_to_img_frame, frame_name, self.mode, source_frames
            )
            print(f"源帧类型: {type(source_frame)}, 形状: {source_frame.shape}")  # 应为 torch.Tensor, [2, H, W]
            source_frames.append(source_frame)

        # print('---unlabel_frames---', unlabel_frames)
        for frame_name in unlabel_frame_names:
            unlabel_frame = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, self.mode, unlabel_frames
            )
            print(f"未标记帧类型: {type(unlabel_frame)}, 形状: {unlabel_frame.shape}")  # 应为 torch.Tensor, [2, H, W]
            unlabel_frames.append(unlabel_frame)

        # 断言列表非空
        assert len(source_frames) > 0, "source_frames 为空，请检查数据加载逻辑"
        assert len(unlabel_frames) > 0, "unlabel_frames 为空，请检查数据加载逻辑"

        # [T, H, W, C] -> [T*C, H, W] -> [C, T, H, W]
        # print('-----transform------', self.transform)
        # print('-----------')
        # print(dir(self.transform))  # 打印所有可用的属性和方法
        # print('-----------')
        source_frames = self.transform.weak_aug(source_frames)
        unlabel_frames = self.transform.weak_aug(unlabel_frames)

        print(f"预处理后 source_frames 形状: {source_frames.shape}")  # 应为 [T, C=2, 224, 384]
        source_frames = source_frames.permute(0, 3, 1, 2)  # 假设输入是 [T, H, W, C]
        unlabel_frames = unlabel_frames.permute(0, 3, 1, 2)

        # mask generation
        mask = self.mask_gen()


        # print(f"修正后 source_frames 形状: {source_frames.shape}")  # 应为 [4, 3, 16, 224, 384]
        return source_frames, unlabel_frames, mask


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

        # _, _, T, H, W = source_frames.shape  # 假设已修正为 [B, C, T, H, W]
        _, T, H, W = source_frames.shape
        mask = self._generate_mask(H, W, T)


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