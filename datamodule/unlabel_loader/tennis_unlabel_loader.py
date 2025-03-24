from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TENNISUnlabelLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # 初始化
        self._dir_to_img_frame = []
        self._uid = []
        self._far_name = []
        self._near_name = []
        self._video_id = []
        self._start_frame = []

        # 读取未标注数据的 JSON 文件
        with open(cfg.unlabel_json_path) as f:
            data = json.load(f)

        # 解析 JSON 数据
        if isinstance(data, list):  # 确保数据是列表
            for clip_dict in data:
                video_id = clip_dict["video"]
                start_frame = clip_dict["events"][0]["frame"]  # 取第一个事件帧号
                far_name = clip_dict["far_name"]
                near_name = clip_dict["near_name"]

                # 构造图像帧路径
                dir_to_img_frame = Path(
                    cfg.target_data_dir,
                    "vid_frames_224",
                    far_name,
                    near_name,
                    video_id,
                )

                # 存储信息
                self._dir_to_img_frame.append(dir_to_img_frame)
                self._uid.append(video_id)
                self._far_name.append(far_name)
                self._near_name.append(near_name)
                self._video_id.append(video_id)
                self._start_frame.append(start_frame)

        logger.info(f"Constructing TENNIS unlabel dataloader (size: {len(self._uid)})")

    def get_frame_str(self, frame_name):
        """格式化帧文件名，例如 frame_0000000123.jpg"""
        return f"frame_{str(frame_name).zfill(10)}.jpg"

    def __len__(self):
        """返回未标注样本的数量"""
        return len(self._uid)
