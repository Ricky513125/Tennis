import logging
from torchvision import transforms
import torch
from PIL import Image as PILImage
from datamodule.utils.masking_generator import (
    RandomMaskingGenerator,
    TubeMaskingGenerator,
)
from datamodule.utils.transform import (
    GroupMultiScaleCrop,
    GroupNormalize,
    GroupScale,
    Stack,
    ToTensor,
    ToTorchFormatTensor,
)

logger = logging.getLogger(__name__)


class DataAugmentationForVideoMAERGB(object):
    def __init__(
        self,
        cfg,
        num_frames=16,
        input_size=224,
        patch_size=[16, 16],
        mean=[0.485, 0.456, 0.406],  # IMAGENET_DEFAULT_MEAN
        std=[0.229, 0.224, 0.225],  # IMAGENET_DEFAULT_STD
        multi_scale_crop=True,
    ):
        aug_list = []

        if multi_scale_crop:
            aug_list.append(GroupMultiScaleCrop(input_size, [1, 0.875, 0.75, 0.66]))
        else:
            aug_list.append(GroupScale(size=(input_size, input_size)))
        aug_list.append(Stack(roll=False))
        aug_list.append(ToTorchFormatTensor(div=True))
        aug_list.append(GroupNormalize(mean, std))
        self.transform = transforms.Compose(aug_list)

        window_size = (
            num_frames // 2,
            input_size // patch_size[0],
            input_size // patch_size[1],
        )

        if cfg.mask_type == "tube":
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


class MaskGeneration(object):
    def __init__(
        self,
        cfg,
        num_frames=16,
        input_size=224,
        patch_size=[16, 16],
    ):
        window_size = (
            num_frames // 2,
            input_size // patch_size[0],
            input_size // patch_size[1],
        )
        if cfg.mask_type == "tube":
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        elif cfg.mask_type == "random":
            self.masked_position_generator = RandomMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self):
        return self.masked_position_generator()


class DataAugmentationForUnlabelRGB(object):
    def __init__(
        self, cfg, input_size=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        self.cfg = cfg
        
        # 从配置中读取 input_size（如果未提供）
        if input_size is None:
            # 尝试从配置中读取
            if hasattr(cfg, 'input_size'):
                if isinstance(cfg.input_size, list) and len(cfg.input_size) > 0:
                    # 多模态配置：input_size 是列表 [[224, 384], [224, 384], [224, 384]]
                    # 取第一个（RGB）
                    if isinstance(cfg.input_size[0], (list, tuple)):
                        input_size = list(cfg.input_size[0])
                    else:
                        input_size = cfg.input_size[0]
                else:
                    input_size = cfg.input_size
            else:
                input_size = [224, 384]  # 默认值
        
        # 处理 input_size
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            self.input_size = list(input_size)
        elif isinstance(input_size, int):
            self.input_size = [input_size, input_size]
        else:
            self.input_size = [224, 384]  # 默认值
        
        logger.info(f"[AUGMENTATION] DataAugmentationForUnlabelRGB initialized with input_size: {self.input_size}")
        
        # 转换为 Tensor 格式
        self.mean = torch.tensor(mean).view(-1, 1, 1) if isinstance(mean, list) else mean
        self.std = torch.tensor(std).view(-1, 1, 1) if isinstance(std, list) else std
        
        self._construct_no_aug()
        self._construct_weak_aug()
        self._construct_strong_aug()

    def _construct_no_aug(self):
        self.no_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _construct_weak_aug(self):
        # 自定义 transform 来处理视频 Tensor [T, C, H, W]
        # 使用 CenterCrop 而不是 RandomResizedCrop，与 Flow 处理保持一致
        class VideoCenterCrop:
            def __init__(self, size):
                # size 可能是 [H, W] 列表或单个整数
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    self.size = tuple(size)  # (H, W)
                elif isinstance(size, int):
                    self.size = (size, size)  # (size, size)
                else:
                    self.size = (224, 384)  # 默认值
                
                logger.debug(f"[VideoCenterCrop] target_size: {self.size}")
            
            def __call__(self, tensor):
                # tensor: [T, C, H, W]
                # 对每一帧分别应用 CenterCrop
                T, C, H, W = tensor.shape
                target_H, target_W = self.size
                cropped_frames = []
                
                for t in range(T):
                    frame = tensor[t]  # [C, H, W]
                    
                    # 如果尺寸已经匹配，直接返回
                    if H == target_H and W == target_W:
                        cropped_frames.append(frame)
                        continue
                    
                    # 居中裁剪：计算裁剪起始位置
                    if H != target_H:
                        start_h = (H - target_H) // 2
                        end_h = start_h + target_H
                    else:
                        start_h = 0
                        end_h = H
                    
                    if W != target_W:
                        start_w = (W - target_W) // 2
                        end_w = start_w + target_W
                    else:
                        start_w = 0
                        end_w = W
                    
                    # 执行裁剪: [C, H, W] -> [C, target_H, target_W]
                    cropped_frame = frame[:, start_h:end_h, start_w:end_w]
                    
                    # 如果裁剪后尺寸仍不匹配（可能因为原始尺寸小于目标尺寸），进行 resize
                    if cropped_frame.shape[1] != target_H or cropped_frame.shape[2] != target_W:
                        # 转换为 PIL Image 进行 resize
                        frame_pil = transforms.ToPILImage()(cropped_frame)
                        cropped_frame = transforms.ToTensor()(
                            frame_pil.resize((target_W, target_H), PILImage.BILINEAR)  # PIL 使用 (W, H)
                        )
                    
                    cropped_frames.append(cropped_frame)
                
                return torch.stack(cropped_frames, dim=0)  # [T, C, H, W]
        
        class VideoRandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p
                self.flip = transforms.RandomHorizontalFlip(p=p)
            
            def __call__(self, tensor):
                # tensor: [T, C, H, W]
                # 对每一帧分别应用 RandomHorizontalFlip（使用相同的随机性）
                if torch.rand(1) < self.p:
                    return torch.flip(tensor, dims=[3])  # 水平翻转最后一维（宽度）
                return tensor
        
        self.weak_aug = transforms.Compose(
            [
                ToTensor(),  # PIL Image 列表 -> [T, C, H, W]
                VideoCenterCrop(self.input_size),  # [T, C, H, W] -> [T, C, H, W] (居中裁剪，与 Flow 一致)
                VideoRandomHorizontalFlip(p=0.5),  # [T, C, H, W] -> [T, C, H, W]
                transforms.Lambda(lambda x: self._normalize_tensor(x)),  # 归一化
            ]
        )
    
    def _normalize_tensor(self, tensor):
        # tensor: [T, C, H, W]
        # mean/std: [C] -> [1, C, 1, 1]
        mean = self.mean.view(1, -1, 1, 1)
        std = self.std.view(1, -1, 1, 1)
        return (tensor - mean) / std

    def _construct_strong_aug(self):
        self.strong_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )


class DataAugmentationForVideoMAEMM(object):
    def __init__(self, cfg, mean, std):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose(
            [ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )

        if cfg.mask_type == "tube":
            window_size = (
                cfg.num_frames // 2,
                cfg.input_size // cfg.patch_size[0],
                cfg.input_size // cfg.patch_size[1],
            )
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self, images):
        process_data = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


class DataAugmentationForUnlabelMM(object):
    def __init__(self, cfg, mean, std, mode=None):
        self.cfg = cfg
        # self.mean = mean
        # self.std = std

        # 确定模态类型：支持两种配置格式
        # 1. 单模态配置：cfg.data_module.modality.mode (用于预训练)
        # 2. 多模态配置：cfg.data_module.mode 是列表 (用于多模态蒸馏)
        if mode is not None:
            # 如果显式传入 mode 参数，使用它
            self.modality_mode = mode
        elif hasattr(cfg, 'data_module') and hasattr(cfg.data_module, 'modality') and hasattr(cfg.data_module.modality, 'mode'):
            # 单模态配置格式
            self.modality_mode = cfg.data_module.modality.mode
        elif hasattr(cfg, 'data_module') and hasattr(cfg.data_module, 'mode'):
            # 多模态配置格式：mode 是列表，需要根据 mean/std 的长度推断
            # mean 和 std 的长度对应模态索引：0=RGB, 1=flow, 2=skeleton
            if isinstance(cfg.data_module.mode, list):
                # 根据 mean 的长度推断模态（但这里 mean 是传入的参数，不是列表）
                # 实际上，我们应该通过其他方式推断，或者要求传入 mode
                # 暂时使用默认值
                self.modality_mode = "flow"  # 默认，但应该通过参数传入
            else:
                self.modality_mode = cfg.data_module.mode
        else:
            # 无法确定，使用默认值
            self.modality_mode = "flow"
            logger.warning(f"[AUGMENTATION] Cannot determine modality mode, using default: {self.modality_mode}")

        # 从配置中读取 input_size
        # 对于多模态配置，input_size 是列表，需要根据模态索引选择
        if hasattr(cfg, 'data_module') and hasattr(cfg.data_module, 'modality') and hasattr(cfg.data_module.modality, 'input_size'):
            # 单模态配置格式
            input_size = cfg.data_module.modality.input_size
        elif hasattr(cfg, 'data_module') and hasattr(cfg.data_module, 'input_size'):
            # 多模态配置格式：input_size 是列表 [[224, 384], [224, 384], [224, 384]]
            if isinstance(cfg.data_module.input_size, list) and len(cfg.data_module.input_size) > 0:
                # 根据 mode 推断索引：0=RGB, 1=flow, 2=skeleton
                if self.modality_mode == "flow":
                    input_size = cfg.data_module.input_size[1] if len(cfg.data_module.input_size) > 1 else cfg.data_module.input_size[0]
                elif self.modality_mode == "skeleton":
                    input_size = cfg.data_module.input_size[2] if len(cfg.data_module.input_size) > 2 else cfg.data_module.input_size[0]
                elif self.modality_mode == "rgb":
                    input_size = cfg.data_module.input_size[0]
                else:
                    input_size = cfg.data_module.input_size[0]
            else:
                input_size = cfg.data_module.input_size
        else:
            input_size = None

        # 处理 input_size
        if input_size is not None:
            if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                self.input_size = list(input_size)
            elif isinstance(input_size, int):
                self.input_size = [input_size, input_size]
            else:
                self.input_size = [224, 384]  # 默认值
        else:
            # 根据模态设置默认值
            if self.modality_mode == "skeleton":
                self.input_size = [224, 384]  # Skeleton 应该与 RGB/Flow 保持一致
            else:
                self.input_size = [224, 384]  # 默认值
        
        logger.info(f"[AUGMENTATION] DataAugmentationForUnlabelMM initialized with input_size: {self.input_size}, modality: {self.modality_mode}")
        
        # self.mean = torch.tensor(mean).view(-1, 1, 1)  # 形状 [2, 1, 1]
        # self.std = torch.tensor(std).view(-1, 1, 1)  # 形状 [2, 1, 1]

        # 4.15 1619
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)  # [1, C, 1, 1]
        self.std = torch.tensor(std).view(1, -1, 1, 1)  # [1, C, 1, 1]


        self._construct_weak_aug()
        self._construct_strong_aug()


        # print("DataAugmentationForUnlabelMM")
        #
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),  # 将 numpy 数组或张量转为 PIL 图像
        #     transforms.Resize([224, 384]),  # 调整为 (H, W) = (224, 384)
        #     transforms.ToTensor(),  # 转换为张量 [C, H, W]
        #     transforms.Normalize(mean=self.mean, std=self.std),
        # ])

    def weak_aug(self, frames):
        """
        frames: 列表，每个元素是形状为 [C=2, H, W] 的张量
        返回: 形状为 [T, C=2, H, W] 的张量
        """
        # 调整尺寸（如果需要）
        resized_frames = [
            torch.nn.functional.interpolate(
                frame.unsqueeze(0),  # 添加 batch 维度 [1, C, H, W]
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # 移除 batch 维度 [C, H, W]
            for frame in frames
        ]

        # 堆叠并归一化
        frames_tensor = torch.stack(resized_frames)  # [T, C, H, W]
        frames_tensor = (frames_tensor - self.mean) / self.std
        return frames_tensor

    def _construct_weak_aug(self):
        # 确保 input_size 是元组格式，用于 resize
        input_size_tuple = tuple(self.input_size) if isinstance(self.input_size, (list, tuple)) else (self.input_size, self.input_size)
        
        def resize_frames(x):
            """
            x: [T, C, H, W]
            返回: [T, C, H_out, W_out]
            """
            T, C, H, W = x.shape
            target_H, target_W = self.input_size
            logger.info(f"[RESIZE] Input shape: {x.shape}, target input_size: {self.input_size}, modality: {self.modality_mode}")
            # 检查是否需要 resize
            if (H, W) == (target_H, target_W):
                logger.debug(f"[RESIZE] No resize needed, already at target size")
                return x
            
            logger.info(f"[RESIZE] Resizing from ({H}, {W}) to ({target_H}, {target_W})")
            
            # 对每一帧进行 resize
            resized_frames = []
            for t in range(T):
                frame = x[t]  # [C, H, W]
                # resize 需要 [N, C, H, W] 格式，所以添加 batch 维度
                frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
                # resize 到目标尺寸
                frame_resized = torch.nn.functional.interpolate(
                    frame_batch,
                    size=input_size_tuple,  # (H, W)
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )  # [1, C, H_out, W_out]
                resized_frames.append(frame_resized.squeeze(0))  # [C, H_out, W_out]
            
            result = torch.stack(resized_frames, dim=0)  # [T, C, H_out, W_out]
            logger.debug(f"[RESIZE] Output shape: {result.shape}")
            return result
        
        self.weak_aug = transforms.Compose(
            [
                # 输入格式: [T, H, W, C]
                # 第一步: 转换为 [T, C, H, W]
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) if x.dim() == 4 and x.shape[-1] in [2, 3, 17, 21] else x),
                # 第二步: 调整每帧的尺寸（如果需要）
                transforms.Lambda(resize_frames),  # [T, C, H, W] -> [T, C, H_out, W_out]
                # 第三步: 归一化
                # x 形状: [T, C, H, W], mean/std 形状: [1, C, 1, 1]
                transforms.Lambda(lambda x: (x - self.mean) / self.std),
            ]
        )
        
        # 添加验证：确保 weak_aug 输出正确的尺寸
        logger.info(f"[AUGMENTATION] weak_aug constructed for modality {self.modality_mode}, target input_size: {self.input_size}")

    def _construct_strong_aug(self):
        self.strong_aug = transforms.Compose(
            [
                # ToTensor(),
                # transforms.Normalize(mean=self.mean, std=self.std),
                # # GaussianNoise(variance=self.variance),

                # ToTensor(),
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )



