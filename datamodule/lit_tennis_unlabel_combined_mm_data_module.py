"""
Tennis 数据集的多模态数据模块（RGB、Flow、Skeleton）
用于多模态蒸馏训练
"""
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler

from datamodule.dataset.tennis_unlabel_combined_multimodal_dataset import (
    TennisUnlabelCombinedMMDataset,
)
from datamodule.utils.augmentation import (
    DataAugmentationForUnlabelMM,
    DataAugmentationForUnlabelRGB,
    DataAugmentationForVideoMAERGB,
    MaskGeneration,
)
from datamodule.utils.episodic_batch_sampler import EpisodicBatchSampler
from netscripts.get_fewshot_eval_dataset import get_fewshot_eval_dataset


class TennisUnlabelCombinedMMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(TennisUnlabelCombinedMMDataModule, self).__init__()
        self.cfg = cfg
        self.data_module_cfg = cfg.data_module
        self.n_way = cfg.data_module.n_way
        self.k_shot = cfg.data_module.k_shot
        self.q_sample = cfg.data_module.q_sample
        self.episodes = cfg.data_module.episodes
        self.eval_batch_size = self.n_way * (self.k_shot + self.q_sample)
        
        # transform
        self.mask_gen = MaskGeneration(cfg.data_module)
        self.transform_train_rgb = DataAugmentationForUnlabelRGB(cfg.data_module)
        self.transform_train_flow = DataAugmentationForUnlabelMM(
            cfg.data_module, mean=cfg.data_module.mean[1], std=cfg.data_module.std[1], mode="flow"
        )
        self.transform_train_skeleton = DataAugmentationForUnlabelMM(
            cfg.data_module, mean=cfg.data_module.mean[2], std=cfg.data_module.std[2], mode="skeleton"
        )
        
        self.transform_train = [
            self.transform_train_rgb,
            self.transform_train_flow,
            self.transform_train_skeleton,
        ]
        
        # 评估用的 transform（只使用 RGB）
        # 使用 DataAugmentationForUnlabelRGB，确保与训练时使用相同的 224x384 尺寸
        # 从配置中获取 RGB 的 input_size: [224, 384]
        rgb_input_size = None
        if hasattr(cfg.data_module, 'input_size'):
            if isinstance(cfg.data_module.input_size, list) and len(cfg.data_module.input_size) > 0:
                # 多模态配置：input_size 是列表 [[224, 384], [224, 384], [224, 384]]
                # 取第一个（RGB）
                if isinstance(cfg.data_module.input_size[0], (list, tuple)):
                    rgb_input_size = list(cfg.data_module.input_size[0])
                else:
                    rgb_input_size = cfg.data_module.input_size[0]
        
        # 创建评估用的 RGB transform，使用 weak_aug（包含 VideoCenterCrop 到 224x384）
        self.transform_eval_rgb = DataAugmentationForUnlabelRGB(
            cfg.data_module,
            input_size=rgb_input_size,  # [224, 384]
            mean=cfg.data_module.mean[0] if isinstance(cfg.data_module.mean, list) else cfg.data_module.mean,
            std=cfg.data_module.std[0] if isinstance(cfg.data_module.std, list) else cfg.data_module.std,
        )
        # 评估时只使用 weak_aug（不进行随机翻转，保持一致性）
        # 但 TennisFewshotEvalDataset 会调用 transform((frames, None))，需要适配
        # 创建一个包装类，使其兼容评估数据集的调用方式
        # TennisFewshotEvalDataset 期望输出格式: [T*C, H, W]
        import torch
        from datamodule.utils.augmentation import ToTensor
        from PIL import Image as PILImage
        from torchvision import transforms
        
        class VideoCenterCrop:
            """评估时使用的居中裁剪类（与训练时一致）"""
            def __init__(self, size):
                # size 可能是 [H, W] 列表或单个整数
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    self.size = tuple(size)  # (H, W)
                elif isinstance(size, int):
                    self.size = (size, size)
                else:
                    self.size = (224, 384)  # 默认值
                self.target_H, self.target_W = self.size
            
            def __call__(self, tensor):
                # tensor: [T, C, H, W]
                # 对每一帧分别应用 CenterCrop
                T, C, H, W = tensor.shape
                cropped_frames = []
                
                for t in range(T):
                    frame = tensor[t]  # [C, H, W]
                    
                    # 如果尺寸已经匹配，直接返回
                    if H == self.target_H and W == self.target_W:
                        cropped_frames.append(frame)
                        continue
                    
                    # 居中裁剪：计算裁剪起始位置
                    if H != self.target_H:
                        start_h = (H - self.target_H) // 2
                        end_h = start_h + self.target_H
                    else:
                        start_h = 0
                        end_h = H
                    
                    if W != self.target_W:
                        start_w = (W - self.target_W) // 2
                        end_w = start_w + self.target_W
                    else:
                        start_w = 0
                        end_w = W
                    
                    # 执行裁剪: [C, H, W] -> [C, target_H, target_W]
                    cropped_frame = frame[:, start_h:end_h, start_w:end_w]
                    
                    # 如果裁剪后尺寸仍不匹配（可能因为原始尺寸小于目标尺寸），进行 resize
                    if cropped_frame.shape[1] != self.target_H or cropped_frame.shape[2] != self.target_W:
                        # 转换为 PIL Image 进行 resize
                        frame_pil = transforms.ToPILImage()(cropped_frame)
                        cropped_frame = transforms.ToTensor()(
                            frame_pil.resize((self.target_W, self.target_H), PILImage.BILINEAR)  # PIL 使用 (W, H)
                        )
                    
                    cropped_frames.append(cropped_frame)
                
                return torch.stack(cropped_frames, dim=0)  # [T, C, H, W]
        
        class EvalTransformWrapper:
            def __init__(self, base_transform, input_size, mean, std):
                self.base_transform = base_transform
                self.input_size = input_size  # [224, 384]
                # 确保 mean 和 std 是 PyTorch tensor
                # 处理 omegaconf.ListConfig 或其他类型
                if not isinstance(mean, torch.Tensor):
                    if hasattr(mean, '__iter__') and not isinstance(mean, str):
                        mean = list(mean)  # 转换为 Python list
                    self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
                else:
                    self.mean = mean.view(-1, 1, 1)
                
                if not isinstance(std, torch.Tensor):
                    if hasattr(std, '__iter__') and not isinstance(std, str):
                        std = list(std)  # 转换为 Python list
                    self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
                else:
                    self.std = std.view(-1, 1, 1)
                
                # 评估时只进行居中裁剪和归一化，不进行随机翻转
                self.eval_transform = transforms.Compose([
                    ToTensor(),  # PIL Image 列表 -> [T, C, H, W]
                    VideoCenterCrop(self.input_size),  # [T, C, H, W] -> [T, C, H, W] (居中裁剪到 224x384)
                ])
            
            def _normalize_tensor(self, tensor):
                # tensor: [T, C, H, W]
                # mean/std: [C, 1, 1] -> [1, C, 1, 1]
                mean = self.mean.view(1, -1, 1, 1)
                std = self.std.view(1, -1, 1, 1)
                return (tensor - mean) / std
            
            def __call__(self, frames_tuple):
                # frames_tuple 是 (frames, None) 格式
                frames, _ = frames_tuple
                # 应用评估 transform: ToTensor + VideoCenterCrop
                frames_tensor = self.eval_transform(frames)  # [T, C, H, W]
                # 归一化
                frames_tensor = self._normalize_tensor(frames_tensor)  # [T, C, H, W]
                # 转换为 TennisFewshotEvalDataset 期望的格式: [T*C, H, W]
                T, C, H, W = frames_tensor.shape
                frames_reshaped = frames_tensor.view(T * C, H, W)  # [T*C, H, W]
                return frames_reshaped, None
        
        # 正确提取 RGB 的 mean 和 std
        # cfg.data_module.mean 是列表: [[RGB_mean], [Flow_mean], [Skeleton_mean]]
        if hasattr(cfg.data_module, 'mean'):
            if isinstance(cfg.data_module.mean, (list, tuple)) and len(cfg.data_module.mean) > 0:
                rgb_mean_raw = cfg.data_module.mean[0]
                # 处理 omegaconf.ListConfig 或其他类型
                if hasattr(rgb_mean_raw, '__iter__') and not isinstance(rgb_mean_raw, str):
                    rgb_mean = list(rgb_mean_raw)
                else:
                    rgb_mean = rgb_mean_raw
            else:
                rgb_mean = cfg.data_module.mean
        else:
            rgb_mean = [0.485, 0.456, 0.406]  # ImageNet 默认值
        
        if hasattr(cfg.data_module, 'std'):
            if isinstance(cfg.data_module.std, (list, tuple)) and len(cfg.data_module.std) > 0:
                rgb_std_raw = cfg.data_module.std[0]
                # 处理 omegaconf.ListConfig 或其他类型
                if hasattr(rgb_std_raw, '__iter__') and not isinstance(rgb_std_raw, str):
                    rgb_std = list(rgb_std_raw)
                else:
                    rgb_std = rgb_std_raw
            else:
                rgb_std = cfg.data_module.std
        else:
            rgb_std = [0.229, 0.224, 0.225]  # ImageNet 默认值
        
        # 确保 rgb_mean 和 rgb_std 是长度为 3 的列表（RGB 有 3 个通道）
        if not isinstance(rgb_mean, (list, tuple)) or len(rgb_mean) != 3:
            logger.warning(f"RGB mean length is {len(rgb_mean) if isinstance(rgb_mean, (list, tuple)) else 'not a list'}, expected 3. Using ImageNet defaults.")
            rgb_mean = [0.485, 0.456, 0.406]
        
        if not isinstance(rgb_std, (list, tuple)) or len(rgb_std) != 3:
            logger.warning(f"RGB std length is {len(rgb_std) if isinstance(rgb_std, (list, tuple)) else 'not a list'}, expected 3. Using ImageNet defaults.")
            rgb_std = [0.229, 0.224, 0.225]
        
        self.transform_eval = EvalTransformWrapper(
            self.transform_eval_rgb,
            input_size=rgb_input_size or [224, 384],
            mean=rgb_mean,
            std=rgb_std,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TennisUnlabelCombinedMMDataset(
                self.data_module_cfg, self.transform_train, self.mask_gen
            )
            
            # 打印训练集信息
            train_size = len(self.train_dataset)
            print("=" * 80)
            print("📊 数据集统计信息")
            print("=" * 80)
            print(f"✅ 训练集 (Train Dataset):")
            print(f"   - 总样本数: {train_size}")
            if hasattr(self.train_dataset, 'unlabel_loader'):
                if hasattr(self.train_dataset.unlabel_loader, '_dir_to_img_frame'):
                    print(f"   - 视频/目录数: {len(self.train_dataset.unlabel_loader._dir_to_img_frame)}")
            
            self.batch_sampler_train = BatchSampler(
                sampler=DistributedSampler(
                    dataset=self.train_dataset,
                    num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                    shuffle=True,
                ),
                batch_size=self.cfg.batch_size,
                drop_last=True,
            )
            self.val_dataset = get_fewshot_eval_dataset(
                self.data_module_cfg.dataset,
                self.transform_eval,
                self.mask_gen,
                self.data_module_cfg.num_frames,
                "RGB",
            )
            
            # 打印验证集信息
            val_size = len(self.val_dataset)
            print(f"\n✅ 验证集 (Validation Dataset):")
            print(f"   - 总样本数: {val_size}")
            
            # 统计每个 action 类别的样本数
            if hasattr(self.val_dataset, '_action_idx') and hasattr(self.val_dataset, '_action_label'):
                action_counts = {}
                action_labels_map = {}
                
                # 统计每个 action_idx 的样本数
                for i, action_idx in enumerate(self.val_dataset._action_idx):
                    action_counts[action_idx] = action_counts.get(action_idx, 0) + 1
                    # 记录每个 action_idx 对应的 label
                    if action_idx not in action_labels_map:
                        action_labels_map[action_idx] = self.val_dataset._action_label[i] if i < len(self.val_dataset._action_label) else "N/A"
                
                print(f"   - Action 类别数: {len(action_counts)}")
                print(f"   - 每个类别的样本数:")
                for action_idx, count in sorted(action_counts.items()):
                    action_label = action_labels_map.get(action_idx, "N/A")
                    print(f"     * Action {action_idx} ({action_label}): {count} 个样本")
            
            # 检查 few-shot 评估的可行性
            expected_batch_size = self.n_way * (self.k_shot + self.q_sample)
            print(f"\n📋 Few-shot 评估配置:")
            print(f"   - N-way: {self.n_way}")
            print(f"   - K-shot: {self.k_shot}")
            print(f"   - Q-sample: {self.q_sample}")
            print(f"   - 期望的 batch size: {expected_batch_size}")
            print(f"   - Episodes: {self.episodes}")
            
            if val_size < expected_batch_size:
                print(f"\n⚠️  警告: 验证集样本数 ({val_size}) 小于期望的 batch size ({expected_batch_size})")
                print(f"   无法创建完整的 few-shot episode。")
                print(f"   建议:")
                print(f"   1. 增加验证数据集的大小")
                print(f"   2. 或者减小 n_way, k_shot, q_sample 的值")
            else:
                max_episodes = val_size // expected_batch_size
                print(f"   - 理论上可以创建的最大 episodes: {max_episodes}")
                if self.episodes > max_episodes:
                    print(f"   ⚠️  警告: 配置的 episodes ({self.episodes}) 大于可创建的最大值 ({max_episodes})")
            
            print("=" * 80)
            batch_sampler = BatchSampler(
                sampler=DistributedSampler(
                    dataset=range(self.eval_batch_size * self.episodes),
                    num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                ),
                batch_size=self.eval_batch_size,
                drop_last=False,
            )
            self.episodic_batch_sampler_val = EpisodicBatchSampler(
                dataset=self.val_dataset,
                batch_sampler=batch_sampler,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_sample=self.q_sample,
                episodes=self.episodes,
            )
        elif stage == "test":
            self.test_dataset = get_fewshot_eval_dataset(
                self.data_module_cfg.dataset,
                self.transform_eval,
                self.mask_gen,
                self.data_module_cfg.num_frames,
                "RGB",
            )
            
            # 为 test stage 创建 batch_sampler
            batch_sampler = BatchSampler(
                sampler=DistributedSampler(
                    dataset=range(self.eval_batch_size * self.episodes),
                    num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                ),
                batch_size=self.eval_batch_size,
                drop_last=False,
            )
            self.episodic_batch_sampler_test = EpisodicBatchSampler(
                dataset=self.test_dataset,
                batch_sampler=batch_sampler,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_sample=self.q_sample,
                episodes=self.episodes,
            )

    def train_dataloader(self):
        # 评估阶段可能没有 train_dataset
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler_train,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        # 测试阶段可能没有 val_dataset
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.episodic_batch_sampler_val,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_sampler=self.episodic_batch_sampler_test,
            num_workers=self.cfg.num_workers,
        )
