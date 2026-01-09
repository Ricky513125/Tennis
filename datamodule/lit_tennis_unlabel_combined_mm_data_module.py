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
        # 需要从配置中获取 input_size，Tennis 使用 [224, 384]
        input_size = cfg.data_module.input_size[0] if isinstance(cfg.data_module.input_size, list) else 224
        if isinstance(input_size, list):
            input_size = input_size[0]  # 取高度
        
        self.transform_eval_rgb = DataAugmentationForVideoMAERGB(
            cfg.data_module, 
            input_size=input_size,
            multi_scale_crop=False
        )
        self.transform_eval = self.transform_eval_rgb

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
            self.episodic_batch_sampler_test = EpisodicBatchSampler(
                dataset=self.test_dataset,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_sample=self.q_sample,
                episodes=self.episodes,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler_train,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
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
