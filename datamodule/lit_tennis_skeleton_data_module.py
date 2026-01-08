"""
只使用 Tennis skeleton 数据的数据模块
用于在无标签目标域上进行 skeleton 数据的微调
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodule.dataset.tennis_skeleton_dataset import TennisSkeletonDataset
from datamodule.utils.augmentation import (
    DataAugmentationForUnlabelMM,
    MaskGeneration,
)


class TennisSkeletonDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_module_cfg = cfg.data_module
        self.skeleton_dir = cfg.data_module.skeleton_dir
        self.mask_gen = MaskGeneration(cfg.data_module)
        # skeleton 数据使用与 pose 相同的 transform
        self.transform_train = DataAugmentationForUnlabelMM(
            cfg,
            mean=cfg.data_module.modality.mean,
            std=cfg.data_module.modality.std,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TennisSkeletonDataset(
                self.data_module_cfg,
                self.transform_train,
                self.mask_gen,
                skeleton_dir=self.skeleton_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
