import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodule.dataset.tennis_dataset import TennisDataset  # ✅ 你需要实现这个 Dataset
from datamodule.utils.augmentation import (
    DataAugmentationForUnlabelMM,
    DataAugmentationForUnlabelRGB,
    MaskGeneration,
)

class TennisDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_module_cfg = cfg.data_module
        self.mode = cfg.data_module.modality.mode
        self.mask_gen = MaskGeneration(cfg.data_module)
        # add
        self.transform_train = None
        if self.mode == "RGB":
            self.transform_train = DataAugmentationForUnlabelRGB(
                cfg, input_size=cfg.data_module.modality.input_size
            )
        elif self.mode == "flow" or self.mode == "pose":
            self.transform_train = DataAugmentationForUnlabelMM(
                cfg,
                mean=cfg.data_module.modality.mean,
                std=cfg.data_module.modality.std,
            )

    def setup(self, stage=None):
        # self.train_dataset = TennisDataset(self.cfg, split="train")
        # self.val_dataset = TennisDataset(self.cfg, split="val")
        if stage == "fit" or stage is None:
            # 这里可能需要检查 transform_train 是否正确赋值
            # self.transform_train = self.get_train_transforms()

            # 根据模态选择 transform
            if self.mode == "RGB":
                transform = DataAugmentationForUnlabelRGB(...)
            elif self.mode == "flow":
                transform = DataAugmentationForUnlabelMM(cfg=self.cfg, mean=self.cfg.data_module.modality.mean,
                std=self.cfg.data_module.modality.std)

            self.train_dataset = TennisDataset(
                self.data_module_cfg,
                self.transform_train,
                self.mask_gen,
                mode=self.mode,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,

            precision="16-mixed"
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.cfg.trainer.batch_size,
    #         shuffle=False,
    #         num_workers=self.cfg.num_workers,
    #         pin_memory=True,
    #     )

    def get_train_transforms(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
