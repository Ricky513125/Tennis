import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers

from datamodule.lit_tennis_skeleton_data_module import (
    TennisSkeletonDataModule
)

from models.lit_VideoMAETrainer_unlabel_only import VideoMAETrainer as VideoMAETrainerUnlabelOnly

from omegaconf import OmegaConf


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_pretrain_skeleton.yaml")
def main(cfg):
    print(cfg.trainer)
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module - 只使用 Tennis skeleton 数据
    data_module = TennisSkeletonDataModule(cfg)
    
    # model - 只使用重建损失
    model = VideoMAETrainerUnlabelOnly(cfg)

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.trainer.epochs,
        logger=train_logger,
        detect_anomaly=True,
    )

    if cfg.train:
        trainer.fit(model, data_module)
        print(trainer.callback_metrics)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
