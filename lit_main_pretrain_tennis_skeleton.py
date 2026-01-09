import logging
import os
import random
import sys
import warnings
from pathlib import Path

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


def setup_logging(log_dir):
    """设置日志，同时输出到控制台和文件"""
    log_file = Path(log_dir) / "debug.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有的 handlers
    root_logger.handlers = []
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_file}")
    return log_file


@hydra.main(config_path="configs", config_name="config_pretrain_skeleton.yaml")
def main(cfg):
    # 设置日志（Hydra 会自动切换到输出目录）
    log_file = setup_logging(os.getcwd())
    logger.info("=" * 80)
    logger.info("Starting Skeleton Training")
    logger.info("=" * 80)
    
    logger.info(f"Configuration:\n{cfg.trainer}")
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
        logger.info(f"Using {len(cfg.devices)} GPUs !")

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
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        logger.info(f"Training completed. Metrics: {trainer.callback_metrics}")
        logger.info(f"Debug log saved to: {log_file}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
