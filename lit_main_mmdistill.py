import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.lit_unlabel_combined_mm_data_module import UnlabelCombinedMMDataModule
from datamodule.lit_tennis_unlabel_combined_mm_data_module import TennisUnlabelCombinedMMDataModule
from models.lit_MMDistillTrainer import MMDistillTrainer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_mmdistill.yaml")
def main(cfg):
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module - 根据数据集类型选择
    # 检查 target_dataset 或 dataset 字段
    dataset_name = None
    if hasattr(cfg.data_module, 'dataset') and hasattr(cfg.data_module.dataset, 'target_dataset'):
        dataset_name = cfg.data_module.dataset.target_dataset.lower()
    elif hasattr(cfg.data_module, 'target_dataset'):
        dataset_name = cfg.data_module.target_dataset.lower()
    elif hasattr(cfg.data_module, 'dataset'):
        # 如果 dataset 是字符串，直接使用
        dataset_name = str(cfg.data_module.dataset).lower() if not hasattr(cfg.data_module.dataset, 'target_dataset') else None
    
    if dataset_name == "tennis" or (hasattr(cfg.data_module, 'dataset') and str(cfg.data_module.dataset).lower() == "tennis"):
        logger.info("Using Tennis dataset for multimodal distillation")
        data_module = TennisUnlabelCombinedMMDataModule(cfg)
    else:
        logger.info("Using Ego4D dataset for multimodal distillation")
        data_module = UnlabelCombinedMMDataModule(cfg)

    # model
    model = MMDistillTrainer(cfg)

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    # 由于验证被禁用，使用训练损失来保存 checkpoint
    # 如果验证启用，则使用验证指标
    check_val_every_n_epoch = getattr(cfg, 'check_val_every_n_epoch', 999)
    if check_val_every_n_epoch >= 999:
        # 验证被禁用，监控训练损失
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath="checkpoints/",
            filename="{epoch:02d}-{train_loss:.4f}",
            save_top_k=5,
            mode="min",
            save_last=True,  # 总是保存最后一个 epoch
        )
    else:
        # 验证启用，监控验证指标
        checkpoint_callback = ModelCheckpoint(
            monitor="val_top1_action",
            dirpath="checkpoints/",
            filename="{epoch:02d}-{val_top1_action:.4f}",
            save_top_k=5,
            mode="max",
            save_last=True,  # 总是保存最后一个 epoch
        )

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.trainer.epochs,
        logger=train_logger,
        callbacks=[checkpoint_callback],
        detect_anomaly=True,
        use_distributed_sampler=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    if cfg.train:
        trainer.fit(model, data_module)
        print(trainer.callback_metrics)

    if cfg.test:
        logging.basicConfig(level=logging.DEBUG)
        trainer.test(model, data_module)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
