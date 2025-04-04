import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers

from datamodule.lit_unlabel_combined_pretrain_data_module import (
    UnlabelCombinedPretrainDataModule,
)

from datamodule.lit_tennis_data_module import (
    TennisDataModule
)

from models.lit_VideoMAETrainer import VideoMAETrainer

from omegaconf import OmegaConf




warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_pretrain.yaml")
def main(cfg):
    # gpt
    # print("注意这里！", OmegaConf.to_yaml(cfg))  # 打印配置，确保 `target_json_path` 在里面

    print(cfg.trainer)
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module
    # data_module = UnlabelCombinedPretrainDataModule(cfg)
    # use Tennis data module
    data_module = TennisDataModule(cfg)

    # model
    model = VideoMAETrainer(cfg)


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
