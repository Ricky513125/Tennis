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

from datamodule.lit_tennis_unlabel_only_data_module import (
    TennisUnlabelOnlyDataModule
)

from models.lit_VideoMAETrainer import VideoMAETrainer
from models.lit_VideoMAETrainer_unlabel_only import VideoMAETrainer as VideoMAETrainerUnlabelOnly

from omegaconf import OmegaConf




warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_pretrain_flow.yaml")
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
    # 选项1：使用 Ego4D 作为 source，Tennis 作为 unlabel（原始 MM-CDFSL 方法）
    # data_module = UnlabelCombinedPretrainDataModule(cfg)
    # model = VideoMAETrainer(cfg)
    
    # 选项2：只使用 Tennis unlabel 数据，不使用源域数据（无标签微调）
    data_module = TennisUnlabelOnlyDataModule(cfg)
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
