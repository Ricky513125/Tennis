import logging

# from models.modeling_finetune import *
from models.modeling_pretrain import (
    # pretrain_videomae_base_patch16_224,
    # pretrain_videomae_huge_patch16_224,
    # pretrain_videomae_large_patch16_224,
    # pretrain_videomae_small_patch16_224,
    VideoMAE_ViT_B_1600
)
from models.videomae_classifier import videomae_classifier_small_patch16_224

logger = logging.getLogger(__name__)


def get_model(cfg, ckpt_pth=None, input_size=224, patch_size=16, in_chans=None):
    print(f"Creating model: {cfg.trainer.model}")
    model = None
    
    if cfg.trainer.model.split("_")[0] == "VideoMAE":
        func = VideoMAE_ViT_B_1600
        # 检查配置格式：单模态或多模态
        if hasattr(cfg.data_module, 'modality') and hasattr(cfg.data_module.modality, 'mode'):
            # 单模态配置
            modality_mode = cfg.data_module.modality.mode
            if modality_mode == "RGB":
                assert cfg.trainer.modality.in_chans == 3
                assert cfg.trainer.pretrain is not None
            elif modality_mode == "flow":
                assert cfg.trainer.modality.in_chans == 2
                assert cfg.trainer.pretrain is not None
            elif modality_mode == "pose":
                assert cfg.trainer.modality.in_chans == 21
                assert cfg.trainer.pretrain is not None
            elif modality_mode == "skeleton":
                assert cfg.trainer.modality.in_chans == 17
                assert cfg.trainer.pretrain is not None
            else:
                raise Exception(f"{modality_mode} is not supported!")
            
            logger.info(f"[GET_MODEL] Creating model with img_size: {cfg.data_module.modality.input_size}, patch_size: {cfg.data_module.modality.patch_size[0]}")
            logger.info(f"[GET_MODEL] encoder_in_chans: {cfg.trainer.modality.in_chans}, decoder_num_classes: {cfg.trainer.modality.decoder_num_classes}")
            model = func(
                ckpt_pth=cfg.trainer.pretrain,
                img_size=cfg.data_module.modality.input_size,
                patch_size=cfg.data_module.modality.patch_size[0],
                encoder_in_chans=cfg.trainer.modality.in_chans,
                decoder_num_classes=cfg.trainer.modality.decoder_num_classes,
            )
        else:
            # 多模态配置：使用传入的参数
            logger.info(f"[GET_MODEL] Creating model with input_size: {input_size}, patch_size: {patch_size}, in_chans: {in_chans}")
            # 从 checkpoint 路径推断模态类型（用于确定 decoder_num_classes）
            # 或者从配置中获取
            if hasattr(cfg.trainer, 'modality') and hasattr(cfg.trainer.modality, 'decoder_num_classes'):
                decoder_num_classes = cfg.trainer.modality.decoder_num_classes
            else:
                # 根据 in_chans 推断
                if in_chans == 3:
                    decoder_num_classes = 1536  # RGB: 3 * 2 * 16^2
                elif in_chans == 2:
                    decoder_num_classes = 1024  # Flow: 2 * 2 * 16^2
                elif in_chans == 17:
                    decoder_num_classes = 8704  # Skeleton: 17 * 2 * 16^2
                else:
                    decoder_num_classes = 1536  # 默认
            
            # 获取 num_frames
            num_frames = getattr(cfg.data_module, 'num_frames', 16)
            
            model = func(
                ckpt_pth=ckpt_pth,
                img_size=input_size,
                patch_size=patch_size,
                encoder_in_chans=in_chans,
                decoder_num_classes=decoder_num_classes,
                num_frames=num_frames,  # 传递 num_frames
            )
        print(f"[GET_MODEL] Model created successfully")
    elif len(cfg.trainer.model.split("_")) > 1 and cfg.trainer.model.split("_")[1] == "classifier":
        scale = cfg.trainer.model.split("_")[2]
        if scale == "small":
            func = videomae_classifier_small_patch16_224
        else:
            raise Exception(f"{scale} is not supported!")

        # 确保 img_size 是正确的格式
        # input_size 可能是列表 [H, W]、OmegaConf ListConfig 或单个整数
        from omegaconf import ListConfig
        
        if isinstance(input_size, (list, ListConfig)):
            # 转换为元组 (H, W)，注意顺序：input_size 通常是 [H, W]
            # OmegaConf ListConfig 需要转换为普通列表再转元组
            img_size = tuple(list(input_size))
        elif isinstance(input_size, (tuple, int)):
            img_size = input_size
        else:
            raise ValueError(f"Unsupported input_size type: {type(input_size)}, value: {input_size}")
        
        logger.info(f"[GET_MODEL] Classifier model - img_size: {img_size}, patch_size: {patch_size}, in_chans: {in_chans}")
        
        model = func(
            ckpt_pth=ckpt_pth,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes_action=cfg.data_module.num_classes_action,
            use_mean_pooling=cfg.trainer.use_mean_pooling,
        )
        logger.info(f"[GET_MODEL] Classifier model created successfully")
    
    if model is None:
        raise Exception(f"Model creation failed for {cfg.trainer.model}. Please check the model name format.")

    # if cfg.trainer.model.split("_")[0] == "pretrain":
    #     scale = cfg.trainer.model.split("_")[2]
    #     if scale == "small":
    #         func = pretrain_videomae_small_patch16_224
    #     elif scale == "base":
    #         func = pretrain_videomae_base_patch16_224
    #     elif scale == "large":
    #         func = pretrain_videomae_large_patch16_224
    #     elif scale == "huge":
    #         func = pretrain_videomae_huge_patch16_224
    #     else:
    #         raise Exception(f"{scale} is not supported!")
    #     if cfg.data_module.modality.mode == "RGB":
    #         assert cfg.trainer.modality.in_chans == 3
    #         assert cfg.trainer.pretrain is not None
    #     elif cfg.data_module.modality.mode == "flow":
    #         assert cfg.trainer.modality.in_chans == 2
    #         assert cfg.trainer.pretrain is not None
    #     elif cfg.data_module.modality.mode == "pose":
    #         assert cfg.trainer.modality.in_chans == 21
    #         assert cfg.trainer.pretrain is not None
    #     else:
    #         raise Exception(f"{cfg.data_module.modality.mode} is not supported!")
    #
    #     model = func(
    #         ckpt_pth=cfg.trainer.pretrain,
    #         img_size=cfg.data_module.modality.input_size,
    #         patch_size=cfg.data_module.modality.patch_size[0],
    #         in_chans=cfg.trainer.modality.in_chans,
    #         decoder_num_classes=cfg.trainer.modality.decoder_num_classes,
    #     )
    # elif cfg.trainer.model.split("_")[1] == "classifier":
    #     scale = cfg.trainer.model.split("_")[2]
    #     if scale == "small":
    #         func = videomae_classifier_small_patch16_224
    #     else:
    #         raise Exception(f"{scale} is not supported!")
    #
    #     model = func(
    #         ckpt_pth=ckpt_pth,
    #         img_size=input_size,
    #         patch_size=patch_size,
    #         in_chans=in_chans,
    #         num_classes_action=cfg.data_module.num_classes_action,
    #         use_mean_pooling=cfg.trainer.use_mean_pooling,
    #     )
    return model
