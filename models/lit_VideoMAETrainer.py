import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange

from netscripts.get_model import get_model
from netscripts.get_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class VideoMAETrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(VideoMAETrainer, self).__init__()
        self.cfg = cfg

        # # 验证分辨率能被patch_size整除
        # H = cfg.data_module.modality.img_size[0]
        # W = cfg.data_module.modality.img_size[1]




        self.model = get_model(cfg)
        self.normalize_target = cfg.trainer.normalize_target
        self.patch_size =16
        self.training_step_outputs = []
        # assert H % self.patch_size == 0, f"高度{H}必须能被patch_size {self.patch_size}整除"
        # assert W % self.patch_size == 0, f"宽度{W}必须能被patch_size {self.patch_size}整除"




    def configure_optimizers(self):
        total_batch_size = self.scale_lr()
        self.trainer.fit_loop.setup_data()
        dataset = self.trainer.train_dataloader.dataset
        self.niter_per_epoch = len(dataset) // total_batch_size
        print("Number of training steps = %d" % self.niter_per_epoch)
        print(
            "Number of training examples per epoch = %d"
            % (total_batch_size * self.niter_per_epoch)
        )
        optimizer, scheduler = get_optimizer(
            self.cfg.trainer, self.model, self.niter_per_epoch
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        cur_iter = self.trainer.global_step
        next_lr = scheduler.get_epoch_values(cur_iter + 1)[0]
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group["lr"] = next_lr

    def normalize_videos(self, unnorm_videos):
        # print("Normalizing videos")
        # print("--------------------")
        # print(unnorm_videos)
        # if unnorm_videos.shape[2] == 1 :
            # print(self.cfg)
            # print("--------------------")

        # 添加输入形状检查
        B, T, C, H, W = unnorm_videos.shape
        # B, C, T, H, W = unnorm_videos.shape # **********
        print(f"输入视频形状: [B={B}, T={T}, C={C}, H={H}, W={W}]")

        # 计算预期分块数
        expected_patches = (H // self.patch_size) * (W // self.patch_size) * (T // 2)
        print(f"预期序列长度: {expected_patches}")


        if self.normalize_target:
            videos_squeeze = rearrange(
                unnorm_videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                p0=2,
                p1=self.patch_size,
                p2=self.patch_size,
            )
            ''' 
            这里使用 einops.rearrange 将 unnorm_videos 重新排列成patch-based 格式：
            p0=2 表示时间轴（T 维度）以 2 帧 为一个 patch。
            p1=self.patch_size，p2=self.patch_size，表示空间维度被分割成 patch_size × patch_size 的小块。
(T' H' W'): 这是视频被划分成 patch 后的总 patch 数量。
(p0 * p1 * p2): 每个 patch 内的像素点数量。
C: 通道数。
                            '''
            videos_norm = (
                videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
            ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
            # 将p*c 展平成一个向量
        else:
            videos_patch = rearrange(
                unnorm_videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                p0=2,
                p1=self.patch_size,
                p2=self.patch_size,
            )
        return videos_patch

    def training_step(self, batch, batch_idx):
        input = batch
        source_frames = input["source_frames"] # 有标签的视频帧(B,T, C, H, W) 表示B(Batch)个T帧，每帧C个通道(RGB)
        unlabel_frames = input["unlabel_frames"]
        action_label = input["action_label"] # 动作分类标签

        # ========== 动态生成掩码 ==========
        B, T, C, H, W = source_frames.shape
        # 计算序列长度
        seq_length = (H // self.patch_size) * (W // self.patch_size) * (T // 2)
        # tubelet_size=2
        # 生成随机掩码
        mask_ratio = 0.75  # 或从配置中读取
        bool_masked_pos = torch.rand(B, seq_length) < mask_ratio
        bool_masked_pos = bool_masked_pos.to(source_frames.device)
        # ========== 结束修改 ==========

        # bool_masked_pos = input["mask"] # 掩码标记，用于视频MAE的mask
        # bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool) # flatten(1) from to [B, T*H*W]，再转为bool类型
        #
        # # print("Input shape:", batch.shape)  # 应为 [B, T, 3, H, W]
        # print("------original---------source_frames-----------", source_frames.shape)
        # print("------original---------unlabel_frames-----------", unlabel_frames.shape)

        # add: 调整维度的顺序以适应原模型，把channel和t帧进行位置对调
        # source_frames = source_frames.permute(0, 2, 1, 3, 4)
        # unlabel_frames = unlabel_frames.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            # calculate the predict label
            # 计算图像均值和标准差， （用于归一化）
            if self.cfg.data_module.modality.mode == "RGB":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                # print('----mean----', mean, mean.shape)

                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)
                # print('----std----', std, std.shape)
            elif self.cfg.data_module.modality.mode == "flow":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)
            elif self.cfg.data_module.modality.mode == "pose":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)

            print('---source_frames---', source_frames.shape)
            print('---unlabel_frames---', unlabel_frames.shape)
            print('---std---', std, '---std.shape---', std.shape)
            print('---mean---', mean, '---mean.shape---', mean.shape)

            print('lit_VideoMAETrainer.training_step', source_frames.shape)
            # 反归一化视频，将其恢复到原始数据范围[0, 1]
            unnorm_videos_source = source_frames * std + mean  # in [0, 1]
            unnorm_videos_target = unlabel_frames * std + mean  # in [0, 1]


            # 将视频帧转为token
            videos_patch_source = self.normalize_videos(unnorm_videos_source)
            videos_patch_target = self.normalize_videos(unnorm_videos_target)

            # b t c 只保留被mask的部分数据
            B, _, C = videos_patch_source.shape
            labels_source = videos_patch_source[bool_masked_pos].reshape(B, -1, C)
            labels_target = videos_patch_target[bool_masked_pos].reshape(B, -1, C)

        preds_source, logits_source = self.model(source_frames, bool_masked_pos)
        preds_target, _ = self.model(unlabel_frames, bool_masked_pos)

        loss_mse = nn.MSELoss()
        loss_ce = nn.CrossEntropyLoss()
        recon_loss_source = loss_mse(input=preds_source, target=labels_source)
        recon_loss_target = loss_mse(input=preds_target, target=labels_target)
        ce_loss = loss_ce(logits_source, action_label)

        loss = recon_loss_source + self.cfg.trainer.modality.lambda_ce * ce_loss

        output = {
            "loss": loss.item(),
            "recon_loss_source": recon_loss_source.item(),
            "recon_loss_target": recon_loss_target.item(),
            "ce_loss": ce_loss.item(),
        }

        self.training_step_outputs.append(output)
        return loss

    def on_train_epoch_start(self):
        # shuffle the unlabel data loader
        unlabel_dir_to_img_frame = (
            self.trainer.train_dataloader.dataset.unlabel_loader._dir_to_img_frame
        )
        unlabel_start_frame = (
            self.trainer.train_dataloader.dataset.unlabel_loader._start_frame
        )
        lists = list(zip(unlabel_dir_to_img_frame, unlabel_start_frame))
        random.shuffle(lists)
        unlabel_dir_to_img_frame, unlabel_start_frame = zip(*lists)
        self.trainer.train_dataloader.dataset.unlabel_loader._dir_to_img_frame = list(
            unlabel_dir_to_img_frame
        )
        self.trainer.train_dataloader.dataset.unlabel_loader._start_frame = list(
            unlabel_start_frame
        )

    def on_train_epoch_end(self):
        train_loss = np.mean([output["loss"] for output in self.training_step_outputs])
        train_recon_loss_source = np.mean(
            [output["recon_loss_source"] for output in self.training_step_outputs]
        )
        train_recon_loss_target = np.mean(
            [output["recon_loss_target"] for output in self.training_step_outputs]
        )
        train_ce_loss = np.mean(
            [output["ce_loss"] for output in self.training_step_outputs]
        )
        self.log("train_loss", train_loss, on_step=False)
        self.log("train_recon_loss_source", train_recon_loss_source, on_step=False)
        self.log("train_recon_loss_target", train_recon_loss_target, on_step=False)
        self.log("train_ce_loss", train_ce_loss, on_step=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.training_step_outputs.clear()
        # save the model parameters
        if (self.trainer.current_epoch + 1) % self.cfg.save_ckpt_freq == 0:
            self.trainer.save_checkpoint(
                f"checkpoints/epoch={self.trainer.current_epoch:02d}-loss={train_loss:.4f}"
            )

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def on_validation_epoch_end(self):
        return NotImplementedError

    def test_step(self, batch, batch_idx):
        return NotImplementedError

    def on_test_epoch_end(self):
        return NotImplementedError

    def scale_lr(self):
        total_batch_size = self.cfg.batch_size * len(self.cfg.devices)
        self.cfg.trainer.lr = self.cfg.trainer.lr * total_batch_size / 256
        self.cfg.trainer.min_lr = self.cfg.trainer.min_lr * total_batch_size / 256
        self.cfg.trainer.warmup_lr = self.cfg.trainer.warmup_lr * total_batch_size / 256
        print("LR = %.8f" % self.cfg.trainer.lr)
        print("Batch size = %d" % total_batch_size)
        return total_batch_size