"""
只使用 unlabel 数据的 VideoMAE 训练器
用于在无标签目标域上进行微调，不使用源域数据和分类损失
"""
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

        self.model = get_model(cfg)
        self.normalize_target = getattr(cfg.trainer, 'normalize_target', False)
        self.patch_size = 16
        self.training_step_outputs = []

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
        """将视频归一化并转换为 patch tokens"""
        # unnorm_videos: [B, C, T, H, W]
        if self.normalize_target:
            videos_squeeze = rearrange(
                unnorm_videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                p0=2,
                p1=self.patch_size,
                p2=self.patch_size,
            )
            videos_norm = (
                videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
            ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
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
        """只使用 unlabel 数据进行重建损失"""
        input = batch
        
        # 只使用 unlabel_frames
        unlabel_frames = input["unlabel_frames"]
        bool_masked_pos = input["mask"].flatten(1).to(torch.bool)  # [B, seq_length]

        # 输入: [B, T, H, W, C]
        B, T, H, W, C = unlabel_frames.shape
        
        # 计算序列长度
        seq_length = (H // self.patch_size) * (W // self.patch_size) * (T // 2)
        
        # 确保 mask 长度匹配
        if bool_masked_pos.shape[1] != seq_length:
            # 如果 mask 长度不匹配，重新生成
            mask_ratio = 0.75
            num_masked_per_batch = int(seq_length * mask_ratio)
            rand_indices = torch.rand(B, seq_length, device=unlabel_frames.device).argsort(dim=-1)
            bool_masked_pos = torch.zeros(B, seq_length, dtype=torch.bool, device=unlabel_frames.device)
            for i in range(B):
                bool_masked_pos[i, rand_indices[i, :num_masked_per_batch]] = True

        with torch.no_grad():
            # 先转换为 [B, C, T, H, W] 格式
            unlabel_frames = unlabel_frames.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]
            
            # 计算均值和标准差
            if self.cfg.data_module.modality.mode == "RGB":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
            elif self.cfg.data_module.modality.mode == "flow":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
            elif self.cfg.data_module.modality.mode == "pose":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
            elif self.cfg.data_module.modality.mode == "skeleton":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(unlabel_frames)

            # 反归一化
            unnorm_videos = unlabel_frames * std + mean  # in [0, 1]

            # 转换为 patch tokens
            videos_patch = self.normalize_videos(unnorm_videos)  # [B, num_patches, patch_dim]

            # 提取被 mask 的部分作为 label
            B, num_patches, C = videos_patch.shape
            # 确保 mask 长度匹配
            new_masked_pos = bool_masked_pos
            if bool_masked_pos.shape[1] > videos_patch.shape[1]:
                new_masked_pos = bool_masked_pos[:, :videos_patch.shape[1]]
            elif bool_masked_pos.shape[1] < videos_patch.shape[1]:
                # 如果 mask 长度小于 patches 数量，需要扩展
                pad_length = videos_patch.shape[1] - bool_masked_pos.shape[1]
                new_masked_pos = torch.cat([
                    bool_masked_pos,
                    torch.zeros(B, pad_length, dtype=torch.bool, device=bool_masked_pos.device)
                ], dim=1)
            
            # 按 batch 提取被 mask 的 patches
            # videos_patch: [B, num_patches, C]
            # new_masked_pos: [B, num_patches]
            labels_list = []
            for i in range(B):
                masked_patches = videos_patch[i][new_masked_pos[i]]  # [num_masked_i, C]
                labels_list.append(masked_patches)
            
            # 找到每个 batch 中被 mask 的 patch 数量
            num_masked_per_batch = [new_masked_pos[i].sum().item() for i in range(B)]
            max_masked = max(num_masked_per_batch)
            
            # 如果每个 batch 的 mask 数量不同，需要 padding 或截断
            # 但通常 mask_ratio 是固定的，所以应该相同
            if len(set(num_masked_per_batch)) == 1:
                # 所有 batch 的 mask 数量相同，可以直接 stack
                labels = torch.stack(labels_list, dim=0)  # [B, num_masked, C]
            else:
                # 如果不同，需要 padding 到最大长度
                padded_labels = []
                for i, label in enumerate(labels_list):
                    if label.shape[0] < max_masked:
                        padding = torch.zeros(max_masked - label.shape[0], C, 
                                             dtype=label.dtype, device=label.device)
                        label = torch.cat([label, padding], dim=0)
                    padded_labels.append(label)
                labels = torch.stack(padded_labels, dim=0)  # [B, max_masked, C]

        # 前向传播
        preds, _ = self.model(unlabel_frames, bool_masked_pos)

        # 只计算重建损失（不使用分类损失）
        loss_mse = nn.MSELoss()
        recon_loss = loss_mse(input=preds, target=labels)

        loss = recon_loss

        output = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
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
        train_recon_loss = np.mean(
            [output["recon_loss"] for output in self.training_step_outputs]
        )
        self.log("train_loss", train_loss, on_step=False)
        self.log("train_recon_loss", train_recon_loss, on_step=False)
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
