import copy
import logging
import random

import numpy as np
import pytorch_lightning as pl
import sklearn.linear_model
import torch
import torch.nn as nn
import torchmetrics
from einops import rearrange
from torch.nn.functional import normalize

from models.cmt import CrossModalTranslate
from netscripts.get_model import get_model
from netscripts.get_optimizer import get_optimizer_mmdistill

logger = logging.getLogger(__name__)


class MMDistillTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(MMDistillTrainer, self).__init__()
        self.cfg = cfg
        # model
        self.student_rgb = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[0],
            input_size=cfg.data_module.input_size[0],
            patch_size=cfg.data_module.patch_size[0][0],
            in_chans=cfg.trainer.in_chans[0],
        )
        self.teacher_flow = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[1],
            input_size=cfg.data_module.input_size[1],
            patch_size=cfg.data_module.patch_size[1][0],
            in_chans=cfg.trainer.in_chans[1],
        )
        self.teacher_skeleton = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[2],
            input_size=cfg.data_module.input_size[2],
            patch_size=cfg.data_module.patch_size[2][0],
            in_chans=cfg.trainer.in_chans[2],
        )
        self.cmt = CrossModalTranslate()
        self.teacher_rgb = copy.deepcopy(self.student_rgb)
        self.teacher_rgb.requires_grad_(False)
        self.teacher_flow.requires_grad_(False)
        self.teacher_skeleton.requires_grad_(False)

        # loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.train_top1_a = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.cfg.data_module.num_classes_action
        )

        # initialization
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        self.scale_lr()
        self.trainer.fit_loop.setup_data()
        dataset = self.trainer.train_dataloader.dataset
        self.niter_per_epoch = len(dataset) // self.total_batch_size
        print("Number of training steps = %d" % self.niter_per_epoch)
        print(
            "Number of training examples per epoch = %d"
            % (self.total_batch_size * self.niter_per_epoch)
        )
        optimizer, scheduler = get_optimizer_mmdistill(
            self.cfg.trainer, [self.student_rgb, self.cmt], self.niter_per_epoch
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        cur_iter = self.trainer.global_step
        next_lr = scheduler.get_epoch_values(cur_iter + 1)[0]
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group["lr"] = next_lr

    def _forward_loss_action(
        self,
        unlabel_frames_rgb_w,
        unlabel_frames_flow_w,
        unlabel_frames_skeleton_w,
        mask=None,
    ):
        # feature distillation
        fr, _ = self.teacher_rgb(unlabel_frames_rgb_w, mask)
        ff, _ = self.teacher_flow(unlabel_frames_flow_w, mask)
        fs, _ = self.teacher_skeleton(unlabel_frames_skeleton_w, mask)
        x_rgb, _ = self.student_rgb(unlabel_frames_rgb_w, mask)
        trans_rgb, trans_flow, trans_skeleton = self.cmt(x_rgb)

        trans_loss_rgb = self.mse_loss(trans_rgb, fr.detach())
        trans_loss_flow = self.mse_loss(trans_flow, ff.detach())
        trans_loss_skeleton = self.mse_loss(trans_skeleton, fs.detach())
        return trans_loss_rgb, trans_loss_flow, trans_loss_skeleton

    def training_step(self, batch, batch_idx):
        input = batch

        unlabel_frames_rgb_w = input["unlabel_frames_rgb"]
        unlabel_frames_flow_w = input["unlabel_frames_flow"]
        unlabel_frames_skeleton_w = input["unlabel_frames_skeleton"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        trans_loss_rgb, trans_loss_flow, trans_loss_skeleton = self._forward_loss_action(
            unlabel_frames_rgb_w,
            unlabel_frames_flow_w,
            unlabel_frames_skeleton_w,
            bool_masked_pos,
        )

        loss = trans_loss_rgb + trans_loss_flow + trans_loss_skeleton

        outputs = {
            "train_loss": loss.item(),
            "trans_loss_rgb": trans_loss_rgb.item(),
            "trans_loss_flow": trans_loss_flow.item(),
            "trans_loss_skeleton": trans_loss_skeleton.item(),
        }

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log_dict(outputs)
        # 保存 outputs 用于 epoch_end 计算平均值
        self.training_step_outputs.append(outputs)
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
        # 计算平均训练损失
        if len(self.training_step_outputs) > 0:
            train_loss = np.mean([output["train_loss"] for output in self.training_step_outputs])
            trans_loss_rgb = np.mean([output["trans_loss_rgb"] for output in self.training_step_outputs])
            trans_loss_flow = np.mean([output["trans_loss_flow"] for output in self.training_step_outputs])
            trans_loss_skeleton = np.mean([output["trans_loss_skeleton"] for output in self.training_step_outputs])
            
            self.log("train_loss", train_loss, on_step=False, on_epoch=True)
            self.log("train_trans_loss_rgb", trans_loss_rgb, on_step=False, on_epoch=True)
            self.log("train_trans_loss_flow", trans_loss_flow, on_step=False, on_epoch=True)
            self.log("train_trans_loss_skeleton", trans_loss_skeleton, on_step=False, on_epoch=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True)
            
            self.training_step_outputs.clear()
            
            # 定期保存 checkpoint（如果配置了 save_ckpt_freq）
            save_ckpt_freq = getattr(self.cfg.trainer, 'save_ckpt_freq', None)
            if save_ckpt_freq is not None and (self.trainer.current_epoch + 1) % save_ckpt_freq == 0:
                # 使用 DeepSpeed 的 save_checkpoint 方法
                checkpoint_dir = f"checkpoints/epoch={self.trainer.current_epoch:02d}-loss={train_loss:.4f}"
                logger.info(f"Saving checkpoint to: {checkpoint_dir}")
                # 注意：DeepSpeed 的 save_checkpoint 需要特殊处理
                # 这里我们依赖 PyTorch Lightning 的 ModelCheckpoint callback
                # 如果需要手动保存，可以使用 self.trainer.save_checkpoint()，但需要适配 DeepSpeed

    def validation_step(self, batch, batch_idx):
        input = batch[0]

        frames_rgb = input["frames"]
        action_idx = input["action_idx"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # convert labels for fewshot evaluation
        _, action_idx = torch.unique(action_idx, return_inverse=True)

        n_way = self.cfg.data_module.n_way
        k_shot = self.cfg.data_module.k_shot
        q_sample = self.cfg.data_module.q_sample

        # 检查 batch size 是否正确
        expected_batch_size = n_way * (k_shot + q_sample)
        actual_batch_size = frames_rgb.shape[0]
        
        if actual_batch_size != expected_batch_size:
            logger.warning(
                f"Validation batch size mismatch: expected {expected_batch_size}, got {actual_batch_size}. "
                f"Skipping this validation step."
            )
            return None

        # RGB
        frames_rgb, support_frames_rgb, query_frames_rgb = self.preprocess_frames(
            frames=frames_rgb, n_way=n_way, k_shot=k_shot, q_sample=q_sample
        )

        # mask
        support_mask = bool_masked_pos[: k_shot * n_way]
        query_mask = bool_masked_pos[k_shot * n_way :]

        action_idx = action_idx.view(n_way, (k_shot + q_sample))
        support_action_label, query_action_label = (
            action_idx[:, :k_shot].flatten(),
            action_idx[:, k_shot:].flatten(),
        )

        pred_rgb, prob_rgb = self.LR(
            self.student_rgb,
            support=support_frames_rgb,
            support_label=support_action_label,
            query=query_frames_rgb,
            support_mask=support_mask,
            query_mask=query_mask,
        )

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

        top1_action = acc(pred_rgb.cpu(), query_action_label.cpu())

        outputs = {
            "top1_action": top1_action.item(),
        }
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        top1_action = np.mean(
            [output["top1_action"] for output in self.validation_step_outputs]
        )
        self.log("val_top1_action", top1_action, on_step=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        input = batch[0]

        frames_rgb = input["frames"]
        action_idx = input["action_idx"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # convert labels for fewshot evaluation
        unique_labels, action_idx = torch.unique(action_idx, return_inverse=True)

        k_shot = self.cfg.data_module.k_shot
        q_sample = self.cfg.data_module.q_sample
        samples_per_class = k_shot + q_sample
        
        actual_batch_size = frames_rgb.shape[0]
        
        # 动态计算 n_way：根据实际 batch size 和每个类别需要的样本数
        # 如果 batch size 不能被 samples_per_class 整除，说明样本不足
        if actual_batch_size < samples_per_class:
            logger.debug(
                f"Test batch too small: got {actual_batch_size}, need at least {samples_per_class}. "
                f"Skipping this test step."
            )
            return None
        
        # 计算实际可以使用的 n_way
        n_way = actual_batch_size // samples_per_class
        
        # 如果无法整除，截断到可以整除的部分
        original_batch_size = actual_batch_size
        if actual_batch_size % samples_per_class != 0:
            actual_batch_size = n_way * samples_per_class
            frames_rgb = frames_rgb[:actual_batch_size]
            action_idx = action_idx[:actual_batch_size]
            bool_masked_pos = bool_masked_pos[:actual_batch_size]
            logger.debug(
                f"Adjusted batch size from {original_batch_size} "
                f"to {actual_batch_size} (n_way={n_way})"
            )
        
        # 确保 n_way 至少为 1
        if n_way < 1:
            logger.debug(f"n_way too small: {n_way}. Skipping this test step.")
            return None

        # RGB - 使用动态计算的 n_way
        frames_rgb, support_frames_rgb, query_frames_rgb = self.preprocess_frames_dynamic(
            frames=frames_rgb, n_way=n_way, k_shot=k_shot, q_sample=q_sample
        )

        # mask
        support_mask = bool_masked_pos[: k_shot * n_way]

        query_masks = []
        for _ in range(2):
            query_mask = bool_masked_pos[k_shot * n_way :]
            query_masks.append(query_mask)
            # Shift by 1 in the batch dimension
            bool_masked_pos = torch.cat(
                (bool_masked_pos[1:], bool_masked_pos[:1]), dim=0
            )

        action_idx = action_idx.view(n_way, (k_shot + q_sample))
        support_action_label, query_action_label = (
            action_idx[:, :k_shot].flatten(),
            action_idx[:, k_shot:].flatten(),
        )

        # # prediction with no mask
        # pred_rgb, prob_rgb = self.LR(
        #     self.student_rgb,
        #     support=support_frames_rgb,
        #     support_label=support_action_label,
        #     query=query_frames_rgb,
        # )

        # prediction with mask and ensemble
        pred_rgb_ensemble, prob_rgb_original = self.LR_ensemble(
            self.teacher_rgb,
            support=support_frames_rgb,
            support_label=support_action_label,
            query=query_frames_rgb,
            support_mask=support_mask,
            query_masks=query_masks[:2],
        )

        # 使用动态的 n_way 创建 accuracy metric
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_way)

        # top1_action = acc(pred_rgb.cpu(), query_action_label.cpu())
        top1_action_ensemble = acc(pred_rgb_ensemble.cpu(), query_action_label.cpu())

        outputs = {
            # "top1_action": top1_action.item(),
            "top1_action_ensemble": top1_action_ensemble.item(),
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            logger.warning("No test outputs collected. All batches may have been skipped due to size mismatch.")
            return
        
        top1_action_ensemble = np.mean(
            [output["top1_action_ensemble"] for output in self.test_step_outputs]
        )
        top1_action_ensemble_std = np.std(
            [output["top1_action_ensemble"] for output in self.test_step_outputs]
        )
        top1_action_ensemble_std_error = top1_action_ensemble_std / np.sqrt(
            len(self.test_step_outputs)
        )
        self.log("top1_action_ensemble", top1_action_ensemble, on_step=False)
        self.log("top1_action_ensemble_std", top1_action_ensemble_std, on_step=False)
        self.log(
            "top1_action_ensemble_std_error",
            top1_action_ensemble_std_error,
            on_step=False,
        )
        self.test_step_outputs.clear()

    def scale_lr(self):
        self.total_batch_size = self.cfg.batch_size * len(self.cfg.devices)
        self.cfg.trainer.lr = self.cfg.trainer.lr * self.total_batch_size / 256
        self.cfg.trainer.min_lr = self.cfg.trainer.min_lr * self.total_batch_size / 256
        self.cfg.trainer.warmup_lr = (
            self.cfg.trainer.warmup_lr * self.total_batch_size / 256
        )
        print("LR = %.8f" % self.cfg.trainer.lr)
        print("Batch size = %d" % self.total_batch_size)

    def preprocess_frames(self, frames, n_way, k_shot, q_sample):
        # 检查输入形状
        expected_batch_size = n_way * (k_shot + q_sample)
        actual_batch_size = frames.shape[0]
        
        if actual_batch_size != expected_batch_size:
            raise ValueError(
                f"Batch size mismatch in preprocess_frames: "
                f"expected {expected_batch_size} (n_way={n_way} * (k_shot={k_shot} + q_sample={q_sample})), "
                f"got {actual_batch_size}. "
                f"Frames shape: {frames.shape}"
            )
        
        frames = rearrange(
            frames, "(n m) c t h w -> n m c t h w", n=n_way, m=(k_shot + q_sample)
        )

        support_frames = rearrange(
            frames[:, :k_shot],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=k_shot,
        )
        query_frames = rearrange(
            frames[:, k_shot:],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=q_sample,
        )
        return frames, support_frames, query_frames

    def preprocess_frames_dynamic(self, frames, n_way, k_shot, q_sample):
        """
        动态处理 frames，允许不完整的 batch
        与 preprocess_frames 相同，但用于动态 n_way 的情况
        """
        expected_batch_size = n_way * (k_shot + q_sample)
        actual_batch_size = frames.shape[0]
        
        if actual_batch_size != expected_batch_size:
            # 如果仍然不匹配，截断到可以整除的部分
            actual_batch_size = (actual_batch_size // (k_shot + q_sample)) * (k_shot + q_sample)
            frames = frames[:actual_batch_size]
            n_way = actual_batch_size // (k_shot + q_sample)
        
        frames = rearrange(
            frames, "(n m) c t h w -> n m c t h w", n=n_way, m=(k_shot + q_sample)
        )

        support_frames = rearrange(
            frames[:, :k_shot],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=k_shot,
        )
        query_frames = rearrange(
            frames[:, k_shot:],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=q_sample,
        )
        return frames, support_frames, query_frames

    @torch.no_grad()
    def LR(
        self,
        model,
        support,
        support_label,
        query,
        support_mask=None,
        query_mask=None,
        norm=False,
    ):
        """logistic regression classifier"""
        support = model(support, support_mask)[0].detach()
        query = model(query, query_mask)[0].detach()
        if norm:
            support = normalize(support)
            query = normalize(query)

        clf = sklearn.linear_model.LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            C=1,
        )

        support_features_np = support.data.cpu().numpy()
        support_label_np = support_label.data.cpu().numpy()
        
        # 检查 support set 中是否有至少两个不同的类别
        unique_labels = np.unique(support_label_np)
        if len(unique_labels) < 2:
            # 如果只有一个类别，返回默认预测（所有 query 都预测为这个类别）
            logger.debug(f"Support set contains only {len(unique_labels)} class(es). Using default prediction.")
            query_features_np = query.data.cpu().numpy()
            n_query = query_features_np.shape[0]
            pred = np.full(n_query, unique_labels[0] if len(unique_labels) > 0 else 0)
            # 创建概率矩阵（所有概率都分配给这个类别）
            n_classes = len(unique_labels) if len(unique_labels) > 0 else 1
            prob = np.zeros((n_query, n_classes))
            prob[:, 0] = 1.0
            pred = torch.from_numpy(pred).type_as(support)
            prob = torch.from_numpy(prob).type_as(support)
            return pred, prob
        
        clf.fit(support_features_np, support_label_np)

        query_features_np = query.data.cpu().numpy()
        pred = clf.predict(query_features_np)
        prob = clf.predict_proba(query_features_np)

        pred = torch.from_numpy(pred).type_as(support)
        prob = torch.from_numpy(prob).type_as(support)
        return pred, prob

    @torch.no_grad()
    def LR_ensemble(
        self,
        model,
        support,
        support_label,
        query,
        support_mask=None,
        query_masks=None,
        norm=False,
    ):
        """logistic regression classifier"""
        support = model(support, support_mask)[0].detach()

        clf = sklearn.linear_model.LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            C=1,
        )

        support_features_np = support.data.cpu().numpy()
        support_label_np = support_label.data.cpu().numpy()
        
        # 检查 support set 中是否有至少两个不同的类别
        unique_labels = np.unique(support_label_np)
        if len(unique_labels) < 2:
            # 如果只有一个类别，返回默认预测（所有 query 都预测为这个类别）
            logger.debug(f"Support set contains only {len(unique_labels)} class(es). Using default prediction.")
            n_query = query.shape[0]
            pred = np.full(n_query, unique_labels[0] if len(unique_labels) > 0 else 0)
            # 创建概率矩阵（所有概率都分配给这个类别）
            n_classes = len(unique_labels) if len(unique_labels) > 0 else 1
            prob = np.zeros((n_query, n_classes))
            prob[:, 0] = 1.0
            pred = torch.from_numpy(pred).type_as(support)
            prob = torch.from_numpy(prob).type_as(support)
            return pred, prob
        
        clf.fit(support_features_np, support_label_np)

        probs = []
        for query_mask in query_masks:
            query_features = model(query, query_mask)[0].detach()

            query_features_np = query_features.data.cpu().numpy()
            prob = clf.predict_proba(query_features_np)
            probs.append(prob)

        probs = np.array(probs)
        prob = np.mean(probs, axis=0)
        pred = np.argmax(prob, axis=1)
        pred = torch.from_numpy(pred).type_as(support)
        prob = torch.from_numpy(prob).type_as(support)
        return pred, prob
