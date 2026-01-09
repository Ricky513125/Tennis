from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from models.modeling_finetune import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# __all__ = [
#     "pretrain_videomae_small_patch16_224",
#     "pretrain_videomae_base_patch16_224",
#     "pretrain_videomae_large_patch16_224",
#     "pretrain_videomae_huge_patch16_224",
# ]


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        # img_size=224,
        # patch_size=16,
        img_size=(384, 224),  # 设为 tuple 以支持非正方形输入 384
        patch_size = 16, # 使其能够被整除
        in_chans=2,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask=None):
        print(f"[MODEL] Input to forward_features - x shape: {x.shape}")
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        print(f"[MODEL] After patch_embed - x shape: {x.shape}, pos_embed shape: {self.pos_embed.shape}")
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, num_patches, C = x.shape
        # 在forward_features方法中
        if mask is not None:
            # 确保 mask 形状匹配
            if mask.shape[1] != num_patches:
                # 如果 mask 长度不匹配，调整 mask
                if mask.shape[1] > num_patches:
                    mask = mask[:, :num_patches]
                else:
                    # 如果 mask 长度小于 patches，padding False
                    pad_length = num_patches - mask.shape[1]
                    mask = torch.cat([
                        mask,
                        torch.zeros(B, pad_length, dtype=mask.dtype, device=mask.device)
                    ], dim=1)
            
            # 按 batch 分别提取可见的 patches
            x_vis_list = []
            for i in range(B):
                visible_patches = x[i][~mask[i]]  # [num_visible_i, C]
                x_vis_list.append(visible_patches)
            
            # 找到每个 batch 中可见的 patch 数量
            num_visible_per_batch = [(~mask[i]).sum().item() for i in range(B)]
            max_visible = max(num_visible_per_batch)
            
            # 如果每个 batch 的可见数量相同，直接 stack
            if len(set(num_visible_per_batch)) == 1:
                x = torch.stack(x_vis_list, dim=0)  # [B, num_visible, C]
            else:
                # 如果不同，需要 padding 到最大长度（理论上不应该发生，但为了安全）
                padded_x_vis = []
                for i, x_vis in enumerate(x_vis_list):
                    if x_vis.shape[0] < max_visible:
                        padding = torch.zeros(max_visible - x_vis.shape[0], C,
                                             dtype=x_vis.dtype, device=x_vis.device)
                        x_vis = torch.cat([x_vis, padding], dim=0)
                    padded_x_vis.append(x_vis)
                x = torch.stack(padded_x_vis, dim=0)  # [B, max_visible, C]

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16, # 空间 patch 大小 14
        num_classes=768,
        out_chans=2, # 输出通道 光流 2
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2, # 时间维度的 patch 数
        use_checkpoint=False,
    ):
        # print(f"Config: patch_size={patch_size}, out_chans={out_chans}, tubelet_size={tubelet_size}")
        # print(f"Required num_classes: {out_chans * tubelet_size * patch_size ** 2}")

        super().__init__()
        self.num_classes = num_classes
        # print(f"num_classes: {self.num_classes}")
        # print(f"out_chans: {out_chans}")
        # print(f"tubelet_size: {tubelet_size}")
        # print(f"patch_size: {patch_size}")

        # modeling_pretrain.py
        # # print(f"[DEBUG] 输入通道: {in_chans}")
        # print(f"[DEBUG] 时间分块: {tubelet_size}")
        # print(f"[DEBUG] 空间分块: {patch_size}x{patch_size}")
        # print(f"[DEBUG] 需要 decoder_num_classes = {in_chans * tubelet_size * patch_size ** 2}")

        print(
            f"Dimension Check: {out_chans}*{tubelet_size}*{patch_size}^2 = {out_chans * tubelet_size * patch_size ** 2}")
        print(f"decoder_num_classes = {num_classes}")

        assert num_classes == out_chans * tubelet_size * patch_size**2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        # img_size=224,
        img_size=(384, 224), # 384, 224
        h = 384,
        w = 224,
        patch_size=16, # 14
        encoder_in_chans=2,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  # decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        fc_drop_rate=0.5,
        use_mean_pooling=True,
        num_classes_action=204,
        num_frames=16,

    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
        ) # 一个基于 Vision Transformer 的 encoder，用于提取特征。

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            out_chans=encoder_in_chans,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        ) # 用于将特征解码回原始输入形式。

        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.fc_norm = norm_layer(encoder_embed_dim) if use_mean_pooling else None
        self.head_action = nn.Linear(encoder_embed_dim, num_classes_action) # 用于动作分类任务

        # encoder to decoder
        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        # Masked Autoencoder(MAE) 任务
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 位置编码
        self.pos_embed = None  # 不再预先生成 用于动态生成， 基于不同视频的长度

        # 这是被替代的
        # self.pos_embed = get_sinusoid_encoding_table(
        #     self.encoder.patch_embed.num_patches, decoder_embed_dim
        # )

        trunc_normal_(self.mask_token, std=0.02)

        # 元宝添加
        h, w = img_size
        self.grid_size = (h//patch_size, w//patch_size)
        # self.num_patches = self.grid_size[0] * self.grid_size[1]

        # self.temporal_length = img_size[2] // 2  # 假设输入尺寸包含时间维度
        self.temporal_length = num_frames // 2
        self.spatial_patches = self.grid_size[0] * self.grid_size[1]
        self.seq_length = self.temporal_length * self.spatial_patches
        
        # 打印模型初始化信息
        print(f"[MODEL INIT] img_size: {img_size}, patch_size: {patch_size}, num_frames: {num_frames}")
        print(f"[MODEL INIT] grid_size: {self.grid_size}, spatial_patches: {self.spatial_patches}, temporal_length: {self.temporal_length}")
        print(f"[MODEL INIT] seq_length: {self.seq_length}, encoder num_patches: {self.encoder.patch_embed.num_patches}")
        if hasattr(self.encoder, 'pos_embed') and self.encoder.pos_embed is not None:
            print(f"[MODEL INIT] encoder pos_embed shape: {self.encoder.pos_embed.shape}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask):
        B, _, T, H, W = x.shape
        # print("--------------", x.shape)

        # # add 使用 Sinusoidal 位置编码 动态生成pos_embed
        # if self.pos_embed is None or self.pos_embed.shape[1] != T:
        #     self.pos_embed = get_sinusoid_encoding_table(T, self.decoder.embed_dim).to(x.device)

        patch_size = 16
        # 计算时空分块后的序列长度
        t_chunks = T // 2
        h_chunks = H // patch_size
        w_chunks = W // patch_size
        seq_length = t_chunks * h_chunks * w_chunks

        # 动态生成位置编码
        if self.pos_embed is None or self.pos_embed.shape[1] != seq_length:
            self.pos_embed = get_sinusoid_encoding_table(seq_length, self.decoder.embed_dim).to(x.device)

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).clone().detach()

        # TODO 这个地方完全重新生成了，不知道会不会有影响
        # 重新生成一个形状与 expand_pos_embed 匹配的 mask
        # mask = torch.randint(0, 2, (4, 16, 512), dtype=torch.bool, device="cuda")

        x_vis = self.encoder(x, mask)  # [B, N_encoded, C_e]
        # N- 编码后的Token数量
        # C_e 编码后的特征维度e
        # classifier branch
        x = self.fc_norm(x_vis.mean(1))
        # x_vis.mean(1): 对所有 token 取均值，得到视频级特征
        # fc_norm: 归一化层。
        logits = self.head_action(self.fc_dropout(x))
        # head_action: 线性分类头，输出 num_classes_action=204 维的分类结果
        # decoder branch
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        # encoder_to_decoder 是一个线性层，将 encoder 输出 (C_e=768) 投影到 decoder 维度 (C_d=512)。
        B, N, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        # print('---pos_embed---', self.pos_embed.shape)
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        # 确保 mask 形状匹配
        if mask.shape[1] != expand_pos_embed.shape[1]:
            if mask.shape[1] > expand_pos_embed.shape[1]:
                mask = mask[:, :expand_pos_embed.shape[1]]
            else:
                pad_length = expand_pos_embed.shape[1] - mask.shape[1]
                mask = torch.cat([
                    mask,
                    torch.zeros(B, pad_length, dtype=mask.dtype, device=mask.device)
                ], dim=1)
        
        # 按 batch 分别提取位置编码
        pos_emd_vis_list = []
        pos_emd_mask_list = []
        for i in range(B):
            pos_emd_vis_i = expand_pos_embed[i][~mask[i]]  # [num_visible_i, C]
            pos_emd_mask_i = expand_pos_embed[i][mask[i]]  # [num_masked_i, C]
            pos_emd_vis_list.append(pos_emd_vis_i)
            pos_emd_mask_list.append(pos_emd_mask_i)
        
        # 检查每个 batch 的可见和 mask 数量
        num_visible_per_batch = [(~mask[i]).sum().item() for i in range(B)]
        num_masked_per_batch = [mask[i].sum().item() for i in range(B)]
        
        # 如果所有 batch 的数量相同，直接 stack
        if len(set(num_visible_per_batch)) == 1 and len(set(num_masked_per_batch)) == 1:
            pos_emd_vis = torch.stack(pos_emd_vis_list, dim=0)  # [B, num_visible, C]
            pos_emd_mask = torch.stack(pos_emd_mask_list, dim=0)  # [B, num_masked, C]
        else:
            # 如果不同，需要 padding（理论上不应该发生）
            max_visible = max(num_visible_per_batch)
            max_masked = max(num_masked_per_batch)
            padded_vis = []
            padded_mask = []
            for i in range(B):
                if pos_emd_vis_list[i].shape[0] < max_visible:
                    padding = torch.zeros(max_visible - pos_emd_vis_list[i].shape[0], C,
                                         dtype=pos_emd_vis_list[i].dtype, 
                                         device=pos_emd_vis_list[i].device)
                    pos_emd_vis_list[i] = torch.cat([pos_emd_vis_list[i], padding], dim=0)
                if pos_emd_mask_list[i].shape[0] < max_masked:
                    padding = torch.zeros(max_masked - pos_emd_mask_list[i].shape[0], C,
                                         dtype=pos_emd_mask_list[i].dtype,
                                         device=pos_emd_mask_list[i].device)
                    pos_emd_mask_list[i] = torch.cat([pos_emd_mask_list[i], padding], dim=0)
                padded_vis.append(pos_emd_vis_list[i])
                padded_mask.append(pos_emd_mask_list[i])
            pos_emd_vis = torch.stack(padded_vis, dim=0)  # [B, max_visible, C]
            pos_emd_mask = torch.stack(padded_mask, dim=0)  # [B, max_masked, C]
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x, logits


# 解决第一个问题， 添加当前使用类
def VideoMAE_ViT_B_1600(ckpt_pth=None, **kwargs):
    model = PretrainVisionTransformer(
        # img_size=(384, 224),  # 确保输入尺寸是16的倍数 上面init有指定了
        # patch_size=16,  # 明确指定
        # decoder_num_classes=1536,

        encoder_embed_dim=1024,  # 原值 768 编码器的嵌入维度
        encoder_depth=12, # 编码器层数(Transformer blocks 数量)
        encoder_num_heads=12, # 注意力头数
        encoder_num_classes=0, # 分类类别数（0 无分类头）
        decoder_embed_dim=512,  # 原值 384 解码器
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4, # mlp 扩展比例（隐藏层维度=embed_dim * mlp_ratio)
        qkv_bias=True, # 是否在QKV计算中使用偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # 归一化层类型
        **kwargs # 其他动态参数
    )
    model.default_cfg = _cfg()

    if ckpt_pth is not None:
        try:
            checkpoint = torch.load(ckpt_pth, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

            # 处理可能的 "module." 前缀问题
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v

            # **过滤掉维度不匹配的参数**
            model_dict = model.state_dict()
            filtered_state_dict = {
                k: v for k, v in new_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            # **只加载匹配的参数**
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict, strict=False)

            print(f"Checkpoint partially loaded from {ckpt_pth}")
            print(f"Loaded {len(filtered_state_dict)} / {len(new_state_dict)} parameters successfully.")

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    return model




