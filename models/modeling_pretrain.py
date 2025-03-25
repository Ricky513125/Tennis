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


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
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
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        print(f"x.shape: {x.shape}")  # (B, N, C)
        print(f"pos_embed.shape: {self.pos_embed.shape}")  # (1, ?, C)
        # 修正 pos_embed 维度
        B, N, C = x.shape  # 获取 x 的 token 数 N
        pos_embed = self.pos_embed[:, :N, :].to(x.device)  # 取前 N 个位置编码
        x = x + pos_embed  # 现在 x.shape 和 pos_embed.shape 匹配

        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape

        if mask is not None:
            mask = mask[:, :N]  # 保证 mask 维度匹配 x.shape[1]
            x = x[~mask].reshape(B, -1, C)  # ~mask means visible

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
        patch_size=16,
        num_classes=768,
        out_chans=3,
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
        tubelet_size=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
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
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  # decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=4,
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
        pretrained=False,
        all_frames=16,
        with_cp=True
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
        )

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
        )

        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.fc_norm = norm_layer(encoder_embed_dim) if use_mean_pooling else None
        self.head_action = nn.Linear(encoder_embed_dim, num_classes_action)

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)
        # add
        # self.config = self.model(hidden_size=384)

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
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_encoded, C_e]
        # classifier branch
        x = self.fc_norm(x_vis.mean(1))
        logits = self.head_action(self.fc_dropout(x))
        # decoder branch
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        # add
        print(f"x_vis shape: {x_vis.shape}, pos_emd_vis shape: {pos_emd_vis.shape}")
        print(f"mask_token shape: {self.mask_token.shape}, pos_emd_mask shape: {pos_emd_mask.shape}")

        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x, logits



# @register_model
# def VideoMAE_ViT_B_1600(ckpt_pth=None, **kwargs):
#     model = PretrainVisionTransformer(
#         # img_size=_cfg["inputsize"][1],
#         # patch_size=16,
#         encoder_embed_dim=1024, # 768
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_num_classes=0,
#         # decoder_num_classes=1536,
#         decoder_embed_dim=512, # 384
#         decoder_depth=4,
#         decoder_num_heads=6,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     model.default_cfg = _cfg()
#     if ckpt_pth is not None:
#         p = torch.load(ckpt_pth)
#         if "state_dict" in p:
#             state_dict = p["state_dict"]
#         elif "model" in p:
#             state_dict = p["model"]  # 有些预训练模型存的是 "model" 而不是 "state_dict"
#         else:
#             raise KeyError(f"Invalid checkpoint format: {p.keys()}")
#         model.load_state_dict(state_dict)
#     return model

def VideoMAE_ViT_B_1600(ckpt_pth=None, **kwargs):
    model = PretrainVisionTransformer(
        encoder_embed_dim=1024,  # 原值 768
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=512,  # 原值 384
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
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

