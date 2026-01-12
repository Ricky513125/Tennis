# Tennis RGB 数据处理完整流程

## 1. 数据加载阶段

### 1.1 文件路径
- **数据目录**: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224/{video_id}/`
- **文件格式**: `{frame_id:06d}.jpg` (例如: `000001.jpg`)
- **数据格式**: JPEG 图像，PIL Image 对象
- **原始尺寸**: `(398, 224)` (W × H) - 注意：宽度为 398，高度为 224

### 1.2 数据加载过程 (`tennis_unlabel_only_dataset.py`)

```python
# 1. 从 .jpg 文件加载 RGB 图像
frame = Image.open(path)  # PIL Image, mode='RGB'
# 维度: PIL Image 格式，实际像素为 [H, W, C] = [224, 398, 3] (原始尺寸)
# 注意：原始图片宽度为 398，会在后续处理中裁剪到 384

# 2. 加载 16 帧 RGB 图像，组成列表
unlabel_frames = [frame_1, frame_2, ..., frame_16]
# 每个 frame: PIL Image 对象，尺寸为 (398, 224)
```

### 1.3 数据增强处理

```python
# RGB 模式：PIL Image 列表
# weak_aug 期望 PIL Image 列表，返回 Tensor [T, C, H, W]
# 处理流程：
#   1. ToTensor: PIL Image 列表 -> [T, C, H, W] = [16, 3, 224, 398] (原始尺寸)
#   2. VideoCenterCrop: 居中裁剪 398 -> 384 -> [16, 3, 224, 384]
#   3. VideoRandomHorizontalFlip: 统一水平翻转 -> [16, 3, 224, 384]
#   4. 归一化: 应用 mean/std -> [16, 3, 224, 384]
unlabel_frames = self.transform.weak_aug(unlabel_frames)
# 输出: [T, C, H, W] = [16, 3, 224, 384]

# 转换为 [T, H, W, C] 格式用于后续处理
unlabel_frames = unlabel_frames.permute(0, 2, 3, 1)
# 输出: [T, H, W, C] = [16, 224, 384, 3]
```

**⚠️ 重要说明**：
- 原始 RGB 图片尺寸为 `224×398`，通过居中裁剪到 `224×384`
- 裁剪方式与 Flow 数据保持一致（Flow 也是从 `224×398` 居中裁剪到 `224×384`）
- 这确保了 RGB 和 Flow 在空间位置上的对应关系，对多模态训练至关重要

## 2. 数据预处理阶段

### 2.1 归一化参数
- **Mean**: `[0.485, 0.456, 0.406]` (三个通道的均值，ImageNet 默认值)
- **Std**: `[0.229, 0.224, 0.225]` (三个通道的标准差，ImageNet 默认值)

⚠️ **重要提示**: 这些参数目前使用的是 ImageNet 的默认值，**应该根据实际数据计算**！

**为什么需要计算？**
- ImageNet 的统计信息可能不适合 Tennis 数据集
- 不同数据集的颜色分布可能不同
- 正确的归一化参数可以提高训练稳定性和效果

**如何计算？**
使用提供的脚本 `calculate_rgb_statistics.py`：
```bash
# 计算所有 RGB 图像的统计信息
python3 calculate_rgb_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224

# 只采样部分文件（更快）
python3 calculate_rgb_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224 --sample 1000

# 保存结果到文件
python3 calculate_rgb_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224 --output rgb_stats.json
```

计算完成后，将结果更新到 `configs/data_module/modality/RGB.yaml`：
```yaml
mean: [计算得到的均值R, 计算得到的均值G, 计算得到的均值B]
std: [计算得到的标准差R, 计算得到的标准差G, 计算得到的标准差B]
```

### 2.2 数据增强 (`DataAugmentationForUnlabelRGB`)

数据增强在 `weak_aug` 方法中进行，处理流程如下：

```python
# 输入: PIL Image 列表 [img_1, img_2, ..., img_16]
# 注意：原始图片尺寸可能是 (398, 224) 或 (W, H)

# 第一步: ToTensor() - 将 PIL Image 列表转换为 Tensor
# ToTensor 内部处理：
#   - 对每个 PIL Image 调用 transforms.ToTensor() -> [C, H, W]
#   - Stack 所有帧 -> [T, C, H, W] = [16, 3, 224, 398] (原始尺寸)
video_tensor = ToTensor()(unlabel_frames)  # [T, C, H, W]

# 第二步: VideoCenterCrop - 对每帧分别应用居中裁剪
# 对每一帧：
#   - 从 [C, H, W] = [3, 224, 398] 居中裁剪到 [3, 224, 384]
#   - 计算裁剪起始位置: start_w = (398 - 384) // 2 = 7
#   - 裁剪: frame[:, :, 7:7+384] -> [3, 224, 384]
#   - Stack -> [T, C, H, W] = [16, 3, 224, 384]
video_tensor = VideoCenterCrop([224, 384])(video_tensor)

# 第三步: VideoRandomHorizontalFlip - 统一水平翻转
# 对所有帧使用相同的随机性（要么全部翻转，要么全部不翻转）
if torch.rand(1) < 0.5:
    video_tensor = torch.flip(video_tensor, dims=[3])  # 翻转宽度维度
# 输出: [T, C, H, W] = [16, 3, 224, 384]

# 第四步: 归一化
# mean/std 形状: [1, C, 1, 1] = [1, 3, 1, 1]
# video_tensor 形状: [T, C, H, W] = [16, 3, 224, 384]
video_tensor = (video_tensor - mean) / std  # 广播归一化

# 最终输出: [T, C, H, W] = [16, 3, 224, 384]
```

**关键实现细节**：
- ✅ `ToTensor` 类期望 PIL Image 列表，内部会 stack 成 `[T, C, H, W]`
- ✅ `VideoCenterCrop` 对每帧分别应用居中裁剪，与 Flow 处理方式一致，保持位置对应关系
- ✅ **居中裁剪的优势**：
  - 与 Flow 数据（也是居中裁剪）保持一致，确保 RGB 和 Flow 的位置对应
  - 避免随机裁剪导致的信息丢失和位置不对齐
  - 训练时每次看到相同的中心区域，提高训练稳定性
- ✅ `VideoRandomHorizontalFlip` 对所有帧统一翻转，保持时间一致性
- ✅ 归一化使用广播机制，`mean/std` 形状为 `[1, C, 1, 1]` 与 `[T, C, H, W]` 兼容

**⚠️ 重要说明**：
- 原始 RGB 图片尺寸为 `224×398`，通过居中裁剪到 `224×384`
- 裁剪方式与 Flow 数据保持一致（Flow 也是从 `224×398` 居中裁剪到 `224×384`）
- 这确保了 RGB 和 Flow 在空间位置上的对应关系，对多模态训练至关重要

### 2.3 输入到训练器的形状
- **Batch 输入**: `[B, T, H, W, C]` = `[4, 16, 224, 384, 3]`
  - `B = 4`: batch size
  - `T = 16`: 时间帧数
  - `H = 224`: 高度
  - `W = 384`: 宽度
  - `C = 3`: RGB 通道数

## 3. 训练步骤 (`training_step`)

### 3.1 维度调整和序列长度计算
```python
# 输入: [B, T, H, W, C] = [4, 16, 224, 384, 3]
unlabel_frames = input["unlabel_frames"]
bool_masked_pos = input["mask"].flatten(1).to(torch.bool)  # [B, seq_length]

# 获取输入维度
B, T, H, W, C = unlabel_frames.shape  # [4, 16, 224, 384, 3]

# 计算序列长度（patch 数量）
seq_length = (H // self.patch_size) * (W // self.patch_size) * (T // 2)
         = (224 // 16) * (384 // 16) * (16 // 2)
         = 14 * 24 * 8
         = 2688
```

### 3.2 生成 Mask
```python
# 如果 mask 长度不匹配，重新生成
if bool_masked_pos.shape[1] != seq_length:
    mask_ratio = 0.75
    num_masked_per_batch = int(seq_length * mask_ratio)  # 2016
    
    # 为每个 batch 独立生成随机 mask
    rand_indices = torch.rand(B, seq_length, device=unlabel_frames.device).argsort(dim=-1)
    bool_masked_pos = torch.zeros(B, seq_length, dtype=torch.bool, device=unlabel_frames.device)
    for i in range(B):
        bool_masked_pos[i, rand_indices[i, :num_masked_per_batch]] = True

# 形状: [B, seq_length] = [4, 2688]
# 每个样本随机 mask 2016 个 patch（75%）
```

### 3.3 反归一化（准备 label）
```python
# 转换为 [B, C, T, H, W] 格式
unlabel_frames = unlabel_frames.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]
# 形状: [4, 3, 16, 224, 384]

# 计算均值和标准差（用于反归一化）
mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
    None, :, None, None, None
].type_as(unlabel_frames)  # shape: [1, 3, 1, 1, 1]
std = torch.as_tensor(self.cfg.data_module.modality.std)[
    None, :, None, None, None
].type_as(unlabel_frames)  # shape: [1, 3, 1, 1, 1]

# 反归一化（恢复到原始范围 [0, 1]）
unnorm_videos = unlabel_frames * std + mean
# 形状: [4, 3, 16, 224, 384]
```

### 3.4 转换为 Patch Tokens
```python
# 使用 normalize_videos 方法
# 输入: [B, C, T, H, W] = [4, 3, 16, 224, 384]

# 如果不使用 normalize_target (RGB 模式通常使用):
videos_patch = rearrange(
    unnorm_videos,
    "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
    p0=2,      # tubelet_size: 时间维度每 2 帧一个 patch
    p1=16,     # patch_size: 空间高度
    p2=16,     # patch_size: 空间宽度
)

# 转换过程:
# [4, 3, 16, 224, 384]
# -> [4, 3, 8, 14, 24, 2, 16, 16]  (展开)
# -> [4, 8*14*24, 2*16*16*3]  (重组)
# -> [4, 2688, 1536]

# 最终形状: [B, num_patches, patch_dim]
# [4, 2688, 1536]
# 其中 patch_dim = 2 * 16 * 16 * 3 = 1536
#   (tubelet_size * patch_size^2 * in_chans)
```

### 3.5 提取被 Mask 的部分作为 Label
```python
# videos_patch: [B, num_patches, patch_dim] = [4, 2688, 1536]
# bool_masked_pos: [B, seq_length] = [4, 2688] (布尔掩码)

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

# 按 batch 分别提取被 mask 的 patches
labels_list = []
for i in range(B):
    masked_patches = videos_patch[i][new_masked_pos[i]]  # [num_masked_i, patch_dim]
    labels_list.append(masked_patches)

# 如果所有 batch 的 mask 数量相同，直接 stack
# 形状: [B, num_masked, patch_dim] = [4, 2016, 1536]
labels = torch.stack(labels_list, dim=0)
```

## 4. 模型前向传播

### 4.1 Encoder 阶段 (`PretrainVisionTransformerEncoder`)

#### 4.1.1 Patch Embedding
```python
# 输入: [B, C, T, H, W] = [4, 3, 16, 224, 384]

# PatchEmbed 使用 Conv3d 进行 patch 嵌入
proj = nn.Conv3d(
    in_channels=3,      # 输入通道数
    out_channels=1024,  # embed_dim (encoder_embed_dim)
    kernel_size=(2, 16, 16),  # (tubelet_size, patch_size, patch_size)
    stride=(2, 16, 16),
)

# 输出: [B, embed_dim, T', H', W']
# T' = 16 // 2 = 8
# H' = 224 // 16 = 14
# W' = 384 // 16 = 24

# Flatten 和 Transpose
x = proj(x).flatten(2).transpose(1, 2)
# 形状: [4, 2688, 1024]
# 2688 = 8 * 14 * 24 (时间 * 高度 * 宽度 patches)
```

#### 4.1.2 位置编码
```python
# 添加正弦位置编码
pos_embed = get_sinusoid_encoding_table(2688, 1024)
x = x + pos_embed  # [4, 2688, 1024]
```

#### 4.1.3 Mask 处理
```python
# 确保 mask 形状匹配
if mask.shape[1] != x.shape[1]:
    if mask.shape[1] > x.shape[1]:
        mask = mask[:, :x.shape[1]]
    else:
        pad_length = x.shape[1] - mask.shape[1]
        mask = torch.cat([
            mask,
            torch.zeros(B, pad_length, dtype=mask.dtype, device=mask.device)
        ], dim=1)

# 按 batch 分别提取可见的 patches
x_vis_list = []
for i in range(B):
    visible_patches = x[i][~mask[i]]  # [num_visible_i, C]
    x_vis_list.append(visible_patches)

# 如果所有 batch 的可见数量相同，直接 stack
# 形状: [B, num_visible, C] = [4, 672, 1024]  (672 = 2688 * 0.25, 25% 可见)
x_vis = torch.stack(x_vis_list, dim=0)
```

#### 4.1.4 Transformer Blocks
```python
# 通过 12 层 Transformer Encoder
for blk in self.blocks:  # 12 层
    x_vis = blk(x_vis)  # Self-Attention + MLP

# 输出: [4, 672, 1024]
x_vis = self.norm(x_vis)
```

### 4.2 Decoder 阶段 (`PretrainVisionTransformerDecoder`)

#### 4.2.1 Encoder 到 Decoder 投影
```python
# 投影到 decoder 维度
encoder_to_decoder = nn.Linear(1024, 512)
x_vis = encoder_to_decoder(x_vis)
# 形状: [4, 672, 512]  (embed_dim: 1024 -> 512)
```

#### 4.2.2 添加 Mask Tokens
```python
# 为被 mask 的位置添加可学习的 mask token
mask_token = nn.Parameter(torch.zeros(1, 1, 512))

# 确保 mask 形状匹配 expand_pos_embed
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

# Stack 位置编码
pos_emd_vis = torch.stack(pos_emd_vis_list, dim=0)  # [B, num_visible, C]
pos_emd_mask = torch.stack(pos_emd_mask_list, dim=0)  # [B, num_masked, C]

# 组合可见 tokens 和 mask tokens
x_full = torch.cat([
    x_vis + pos_emd_vis,      # 可见 patches + 位置编码
    self.mask_token + pos_emd_mask  # mask tokens + 位置编码
], dim=1)
# 形状: [4, 2688, 512]
```

#### 4.2.3 Decoder Transformer Blocks
```python
# 通过 4 层 Transformer Decoder
for blk in self.decoder.blocks:  # 4 层
    x_full = blk(x_full)

# 输出: [4, 2688, 512]
x_full = self.decoder.norm(x_full)
```

#### 4.2.4 重建头 (Reconstruction Head)
```python
# 线性层：将 decoder 输出投影到 patch 维度
head = nn.Linear(512, decoder_num_classes)  # decoder_num_classes = 1536
preds = head(x_full)  # [4, 2688, 1536]

# 只保留被 mask 的部分（在 training_step 中处理）
# 注意：preds 和 labels 的提取方式相同，都是按 batch 分别提取
```

## 5. 损失计算

### 5.1 重建损失 (Reconstruction Loss)
```python
# 预测值: preds [4, 2016, 1536]
# 目标值: labels [4, 2016, 1536]

loss_mse = nn.MSELoss()
recon_loss = loss_mse(preds, labels)
# 计算每个 patch 的像素级重建误差
```

### 5.2 分类损失 (Classification Loss) - 仅在有 source 数据时
```python
# 如果有 source 数据，还会计算分类损失
logits = model.classifier_head(x_vis)  # [4, num_classes]
ce_loss = nn.CrossEntropyLoss()(logits, action_label)
```

### 5.3 总损失
```python
# 对于 unlabel-only 模式:
loss = recon_loss

# 对于 source + unlabel 模式:
loss = recon_loss_source + lambda_ce * ce_loss
# lambda_ce = 0.05 (RGB 模式)
```

## 6. 反向传播

```python
# PyTorch Lightning 自动处理
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数
optimizer.zero_grad()  # 清零梯度
```

## 7. 维度变化总结

| 阶段 | 形状 | 说明 |
|------|------|------|
| **原始数据** | `[224, 398, 3]` | PIL Image [H, W, C] (原始尺寸) |
| **转换为 Tensor** | `[3, 224, 398]` | ToTensor() [C, H, W] |
| **PIL Image 列表** | `[16个 PIL Image]` | 时间序列 |
| **ToTensor** | `[16, 3, 224, 398]` | 转换为 Tensor [T, C, H, W] |
| **VideoCenterCrop** | `[16, 3, 224, 384]` | 每帧居中裁剪 (398→384) |
| **VideoRandomHorizontalFlip** | `[16, 3, 224, 384]` | 统一水平翻转 |
| **归一化** | `[16, 3, 224, 384]` | 广播归一化 |
| **转换为 [T, H, W, C]** | `[16, 224, 384, 3]` | 用于后续处理 |
| **Batch 输入** | `[4, 16, 224, 384, 3]` | Batch size=4 |
| **转换为 [B, C, T, H, W]** | `[4, 3, 16, 224, 384]` | 用于模型处理 |
| **Patch Embedding** | `[4, 2688, 1024]` | 转换为 tokens |
| **Encoder 输出** | `[4, 672, 1024]` | 只保留可见 patches |
| **Decoder 输入** | `[4, 2688, 512]` | 添加 mask tokens |
| **Decoder 输出** | `[4, 2688, 512]` | Decoder 特征 |
| **重建预测** | `[4, 2016, 1536]` | 只保留被 mask 的部分 |
| **Label** | `[4, 2016, 1536]` | 目标重建值 |

## 8. 关键参数

- **输入尺寸**: `[224, 384]` (H × W)
- **通道数**: `3` (RGB)
- **时间帧数**: `16`
- **Patch 大小**: `16 × 16`
- **Tubelet 大小**: `2` (时间维度每 2 帧一个 patch)
- **Encoder 维度**: `1024`
- **Decoder 维度**: `512`
- **Patch 数量**: `2688 = 8 × 14 × 24` (时间 × 高度 × 宽度)
- **Patch 维度**: `1536 = 2 × 16 × 16 × 3` (tubelet × patch_size² × in_chans)
- **Mask 比例**: `75%` (2016 个 patches 被 mask)
- **训练轮数**: `10` (临时调整)

## 9. 数据流图

```
.jpg 文件 (PIL Image: [H, W, C] = [224, 398, 3])
    ↓ (加载 16 帧)
[16个 PIL Image, 每个尺寸 (398, 224)]
    ↓ (ToTensor)
[T=16, C=3, H=224, W=398] (原始尺寸)
    ↓ (VideoCenterCrop: 居中裁剪 398→384)
[T=16, C=3, H=224, W=384] (目标尺寸)
    ↓ (VideoRandomHorizontalFlip)
[T=16, C=3, H=224, W=384]
    ↓ (归一化)
[T=16, C=3, H=224, W=384]
    ↓ (permute to [T, H, W, C])
[T=16, H=224, W=384, C=3]
    ↓ (Batch)
[B=4, T=16, H=224, W=384, C=3]
    ↓ (permute to [B, C, T, H, W])
[B=4, C=3, T=16, H=224, W=384]
    ↓ (Patch Embedding)
[B=4, N=2688, D=1024]
    ↓ (Mask, 保留 25%)
[B=4, N_vis=672, D=1024]
    ↓ (Encoder, 12 层)
[B=4, N_vis=672, D=1024]
    ↓ (投影到 Decoder)
[B=4, N_vis=672, D=512]
    ↓ (添加 Mask Tokens)
[B=4, N=2688, D=512]
    ↓ (Decoder, 4 层)
[B=4, N=2688, D=512]
    ↓ (重建头)
[B=4, N_mask=2016, 1536]
    ↓ (MSE Loss)
loss (标量)
```
