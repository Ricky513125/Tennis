# Tennis Flow 数据处理完整流程

## 1. 数据加载阶段

### 1.1 文件路径
- **数据目录**: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows/{video_id}/`
- **文件格式**: `pair_{frame_id:05d}.npy` (例如: `pair_00041.npy`)
- **数据格式**: NumPy 数组，形状为 `[C, H, W] = [2, 224, 398]`，其中：
  - `C = 2` (光流的两个通道：x 和 y 方向的运动)
  - `H = 224` (高度)
  - `W = 398` (原始宽度，会被裁剪到 384)

### 1.2 数据加载过程 (`tennis_dataset.py`)

```python
# 1. 从 .npy 文件加载光流数据
frame = np.load(path)  # shape: [2, 224, 398] (实际维度是 [C, H, W])

# 2. 宽度裁剪：从 398 裁剪到 384 (居中裁剪)
C, H, original_width = frame.shape  # [2, 224, 398]
start_x = (398 - 384) // 2 = 7
cropped_flow = frame[:, :, 7:7+384]  # [2, 224, 384]
# 注意：维度是 [C, H, W]，所以裁剪的是最后一个维度（宽度）

# 3. 转换为 Tensor（已经是 [C, H, W] 格式，不需要 permute）
flow_tensor = torch.from_numpy(cropped_flow).float()
# 最终形状: [C=2, H=224, W=384]
```

### 1.3 批次数据组装

```python
# 加载 16 帧光流数据
frames = [flow_tensor_1, flow_tensor_2, ..., flow_tensor_16]
# 每个 flow_tensor: [2, 224, 384]

# 堆叠成视频序列
video_frames = torch.stack(frames, dim=0)  # [T=16, C=2, H=224, W=384]

# 应用数据增强（归一化等）
video_frames = video_frames.permute(0, 2, 3, 1)  # [T, H, W, C]
video_frames = transform.weak_aug(video_frames)  # 归一化
# 输出形状: [T=16, H=224, W=384, C=2]
```

## 2. 数据预处理阶段

### 2.1 归一化参数
- **Mean**: `[0.0507, 0.4671]` (两个通道的均值)
- **Std**: `[10.9280, 8.6857]` (两个通道的标准差)

⚠️ **重要提示**: 这些参数目前是硬编码在配置文件中的，**应该根据实际数据计算**！

**为什么需要计算？**
- 不同数据集的光流数据分布可能不同
- 正确的归一化参数可以提高训练稳定性和效果
- 这些参数应该基于训练数据的统计特性

**如何计算？**
使用提供的脚本 `calculate_flow_statistics.py`：
```bash
# 计算所有 flow 文件的统计信息
python3 calculate_flow_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows

# 只采样部分文件（更快）
python3 calculate_flow_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows --sample 1000

# 保存结果到文件
python3 calculate_flow_statistics.py --input /mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows --output flow_stats.json
```

计算完成后，将结果更新到 `configs/data_module/modality/flow.yaml`：
```yaml
mean: [计算得到的均值1, 计算得到的均值2]
std: [计算得到的标准差1, 计算得到的标准差2]
```

### 2.2 数据增强 (`DataAugmentationForUnlabelMM`)
```python
transform = transforms.Compose([
    ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=self.mean, std=self.std)  # 从配置读取
])
```

### 2.3 输入到训练器的形状
- **Batch 输入**: `[B, T, H, W, C]` = `[4, 16, 224, 384, 2]`
  - `B = 4`: batch size
  - `T = 16`: 时间帧数
  - `H = 224`: 高度
  - `W = 384`: 宽度
  - `C = 2`: 光流通道数

## 3. 训练步骤 (`training_step`)

### 3.1 维度调整
```python
# 输入: [B, T, H, W, C] = [4, 16, 224, 384, 2]
unlabel_frames = input["unlabel_frames"]

# 第一次 permute: [B, T, H, W, C] -> [B, H, T, W, C]
unlabel_frames = unlabel_frames.permute(0, 3, 1, 4, 2)
# 形状: [4, 224, 16, 384, 2]

# 计算序列长度
B, H, T, W, C = unlabel_frames.shape  # [4, 224, 16, 384, 2]
seq_length = (H // 16) * (W // 16) * (T // 2)
         = (224 // 16) * (384 // 16) * (16 // 2)
         = 14 * 24 * 8
         = 2688
```

### 3.2 生成 Mask
```python
mask_ratio = 0.75
num_masked_per_batch = int(2688 * 0.75) = 2016

# 随机选择要 mask 的位置
bool_masked_pos = torch.zeros(B, seq_length, dtype=torch.bool)
# 形状: [4, 2688]
# 每个样本随机 mask 2016 个 patch
```

### 3.3 反归一化（准备 label）
```python
# 第二次 permute: [B, H, T, W, C] -> [B, C, T, H, W]
unlabel_frames = unlabel_frames.permute(0, 4, 2, 1, 3)
# 形状: [4, 2, 16, 224, 384]

# 反归一化（恢复到原始范围）
mean = [0.0507, 0.4671]  # shape: [1, 2, 1, 1, 1]
std = [10.9280, 8.6857]  # shape: [1, 2, 1, 1, 1]
unnorm_videos = unlabel_frames * std + mean
# 形状: [4, 2, 16, 224, 384]
```

### 3.4 转换为 Patch Tokens
```python
# 使用 normalize_videos 方法
# 输入: [B, C, T, H, W] = [4, 2, 16, 224, 384]

# 如果不使用 normalize_target (flow 模式):
videos_patch = rearrange(
    unnorm_videos,
    "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
    p0=2,      # tubelet_size: 时间维度每 2 帧一个 patch
    p1=16,     # patch_size: 空间高度
    p2=16,     # patch_size: 空间宽度
)

# 转换过程:
# [4, 2, 16, 224, 384]
# -> [4, 2, 8, 14, 24, 2, 16, 16]  (展开)
# -> [4, 8*14*24, 2*16*16*2]  (重组)
# -> [4, 2688, 1024]

# 最终形状: [B, num_patches, patch_dim]
# [4, 2688, 1024]
# 其中 patch_dim = 2 * 2 * 16 * 16 = 1024
#   (tubelet_size * in_chans * patch_size^2)
```

### 3.5 提取被 Mask 的部分作为 Label
```python
# videos_patch: [4, 2688, 1024]
# bool_masked_pos: [4, 2688] (布尔掩码)

labels = videos_patch[bool_masked_pos].reshape(B, num_masked_per_batch, patch_dim)
# 形状: [4, 2016, 1024]
# 只保留被 mask 的 patch 作为重建目标
```

## 4. 模型前向传播

### 4.1 Encoder 阶段 (`PretrainVisionTransformerEncoder`)

#### 4.1.1 Patch Embedding
```python
# 输入: [B, C, T, H, W] = [4, 2, 16, 224, 384]

# PatchEmbed 使用 Conv3d 进行 patch 嵌入
proj = nn.Conv3d(
    in_channels=2,      # 输入通道数
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
# 只保留可见的 (unmasked) patches
x_vis = x[~bool_masked_pos].reshape(B, -1, C)
# 形状: [4, 672, 1024]  (672 = 2688 * 0.25, 25% 可见)
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

# 组合可见 tokens 和 mask tokens
x_full = torch.cat([
    x_vis + pos_emd_vis,      # 可见 patches + 位置编码
    mask_token + pos_emd_mask  # mask tokens + 位置编码
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
head = nn.Linear(512, decoder_num_classes)
preds = head(x_full)  # [4, 2688, 1024]

# 只保留被 mask 的部分
preds = preds[bool_masked_pos].reshape(B, num_masked_per_batch, patch_dim)
# 形状: [4, 2016, 1024]
```

## 5. 损失计算

### 5.1 重建损失 (Reconstruction Loss)
```python
# 预测值: preds [4, 2016, 1024]
# 目标值: labels [4, 2016, 1024]

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
# lambda_ce = 0.01 (flow 模式)
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
| **原始数据** | `[2, 224, 398]` | 从 .npy 文件加载 [C, H, W] |
| **裁剪后** | `[2, 224, 384]` | 宽度裁剪（最后一个维度） |
| **转换为 Tensor** | `[2, 224, 384]` | 已经是 [C, H, W] 格式，直接转换 |
| **堆叠 16 帧** | `[16, 2, 224, 384]` | 时间序列 |
| **数据增强后** | `[16, 224, 384, 2]` | 归一化 |
| **Batch 输入** | `[4, 16, 224, 384, 2]` | Batch size=4 |
| **第一次 permute** | `[4, 224, 16, 384, 2]` | 调整维度顺序 |
| **第二次 permute** | `[4, 2, 16, 224, 384]` | [B, C, T, H, W] |
| **Patch Embedding** | `[4, 2688, 1024]` | 转换为 tokens |
| **Encoder 输出** | `[4, 672, 1024]` | 只保留可见 patches |
| **Decoder 输入** | `[4, 2688, 512]` | 添加 mask tokens |
| **Decoder 输出** | `[4, 2688, 512]` | Decoder 特征 |
| **重建预测** | `[4, 2016, 1024]` | 只保留被 mask 的部分 |
| **Label** | `[4, 2016, 1024]` | 目标重建值 |

## 8. 关键参数

- **输入尺寸**: `[224, 384]` (H × W)
- **通道数**: `2` (光流的 x 和 y 方向)
- **时间帧数**: `16`
- **Patch 大小**: `16 × 16`
- **Tubelet 大小**: `2` (时间维度每 2 帧一个 patch)
- **Encoder 维度**: `1024`
- **Decoder 维度**: `512`
- **Patch 数量**: `2688 = 8 × 14 × 24` (时间 × 高度 × 宽度)
- **Patch 维度**: `1024 = 2 × 2 × 16 × 16` (in_chans × tubelet × patch_size²)
- **Mask 比例**: `75%` (2016 个 patches 被 mask)

## 9. 数据流图

```
.npy 文件 [224, 398, 2]
    ↓ (裁剪)
[224, 384, 2]
    ↓ (permute)
[2, 224, 384]
    ↓ (堆叠 16 帧)
[16, 2, 224, 384]
    ↓ (归一化)
[16, 224, 384, 2]
    ↓ (Batch)
[B=4, T=16, H=224, W=384, C=2]
    ↓ (permute)
[B=4, C=2, T=16, H=224, W=384]
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
[B=4, N_mask=2016, 1024]
    ↓ (MSE Loss)
loss (标量)
```
