# Tennis Skeleton 数据处理完整流程

## 1. 数据加载阶段

### 1.1 文件路径
- **数据目录**: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis/`
- **文件格式**: `{video_id}.pkl` (例如: `20210220-W-Australian_Open-F-Naomi_Osaka-Jennifer_Brady_70873_71167.pkl`)
- **数据格式**: Python Pickle 文件，包含以下字段：
  - `keypoint`: 关键点坐标，形状为 `[M, N, K, 2]` (多人) 或 `[N, K, 2]` (单人)
    - `M`: 人数（多人情况）
    - `N`: 帧数
    - `K`: 关键点数量（17，COCO 格式）
    - `2`: (x, y) 坐标
  - `keypoint_score`: 关键点置信度，形状为 `[M, N, K]` 或 `[N, K]`（可选）
  - `total_frames`: 视频总帧数
  - `img_shape`: 原始图像尺寸，格式为 `(H, W)`，例如 `(720, 1280)`

### 1.2 数据加载过程 (`tennis_skeleton_dataset.py`)

```python
# 1. 从 PKL 文件加载 skeleton 数据
with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

keypoints = data['keypoint']  # [M, N, K, 2] 或 [N, K, 2]
keypoint_scores = data.get('keypoint_score', None)  # [M, N, K] 或 [N, K]
total_frames = data['total_frames']
img_shape = data.get('img_shape', (720, 1280))  # (H, W)

# 2. 提取指定帧的关键点
frame_idx = min(max(0, int(frame_name) - 1), total_frames - 1)

# 处理多人情况：取第一个人（person_idx=0）
if keypoints.ndim == 4:  # [M, N, K, 2] - 多人
    frame_kpts = keypoints[0, frame_idx]  # [K, 2]
    frame_scores = keypoint_scores[0, frame_idx] if keypoint_scores is not None else np.ones(K)
elif keypoints.ndim == 3:  # [N, K, 2] - 单人
    frame_kpts = keypoints[frame_idx]  # [K, 2]
    frame_scores = keypoint_scores[frame_idx] if keypoint_scores is not None else np.ones(K)

# 3. 归一化坐标到 [0, 1] 范围（基于原始图像尺寸）
H, W = img_shape  # (720, 1280)
frame_kpts_normalized = frame_kpts.copy().astype(np.float32)
frame_kpts_normalized[:, 0] = frame_kpts_normalized[:, 0] / W  # x 坐标归一化
frame_kpts_normalized[:, 1] = frame_kpts_normalized[:, 1] / H  # y 坐标归一化

# 4. 合并坐标和置信度：[K, 2] + [K] -> [K, 3]
frame_scores = frame_scores.astype(np.float32).reshape(-1, 1)
frame_kpts_with_score = np.concatenate([frame_kpts_normalized, frame_scores], axis=-1)
# 最终形状: [K=17, 3] (x_norm, y_norm, confidence)
```

### 1.3 关键点转换为热图

```python
# 将归一化的关键点转换为热图
# 输入: keypoints [K=17, 3] (x_norm, y_norm, confidence)
# 输出: heatmap [K=17, H=56, W=98]

# 注意：使用宽屏格式 (56×98) 以保持宽高比，匹配最终尺寸 [224, 384]
# 宽高比: 56/98 ≈ 0.571 ≈ 224/384 ≈ 0.583

K = 17  # COCO 格式的 17 个关键点
H, W = 56, 98  # 初始热图尺寸（保持宽高比）
sigma = 2.0  # 高斯核标准差

heatmap = np.zeros((K, H, W), dtype=np.float32)

for k in range(K):
    if confidences[k] > 0.1:  # 只处理置信度大于阈值的点
        x_center = x_coords[k] * W  # 转换为热图坐标
        y_center = y_coords[k] * H
        
        # 创建高斯分布
        y_grid, x_grid = np.ogrid[:H, :W]
        gaussian = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))
        gaussian = gaussian * confidences[k]  # 乘以置信度
        
        heatmap[k] = np.maximum(heatmap[k], gaussian)

# 最终形状: [K=17, H=56, W=98]
```

### 1.4 批次数据组装

```python
# 加载 16 帧 skeleton 数据
skeleton_frames = []
for frame_name in frame_names:
    skeleton_frame = _load_skeleton_from_pkl(pkl_path, frame_name)  # [K, 3]
    skeleton_frames.append(skeleton_frame)

# 转换为热图格式 [T, K, H, W]
heatmap_frames = []
for skeleton_frame in skeleton_frames:
    heatmap = _keypoints_to_heatmap(skeleton_frame, H=56, W=98)  # [K, H, W]
    heatmap_frames.append(heatmap)

# 堆叠成视频序列
skeleton_tensor = torch.from_numpy(np.stack(heatmap_frames, axis=0)).float()
# 形状: [T=16, K=17, H=56, W=98]

# 转换为 [T, H, W, K] 格式，因为 weak_aug 期望这个格式作为输入
skeleton_tensor = skeleton_tensor.permute(0, 2, 3, 1)  # [T, H, W, K] = [16, 56, 98, 17]

# 应用数据增强（resize 和归一化）
skeleton_tensor = self.transform.weak_aug(skeleton_tensor)
# weak_aug 输出是 [T, C, H, W] = [16, 17, 224, 384]

# 转换回 [T, H, W, C] 用于后续处理
skeleton_tensor = skeleton_tensor.permute(0, 2, 3, 1)  # [T, H, W, C] = [16, 224, 384, 17]
```

## 2. 数据预处理阶段

### 2.1 归一化参数
- **Mean**: `[0.005979, 0.006135, ..., 0.005492]` (17 个通道的均值，每个关键点一个通道)
- **Std**: `[0.047612, 0.048797, ..., 0.043690]` (17 个通道的标准差)

⚠️ **重要提示**: 这些参数应该根据实际数据计算！

**为什么需要计算？**
- 不同数据集的 skeleton 热图分布可能不同
- 正确的归一化参数可以提高训练稳定性和效果
- 这些参数应该基于训练数据的统计特性

**如何计算？**
使用提供的脚本 `calculate_skeleton_statistics.py`：
```bash
# 计算所有 skeleton 文件的统计信息
python3 calculate_skeleton_statistics.py \
    --skeleton-dir /mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --unlabel-json /mnt/ssd2/lingyu/Tennis/unlabel.json

# 只采样部分文件（更快）
python3 calculate_skeleton_statistics.py \
    --skeleton-dir /mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --unlabel-json /mnt/ssd2/lingyu/Tennis/unlabel.json \
    --sample 1000

# 保存结果到文件
python3 calculate_skeleton_statistics.py \
    --skeleton-dir /mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --unlabel-json /mnt/ssd2/lingyu/Tennis/unlabel.json \
    --output skeleton_stats.json
```

计算完成后，将结果更新到 `configs/data_module/modality/skeleton.yaml`：
```yaml
mean: [计算得到的均值1, 计算得到的均值2, ..., 计算得到的均值17]
std: [计算得到的标准差1, 计算得到的标准差2, ..., 计算得到的标准差17]
```

### 2.2 数据增强 (`DataAugmentationForUnlabelMM`)

数据增强在 `weak_aug` 方法中进行，处理流程如下：

```python
# 输入: [T, H, W, C] = [16, 56, 98, 17]
# 从 tennis_skeleton_dataset.py 传入

# 第一步: 转换为 [T, C, H, W]
x = x.permute(0, 3, 1, 2)  # [16, 17, 56, 98]

# 第二步: Resize 到目标尺寸 [224, 384]
# 使用双线性插值调整每帧的尺寸
# 对每一帧分别 resize
resized_frames = []
for t in range(T):
    frame = x[t:t+1]  # [1, 17, 56, 98]
    frame_resized = torch.nn.functional.interpolate(
        frame, 
        size=(224, 384), 
        mode='bilinear', 
        align_corners=False
    )  # [1, 17, 224, 384]
    resized_frames.append(frame_resized.squeeze(0))  # [17, 224, 384]
x = torch.stack(resized_frames, dim=0)  # [16, 17, 224, 384]

# 第三步: 归一化
# mean/std 形状: [1, C, 1, 1] = [1, 17, 1, 1]
# x 形状: [T, C, H, W] = [16, 17, 224, 384]
x = (x - mean) / std  # 广播归一化

# 最终输出: [T, C, H, W] = [16, 17, 224, 384]
```

## 3. 训练阶段数据处理

### 3.1 输入数据格式
```python
# 从 dataset 返回的数据
input = {
    "unlabel_frames": skeleton_tensor,  # [B, T, H, W, C] = [4, 16, 224, 384, 17]
    "mask": mask  # [B, seq_length] = [4, 2688]
}
```

### 3.2 转换为模型输入格式
```python
# 输入: [B, T, H, W, C] = [4, 16, 224, 384, 17]
unlabel_frames = input["unlabel_frames"]

# 转换为 [B, C, T, H, W] 格式用于模型处理
unlabel_frames = unlabel_frames.permute(0, 4, 1, 2, 3)
# 形状: [B, C, T, H, W] = [4, 17, 16, 224, 384]
```

### 3.3 反归一化（用于生成 Label）
```python
# 计算均值和标准差（用于反归一化）
mean = torch.as_tensor(cfg.data_module.modality.mean)[
    None, :, None, None, None
].type_as(unlabel_frames)  # [1, 17, 1, 1, 1]

std = torch.as_tensor(cfg.data_module.modality.std)[
    None, :, None, None, None
].type_as(unlabel_frames)  # [1, 17, 1, 1, 1]

# 反归一化（恢复到 [0, 1] 范围）
unnorm_videos = unlabel_frames * std + mean
# 形状: [4, 17, 16, 224, 384]
```

### 3.4 转换为 Patch Tokens
```python
# 将视频转换为 patch tokens
# unnorm_videos: [B, C, T, H, W] = [4, 17, 16, 224, 384]

# 使用 einops 进行 reshape
videos_patch = rearrange(
    unnorm_videos,
    "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
    p0=2,  # tubelet_size
    p1=16,  # patch_size
    p2=16,  # patch_size
)

# 最终形状: [B, num_patches, patch_dim]
# [4, 2688, 8704]
# 其中:
#   num_patches = (16 // 2) * (224 // 16) * (384 // 16) = 8 * 14 * 24 = 2688
#   patch_dim = 2 * 16 * 16 * 17 = 8704
#     (tubelet_size * patch_size^2 * in_chans)
```

### 3.5 提取被 Mask 的部分作为 Label
```python
# videos_patch: [4, 2688, 8704]
# bool_masked_pos: [4, 2688] (布尔掩码)

# 确保 mask 长度匹配
new_masked_pos = bool_masked_pos
if bool_masked_pos.shape[1] > videos_patch.shape[1]:
    new_masked_pos = bool_masked_pos[:, :videos_patch.shape[1]]
elif bool_masked_pos.shape[1] < videos_patch.shape[1]:
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
# 形状: [B, num_masked, patch_dim] = [4, 2016, 8704]
labels = torch.stack(labels_list, dim=0)
```

## 4. 模型前向传播

### 4.1 Encoder 阶段 (`PretrainVisionTransformerEncoder`)

#### 4.1.1 Patch Embedding
```python
# 输入: [B, C, T, H, W] = [4, 17, 16, 224, 384]

# PatchEmbed 使用 Conv3d 进行 patch 嵌入
proj = nn.Conv3d(
    in_channels=17,     # 输入通道数（17 个关键点）
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

#### 4.2.4 重建预测
```python
# 通过重建头
preds = self.decoder.head(x_full)
# 形状: [4, 2688, 8704]  (decoder_num_classes = 17 * 2 * 16^2 = 8704)

# 只保留被 mask 的部分
preds = preds[bool_masked_pos]
# 形状: [4, 2016, 8704]  (2016 = 2688 * 0.75, 75% 被 mask)
```

### 4.3 损失计算
```python
# 只计算重建损失（MSE Loss）
loss_mse = nn.MSELoss()
recon_loss = loss_mse(input=preds, target=labels)
# preds: [4, 2016, 8704]
# labels: [4, 2016, 8704]

loss = recon_loss
```

## 5. 数据维度变化总结

| 阶段 | 形状 | 说明 |
|------|------|------|
| **PKL 文件** | `[M, N, K, 2]` 或 `[N, K, 2]` | 原始关键点坐标 |
| **归一化后** | `[K=17, 3]` | (x_norm, y_norm, confidence) |
| **热图生成** | `[K=17, H=56, W=98]` | 初始热图（保持宽高比） |
| **堆叠 16 帧** | `[T=16, K=17, H=56, W=98]` | 视频序列 |
| **Permute** | `[T=16, H=56, W=98, K=17]` | 准备数据增强 |
| **Resize** | `[T=16, K=17, H=224, W=384]` | 调整到目标尺寸 |
| **归一化后** | `[T=16, K=17, H=224, W=384]` | 应用 mean/std |
| **Permute** | `[T=16, H=224, W=384, K=17]` | 准备 batch |
| **Batch 输入** | `[B=4, T=16, H=224, W=384, K=17]` | Batch size=4 |
| **转换为 [B, C, T, H, W]** | `[B=4, K=17, T=16, H=224, W=384]` | 用于模型处理 |
| **Patch Embedding** | `[B=4, 2688, 1024]` | 转换为 tokens |
| **Encoder 输出** | `[B=4, 672, 1024]` | 只保留可见 patches |
| **Decoder 输入** | `[B=4, 2688, 512]` | 添加 mask tokens |
| **Decoder 输出** | `[B=4, 2688, 512]` | Decoder 特征 |
| **重建预测** | `[B=4, 2016, 8704]` | 只保留被 mask 的部分 |
| **Label** | `[B=4, 2016, 8704]` | 目标重建值 |

## 6. 关键参数

- **输入尺寸**: `[224, 384]` (H × W) - 与 RGB/Flow 保持一致，确保位置对应
- **通道数**: `17` (COCO 格式的关键点数量)
- **时间帧数**: `16`
- **Patch 大小**: `16 × 16`
- **Tubelet 大小**: `2` (时间维度每 2 帧一个 patch)
- **Encoder 维度**: `1024`
- **Decoder 维度**: `512`
- **Patch 数量**: `2688 = 8 × 14 × 24` (时间 × 高度 × 宽度)
- **Patch 维度**: `8704 = 2 × 16 × 16 × 17` (tubelet × patch_size² × in_chans)
- **Mask 比例**: `75%` (2016 个 patches 被 mask)
- **训练轮数**: `50`
- **预热轮数**: `5`

## 7. 数据流图

```
PKL 文件 (keypoints: [M, N, K, 2])
    ↓ (加载指定帧)
[K=17, 2] (关键点坐标)
    ↓ (归一化到 [0, 1])
[K=17, 3] (x_norm, y_norm, confidence)
    ↓ (转换为热图)
[K=17, H=56, W=98] (初始热图，保持宽高比)
    ↓ (加载 16 帧)
[T=16, K=17, H=56, W=98]
    ↓ (permute to [T, H, W, K])
[T=16, H=56, W=98, K=17]
    ↓ (Resize to [224, 384])
[T=16, K=17, H=224, W=384]
    ↓ (归一化)
[T=16, K=17, H=224, W=384]
    ↓ (permute to [T, H, W, K])
[T=16, H=224, W=384, K=17]
    ↓ (Batch)
[B=4, T=16, H=224, W=384, K=17]
    ↓ (permute to [B, K, T, H, W])
[B=4, K=17, T=16, H=224, W=384]
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
[B=4, N_mask=2016, 8704]
    ↓ (MSE Loss)
loss (标量)
```

## 8. 重要注意事项

### 8.1 位置对应关系
- **Skeleton 热图使用 `[224, 384]` 尺寸**，与 RGB/Flow 保持一致
- 这确保了关键点位置与 RGB/Flow 图像中的位置正确对应
- 原始图像尺寸通常是 `720×1280`（宽高比 ≈ 1:1.78）
- 目标尺寸 `224×384`（宽高比 ≈ 1:1.71）基本匹配

### 8.2 缺失文件处理
- 如果某个视频的 PKL 文件不存在，会使用零骨架（全零关键点）作为回退
- 每个缺失文件只警告一次，避免日志过多
- 可以使用 `check_skeleton_coverage.py` 脚本统计 coverage

### 8.3 多人处理
- 如果 PKL 文件包含多人数据（`[M, N, K, 2]`），只取第一个人（`person_idx=0`）
- 这确保了数据的一致性

### 8.4 热图生成
- 初始热图尺寸为 `56×98`，保持宽高比
- 然后 resize 到 `224×384`，与 RGB/Flow 对齐
- 高斯核标准差 `sigma=2.0`
- 只处理置信度大于 `0.1` 的关键点

## 9. 相关文件

- **数据集类**: `datamodule/dataset/tennis_skeleton_dataset.py`
- **数据模块**: `datamodule/lit_tennis_skeleton_data_module.py`
- **数据增强**: `datamodule/utils/augmentation.py` (DataAugmentationForUnlabelMM)
- **训练脚本**: `lit_main_pretrain_tennis_skeleton.py`
- **配置文件**:
  - `configs/data_module/modality/skeleton.yaml`
  - `configs/trainer/modality/skeleton.yaml`
  - `configs/data_module/pretrain_data_module_skeleton.yaml`
  - `configs/trainer/pretrain_trainer_skeleton.yaml`
- **统计脚本**: `calculate_skeleton_statistics.py`
- **Coverage 检查**: `check_skeleton_coverage.py`
