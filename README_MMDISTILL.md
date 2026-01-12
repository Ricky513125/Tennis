# 多模态知识蒸馏 (Multi-Modal Knowledge Distillation)

## 📋 目录

- [概述](#概述)
- [架构说明](#架构说明)
- [数据准备](#数据准备)
- [训练步骤](#训练步骤)
- [评估方法](#评估方法)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

## 概述

多模态知识蒸馏是一种将多个预训练模态（RGB、Flow、Skeleton）的知识融合到单一 RGB 学生模型中的方法。通过跨模态转换模块（CMT），学生模型可以学习到来自不同模态的丰富特征表示。

### 核心思想

- **学生模型（Student）**: RGB 模态的 VideoMAE 模型，作为最终使用的模型
- **教师模型（Teachers）**: 三个预训练的模态模型（RGB、Flow、Skeleton），提供知识指导
- **跨模态转换（CMT）**: 将 RGB 特征转换为其他模态的特征空间，实现知识传递

### 优势

- ✅ **多模态融合**: 利用 RGB、Flow、Skeleton 三种模态的互补信息
- ✅ **无标签训练**: 仅使用无标签数据进行蒸馏，无需额外标注
- ✅ **知识传递**: 通过特征蒸馏，将多模态知识融合到单一 RGB 模型中
- ✅ **Few-shot 评估**: 支持 N-way K-shot 的少样本评估

## 架构说明

### 模型架构

```
输入数据:
├── RGB: [B, 3, T, H, W]     (学生模型输入)
├── Flow: [B, 2, T, H, W]    (教师模型输入)
└── Skeleton: [B, 17, T, H, W] (教师模型输入)

前向传播:
1. Student RGB → RGB Features
2. Teacher RGB → RGB Features (冻结)
3. Teacher Flow → Flow Features (冻结)
4. Teacher Skeleton → Skeleton Features (冻结)
5. CMT: RGB Features → [RGB', Flow', Skeleton'] Features

损失计算:
├── Loss_RGB = MSE(CMT_RGB', Teacher_RGB)
├── Loss_Flow = MSE(CMT_Flow', Teacher_Flow)
└── Loss_Skeleton = MSE(CMT_Skeleton', Teacher_Skeleton)

总损失 = Loss_RGB + Loss_Flow + Loss_Skeleton
```

### 关键组件

1. **Student RGB Model**: 可训练的 RGB VideoMAE 模型
2. **Teacher Models**: 冻结的 RGB、Flow、Skeleton 预训练模型
3. **CrossModalTranslate (CMT)**: 跨模态转换模块
   - `mlp_to_rgb`: RGB → RGB 特征转换
   - `mlp_to_flow`: RGB → Flow 特征转换
   - `mlp_to_skeleton`: RGB → Skeleton 特征转换

## 数据准备

### 1. 预训练模型检查点

确保已训练好三个模态的预训练模型：

```bash
# RGB 预训练模型
/mnt/ssd2/lingyu/Tennis/output/.../pretrain_rgb/checkpoints/epoch=49-loss=0.6095

# Flow 预训练模型
/mnt/ssd2/lingyu/Tennis/output/.../pretrain_flow/checkpoints/epoch=49-loss=0.9503

# Skeleton 预训练模型
/mnt/ssd2/lingyu/Tennis/output/.../pretrain_skeleton/checkpoints/epoch=49-loss=0.0141
```

### 2. 数据文件

确保以下文件存在：

- `unlabel.json`: 无标签训练数据列表
- `train.json`: Few-shot 评估数据（包含标签）
- RGB 帧: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/vid_frames_224/{video_id}/`
- Flow 数据: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/tennis_flows/{video_id}/`
- Skeleton 数据: `/mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis/{video_id}.pkl`

### 3. 数据格式要求

- **RGB**: JPEG 图像，尺寸 `224×398`（会自动裁剪到 `224×384`）
- **Flow**: NumPy 数组，形状 `[2, 224, 398]`（会自动裁剪到 `[2, 224, 384]`）
- **Skeleton**: PKL 文件，包含关键点数据（会转换为热图 `[17, 224, 384]`）

## 训练步骤

### 1. 配置检查点路径

编辑 `configs/trainer/mm_distill_trainer_tennis.yaml`:

```yaml
ckpt_path: [
  /path/to/pretrain_rgb/checkpoints/epoch=49-loss=0.6095,    # RGB checkpoint
  /path/to/pretrain_flow/checkpoints/epoch=49-loss=0.9503,   # Flow checkpoint
  /path/to/pretrain_skeleton/checkpoints/epoch=49-loss=0.0141, # Skeleton checkpoint
]
```

⚠️ **注意**: 
- DeepSpeed checkpoint 格式：路径指向**目录**（例如：`epoch=49-loss=0.6095`），不是文件
- 代码会自动查找目录内的 `checkpoint/mp_rank_00_model_states.pt`

### 2. 开始训练

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=True \
    test=False
```

### 3. 训练参数

主要训练参数（在 `configs/trainer/mm_distill_trainer_tennis.yaml` 中配置）：

- `epochs`: 50（训练轮数）
- `lr`: 2e-3（学习率）
- `warmup_epochs`: 5（预热轮数）
- `batch_size`: 4（批次大小，在 `configs/config_mmdistill_tennis.yaml` 中配置）
- `save_ckpt_freq`: 5（每 5 个 epoch 保存一次 checkpoint）

### 4. 训练输出

训练日志和 checkpoint 保存在：

```
./output/YYYY-MM-DD/HH-MM-SS/mmdistill_tennis/
├── checkpoints/
│   ├── last.ckpt                    # 最后一个 epoch 的 checkpoint
│   ├── epoch=04-train_loss=0.XXXX.ckpt
│   ├── epoch=09-train_loss=0.XXXX.ckpt
│   └── ...
└── tensorboard_logs/               # TensorBoard 日志
```

### 5. 训练监控

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir ./output/YYYY-MM-DD/HH-MM-SS/mmdistill_tennis/tensorboard_logs
```

监控指标：
- `train_loss`: 总损失
- `train_trans_loss_rgb`: RGB 转换损失
- `train_trans_loss_flow`: Flow 转换损失
- `train_trans_loss_skeleton`: Skeleton 转换损失
- `lr`: 学习率

## 评估方法

### Few-shot 评估

使用 N-way K-shot 的少样本评估方法：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/path/to/checkpoint/last.ckpt
```

### 评估参数配置

在 `configs/data_module/mm_distill_data_module_tennis.yaml` 中配置：

```yaml
n_way: 5        # N-way: 每个 episode 包含 N 个类别
k_shot: 1       # K-shot: 每个类别有 K 个支持样本
q_sample: 15    # 每个类别有 Q 个查询样本
episodes: 400   # 评估的 episode 数量
```

**注意**:
- `batch_size = n_way * (k_shot + q_sample) = 5 * (1 + 15) = 80`
- 每个 action 类别至少需要 `(k_shot + q_sample) = 16` 个样本
- 如果改为 `k_shot=5`，需要将 `episodes` 调整为 250

### 评估输出

评估结果包括：

- `top1_action_ensemble`: 平均准确率
- `top1_action_ensemble_std`: 标准差
- `top1_action_ensemble_std_error`: 标准误差

### 动态 Batch 处理

评估时会自动处理不完整的 batch：

- 如果 batch size 不匹配，会动态调整 `n_way`
- 如果 batch size 太小（< `k_shot + q_sample`），会跳过该 batch
- 如果 support set 只有一个类别，会使用默认预测

## 配置说明

### 主配置文件

`configs/config_mmdistill_tennis.yaml`:

```yaml
defaults:
  - trainer: mm_distill_trainer_tennis
  - data_module: mm_distill_data_module_tennis

train: True
test: False
batch_size: 4
num_workers: 2
check_val_every_n_epoch: 999  # 禁用验证
```

### 数据模块配置

`configs/data_module/mm_distill_data_module_tennis.yaml`:

- **模态参数**: `mode`, `mean`, `std`, `input_size`, `patch_size`
- **数据路径**: `target_data_dir`, `unlabel_json_path`, `fewshot_eval_json_path`, `skeleton_dir`
- **Few-shot 参数**: `n_way`, `k_shot`, `q_sample`, `episodes`

### 训练器配置

`configs/trainer/mm_distill_trainer_tennis.yaml`:

- **模型参数**: `model`, `ckpt_path`, `in_chans`, `encoder_embed_dim`
- **优化器参数**: `epochs`, `lr`, `warmup_epochs`, `weight_decay`

## 常见问题

### 1. Checkpoint 加载失败

**问题**: `FileNotFoundError: No such file or directory: .../checkpoint/mp_rank_00_model_states.pt`

**解决**: 
- 确保 `ckpt_path` 指向**目录**而不是文件
- 检查 checkpoint 目录结构是否正确
- 对于 PyTorch Lightning checkpoint（`.ckpt` 文件），直接使用文件路径

### 2. 位置编码维度不匹配

**问题**: `RuntimeError: The size of tensor a (1568) must match the size of tensor b (2688)`

**解决**: 
- 代码已自动处理 `pos_embed` 维度不匹配问题
- 如果仍有问题，检查 `input_size` 配置是否正确

### 3. Batch Size 不匹配

**问题**: `ValueError: Batch size mismatch in preprocess_frames: expected 80, got 6`

**解决**: 
- 评估时会自动处理不完整的 batch
- 如果 batch 太小，会自动跳过
- 可以调整 `episodes` 参数来适应数据集大小

### 4. 训练损失不下降

**可能原因**:
- 学习率过大或过小
- 预训练模型 checkpoint 路径错误
- 数据加载问题

**解决**:
- 检查学习率设置（默认 `2e-3`）
- 验证 checkpoint 是否正确加载
- 检查数据路径和格式

### 5. 内存不足

**解决**:
- 减小 `batch_size`（默认 4）
- 减少 `num_workers`（默认 2）
- 使用梯度累积（需要修改代码）

## 文件结构

```
.
├── lit_main_mmdistill.py              # 主训练/评估脚本
├── models/
│   ├── lit_MMDistillTrainer.py        # 多模态蒸馏训练器
│   └── cmt.py                          # 跨模态转换模块
├── datamodule/
│   ├── lit_tennis_unlabel_combined_mm_data_module.py  # 数据模块
│   └── dataset/
│       └── tennis_unlabel_combined_multimodal_dataset.py  # 数据集
├── configs/
│   ├── config_mmdistill_tennis.yaml   # 主配置
│   ├── trainer/
│   │   └── mm_distill_trainer_tennis.yaml  # 训练器配置
│   └── data_module/
│       └── mm_distill_data_module_tennis.yaml  # 数据模块配置
└── README_MMDISTILL.md                 # 本文档
```

## 引用

如果使用本代码，请引用相关论文：

```bibtex
@article{your_paper,
  title={Multi-Modal Knowledge Distillation for Video Understanding},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

[添加许可证信息]

## 联系方式

如有问题，请提交 Issue 或联系维护者。
