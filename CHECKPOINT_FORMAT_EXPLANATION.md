# Checkpoint 格式说明

## 格式差异

### 1. 预训练模型（RGB/Flow/Skeleton）- DeepSpeed 格式

**格式**: 目录结构
```
epoch=49-loss=0.6095/
  └── checkpoint/
      └── mp_rank_00_model_states.pt
```

**特点**:
- 使用 DeepSpeed 训练时自动保存
- 目录名格式：`epoch=XX-loss=X.XXXX`
- 模型权重在 `checkpoint/mp_rank_00_model_states.pt` 文件中

**示例路径**:
```
/mnt/ssd2/lingyu/Tennis/output/2026-01-09/15-29-49/pretrain_rgb/checkpoints/epoch=49-loss=0.6095
```

### 2. 蒸馏模型（MMDistill）- PyTorch Lightning 格式

**格式**: `.ckpt` 文件
```
epoch=49-train_loss=0.0081.ckpt
last.ckpt
```

**特点**:
- 使用 PyTorch Lightning 的 `ModelCheckpoint` callback 保存
- 即使使用 DeepSpeed strategy，PyTorch Lightning 也会保存为标准格式
- 文件名格式：`epoch=XX-train_loss=X.XXXX.ckpt` 或 `last.ckpt`

**示例路径**:
```
/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/epoch=49-train_loss=0.0081.ckpt
/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/last.ckpt
```

## 为什么格式不同？

1. **预训练阶段**:
   - 使用 DeepSpeed 的原生训练循环
   - 直接调用 `model.save_checkpoint()` 保存为 DeepSpeed 格式

2. **蒸馏阶段**:
   - 使用 PyTorch Lightning 框架
   - 通过 `ModelCheckpoint` callback 自动保存
   - PyTorch Lightning 统一保存为标准 `.ckpt` 格式，即使使用 DeepSpeed strategy

## 如何使用

### 查找 Checkpoint

运行脚本查找所有 checkpoint：

```bash
python3 find_checkpoints.py /mnt/ssd2/lingyu/Tennis/output
```

### 评估蒸馏模型

使用蒸馏训练的 checkpoint 进行评估：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/path/to/epoch=49-train_loss=0.0081.ckpt
```

### 加载预训练模型

预训练模型的 checkpoint 在蒸馏训练时已经加载（在配置文件中指定）：

```yaml
# configs/trainer/mm_distill_trainer_tennis.yaml
ckpt_path: [
  /path/to/pretrain_rgb/checkpoints/epoch=49-loss=0.6095,  # DeepSpeed 格式
  /path/to/pretrain_flow/checkpoints/epoch=49-loss=0.9503,
  /path/to/pretrain_skeleton/checkpoints/epoch=49-loss=0.0141,
]
```

## Checkpoint 内容

### DeepSpeed 格式（预训练）
```python
{
    "module": {
        "encoder.patch_embed.proj.weight": ...,
        "encoder.blocks.0.norm1.weight": ...,
        ...
    }
}
```

### PyTorch Lightning 格式（蒸馏）
```python
{
    "state_dict": {
        "student_rgb.encoder.patch_embed.proj.weight": ...,
        "teacher_flow.encoder.blocks.0.norm1.weight": ...,
        "cmt.mlp_to_rgb.0.weight": ...,
        ...
    },
    "epoch": 49,
    "global_step": 28300,
    "lr_schedulers": [...],
    "optimizer_states": [...],
    ...
}
```

## 注意事项

1. **两种格式都是有效的**，只是保存方式不同
2. **评估时使用 `.ckpt` 文件**（PyTorch Lightning 格式）
3. **预训练模型加载时**，代码会自动识别 DeepSpeed 格式并正确加载
4. **不需要手动转换格式**，代码已经处理了格式兼容性

## 常见问题

### Q: 为什么蒸馏训练的 checkpoint 不是 DeepSpeed 格式？

A: 这是 PyTorch Lightning 的设计。即使使用 DeepSpeed strategy，`ModelCheckpoint` callback 也会保存为标准 Lightning 格式，这样可以：
- 统一 checkpoint 格式
- 便于在不同环境下加载
- 包含完整的训练状态（optimizer, scheduler 等）

### Q: 可以手动保存为 DeepSpeed 格式吗？

A: 可以，但不推荐。如果需要，可以在 `on_train_epoch_end` 中手动调用 DeepSpeed 的保存方法，但会失去 PyTorch Lightning 的便利性。

### Q: 如何确认 checkpoint 格式？

A: 
- DeepSpeed 格式：是目录，包含 `checkpoint/mp_rank_00_model_states.pt`
- Lightning 格式：是 `.ckpt` 文件
