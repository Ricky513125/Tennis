# 多模态蒸馏模型评估指南

## 评估命令

### 基本评估命令

使用训练好的 checkpoint 进行评估：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/path/to/your/checkpoint
```

### 完整示例

假设你的 checkpoint 保存在：
```
/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/epoch=49-train_loss=0.1234.ckpt
```

评估命令：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/epoch=49-train_loss=0.1234.ckpt
```

### DeepSpeed Checkpoint 格式

如果使用 DeepSpeed，checkpoint 是目录格式，例如：
```
/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/epoch=49-train_loss=0.1234/
```

评估时直接使用目录路径：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/epoch=49-train_loss=0.1234
```

### 使用 last.ckpt

如果保存了最后一个 epoch 的 checkpoint（`last.ckpt`）：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/mnt/ssd2/lingyu/Tennis/output/2026-01-10/12-30-45/mmdistill_tennis/checkpoints/last.ckpt
```

## 评估配置

### Few-shot 评估参数

在 `configs/data_module/mm_distill_data_module_tennis.yaml` 中配置：

```yaml
n_way: 5        # N-way: 每个 episode 包含 N 个类别
k_shot: 1       # K-shot: 每个类别有 K 个支持样本
q_sample: 15    # Q-sample: 每个类别有 Q 个查询样本
episodes: 400   # Episodes: 评估的 episode 数量
```

### 修改评估参数

可以通过命令行覆盖配置：

```bash
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False \
    test=True \
    ckpt_path=/path/to/checkpoint \
    data_module.n_way=5 \
    data_module.k_shot=1 \
    data_module.q_sample=15 \
    data_module.episodes=400
```

## 评估输出

评估完成后，会输出以下指标：

- `top1_action_ensemble`: 平均 Top-1 准确率
- `top1_action_ensemble_std`: 标准差
- `top1_action_ensemble_std_error`: 标准误差

## 注意事项

1. **Checkpoint 格式**：
   - 标准 PyTorch Lightning checkpoint: `.ckpt` 文件
   - DeepSpeed checkpoint: 目录（包含 `checkpoint/mp_rank_00_model_states.pt`）

2. **GPU 配置**：
   - 确保 `configs/config_mmdistill_tennis.yaml` 中的 `devices` 配置正确
   - 单 GPU: `devices: [0]`
   - 多 GPU: `devices: [0, 1, 2, 3]`

3. **数据路径**：
   - 确保 `fewshot_eval_json_path` 指向正确的评估数据 JSON 文件
   - 默认使用 `train.json` 进行评估

4. **Few-shot 配置检查**：
   - 运行时会自动检查每个 action 类别的样本数是否足够
   - 如果样本不足，会给出警告和建议

## 常见问题

### Q: 如何找到训练好的 checkpoint？

A: 使用 `find_checkpoints.py` 脚本：

```bash
python3 find_checkpoints.py /mnt/ssd2/lingyu/Tennis/output
```

### Q: 评估时出现 batch size 不匹配错误？

A: 检查 few-shot 配置：
- `batch_size = n_way * (k_shot + q_sample)`
- 当前配置：`5 * (1 + 15) = 80`
- 确保验证集样本数足够

### Q: 如何评估不同的 few-shot 设置？

A: 通过命令行参数修改：

```bash
# 5-way 5-shot
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False test=True \
    ckpt_path=/path/to/checkpoint \
    data_module.k_shot=5 \
    data_module.episodes=250

# 5-way 1-shot
python3 lit_main_mmdistill.py \
    --config-name=config_mmdistill_tennis \
    train=False test=True \
    ckpt_path=/path/to/checkpoint \
    data_module.k_shot=1 \
    data_module.episodes=400
```
