# Few-Shot 评估流程详解（5-way 1-shot）

## 概述

Few-shot learning 是一种元学习（meta-learning）方法，用于评估模型在**只有少量样本**的情况下学习新任务的能力。

**核心思想**：给定少量带标签的样本（Support Set），模型需要快速学习并识别新的未标记样本（Query Set）。

## 5-way 1-shot 的含义

- **5-way**：每个 episode 包含 **5 个不同的动作类别**
- **1-shot**：每个动作类别只有 **1 个支持样本**（用于学习）
- **15 query**：每个动作类别有 **15 个查询样本**（用于测试）

## 完整评估流程

### 1. Episode 构建阶段（`EpisodicBatchSampler`）

```python
# 从数据集中随机选择 5 个不同的动作类别
action_ids = [action_1, action_2, action_3, action_4, action_5]

# 对于每个动作类别，选择 k_shot + q_sample = 1 + 15 = 16 个视频样本
for action_id in action_ids:
    # 从该动作类别的所有视频中随机选择 16 个
    samples = select_random_samples(action_id, count=16)
    # 前 1 个作为 Support Set
    # 后 15 个作为 Query Set
```

**示例**：
- 动作类别 A：选择 16 个视频（1 个 support + 15 个 query）
- 动作类别 B：选择 16 个视频（1 个 support + 15 个 query）
- 动作类别 C：选择 16 个视频（1 个 support + 15 个 query）
- 动作类别 D：选择 16 个视频（1 个 support + 15 个 query）
- 动作类别 E：选择 16 个视频（1 个 support + 15 个 query）

**总共**：`5 × 16 = 80` 个视频样本

### 2. 数据组织（`preprocess_frames_dynamic`）

```python
# 输入：80 个视频，形状 [80, C, T, H, W]
# 重新组织为 [n_way, k_shot+q_sample, C, T, H, W] = [5, 16, C, T, H, W]

frames = rearrange(
    frames, 
    "(n m) c t h w -> n m c t h w", 
    n=5,      # 5 个动作类别
    m=16      # 每个类别 16 个样本
)

# 分离 Support Set 和 Query Set
support_frames = frames[:, :1]      # [5, 1, C, T, H, W] = 5 个视频
query_frames = frames[:, 1:]        # [5, 15, C, T, H, W] = 75 个视频
```

**关键点**：
- **不是预测视频序列**，而是**识别动作类别**
- 每个视频是一个完整的 16 帧视频片段，代表一个动作实例
- 所有视频都来自不同的视频文件，但可能属于同一个动作类别

### 3. 特征提取（`LR_ensemble`）

```python
# 使用预训练模型提取特征
support_features = model(support_frames, support_mask)[0]  # [5, feature_dim]
query_features = model(query_frames, query_mask)[0]       # [75, feature_dim]
```

**每个视频**：
- 输入：`[C=3, T=16, H=224, W=384]` - 16 帧 RGB 视频
- 输出：`[feature_dim]` - 一个特征向量（例如 1024 维）

**关键点**：
- 每个视频被编码为一个特征向量
- 这些特征向量包含了该视频的动作信息

### 4. 分类器训练（Logistic Regression）

```python
# 使用 Support Set 训练分类器
# Support Set: 5 个视频（每个动作类别 1 个）
# Support Labels: [0, 1, 2, 3, 4]（5 个动作类别的标签）

clf = LogisticRegression()
clf.fit(support_features, support_labels)
```

**训练数据**：
- **5 个样本**（每个动作类别 1 个）
- **5 个标签**（对应 5 个动作类别）

**关键点**：
- 这是 few-shot learning 的核心：只用 **5 个样本**训练分类器
- 分类器需要从这 5 个样本中学习区分 5 个动作类别

### 5. 预测和评估

```python
# 使用训练好的分类器预测 Query Set
query_predictions = clf.predict(query_features)  # [75] - 75 个预测结果

# 计算准确率
accuracy = accuracy_score(query_predictions, query_labels)
```

**Query Set**：
- **75 个视频**（每个动作类别 15 个）
- **75 个真实标签**（用于评估）

**评估指标**：
- Top-1 准确率：预测正确的查询样本比例

## 完整示例

假设我们有以下动作类别：

```
动作类别 A: "far_ad_stroke_fh_gs_DL_-_in"
动作类别 B: "near_deuce_stroke_bh_volley_CC_-_in"
动作类别 C: "far_deuce_stroke_fh_drop_CC_-_winner"
动作类别 D: "near_ad_stroke_fh_smash_II_-_forced-err"
动作类别 E: "near_middle_stroke_bh_volley_CC_-_unforced-err"
```

### Episode 1：

**Support Set**（5 个视频，用于学习）：
1. 动作 A 的 1 个视频（例如：`video_001` 的第 100 帧开始的 16 帧）
2. 动作 B 的 1 个视频（例如：`video_045` 的第 200 帧开始的 16 帧）
3. 动作 C 的 1 个视频（例如：`video_123` 的第 50 帧开始的 16 帧）
4. 动作 D 的 1 个视频（例如：`video_234` 的第 300 帧开始的 16 帧）
5. 动作 E 的 1 个视频（例如：`video_567` 的第 150 帧开始的 16 帧）

**Query Set**（75 个视频，用于测试）：
- 动作 A 的 15 个视频（来自不同的视频文件或不同的帧）
- 动作 B 的 15 个视频
- 动作 C 的 15 个视频
- 动作 D 的 15 个视频
- 动作 E 的 15 个视频

**评估过程**：
1. 模型提取 Support Set 的 5 个特征向量
2. 使用这 5 个特征向量训练 Logistic Regression 分类器
3. 模型提取 Query Set 的 75 个特征向量
4. 使用分类器预测这 75 个视频的动作类别
5. 计算预测准确率（例如：75 个中预测对了 60 个，准确率 = 80%）

### Episode 2：

**重新随机选择** 5 个不同的动作类别，重复上述过程。

**总共运行 400 个 episodes**，最后计算平均准确率。

## 关键理解点

### ❌ 不是这样的：
- **不是**预测视频序列（预测下一帧）
- **不是**把所有视频聚到一起，学第一张的特征，然后判断后面的

### ✅ 实际是这样的：
1. **识别动作类别**：给定一个视频片段，判断它属于哪个动作类别
2. **Few-shot 学习**：只用 5 个样本（每个类别 1 个）学习区分 5 个动作类别
3. **测试泛化能力**：用训练好的分类器预测 75 个新视频的动作类别
4. **评估模型能力**：准确率越高，说明模型从少量样本中学习的能力越强

## 数据流图

```
Episode 构建
    ↓
[5 个动作类别，每个类别 16 个视频]
    ↓
分离 Support Set 和 Query Set
    ↓
Support Set: [5 个视频] (每个类别 1 个)
Query Set: [75 个视频] (每个类别 15 个)
    ↓
特征提取（使用预训练模型）
    ↓
Support Features: [5, feature_dim]
Query Features: [75, feature_dim]
    ↓
训练 Logistic Regression（只用 5 个样本）
    ↓
预测 Query Set 的动作类别
    ↓
计算准确率
```

## 为什么使用 Few-shot Learning？

1. **模拟真实场景**：在实际应用中，我们往往只有少量标注样本
2. **评估泛化能力**：测试模型是否能从少量样本中快速学习
3. **元学习评估**：评估模型是否学会了"如何学习"（learning to learn）

## 配置参数

```yaml
n_way: 5        # 每个 episode 包含 5 个动作类别
k_shot: 1       # 每个类别只有 1 个支持样本
q_sample: 15    # 每个类别有 15 个查询样本
episodes: 400   # 运行 400 个 episodes，计算平均准确率
```

**Batch Size**：`n_way × (k_shot + q_sample) = 5 × (1 + 15) = 80`

## 总结

**5-way 1-shot 评估**：
- 每个 episode 随机选择 5 个动作类别
- 每个类别提供 1 个支持样本（用于学习）
- 每个类别提供 15 个查询样本（用于测试）
- 使用 Logistic Regression 从 5 个样本中学习分类器
- 用分类器预测 75 个查询样本的动作类别
- 计算准确率，评估模型的 few-shot 学习能力

**核心**：不是预测视频序列，而是**识别动作类别**，并且只用**极少的样本**（5 个）学习分类。
