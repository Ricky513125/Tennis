# 标签到视频帧的映射关系

## 概述

**是的，可以根据一个动作标签找到对应的视频帧（实际上是视频片段）。**

## JSON 数据格式

Tennis 数据集的标注格式如下：

```json
{
  "video": "20220906-M-US_Open-QF-Casper_Ruud-Matteo_Berrettini_108415_108551",
  "far_name": "Matteo Berrettini",
  "near_name": "Casper Ruud",
  "events": [
    {
      "frame": 19,
      "label": "near_deuce_serve_-_-_W_-_unforced-err",
      "outcome": "unforced-err"
    },
    {
      "frame": 50,
      "label": "near_deuce_serve_-_-_B_-_in",
      "outcome": "in"
    }
  ]
}
```

**关键信息**：
- `video`: 视频ID
- `events`: 事件列表，每个事件包含：
  - `frame`: **关键帧号**（整数，例如 19, 50）
  - `label`: **动作标签**（字符串，例如 "near_deuce_serve_-_-_W_-_unforced-err"）
  - `outcome`: 结果（例如 "in", "unforced-err"）

## 数据加载逻辑

### 1. 构建映射关系（`_construct_loader`）

```python
for video_dict in data:
    video_id = video_dict.get("video")
    
    for event in video_dict["events"]:
        action_label = event["label"]      # 动作标签
        frame = event.get("frame", 1)      # 关键帧号
        
        # 存储映射关系
        self._video_id.append(video_id)           # 视频ID
        self._start_frame.append(frame)           # 关键帧号
        self._action_label.append(action_label)   # 动作标签
        self._action_idx.append(action_idx)       # 动作索引
```

**结果**：每个样本包含：
- `video_id`: 视频ID
- `start_frame`: 关键帧号
- `action_label`: 动作标签

### 2. 加载视频片段（`_get_input`）

```python
def _get_input(self, dir_to_img_frame, clip_start_frame):
    # 以关键帧为起点，按照采样率向前采样 16 帧
    frame_names = [
        max(1, clip_start_frame + self.cfg.target_sampling_rate * i)
        for i in range(self.num_frames)  # num_frames = 16
    ]
    
    # 加载这 16 帧
    frames = []
    for frame_name in frame_names:
        frame = self._get_frame(dir_to_img_frame, frame_name, self.mode, frames)
        frames.append(frame)
    
    return frames, mask
```

**关键点**：
- **不是单帧**，而是以关键帧为起点的 **16 帧视频片段**
- 从关键帧开始，按照 `target_sampling_rate` 向前采样
- 例如：关键帧 = 50，采样率 = 1，则加载帧 [50, 51, 52, ..., 65]

## 标签到视频的映射

### 正向映射：标签 → 视频片段

给定一个动作标签，可以找到所有对应的视频片段：

```python
# 示例：查找所有 "near_deuce_serve_-_-_W_-_unforced-err" 标签对应的视频片段

label = "near_deuce_serve_-_-_W_-_unforced-err"

# 方法1：从数据集中查找
matching_indices = [
    i for i, action_label in enumerate(dataset._action_label)
    if action_label == label
]

# 方法2：从 JSON 文件直接查找
import json
with open("train.json") as f:
    data = json.load(f)

matching_videos = []
for video_dict in data:
    for event in video_dict.get("events", []):
        if event.get("label") == label:
            matching_videos.append({
                "video_id": video_dict["video"],
                "key_frame": event["frame"],
                "label": event["label"]
            })
```

**结果**：每个匹配项包含：
- `video_id`: 视频ID（例如：`"20220906-M-US_Open-QF-..."`）
- `key_frame`: 关键帧号（例如：`19`）
- `label`: 动作标签

### 反向映射：视频片段 → 标签

给定一个视频ID和帧号，可以找到对应的标签：

```python
# 示例：查找视频 "video_001" 第 50 帧对应的标签

video_id = "20220906-M-US_Open-QF-Casper_Ruud-Matteo_Berrettini_108415_108551"
frame = 50

# 从 JSON 文件查找
import json
with open("train.json") as f:
    data = json.load(f)

for video_dict in data:
    if video_dict["video"] == video_id:
        for event in video_dict.get("events", []):
            if event.get("frame") == frame:
                label = event["label"]
                print(f"Frame {frame} in {video_id} has label: {label}")
                break
```

## 实际使用示例

### 示例1：查找特定标签的所有视频片段

```python
import json
from pathlib import Path

def find_videos_by_label(json_path, target_label):
    """根据标签查找所有对应的视频片段"""
    with open(json_path) as f:
        data = json.load(f)
    
    results = []
    for video_dict in data:
        video_id = video_dict["video"]
        for event in video_dict.get("events", []):
            if event.get("label") == target_label:
                results.append({
                    "video_id": video_id,
                    "key_frame": event["frame"],
                    "label": event["label"],
                    "outcome": event.get("outcome", "")
                })
    
    return results

# 使用示例
label = "near_deuce_serve_-_-_W_-_unforced-err"
videos = find_videos_by_label("train.json", label)

print(f"Found {len(videos)} video clips with label '{label}':")
for video in videos[:5]:  # 显示前5个
    print(f"  - Video: {video['video_id']}, Frame: {video['key_frame']}")
```

### 示例2：统计每个标签的视频片段数量

```python
import json
from collections import Counter

def count_videos_per_label(json_path):
    """统计每个标签对应的视频片段数量"""
    with open(json_path) as f:
        data = json.load(f)
    
    label_counts = Counter()
    for video_dict in data:
        for event in video_dict.get("events", []):
            label = event.get("label")
            if label:
                label_counts[label] += 1
    
    return label_counts

# 使用示例
counts = count_videos_per_label("train.json")
print(f"Total labels: {len(counts)}")
print(f"\nTop 10 labels by count:")
for label, count in counts.most_common(10):
    print(f"  {label}: {count} videos")
```

### 示例3：加载特定标签的视频片段

```python
from datamodule.dataset.tennis_fewshot_eval_dataset import TennisFewshotEvalDataset
import torch

# 假设已经创建了 dataset
# dataset = TennisFewshotEvalDataset(...)

# 查找特定标签的索引
target_label = "near_deuce_serve_-_-_W_-_unforced-err"
matching_indices = [
    i for i, label in enumerate(dataset._action_label)
    if label == target_label
]

# 加载第一个匹配的视频片段
if matching_indices:
    index = matching_indices[0]
    video_clip, mask = dataset[index]
    
    print(f"Video clip shape: {video_clip.shape}")  # [C, T, H, W] = [3, 16, 224, 384]
    print(f"Video ID: {dataset._video_id[index]}")
    print(f"Key frame: {dataset._start_frame[index]}")
    print(f"Label: {dataset._action_label[index]}")
```

## 关键理解点

### ✅ 可以做到：
1. **根据标签查找视频片段**：给定一个动作标签，可以找到所有对应的视频片段
2. **根据视频片段查找标签**：给定一个视频ID和关键帧号，可以找到对应的标签
3. **统计标签分布**：可以统计每个标签有多少个视频片段

### ⚠️ 注意事项：
1. **不是单帧，而是视频片段**：
   - 每个标签对应一个 **16 帧的视频片段**
   - 以关键帧为起点，按照采样率向前采样
   
2. **一个视频可以有多个标签**：
   - 一个视频可能有多个事件（events）
   - 每个事件对应一个关键帧和一个标签
   - 例如：视频 "video_001" 可能在第 19 帧有标签 A，在第 50 帧有标签 B

3. **一个标签可以出现在多个视频**：
   - 同一个动作标签可以出现在不同的视频中
   - 例如：标签 "near_deuce_serve_-_-_W_-_unforced-err" 可能出现在多个不同的视频中

## 数据结构总结

```
JSON 文件
  └── 视频列表
      └── 每个视频
          ├── video_id: "video_001"
          └── events: [
                {
                  frame: 19,                    # 关键帧号
                  label: "action_A",            # 动作标签
                  outcome: "in"
                },
                {
                  frame: 50,                    # 关键帧号
                  label: "action_B",            # 动作标签
                  outcome: "unforced-err"
                }
              ]

数据集
  └── 样本列表
      └── 每个样本
          ├── video_id: "video_001"
          ├── start_frame: 19                  # 关键帧号
          ├── action_label: "action_A"         # 动作标签
          └── frames: [16帧视频片段]            # 从关键帧开始的16帧
```

## 总结

**是的，可以根据一个标签找到对应的视频帧（实际上是视频片段）。**

- **映射关系**：`label` ↔ `(video_id, key_frame)`
- **视频片段**：以关键帧为起点，采样 16 帧
- **数据结构**：JSON 文件存储了完整的映射关系
- **使用方式**：可以通过遍历 JSON 文件或数据集来查找对应关系
