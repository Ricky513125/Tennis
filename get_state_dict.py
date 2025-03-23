import torch

ckpt_path = "/mnt/ssd2/lingyu/Tennis/ckpt/VideoMAE_ViT_B_1600.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")

# 获取所有 keys
keys = list(state_dict.keys())

# 写入文件
output_path = "state_dict_keys.txt"
with open(output_path, "w") as f:
    for key in keys:
        f.write(key + "\n")

print(f"Keys 已保存到 {output_path}")
