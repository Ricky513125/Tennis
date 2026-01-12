"""
查找预训练模型的 checkpoint 路径
用于多模态蒸馏配置
"""
import os
from pathlib import Path

def find_checkpoints(base_dir="./output"):
    """查找所有预训练模型和蒸馏模型的 checkpoint"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_path}")
        return
    
    checkpoints = {
        "rgb": [],
        "flow": [],
        "skeleton": [],
        "mmdistill": []  # 多模态蒸馏 checkpoint
    }
    
    # 遍历所有输出目录
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir():
            continue
        
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            
            # 检查是否是预训练目录
            if "pretrain_rgb" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                if ckpt_dir.exists():
                    # 查找 DeepSpeed checkpoint 目录（格式：epoch=XX-loss=X.XXXX）
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            # 检查是否有 checkpoint/mp_rank_00_model_states.pt
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["rgb"].append(str(ckpt_item))
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            # 标准 PyTorch Lightning checkpoint
                            checkpoints["rgb"].append(str(ckpt_item))
            
            elif "pretrain_flow" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                if ckpt_dir.exists():
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["flow"].append(str(ckpt_item))
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            checkpoints["flow"].append(str(ckpt_item))
            
            elif "pretrain_skeleton" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                if ckpt_dir.exists():
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["skeleton"].append(str(ckpt_item))
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            checkpoints["skeleton"].append(str(ckpt_item))
            
            # 检查是否是蒸馏训练目录
            elif "mmdistill" in time_dir.name.lower():
                ckpt_dir = time_dir / "checkpoints"
                if ckpt_dir.exists():
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            # PyTorch Lightning checkpoint 格式（.ckpt 文件）
                            checkpoints["mmdistill"].append(str(ckpt_item))
                        elif ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            # 也可能是 DeepSpeed 格式
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["mmdistill"].append(str(ckpt_item))
    
    # 打印结果
    print("=" * 80)
    print("找到的模型 Checkpoint:")
    print("=" * 80)
    
    for modality, ckpt_list in checkpoints.items():
        if modality == "mmdistill":
            print(f"\n多模态蒸馏 (MMDistill):")
        else:
            print(f"\n{modality.upper()} 模态:")
        
        if ckpt_list:
            # 按修改时间排序，最新的在前
            ckpt_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, ckpt_path in enumerate(ckpt_list[:5], 1):  # 只显示最新的5个
                # 判断格式
                ckpt_path_obj = Path(ckpt_path)
                if ckpt_path_obj.is_file() and ckpt_path_obj.suffix == ".ckpt":
                    format_type = "PyTorch Lightning (.ckpt)"
                elif ckpt_path_obj.is_dir():
                    format_type = "DeepSpeed (目录格式)"
                else:
                    format_type = "未知格式"
                print(f"  {i}. {ckpt_path}")
                print(f"     格式: {format_type}")
            if len(ckpt_list) > 5:
                print(f"  ... 还有 {len(ckpt_list) - 5} 个 checkpoint")
        else:
            print(f"  ❌ 未找到 checkpoint")
    
    # 生成配置建议
    print("\n" + "=" * 80)
    print("配置建议:")
    print("=" * 80)
    
    # 预训练模型配置（用于蒸馏训练）
    print("\n1. 预训练模型 Checkpoint (用于蒸馏训练):")
    print("   复制到 configs/trainer/mm_distill_trainer_tennis.yaml")
    print("   ckpt_path:")
    
    rgb_latest = checkpoints["rgb"][0] if checkpoints["rgb"] else "# 请替换为实际的 RGB checkpoint 路径"
    flow_latest = checkpoints["flow"][0] if checkpoints["flow"] else "# 请替换为实际的 Flow checkpoint 路径"
    skeleton_latest = checkpoints["skeleton"][0] if checkpoints["skeleton"] else "# 请替换为实际的 Skeleton checkpoint 路径"
    
    print(f"     - {rgb_latest}  # RGB checkpoint")
    print(f"     - {flow_latest}  # Flow checkpoint")
    print(f"     - {skeleton_latest}  # Skeleton checkpoint")
    
    # 蒸馏模型配置（用于评估）
    if checkpoints["mmdistill"]:
        print("\n2. 蒸馏模型 Checkpoint (用于评估):")
        print("   使用以下命令进行评估:")
        distill_latest = checkpoints["mmdistill"][0]
        print(f"   python3 lit_main_mmdistill.py \\")
        print(f"       --config-name=config_mmdistill_tennis \\")
        print(f"       train=False \\")
        print(f"       test=True \\")
        print(f"       ckpt_path={distill_latest}")
        
        print("\n   注意: 蒸馏训练的 checkpoint 是 PyTorch Lightning 格式 (.ckpt 文件)")
        print("   与预训练的 DeepSpeed 格式（目录）不同，这是正常的。")
    
    print("\n" + "=" * 80)
    print("格式说明:")
    print("=" * 80)
    print("• 预训练模型 (RGB/Flow/Skeleton): DeepSpeed 格式（目录）")
    print("  例如: epoch=49-loss=0.6095/checkpoint/mp_rank_00_model_states.pt")
    print("• 蒸馏模型 (MMDistill): PyTorch Lightning 格式（.ckpt 文件）")
    print("  例如: epoch=49-train_loss=0.0081.ckpt")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    find_checkpoints(base_dir)
