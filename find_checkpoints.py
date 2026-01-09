"""
查找预训练模型的 checkpoint 路径
用于多模态蒸馏配置
"""
import os
from pathlib import Path

def find_checkpoints(base_dir="./output"):
    """查找所有预训练模型的 checkpoint"""
    base_path = Path(base_dir).resolve()
    print(f"🔍 搜索目录: {base_path}")
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_path}")
        print(f"💡 提示: 请确保输出目录存在，或使用绝对路径")
        return
    
    checkpoints = {
        "rgb": [],
        "flow": [],
        "skeleton": []
    }
    
    found_dirs = []
    
    # 遍历所有输出目录
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir() or date_dir.name.startswith('.'):
            continue
        
        print(f"📁 检查日期目录: {date_dir.name}")
        
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir() or time_dir.name.startswith('.'):
                continue
            
            print(f"  📁 检查时间目录: {time_dir.name}")
            
            # 检查是否是预训练目录
            if "pretrain_rgb" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                print(f"    🔍 检查 RGB checkpoint 目录: {ckpt_dir}")
                if ckpt_dir.exists():
                    found_dirs.append(str(ckpt_dir))
                    # 查找 DeepSpeed checkpoint 目录（格式：epoch=XX-loss=X.XXXX）
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            # 检查是否有 checkpoint/mp_rank_00_model_states.pt
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["rgb"].append(str(ckpt_item))
                                print(f"      ✅ 找到 RGB checkpoint: {ckpt_item.name}")
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            # 标准 PyTorch Lightning checkpoint
                            checkpoints["rgb"].append(str(ckpt_item))
                            print(f"      ✅ 找到 RGB checkpoint 文件: {ckpt_item.name}")
                else:
                    print(f"      ❌ Checkpoint 目录不存在: {ckpt_dir}")
            
            elif "pretrain_flow" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                print(f"    🔍 检查 Flow checkpoint 目录: {ckpt_dir}")
                if ckpt_dir.exists():
                    found_dirs.append(str(ckpt_dir))
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["flow"].append(str(ckpt_item))
                                print(f"      ✅ 找到 Flow checkpoint: {ckpt_item.name}")
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            checkpoints["flow"].append(str(ckpt_item))
                            print(f"      ✅ 找到 Flow checkpoint 文件: {ckpt_item.name}")
                else:
                    print(f"      ❌ Checkpoint 目录不存在: {ckpt_dir}")
            
            elif "pretrain_skeleton" in time_dir.name:
                ckpt_dir = time_dir / "checkpoints"
                print(f"    🔍 检查 Skeleton checkpoint 目录: {ckpt_dir}")
                if ckpt_dir.exists():
                    found_dirs.append(str(ckpt_dir))
                    for ckpt_item in ckpt_dir.iterdir():
                        if ckpt_item.is_dir() and "epoch=" in ckpt_item.name:
                            model_states = ckpt_item / "checkpoint" / "mp_rank_00_model_states.pt"
                            if model_states.exists():
                                checkpoints["skeleton"].append(str(ckpt_item))
                                print(f"      ✅ 找到 Skeleton checkpoint: {ckpt_item.name}")
                        elif ckpt_item.is_file() and ckpt_item.suffix == ".ckpt":
                            checkpoints["skeleton"].append(str(ckpt_item))
                            print(f"      ✅ 找到 Skeleton checkpoint 文件: {ckpt_item.name}")
                else:
                    print(f"      ❌ Checkpoint 目录不存在: {ckpt_dir}")
    
    if not found_dirs:
        print(f"\n⚠️  未找到任何预训练目录")
        print(f"💡 提示:")
        print(f"   1. 确保已经运行过预训练脚本")
        print(f"   2. 检查输出目录路径是否正确")
        print(f"   3. 尝试使用绝对路径: python3 find_checkpoints.py /mnt/ssd2/lingyu/Tennis/output")
    
    # 打印结果
    print("=" * 80)
    print("找到的预训练模型 Checkpoint:")
    print("=" * 80)
    
    for modality, ckpt_list in checkpoints.items():
        print(f"\n{modality.upper()} 模态:")
        if ckpt_list:
            # 按修改时间排序，最新的在前
            ckpt_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, ckpt_path in enumerate(ckpt_list[:5], 1):  # 只显示最新的5个
                print(f"  {i}. {ckpt_path}")
            if len(ckpt_list) > 5:
                print(f"  ... 还有 {len(ckpt_list) - 5} 个 checkpoint")
        else:
            print(f"  ❌ 未找到 checkpoint")
    
    # 生成配置建议
    print("\n" + "=" * 80)
    print("配置建议 (复制到 configs/trainer/mm_distill_trainer_tennis.yaml):")
    print("=" * 80)
    print("ckpt_path:")
    
    rgb_latest = checkpoints["rgb"][0] if checkpoints["rgb"] else "# 请替换为实际的 RGB checkpoint 路径"
    flow_latest = checkpoints["flow"][0] if checkpoints["flow"] else "# 请替换为实际的 Flow checkpoint 路径"
    skeleton_latest = checkpoints["skeleton"][0] if checkpoints["skeleton"] else "# 请替换为实际的 Skeleton checkpoint 路径"
    
    print(f"  - {rgb_latest}  # RGB checkpoint")
    print(f"  - {flow_latest}  # Flow checkpoint")
    print(f"  - {skeleton_latest}  # Skeleton checkpoint")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # 尝试常见的输出目录
        possible_dirs = [
            "./output",
            "../output",
            "/mnt/ssd2/lingyu/Tennis/output",
            os.path.expanduser("~/Tennis/output"),
        ]
        base_dir = None
        for dir_path in possible_dirs:
            if Path(dir_path).exists():
                base_dir = dir_path
                break
        
        if base_dir is None:
            base_dir = "./output"
            print(f"⚠️  使用默认目录: {base_dir}")
            print(f"💡 如果找不到 checkpoint，请指定输出目录:")
            print(f"   python3 find_checkpoints.py <output_directory>")
    
    find_checkpoints(base_dir)
