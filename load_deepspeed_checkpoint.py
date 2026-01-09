"""
从 DeepSpeed checkpoint 中加载模型权重
DeepSpeed checkpoint 格式：目录包含 checkpoint/mp_rank_00_model_states.pt
"""
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_deepspeed_checkpoint(ckpt_path):
    """
    从 DeepSpeed checkpoint 目录加载模型权重
    
    Args:
        ckpt_path: DeepSpeed checkpoint 目录路径（例如：epoch=49-loss=0.6095）
                   或 checkpoint 文件路径（例如：epoch=49-loss=0.6095/checkpoint/mp_rank_00_model_states.pt）
    
    Returns:
        state_dict: 模型的状态字典
    """
    ckpt_path = Path(ckpt_path)
    
    # 如果是目录，查找 checkpoint 文件
    if ckpt_path.is_dir():
        # 尝试查找 checkpoint 子目录
        checkpoint_dir = ckpt_path / "checkpoint"
        if checkpoint_dir.exists():
            model_states_file = checkpoint_dir / "mp_rank_00_model_states.pt"
            if model_states_file.exists():
                logger.info(f"Loading DeepSpeed checkpoint from: {model_states_file}")
                state_dict = torch.load(model_states_file, map_location="cpu")
                
                # DeepSpeed checkpoint 格式：{"module": {...}}
                if "module" in state_dict:
                    return state_dict["module"]
                elif "model" in state_dict:
                    return state_dict["model"]
                else:
                    # 直接返回整个字典
                    return state_dict
            else:
                raise FileNotFoundError(f"Model states file not found: {model_states_file}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 如果是文件，直接加载
    elif ckpt_path.is_file():
        logger.info(f"Loading checkpoint file from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        if "module" in state_dict:
            return state_dict["module"]
        elif "model" in state_dict:
            return state_dict["model"]
        else:
            return state_dict
    
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")


def extract_encoder_weights(state_dict, prefix="model."):
    """
    从完整的状态字典中提取 encoder 权重
    
    Args:
        state_dict: 完整的状态字典
        prefix: 权重键的前缀（例如 "model." 或 "_forward_module.model."）
    
    Returns:
        encoder_dict: encoder 的权重字典
    """
    encoder_dict = {}
    
    for k, v in state_dict.items():
        # 查找 encoder 相关的权重
        if "encoder." in k:
            # 移除前缀
            new_key = k
            if prefix in new_key:
                new_key = new_key.replace(prefix, "")
            if "_forward_module." in new_key:
                new_key = new_key.replace("_forward_module.", "")
            
            # 只保留 encoder 部分
            if new_key.startswith("encoder."):
                encoder_dict[new_key] = v
    
    return encoder_dict


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_deepspeed_checkpoint.py <checkpoint_path>")
        print("Example: python load_deepspeed_checkpoint.py ./output/2026-01-09/15-29-49/pretrain_rgb/checkpoints/epoch=49-loss=0.6095")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    
    try:
        state_dict = load_deepspeed_checkpoint(ckpt_path)
        print(f"✅ Successfully loaded checkpoint from: {ckpt_path}")
        print(f"📊 Total keys: {len(state_dict)}")
        
        # 显示前几个键作为示例
        print("\n📋 Sample keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more keys")
        
        # 提取 encoder 权重
        encoder_dict = extract_encoder_weights(state_dict)
        print(f"\n🔧 Encoder keys: {len(encoder_dict)}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
