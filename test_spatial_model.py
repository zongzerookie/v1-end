import torch
import numpy as np
import random
import sys
import os
from transformers import RobertaForMultipleChoice

# 导入我们的新模型
try:
    from aux_methods import SpatiallyAwareISAAQ, ResnetRobertaBUTD

    print("成功导入 aux_methods 中的模型...")
except ImportError as e:
    print(f"FATAL: 导入 aux_methods 失败: {e}")
    sys.exit()

# --- 配置区 ---
# 确保这个列表与 tqa_dq_ensembler.py 中的列表完全一致
MODEL_PATHS_LIST = [
    "checkpoints/tmc_dq_roberta_IR_e4.pth",
    "checkpoints/tmc_dq_roberta_NSP_e4.pth",
    "checkpoints/tmc_dq_roberta_NN_e4.pth",
    "checkpoints/dmc_dq_roberta_SPATIAL_IR_e3.pth",
    "checkpoints/dmc_dq_roberta_SPATIAL_NSP_e4.pth",
    "checkpoints/dmc_dq_roberta_SPATIAL_NN_e4.pth"
]
MODEL_TYPES = ["tmc", "tmc", "tmc", "dmc", "dmc", "dmc"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "./checkpoints/roberta-large"


# --- 结束配置 ---

def main():
    print("--- 完整 Ensembler 加载测试脚本 ---")
    print(f"使用设备: {DEVICE}")

    # 检查 Tokenizer 路径
    if not os.path.exists(TOKENIZER_PATH):
        print(f"错误: 找不到 Tokenizer 路径 {TOKENIZER_PATH}")
        return

    models = []

    # -----------------------------------------------------------------
    # 模拟 tqa_dq_ensembler.py 的加载循环
    # -----------------------------------------------------------------
    for path, model_type in zip(MODEL_PATHS_LIST, MODEL_TYPES):
        print("\n" + "=" * 30)
        print(f"正在测试加载: {path} (类型: {model_type})")

        if not os.path.exists(path):
            print(f"FATAL: 权重文件未找到！跳过。")
            continue

        try:
            # 1. 统一加载文件
            print("  执行 torch.load()...")
            loaded_data = torch.load(path, map_location='cpu')

            if model_type == "tmc":
                # 2a. Tmc 模型：加载的数据 *就是* 模型对象
                if isinstance(loaded_data, torch.nn.Module):
                    model = loaded_data
                    print(f"  加载成功, 类型: {model.__class__.__name__} (来自完整模型保存)")
                else:
                    # 兜底：万一 tmc 也是 state_dict
                    print(f"  TMC 文件是一个 state_dict。正在加载到新实例...")
                    model = RobertaForMultipleChoice.from_pretrained(TOKENIZER_PATH)
                    model.load_state_dict(loaded_data)

            elif model_type == "dmc":
                # 2b. Dmc 模型：加载的数据 *是* state_dict (字典)
                if isinstance(loaded_data, dict):
                    model = SpatiallyAwareISAAQ()
                    model.load_state_dict(loaded_data)
                    print(f"  加载成功, 类型: {model.__class__.__name__} (来自 state_dict)")
                else:
                    print(f"  FATAL: DMC 权重文件 {path} 不是 state_dict, 而是 {type(loaded_data)}!")
                    raise TypeError("DMC checkpoint 格式错误。")

            else:
                print(f"  未知的模型类型: {model_type}")
                continue

            # 3. 尝试将其移动到 GPU
            print(f"  尝试 .to({DEVICE})...")
            model.to(DEVICE)
            print("  .to(device) 成功。")

            models.append(model)

        except Exception as e:
            print(f"\n--- ❌ 测试失败 ---")
            print(f"在加载 {path} 时遇到致命错误:")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "=" * 30)
    print(f"--- ✅ 完整加载测试成功 ---")
    print(f"所有 {len(models)} 个模型均已成功实例化、加载权重并移动到 {DEVICE}。")
    print("您可以安全地运行 tqa_dq_ensembler.py 脚本了。")


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main()