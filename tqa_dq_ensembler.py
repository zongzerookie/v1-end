# tqa_dq_ensembler.py
from transformers import RobertaTokenizer, RobertaForMultipleChoice
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse
import os  # 导入 os

# --- MODIFICATION START: 导入新旧模型 ---
# 导入所有需要的函数和模型
from aux_methods import get_data_ndq, process_data_ndq, get_data_dq, validation_ndq, validation_dq
from aux_methods import get_upper_bound, superensembler, ensembler
# 导入 SpatiallyAwareISAAQ 用于 dmc，导入 ResnetRobertaBUTD 用于兼容旧检查点（如果需要）
from aux_methods import SpatiallyAwareISAAQ, ResnetRobertaBUTD


# --- MODIFICATION END ---


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')
    # 保持您修改后的权重路径
    parser.add_argument('-p', '--pretrainingslist', default=[
        "checkpoints/tmc_dq_roberta_IR_e4.pth",
        "checkpoints/tmc_dq_roberta_NSP_e4.pth",
        "checkpoints/tmc_dq_roberta_NN_e4.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_IR_e4.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_NSP_e3.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_NN_e3.pth"
    ], nargs='+', help='list of paths of the pretrainings model. They must be six.')
    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default=32, type=int, help='size of the batches. Default: 32')
    args = parser.parse_args()
    print(args)

    TOKENIZER_PATH = "./checkpoints/roberta-large"  # 确保 tokenizer 路径正确

    # --- MODIFICATION START: 修复模型加载 ---

    model_paths = args.pretrainingslist
    retrieval_solvers = ["IR", "NSP", "NN", "IR", "NSP", "NN"]
    model_types = ["tmc", "tmc", "tmc", "dmc", "dmc", "dmc"]

    models = []

    for path, model_type in zip(model_paths, model_types):
        print("\n" + "=" * 30)
        print(f"Loading {model_type} model from: {path}")

        if not os.path.exists(path):
            print(f"FATAL: 权重文件未找到！路径: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        try:
            # 1. 统一加载文件
            print("  Executing torch.load()...")
            loaded_data = torch.load(path, map_location='cpu')

            # 2. 智能判断加载的是什么
            if isinstance(loaded_data, dict):
                # A. 加载的是 state_dict (我们的新 dmc 模型，或 state_dict 保存的 tmc)
                print(f"  Loaded data is a state_dict. Instantiating model...")

                if model_type == "dmc":
                    model = SpatiallyAwareISAAQ()
                else:
                    # tmc 权重也是 state_dict
                    model = RobertaForMultipleChoice.from_pretrained(TOKENIZER_PATH)

                model.load_state_dict(loaded_data)
                print(f"  Successfully loaded state_dict into {model.__class__.__name__}.")

            elif isinstance(loaded_data, torch.nn.Module):
                # B. 加载的是完整的模型对象 (旧的 tmc 模型)
                print(f"  Loaded data is a full model object ({loaded_data.__class__.__name__}).")
                model = loaded_data

            else:
                raise TypeError(f"Loaded file {path} is not a dict or nn.Module.")

            models.append(model)

        except Exception as e:
            print(f"\n--- ❌ 加载失败 ---")
            print(f"在加载 {path} 时遇到致命错误:")
            import traceback
            traceback.print_exc()
            raise e

    # --- MODIFICATION END ---

    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)

    max_len = args.maxlen
    batch_size = args.batchsize

    feats_train = []
    feats_val = []
    feats_test = []

    if args.device == "gpu":
        device = torch.device("cuda")
    if args.device == "cpu":
        device = torch.device("cpu")

    for model, model_type, retrieval_solver in zip(models, model_types, retrieval_solvers):

        model.to(device)
        model.eval()

        print("\n")
        print(f"Evaluating {model_type} model with {retrieval_solver} data...")

        if model_type == "dmc":
            print("train (DMC)")
            raw_data_train = get_data_dq("train", retrieval_solver, tokenizer, max_len)
            feats_train.append(validation_dq(model, raw_data_train, batch_size, device))
            labels_train = raw_data_train[-1]

            print("val (DMC)")
            raw_data_val = get_data_dq("val", retrieval_solver, tokenizer, max_len)
            feats_val.append(validation_dq(model, raw_data_val, batch_size, device))
            labels_val = raw_data_val[-1]

            print("test (DMC)")
            raw_data_test = get_data_dq("test", retrieval_solver, tokenizer, max_len)
            feats_test.append(validation_dq(model, raw_data_test, batch_size, device))
            labels_test = raw_data_test[-1]

        if model_type == "tmc":
            print("train (TMC)")
            raw_data_train = get_data_ndq("dq", "train", retrieval_solver, tokenizer, max_len)
            train_dataloader = process_data_ndq(raw_data_train, batch_size, "val")  # 'val' mode for no shuffle
            feats_train.append(validation_ndq(model, train_dataloader, device))
            labels_train = raw_data_train[-1]

            print("val (TMC)")
            raw_data_val = get_data_ndq("dq", "val", retrieval_solver, tokenizer, max_len)
            val_dataloader = process_data_ndq(raw_data_val, batch_size, "val")
            feats_val.append(validation_ndq(model, val_dataloader, device))
            labels_val = raw_data_val[-1]

            print("test (TMC)")
            raw_data_test = get_data_ndq("dq", "test", retrieval_solver, tokenizer, max_len)
            test_dataloader = process_data_ndq(raw_data_test, batch_size, "test")
            feats_test.append(validation_ndq(model, test_dataloader, device))
            labels_test = raw_data_test[-1]

    # upper_bound_train = get_upper_bound(feats_val, labels_val)
    print("\nCalculating Test Set Results...")
    res = superensembler(feats_train, feats_test, labels_train, labels_test)
    print("\nFINAL RESULTS:")
    print("TEST SET: ")
    print(res)

    print("\nCalculating Validation Set Results...")
    res = superensembler(feats_train, feats_val, labels_train, labels_val)
    print("VALIDATION SET: ")
    print(res)


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])