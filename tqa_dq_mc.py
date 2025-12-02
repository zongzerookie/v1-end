# tqa_dq_mc.py
from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse
import os

# --- MODIFICATION START: 导入新旧两个模型类 ---
from aux_methods import get_data_dq, training_dq, SpatiallyAwareISAAQ, ResnetRobertaBUTD


# --- MODIFICATION END ---

def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'],
                          help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/AI2D_e12.pth",
                        help='Path to OLD checkpoint to load RoBERTa weights from. Default: checkpoints/AI2D_e12.pth')
    parser.add_argument('-b', '--batchsize', default=1, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-l', '--lr', default=1e-6, type=float, help='learning rate. Default: 1e-6')
    parser.add_argument('-e', '--epochs', default=4, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training',
                        action='store_true')
    args = parser.parse_args()
    print(args)

    # --- MODIFICATION START: 更改模型初始化和权重加载 ---

    # 1. 实例化我们的新模型
    model = SpatiallyAwareISAAQ()

    # 2. 关键：选择性加载预训练权重
    if args.pretrainings != "" and os.path.exists(args.pretrainings):
        print(f"Loading RoBERTa weights from OLD checkpoint: {args.pretrainings}")
        try:
            # 关键修复:
            # torch.load() 默认使用 pickle，它需要 ResnetRobertaBUTD 类的定义。
            # 我们在 aux_methods.py 中重新添加了这个类，所以现在 load() 会成功。
            # 我们将它加载到 CPU 以避免 GPU 内存冲突。
            old_model_object = torch.load(args.pretrainings, map_location=torch.device('cpu'))

            # 从加载的对象中提取 state_dict
            old_state_dict = old_model_object.state_dict()
            # 结束关键修复

            # 提取旧模型中 RoBERTa 的权重
            roberta_weights = {k.replace('roberta.', ''): v
                               for k, v in old_state_dict.items()
                               if k.startswith('roberta.')}

            if roberta_weights:
                model.roberta.load_state_dict(roberta_weights, strict=False)
                print("Successfully loaded pre-trained RoBERTa weights into new model.")
            else:
                print("WARNING: Could not find 'roberta.' prefix in checkpoint. Training RoBERTa from scratch.")

        except Exception as e:
            print(f"ERROR loading RoBERTa weights: {e}. Training RoBERTa from scratch.")
    else:
        if args.pretrainings != "":
            print(f"WARNING: Pre-training file not found at {args.pretrainings}. Training RoBERTa from scratch.")
        else:
            print("No pre-training checkpoint specified. Training RoBERTa from scratch.")

    # 3. 冻结 ResNet (ISAAQ 原始策略)
    for param in model.resnet.parameters():
        param.requires_grad = False
    print("ResNet parameters frozen.")

    # --- MODIFICATION END ---

    tokenizer = RobertaTokenizer.from_pretrained("./checkpoints/roberta-large")

    if args.device == "gpu":
        device = torch.device("cuda")
        model.cuda()
    if args.device == "cpu":
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save

    # get_data_dq 现在返回 (input_ids, att_mask, images, coords, spatial_adj, labels)
    raw_data_train = get_data_dq("train", retrieval_solver, tokenizer, max_len)
    raw_data_val = get_data_dq("val", retrieval_solver, tokenizer, max_len)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    # raw_data_train[-1] 是 labels_list
    total_steps = len(raw_data_train[-1]) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # training_dq (在 aux_methods.py 中) 已经被修改为接受新数据格式
    training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device,
                save_model)


if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])