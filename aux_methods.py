from transformers import RobertaForSequenceClassification, AdamW, BertConfig, BertTokenizer, BertPreTrainedModel, \
    BertModel, BertForMultipleChoice, RobertaForMultipleChoice, RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
import random
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from PIL import Image

import torchvision.models as models

# --- MODIFICATION START: 导入新模块 ---
try:
    from spatial_utils import build_graph_using_normalized_boxes
    from spatial_transformer import BertSpatialEncoder, SpatialBertConfig

    print("成功导入 spatial_utils 和 spatial_transformer。")
except ImportError:
    print("FATAL ERROR: spatial_utils.py 或 spatial_transformer.py 未找到。")
    print("请确保您已在同一目录下创建了这两个文件。")


# --- MODIFICATION END ---

# -----------------------------------------------------------------
# 辅助函数 (新增与通用)
# -----------------------------------------------------------------

def load_ocr_data(ocr_json_data, image_path_key, img_width, img_height):
    image_path_key_posix = image_path_key.replace("\\", "/")
    ocr_results = ocr_json_data.get(image_path_key_posix)
    if ocr_results is None:
        key_parts = image_path_key_posix.split('/')
        if len(key_parts) >= 2:
            image_path_key_short = f"{key_parts[-2]}/{key_parts[-1]}"
            ocr_results = ocr_json_data.get(image_path_key_short)
    if ocr_results is None:
        key_parts = image_path_key_posix.split('/')
        if key_parts: ocr_results = ocr_json_data.get(key_parts[-1])

    ocr_texts = []
    ocr_coords_abs = []

    if ocr_results:
        for entry in ocr_results:
            word_text = entry.get('WordText')
            coord_info = entry.get('Coordinate')
            if word_text and coord_info:
                ocr_texts.append(word_text)
                if isinstance(coord_info, dict):
                    x1, y1 = coord_info['Left'], coord_info['Top']
                    x2, y2 = x1 + coord_info['Width'], y1 + coord_info['Height']
                else:
                    x1, y1, x2, y2 = 0, 0, 0, 0
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)
                if x1 < x2 and y1 < y2:
                    ocr_coords_abs.append([x1, y1, x2, y2])
    return ocr_texts, ocr_coords_abs


def normalize_coords(coords_list, width, height):
    if len(coords_list) == 0: return np.array([], dtype=np.float32).reshape(0, 4)
    coords_np = np.array(coords_list, dtype=np.float32)
    coords_np[:, [0, 2]] /= width
    coords_np[:, [1, 3]] /= height
    coords_np = np.clip(coords_np, 0.0, 1.0)
    return coords_np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), np.sum(pred_flat != labels_flat)


def get_images(img_path):
    try:
        image = Image.open(img_path)
    except:
        image = Image.new('RGB', (224, 224))
    image = image.resize((224, 224), Image.ANTIALIAS)
    if image.mode != 'RGB': image = image.convert('RGB')
    image = np.array(image)[:, :, :3]
    image = torch.tensor(image).type(torch.FloatTensor).permute(2, 0, 1).cuda()
    return image


def get_rois(img_path, vectors):
    try:
        image = Image.open(img_path)
    except:
        image = Image.new('RGB', (224, 224))

    if image.mode != 'RGB': image = image.convert('RGB')
    rois = []
    if len(vectors) == 0: vectors = [[0, 0, 224, 224]]
    for vector in vectors:
        if len(vector) < 4 or vector[0] >= vector[2] or vector[1] >= vector[3]: vector = [0, 0, 224, 224]
        roi_image = image.crop(vector)
        roi_image = roi_image.resize((224, 224), Image.ANTIALIAS)
        roi_image = np.array(roi_image)
        roi_image = torch.tensor(roi_image).type(torch.FloatTensor).permute(2, 0, 1).cuda()
        rois.append(roi_image)
    return torch.stack(rois, dim=0)


def get_choice_encoded(text, question, answer, max_len, tokenizer):
    if text != "":
        encoded = tokenizer.encode_plus(text, question + " " + answer, max_length=max_len, pad_to_max_length=True,
                                        truncation=True)
    else:
        encoded = tokenizer.encode_plus(question + " " + answer, max_length=max_len, pad_to_max_length=True,
                                        truncation=True)
    return encoded["input_ids"], encoded["attention_mask"]


def get_dq_choice_encoded(text, question, answer, max_len, tokenizer):
    sep = tokenizer.sep_token
    qa_pair = question + f" {sep} " + answer
    kwargs = {"max_length": max_len, "pad_to_max_length": True, "truncation": True, "return_token_type_ids": True}
    if text != "":
        encoded = tokenizer.encode_plus(text, qa_pair, **kwargs)
    else:
        encoded = tokenizer.encode_plus(qa_pair, **kwargs)
    if "token_type_ids" not in encoded: encoded["token_type_ids"] = [0] * len(encoded["input_ids"])
    return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]


# -----------------------------------------------------------------
# 模型类定义
# -----------------------------------------------------------------

class ResnetRoberta(torch.nn.Module):
    def __init__(self):
        super(ResnetRoberta, self).__init__()
        self.roberta = RobertaModel.from_pretrained("./checkpoints/roberta-large")
        self.resnet = models.resnet101(pretrained=True)
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000, 1024))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images=None, labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        img_b = []
        for img_q in images:
            for img_file in img_q: img_b.append(get_images(img_file))
        out_resnet = torch.stack(img_b, dim=0)
        out_resnet = self.resnet(out_resnet)
        out_resnet = self.feats(out_resnet).view(-1, 1024)
        final_out = out_roberta * out_resnet
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output).view(-1, num_choices)
        outputs = (logits,)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class ResnetRobertabd(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertabd, self).__init__()
        self.roberta = RobertaModel.from_pretrained("./checkpoints/roberta-large")
        self.resnet = models.resnet101(pretrained=True)
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000, 1024))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images1=None, images2=None,
                labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        img_b = []
        for img_q in images1:
            for img_file in img_q: img_b.append(get_images(img_file))
        out_resnet1 = torch.stack(img_b, dim=0)
        out_resnet1 = self.resnet(out_resnet1)
        out_resnet1 = self.feats(out_resnet1).view(-1, 1024)
        img_b = []
        for img_q in images2:
            for img_file in img_q: img_b.append(get_images(img_file))
        out_resnet2 = torch.stack(img_b, dim=0)
        out_resnet2 = self.resnet(out_resnet2)
        out_resnet2 = self.feats(out_resnet2).view(-1, 1024)
        out_resnet = out_resnet1 * out_resnet2
        final_out = out_roberta * out_resnet
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output).view(-1, num_choices)
        outputs = (logits,)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class ResnetRobertaBU(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertaBU, self).__init__()
        self.roberta = RobertaModel.from_pretrained("./checkpoints/roberta-large")
        self.resnet = models.resnet101(pretrained=True)
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000, 1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))
        self.boxes = torch.nn.Sequential(torch.nn.Linear(4, 1024), torch.nn.LayerNorm(1024, eps=1e-12))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images=None, coords=None, labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        img_b, coord_b = [], []
        for img_q, coord_q in zip(images, coords):
            for img_file, coord in zip(img_q, coord_q): img_b.append(img_file); coord_b.append(coord)
        roi_b = []
        for image, coord, roberta_b in zip(img_b, coord_b, out_roberta):
            img_v = get_rois(image, coord[:32])
            coord_v = torch.tensor(coord[:32]).cuda()
            out_boxes = self.boxes(coord_v)
            out_resnet = self.resnet(img_v)
            out_resnet = self.feats(out_resnet)
            out_resnet = self.feats2(out_resnet).view(-1, 1024)
            out_roi = torch.sum((out_resnet + out_boxes) / 2, dim=0)
            roi_b.append(out_roi)
        out_visual = torch.stack(roi_b, dim=0)
        final_out = out_roberta * out_visual
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output).view(-1, num_choices)
        outputs = (logits,)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class ResnetRobertaBUTD(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertaBUTD, self).__init__()
        config = BertConfig.from_pretrained("./checkpoints/roberta-large")
        self.roberta = RobertaModel(config)
        self.resnet = models.resnet101(pretrained=False)
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000, 1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))
        self.boxes = torch.nn.Sequential(torch.nn.Linear(4, 1024), torch.nn.LayerNorm(1024, eps=1e-12))
        self.att1 = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.Tanh())
        self.att2 = torch.nn.Sequential(torch.nn.Linear(1024, 1), torch.nn.Sigmoid())
        self.att3 = torch.nn.Sequential(torch.nn.Linear(1024, 1), torch.nn.Softmax(dim=0))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)

    def forward(self): raise NotImplementedError


# --- 新模型 SpatiallyAwareISAAQ (完全体) ---
class SpatiallyAwareISAAQ(nn.Module):
    def __init__(self, hidden_size=1024, text_hidden_size=1024):
        super(SpatiallyAwareISAAQ, self).__init__()

        self.roberta = RobertaModel.from_pretrained("./checkpoints/roberta-large")
        resnet_base = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet_base.children())[:-1])

        # 1. Type Embeddings (0:Visual, 1:Question, 2:Option)
        self.type_embeddings = nn.Embedding(3, hidden_size)

        # 2. 特征投影
        self.vis_projection = nn.Sequential(
            nn.Linear(2048 + 4 + 1024, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.txt_projection = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 3. 空间感知编码器
        self.spatial_config = SpatialBertConfig(
            hidden_size=hidden_size,
            num_attention_heads=16,
            num_hidden_layers=2,
            context_width=1
        )
        self.spatial_encoder = BertSpatialEncoder(self.spatial_config)

        # 4. 判别头 (BUTD)
        self.att1 = nn.Sequential(nn.Linear(hidden_size * 2, 1024), nn.Tanh())
        self.att_gate = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.att_weights = nn.Sequential(nn.Linear(1024, 1), nn.Softmax(dim=1))

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)

    def _prepare_visual_nodes(self, image_path, obj_coords, ocr_coords, ocr_input_ids, device):
        if len(obj_coords) == 0 and len(ocr_coords) == 0:
            all_abs_coords = [[0, 0, 224, 224]]
            n_obj, n_ocr = 1, 0
        else:
            all_abs_coords = obj_coords + ocr_coords
            n_obj = len(obj_coords)
            n_ocr = len(ocr_coords)

        max_nodes = 64
        if len(all_abs_coords) > max_nodes:
            all_abs_coords = all_abs_coords[:max_nodes]
            if n_obj >= max_nodes:
                n_obj = max_nodes; n_ocr = 0
            else:
                n_ocr = max_nodes - n_obj

        all_rois = get_rois(image_path, all_abs_coords)
        vis_feats = self.resnet(all_rois).squeeze(-1).squeeze(-1)
        coords_tensor = torch.tensor(all_abs_coords, dtype=torch.float32, device=device)

        hidden_size = self.roberta.config.hidden_size
        yolo_txt = torch.zeros(n_obj, hidden_size, device=device)

        if n_ocr > 0:
            curr_ocr_ids = ocr_input_ids[:n_ocr].to(device)
            ocr_out = self.roberta(curr_ocr_ids)[1]
            text_feats = torch.cat([yolo_txt, ocr_out], dim=0)
        else:
            text_feats = yolo_txt

        combined = torch.cat([vis_feats, coords_tensor, text_feats], dim=-1)
        vis_nodes = self.vis_projection(combined)
        vis_nodes = vis_nodes + self.type_embeddings(torch.tensor(0, device=device))

        return vis_nodes

    def forward(self, input_ids, attention_mask, token_type_ids,
                images, obj_coords_list, ocr_coords_list,
                ocr_input_ids_list, spatial_adj_matrix_list, labels=None):

        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type = token_type_ids.view(-1, token_type_ids.size(-1))  # 接收 token_type_ids

        roberta_out = self.roberta(flat_input_ids, attention_mask=flat_mask)
        sequence_output = roberta_out[0]

        logits_list = []
        current_idx = 0

        for b in range(batch_size):
            vis_nodes = self._prepare_visual_nodes(
                images[b],
                obj_coords_list[b],
                ocr_coords_list[b],
                ocr_input_ids_list[b],
                device
            )
            n_vis = vis_nodes.size(0)
            vis_adj = spatial_adj_matrix_list[b].to(device)
            vis_adj = vis_adj[:n_vis, :n_vis]

            for c in range(num_choices):
                seq_emb = sequence_output[current_idx]
                curr_input_ids = flat_input_ids[current_idx]
                curr_token_types = flat_token_type[current_idx]  # 使用 token_type_ids

                # --- 修复：使用 token_type_ids (0=Para/Q, 1=Opt) ---
                # (假设 RoBERTa 的 token_type_ids 在 Para/Q 为 0, Opt 为 1)
                # (如果 RoBERTa 全为0, 则需要用 SEP 索引)

                # --- Fallback: 使用 SEP 索引 (更可靠) ---
                sep_indices = (curr_input_ids == 2).nonzero(as_tuple=True)[0]

                cls_token = seq_emb[0:1]  # [1, 1024]
                cls_node = self.txt_projection(cls_token)
                cls_node = cls_node + self.type_embeddings(torch.tensor(1, device=device))  # CLS 视为 Type 1

                text_nodes = [cls_node]

                if len(sep_indices) >= 3:
                    # [CLS] Para [SEP] [SEP] Q [SEP] Opt [SEP]
                    opt_end_idx = sep_indices[-1]
                    q_end_idx = sep_indices[-2]
                    q_start_idx = sep_indices[-3]

                    q_tokens = seq_emb[q_start_idx + 1: q_end_idx]
                    if len(q_tokens) > 0:
                        q_nodes = self.txt_projection(q_tokens)
                        q_nodes += self.type_embeddings(torch.tensor(1, device=device))  # Type 1
                        text_nodes.append(q_nodes)

                    opt_tokens = seq_emb[q_end_idx + 1: opt_end_idx]
                    if len(opt_tokens) > 0:
                        opt_nodes = self.txt_projection(opt_tokens)
                        opt_nodes += self.type_embeddings(torch.tensor(2, device=device))  # Type 2
                        text_nodes.append(opt_nodes)
                else:
                    fallback_tokens = seq_emb[1:]  # 取 [CLS] 之后的所有内容
                    if len(fallback_tokens) > 0:
                        fallback_nodes = self.txt_projection(fallback_tokens)
                        # 统一标记为 Type 1 (Question/General Text)，利用 RoBERTa 自身的注意力
                        fallback_nodes += self.type_embeddings(torch.tensor(1, device=device))
                        text_nodes.append(fallback_nodes)

                all_text_nodes = torch.cat(text_nodes, dim=0)
                n_txt = all_text_nodes.size(0)

                all_nodes = torch.cat([vis_nodes, all_text_nodes], dim=0)
                total_nodes = n_vis + n_txt

                full_adj = torch.full((total_nodes, total_nodes), 14, dtype=torch.long, device=device)  # T-T Global
                full_adj[:n_vis, :n_vis] = vis_adj
                full_adj[:n_vis, n_vis:] = 13  # V-T Implicit
                full_adj[n_vis:, :n_vis] = 13  # T-V Implicit

                encoded_nodes = self.spatial_encoder(
                    all_nodes.unsqueeze(0),
                    full_adj.unsqueeze(0)
                ).squeeze(0)

                vis_nodes_new = encoded_nodes[:n_vis]
                # --- 修复 Query 来源 ---
                # q_final_updated 是更新后的 [CLS] 节点 (它在 all_text_nodes 的第0个)
                q_final_updated = encoded_nodes[n_vis]

                q_expanded = q_final_updated.unsqueeze(0).repeat(n_vis, 1)
                att_in = torch.cat([vis_nodes_new, q_expanded], dim=1)

                att = self.att1(att_in)
                gate = self.att_gate(att)
                weights = self.att_weights(att * gate)

                v_pooled = torch.sum(weights * vis_nodes_new, dim=0)
                final_vec = q_final_updated * v_pooled
                score = self.classifier(final_vec)

                logits_list.append(score)
                current_idx += 1

        logits = torch.stack(logits_list).view(batch_size, num_choices)
        outputs = (logits,)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs


# -----------------------------------------------------------------
# 训练流程函数 (包含被遗漏的函数)
# -----------------------------------------------------------------

def get_data_tf(split, retrieval_solver, tokenizer, max_len):
    input_ids_list, att_mask_list, labels_list = [], [], []
    json_path = os.path.join("jsons", "tqa_tf.json")
    with open(json_path, "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["sentence_" + retrieval_solver]
        encoded = tokenizer.encode_plus(text, question, max_length=max_len, pad_to_max_length=True, truncation=True)
        input_ids_list.append(encoded["input_ids"])
        att_mask_list.append(encoded["attention_mask"])
        labels_list.append(1 if doc["correct_answer"] == "true" else 0)
    return [input_ids_list, att_mask_list, labels_list]


def get_data_ndq(dataset_name, split, retrieval_solver, tokenizer, max_len):
    input_ids_list, att_mask_list, labels_list = [], [], []
    json_path = os.path.join("jsons", f"tqa_{dataset_name}.json")
    with open(json_path, "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    counter = 7 if dataset_name == "ndq" else 4
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q, att_mask_q = [], []
        for count_i in range(counter):
            answer = answers[count_i] if count_i < len(answers) else ""
            ids, mask = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(ids)
            att_mask_q.append(mask)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        labels_list.append(list(doc["answers"].keys()).index(doc["correct_answer"]))
    return [input_ids_list, att_mask_list, labels_list]


def process_data_ndq(raw_data, batch_size, split):
    # raw_data MIGHT include token_type_ids, adjust if necessary
    if len(raw_data) == 3:
        input_ids_list, att_mask_list, labels_list = raw_data
        inputs = torch.tensor(input_ids_list)
        masks = torch.tensor(att_mask_list)
        labels = torch.tensor(labels_list)
        data = TensorDataset(inputs, masks, labels)
    else:
        # Assuming format [ids, mask, types, labels] or similar
        # This part might need adjustment based on what get_data_ndq returns
        input_ids_list, att_mask_list, *_, labels_list = raw_data
        inputs = torch.tensor(input_ids_list)
        masks = torch.tensor(att_mask_list)
        labels = torch.tensor(labels_list)
        data = TensorDataset(inputs, masks, labels)  # 简化版，ensembler 可能不需要 types

    if split == "train":
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_data_dq(split, retrieval_solver, tokenizer, max_len):
    ocr_json_path = os.path.join("ocr_results", f"{split}_ocr_results.json")
    try:
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_json_data = json.load(f)
        print(f"Loaded OCR data from {ocr_json_path}")
    except Exception as e:
        print(f"Warning: Failed to load OCR data ({e}). Proceeding without OCR.")
        ocr_json_data = {}

    input_ids_list, att_mask_list, token_type_list = [], [], []
    images_list, obj_coords_list, ocr_coords_list = [], [], []
    ocr_input_ids_list, spatial_adj_matrix_list, labels_list = [], [], []

    with open(os.path.join("jsons", "tqa_dq.json"), "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]

    img_size_cache = {}
    missing_imgs = set()

    MAX_OCR_LEN = 10
    MAX_OCR_NUM = 64

    for doc in tqdm(dataset, desc=f"Loading DQ {split}"):
        question = doc["question"]
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        inp_q, mask_q, type_q = [], [], []
        for count_i in range(4):
            answer = answers[count_i] if count_i < len(answers) else ""
            ids, mask, tt = get_dq_choice_encoded(text, question, answer, max_len, tokenizer)
            inp_q.append(ids);
            mask_q.append(mask);
            type_q.append(tt)

        image_path = doc["image_path"]
        if not os.path.exists(image_path):
            if image_path not in missing_imgs: missing_imgs.add(image_path)
            continue

        if image_path in img_size_cache:
            w, h = img_size_cache[image_path]
        else:
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
                img_size_cache[image_path] = (w, h)
            except:
                continue

        obj_coords = [c[:4] for c in doc["coords"]]
        key_parts = image_path.replace("\\", "/").split('/')
        key = f"{key_parts[-2]}/{key_parts[-1]}" if len(key_parts) >= 2 else key_parts[-1]
        ocr_texts, ocr_coords = load_ocr_data(ocr_json_data, key, w, h)

        current_ocr_ids = []
        for text in ocr_texts[:MAX_OCR_NUM]:
            tokens = tokenizer.encode(text, add_special_tokens=False, max_length=MAX_OCR_LEN, truncation=True)
            tokens = tokens + [tokenizer.pad_token_id] * (MAX_OCR_LEN - len(tokens))
            current_ocr_ids.append(tokens)

        # 截断 OCR 坐标列表以匹配 tokenized 的文本
        ocr_coords = ocr_coords[:len(current_ocr_ids)]

        if len(current_ocr_ids) == 0:
            ocr_ids_tensor = torch.zeros(1, MAX_OCR_LEN, dtype=torch.long)
        else:
            ocr_ids_tensor = torch.tensor(current_ocr_ids, dtype=torch.long)

        all_coords = obj_coords + ocr_coords
        all_coords = all_coords[:MAX_OCR_NUM]
        if not all_coords: all_coords = [[0, 0, w, h]]

        norm_coords = normalize_coords(all_coords, w, h)
        adj = build_graph_using_normalized_boxes(norm_coords)

        input_ids_list.append(inp_q)
        att_mask_list.append(mask_q)
        token_type_list.append(type_q)
        images_list.append(image_path)
        obj_coords_list.append(obj_coords)
        ocr_coords_list.append(ocr_coords)
        ocr_input_ids_list.append(ocr_ids_tensor)
        spatial_adj_matrix_list.append(torch.tensor(adj, dtype=torch.long))
        labels_list.append(list(doc["answers"].keys()).index(doc["correct_answer"]))

    return [input_ids_list, att_mask_list, token_type_list, images_list, obj_coords_list, ocr_coords_list,
            ocr_input_ids_list, spatial_adj_matrix_list, labels_list]


def get_data_dq_bd(split, retrieval_solver, tokenizer, max_len):
    input_ids_list, att_mask_list = [], []
    images1_list, images2_list, coords_list, labels_list = [], [], [], []
    json_path = os.path.join("jsons", "tqa_dq_bd.json")
    if not os.path.exists(json_path): return [], [], [], [], [], []
    with open(json_path, "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q, att_mask_q = [], []
        for count_i in range(4):
            answer = answers[count_i] if count_i < len(answers) else ""
            ids, mask = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(ids)
            att_mask_q.append(mask)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images1_list.append(doc["image_path"])
        images2_list.append(doc["context_image_path"])
        coords_list.append([c[:4] for c in doc["coords"]])
        labels_list.append(list(doc["answers"].keys()).index(doc["correct_answer"]))
    return [input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list]


def training_tf(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device,
                save_model=False):
    for epoch_i in range(epochs):
        print(f'\nEpoch {epoch_i + 1}/{epochs} Training...')
        model.train()
        total_points, total_errors = 0, 0
        loss_accum = []
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_ids, b_mask, b_lbls = batch
            outputs = model(b_ids, attention_mask=b_mask, labels=b_lbls)
            loss, logits = outputs

            preds = logits.detach().cpu().numpy()
            lbls = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls)
            total_points += p;
            total_errors += e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            loss_accum.append(loss.item())

        if save_model:
            torch.save(model.state_dict(), f"checkpoints/tf_roberta_{retrieval_solver}_e{epoch_i + 1}.pth")
        validation_tf(model, val_dataloader, device)


def validation_tf(model, val_dataloader, device):
    print("Running Validation...")
    model.eval()
    total_points, total_errors = 0, 0
    loss_list = []
    final_res = []  # --- 修复：返回 logits ---
    for batch in tqdm(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_ids, b_mask, b_lbls = batch
        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask, labels=b_lbls)
        loss, logits = outputs
        loss_list.append(loss.item())
        preds = logits.detach().cpu().numpy()
        for l in preds: final_res.append(l)  # --- 修复：填充 logits ---
        lbls = b_lbls.cpu().numpy()
        p, e = flat_accuracy(preds, lbls)
        total_points += p;
        total_errors += e

    acc = total_points / (total_points + total_errors + 1e-9)
    print(f"Val Acc: {acc:.4f} Loss: {np.mean(loss_list):.4f}")
    return final_res  # --- 修复：返回 logits ---


def training_ndq(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device,
                 save_model, dataset_name):
    for epoch_i in range(epochs):
        print(f'\nEpoch {epoch_i + 1}/{epochs} Training...')
        model.train()
        total_points, total_errors = 0, 0
        loss_accum = []
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_ids, b_mask, b_lbls = batch
            outputs = model(b_ids, attention_mask=b_mask, labels=b_lbls)
            loss, logits = outputs
            preds = logits.detach().cpu().numpy()
            lbls = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls)
            total_points += p;
            total_errors += e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            loss_accum.append(loss.item())

        if save_model:
            torch.save(model.state_dict(),
                       f"checkpoints/tmc_{dataset_name}_roberta_{retrieval_solver}_e{epoch_i + 1}.pth")
        validation_ndq(model, val_dataloader, device)


# --- 重新添加 validation_ndq ---
def validation_ndq(model, val_dataloader, device):
    print("Running Validation...")
    model.eval()
    total_points, total_errors = 0, 0
    loss_list = []
    final_res = []  # --- 修复：返回 logits ---
    for batch in tqdm(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_ids, b_mask, b_lbls = batch
        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask, labels=b_lbls)
        loss, logits = outputs
        loss_list.append(loss.item())
        preds = logits.detach().cpu().numpy()
        for l in preds: final_res.append(l)  # --- 修复：填充 logits ---
        lbls = b_lbls.cpu().numpy()
        p, e = flat_accuracy(preds, lbls)
        total_points += p;
        total_errors += e

    acc = total_points / (total_points + total_errors + 1e-9)
    print(f"Val Acc: {acc:.4f} Loss: {np.mean(loss_list):.4f}")
    return final_res  # --- 修复：返回 logits ---


# --- 结束重新添加 ---

def training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device,
                save_model):
    input_ids, att_mask, token_type, images, obj_coords, ocr_coords, ocr_ids, spatial_adj, labels = raw_data_train

    for epoch_i in range(epochs):
        print(f'\nEpoch {epoch_i + 1}/{epochs} Training...')
        model.train()
        total_points, total_errors = 0, 0
        loss_accum = []
        indices = list(range(len(labels)))
        random.shuffle(indices)

        pbar = tqdm(range(0, len(indices), batch_size))
        for i in pbar:
            batch_idx = indices[i:i + batch_size]

            b_ids = torch.tensor([input_ids[k] for k in batch_idx]).to(device)
            b_mask = torch.tensor([att_mask[k] for k in batch_idx]).to(device)
            b_type = torch.tensor([token_type[k] for k in batch_idx]).to(device)
            b_imgs = [images[k] for k in batch_idx]
            b_obj_c = [obj_coords[k] for k in batch_idx]
            b_ocr_c = [ocr_coords[k] for k in batch_idx]
            b_adj = [spatial_adj[k] for k in batch_idx]
            b_lbls = torch.tensor([labels[k] for k in batch_idx]).to(device)

            # Pad OCR IDs
            batch_ocr_ids_list = [ocr_ids[k] for k in batch_idx]
            # 找到批次中最大的 OCR 节点数 (确保不为空)
            max_nodes = max([t.size(0) for t in batch_ocr_ids_list if t.size(0) > 0] + [1])
            padded_ocr_ids = torch.zeros(len(batch_idx), max_nodes, 10, dtype=torch.long).to(
                device)  # 10 is MAX_OCR_LEN
            for j, t in enumerate(batch_ocr_ids_list):
                padded_ocr_ids[j, :t.size(0), :] = t.to(device)

            outputs = model(
                input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type,
                images=b_imgs,
                obj_coords_list=b_obj_c,
                ocr_coords_list=b_ocr_c,
                ocr_input_ids_list=padded_ocr_ids,
                spatial_adj_matrix_list=b_adj,
                labels=b_lbls
            )

            loss, logits = outputs
            preds = logits.detach().cpu().numpy()
            lbls = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls)
            total_points += p;
            total_errors += e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            loss_accum.append(loss.item())
            pbar.set_description(
                f"Acc: {total_points / (total_points + total_errors + 1e-9):.4f} Loss: {np.mean(loss_accum[-10:]):.4f}")

        if save_model:
            torch.save(model.state_dict(), f"checkpoints/dmc_dq_roberta_SPATIAL_{retrieval_solver}_e{epoch_i + 1}.pth")
        validation_dq(model, raw_data_val, batch_size, device)


def validation_dq(model, raw_data_val, batch_size, device):
    print("Running Validation...")
    input_ids, att_mask, token_type, images, obj_coords, ocr_coords, ocr_ids, spatial_adj, labels = raw_data_val
    model.eval()
    total_points, total_errors = 0, 0
    loss_list = []
    final_res = []  # --- 修复：返回 logits ---

    indices = list(range(len(labels)))
    for i in tqdm(range(0, len(indices), batch_size)):
        batch_idx = indices[i:i + batch_size]

        batch_ocr_ids_list = [ocr_ids[k] for k in batch_idx]
        max_nodes = max([t.size(0) for t in batch_ocr_ids_list if t.size(0) > 0] + [1])
        padded_ocr_ids = torch.zeros(len(batch_idx), max_nodes, 10, dtype=torch.long).to(device)
        for j, t in enumerate(batch_ocr_ids_list): padded_ocr_ids[j, :t.size(0), :] = t.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([input_ids[k] for k in batch_idx]).to(device),
                attention_mask=torch.tensor([att_mask[k] for k in batch_idx]).to(device),
                token_type_ids=torch.tensor([token_type[k] for k in batch_idx]).to(device),
                images=[images[k] for k in batch_idx],
                obj_coords_list=[obj_coords[k] for k in batch_idx],
                ocr_coords_list=[ocr_coords[k] for k in batch_idx],
                ocr_input_ids_list=padded_ocr_ids,
                spatial_adj_matrix_list=[spatial_adj[k] for k in batch_idx],
                labels=torch.tensor([labels[k] for k in batch_idx]).to(device)
            )
        loss, logits = outputs
        loss_list.append(loss.item())

        # --- 修复：AxisError ---
        preds = logits.detach().cpu().numpy()  # 之前是 logits[0]
        for l in preds: final_res.append(l)  # --- 修复：填充 logits ---
        # ------------------------

        lbls = torch.tensor([labels[k] for k in batch_idx]).numpy()
        p, e = flat_accuracy(preds, lbls)
        total_points += p;
        total_errors += e

    acc = total_points / (total_points + total_errors + 1e-9)
    print(f"Val Acc: {acc:.4f} Loss: {np.mean(loss_list):.4f}")
    return final_res  # --- 修复：返回 logits ---


def get_data_dq_bd(split, retrieval_solver, tokenizer, max_len):
    input_ids_list, att_mask_list = [], []
    images1_list, images2_list, coords_list, labels_list = [], [], [], []
    json_path = os.path.join("jsons", "tqa_dq_bd.json")
    if not os.path.exists(json_path): return [], [], [], [], [], []
    with open(json_path, "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q, att_mask_q = [], []
        for count_i in range(4):
            answer = answers[count_i] if count_i < len(answers) else ""
            ids, mask = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(ids)
            att_mask_q.append(mask)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images1_list.append(doc["image_path"])
        images2_list.append(doc["context_image_path"])
        coords_list.append([c[:4] for c in doc["coords"]])
        labels_list.append(list(doc["answers"].keys()).index(doc["correct_answer"]))
    return [input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list]


def training_dq_bd(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver,
                   device, save_model):
    input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list = raw_data_train
    for epoch_i in range(epochs):
        print(f'\nEpoch {epoch_i + 1}/{epochs} Training...')
        model.train()
        total_points, total_errors = 0, 0
        indices = list(range(len(labels_list)))
        random.shuffle(indices)
        pbar = tqdm(range(0, len(indices), batch_size))
        for i in pbar:
            batch_idx = indices[i:i + batch_size]
            b_ids = torch.tensor([input_ids_list[k] for k in batch_idx]).to(device)
            b_mask = torch.tensor([att_mask_list[k] for k in batch_idx]).to(device)
            b_im1 = [images1_list[k] for k in batch_idx]
            b_im2 = [images2_list[k] for k in batch_idx]
            b_lbls = torch.tensor([labels_list[k] for k in batch_idx]).to(device)

            outputs = model(b_ids, attention_mask=b_mask, images1=b_im1, images2=b_im2, labels=b_lbls)
            loss, logits = outputs
            preds = logits.detach().cpu().numpy()
            lbls = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls)
            total_points += p;
            total_errors += e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if save_model:
            torch.save(model.state_dict(), f"checkpoints/dmc_dq_roberta_{retrieval_solver}_e{epoch_i + 1}.pth")
        validation_dq_bd(model, raw_data_val, batch_size, device)


def validation_dq_bd(model, raw_data_val, batch_size, device):
    print("Running Validation...")
    input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list = raw_data_val
    model.eval()
    total_points, total_errors = 0, 0
    loss_list = []
    final_res = []  # --- 修复：返回 logits ---
    indices = list(range(len(labels_list)))
    for i in tqdm(range(0, len(indices), batch_size)):
        batch_idx = indices[i:i + batch_size]
        b_ids = torch.tensor([input_ids_list[k] for k in batch_idx]).to(device)
        b_mask = torch.tensor([att_mask_list[k] for k in batch_idx]).to(device)
        b_im1 = [images1_list[k] for k in batch_idx]
        b_im2 = [images2_list[k] for k in batch_idx]
        b_lbls = torch.tensor([labels_list[k] for k in batch_idx]).to(device)
        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask, images1=b_im1, images2=b_im2, labels=b_lbls)
        loss, logits = outputs
        loss_list.append(loss.item())
        preds = logits.detach().cpu().numpy()
        for l in preds: final_res.append(l)  # --- 修复：填充 logits ---
        lbls = b_lbls.cpu().numpy()
        p, e = flat_accuracy(preds, lbls)
        total_points += p;
        total_errors += e
    acc = total_points / (total_points + total_errors + 1e-9)
    print(f"Val Acc: {acc:.4f} Loss: {np.mean(loss_list):.4f}")
    return final_res  # --- 修复：返回 logits ---


def generate_interagreement_chart(feats, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts: list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max), len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            res = sum(x == y for x, y in zip(i_solver, j_solver)) / len(i_solver)
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split + '_interagreement.png')


def generate_complementarity_chart(feats, labels, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts: list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max), len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            points = 0
            totals = 0
            for e1, e2, lab in zip(i_solver, j_solver, labels):
                if e1 != lab:
                    if e2 == lab: points += 1
                    totals += 1
            res = points / (totals + 1e-9)
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split + '_complementarity.png')


def get_upper_bound(feats, labels):
    points = 0
    for e1, e2, e3, lab in zip(feats[0], feats[1], feats[2], labels):
        if np.argmax(e1) == lab:
            points += 1
        else:
            if np.argmax(e2) == lab:
                points += 1
            else:
                if np.argmax(e3) == lab: points += 1
    upper_bound = points / len(labels)
    return upper_bound


def ensembler(feats_train, feats_test, labels_train, labels_test):
    softmax = torch.nn.Softmax(dim=1)
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i], soft_ft[i]])
                list_of_labels.append(1 if lab == i else 0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems,
                                                                                                       list_of_labels))

    list_of_elems = []
    list_of_labels = []
    for feats1, feats2, feats3, lab in zip(feats_train[0], feats_train[1], feats_train[2], labels_train):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output = output1 + output2 + output3
            list_of_elems.append(output)
            list_of_labels.append(1 if lab == i else 0)
    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems,
                                                                                                  list_of_labels)

    points = 0
    for feats1, feats2, feats3, lab in zip(feats_test[0], feats_test[1], feats_test[2], labels_test):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        outs = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output = output1 + output2 + output3
            outs.append(output)
        outs = [list(x) for x in outs]
        outs2 = final_model.predict_proba(outs)
        feats = [x[1] for x in outs2]
        outs3 = np.argmax(feats)
        if outs3 == lab: points += 1
    return points / len(labels_test)


def superensembler(feats_train, feats_test, labels_train, labels_test):
    softmax = torch.nn.Softmax(dim=1)
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i], soft_ft[i]])
                list_of_labels.append(1 if lab == i else 0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems,
                                                                                                       list_of_labels))

    list_of_elems = []
    list_of_labels = []
    if len(feats_train) == 6:
        for feats_list, lab in zip(zip(*feats_train), labels_train):
            soft_feats = [list(softmax(torch.tensor([f]))[0].detach().cpu().numpy()) for f in feats_list]
            for i in range(len(soft_feats[0])):
                outputs = []
                for j in range(len(solvers)):
                    outputs.append(solvers[j].predict_proba([[feats_list[j][i], soft_feats[j][i]]])[0])
                output = sum(outputs)
                list_of_elems.append(output)
                list_of_labels.append(1 if lab == i else 0)
    else:
        for feats1, feats2, feats3, lab in zip(feats_train[0], feats_train[1], feats_train[2], labels_train):
            soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
            soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
            soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
            for i in range(len(soft1)):
                output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
                output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
                output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
                output = output1 + output2 + output3
                list_of_elems.append(output)
                list_of_labels.append(1 if lab == i else 0)

    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems,
                                                                                                  list_of_labels)

    points = 0
    if len(feats_test) == 6:
        for feats_list, lab in zip(zip(*feats_test), labels_test):
            soft_feats = [list(softmax(torch.tensor([f]))[0].detach().cpu().numpy()) for f in feats_list]
            outs = []
            for i in range(len(soft_feats[0])):
                outputs = []
                for j in range(len(solvers)):
                    outputs.append(solvers[j].predict_proba([[feats_list[j][i], soft_feats[j][i]]])[0])
                output = sum(outputs)
                outs.append(output)
            outs = [list(x) for x in outs]
            outs2 = final_model.predict_proba(outs)
            feats = [x[1] for x in outs2]
            outs3 = np.argmax(feats)
            if outs3 == lab: points += 1
    else:
        for feats1, feats2, feats3, lab in zip(feats_test[0], feats_test[1], feats_test[2], labels_test):
            soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
            soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
            soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
            outs = []
            for i in range(len(soft1)):
                output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
                output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
                output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
                output = output1 + output2 + output3
                outs.append(output)
            outs = [list(x) for x in outs]
            outs2 = final_model.predict_proba(outs)
            feats = [x[1] for x in outs2]
            outs3 = np.argmax(feats)
            if outs3 == lab: points += 1
    return points / len(labels_test)