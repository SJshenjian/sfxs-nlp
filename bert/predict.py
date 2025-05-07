from queue import Queue
from threading import Lock

import pandas as pd
import torch
import torch.nn as nn
import re

from elasticsearch import helpers
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

from utils.esutils import query_enterprises_scroll, es_client
import utils.processes as processes

# 模型定义（与训练脚本一致）
class BertCRFModel(nn.Module):
    def __init__(self, num_labels, bert_model_name='bert-base-chinese'):
        super(BertCRFModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return predictions


# 预测函数
def predict(model, tokenizer, address, device, id2label, max_len=128):
    model.eval()
    chars = list(address)
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)[0]

    # 转换为标签
    pred_labels = [id2label[pred] for pred in predictions[1:len(chars) + 1]]

    # 结构化输出
    result = {"PROV": [], "CITY": [], "DISTRICT": [], "DEV": [], "TOWN": [], "EXTRA": []}
    current_key = None
    for char, label in zip(chars, pred_labels):
        if label.startswith("B-"):
            current_key = label[2:]
            result[current_key].append(char)
        elif label.startswith(("I-", "E-")) and current_key == label[2:]:
            result[current_key].append(char)
        else:
            current_key = None

    # 合并字符
    for key in result:
        result[key] = "".join(result[key]) if result[key] else None

    return result


# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = 'bert-base-chinese'
max_len = 128
checkpoint_path = './model_checkpoints/checkpoint_epoch_1.pt'  # 替换为实际保存的模型路径

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 加载模型和标签映射
checkpoint = torch.load(checkpoint_path, map_location=device)
num_labels = len(checkpoint['label2id'])
id2label = checkpoint['id2label']

model = BertCRFModel(num_labels=num_labels, bert_model_name=bert_model_name)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)


# 主程序
def cut(addr):
    result = predict(model, tokenizer, addr, device, id2label, max_len)
    return result

actions = list()  # 使用队列替代全局actions列表
lock = Lock()  # 保护action_queue的锁
processed_count = 0
batch_size = 10000
def deal(row):
    global processed_count
    address = row["registered_address"]
    province = row["province"]
    city = row["city"]
    district = row["district"]
    dev = row["dev"] if 'dev' in row else None  # 如果有开发区，则赋值给变量；否则设为空

    row["extra_address"] = None
    row["matched_province"] = None
    row["matched_city"] = None
    row["matched_district"] = None

    if pd.isna(address) or address == "-":
        return row

    extra_address = address
    matched_province = None
    matched_city = None
    matched_district = None
    matched_dev = None
    matched_town_or_street = None

    if not pd.isna(province) and province != "-" and province in extra_address:
        matched_province = province
        extra_address = extra_address.replace(province, "", 1)

    if not pd.isna(city) and city != "-" and city in extra_address:
        matched_city = city
        extra_address = extra_address.replace(city, "", 1)

    if not pd.isna(district) and district != "-" and district in extra_address:
        matched_district = district
        extra_address = extra_address.replace(district, "", 1)

    dev_pattern = r"[\u4e00-\u9fa5]+(?:经济|高新|技术|科技|产业)?开发区"
    match = re.search(dev_pattern, extra_address)
    if match:
        matched_dev = match.group(0)
        extra_address = extra_address.replace(matched_dev, "", 1)

    # 提取镇/街道（以“镇”或“街道”结尾的词）
    town_or_street_pattern = r"[^\s]{2,10}(?:镇|街道)"
    match = re.search(town_or_street_pattern, extra_address)
    if match:
        matched_town_or_street = match.group(0)
        extra_address = extra_address.replace(matched_town_or_street, "", 1)

    extra_address = extra_address.strip()
    if not extra_address:
        extra_address = None

    row["extra_address"] = extra_address
    row["matched_province"] = matched_province
    row["matched_city"] = matched_city
    row["matched_district"] = matched_district
    row["matched_dev"] = matched_dev
    row["matched_town"] = matched_town_or_street

    ret = cut(address)
    if ret['PROV'] != row["matched_province"] or ret['CITY'] != row["matched_city"] or ret['DISTRICT'] != row["matched_district"] \
        or ret['DEV'] != row["matched_dev"] or ret['TOWN'] != row["matched_town"] or ret['EXTRA'] != row["extra_address"]:
        action = {
            "_op_type": "index",
            "_index": "test_address",
            "_id": row['id'],
            "_source": {
                "address": address,
                "province": province,
                "city": city,
                "district": district,
                "dev": dev,
                'town': row["matched_town"],
                "extra_address": extra_address,
                "wrong_province": ret['PROV'],
                "wrong_city": ret['CITY'],
                "wrong_district": ret['DISTRICT'],
                "wrong_dev": ret['DEV'],
                "wrong_town": ret["TOWN"],
                "wrong_extra": ret['EXTRA']
            }
        }
        print(action)

    return row

if __name__ == "__main__":
    ret = cut('山东省临沂锡山区锡山区锡北镇锡北镇山东省八士菜场')
    df = query_enterprises_scroll(max=1000)
    processes.process_dataframe_multiprocess(df, deal, num_processes=4, chunk_size=100000)

