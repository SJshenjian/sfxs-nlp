import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os

# 定义 BME 标签集合
label_list = [
    "O",
    "B-PROV", "I-PROV", "E-PROV",
    "B-CITY", "I-CITY", "E-CITY",
    "B-DISTRICT", "I-DISTRICT", "E-DISTRICT",
    "B-DEV", "I-DEV", "E-DEV",
    "B-TOWN", "I-TOWN", "E-TOWN",
    "B-EXTRA", "I-EXTRA", "E-EXTRA"
]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}


# 自定义数据集
class AddressDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def load_data(self, data_file):
        """从 txt 文件加载数据"""
        data = []
        current_address = []
        current_labels = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_address and current_labels:
                        data.append({
                            "address": "".join(current_address),
                            "labels": current_labels
                        })
                        current_address = []
                        current_labels = []
                    continue

                try:
                    char, label = line.split()
                    current_address.append(char)
                    current_labels.append(label)
                except ValueError:
                    continue

            if current_address and current_labels:
                data.append({
                    "address": "".join(current_address),
                    "labels": current_labels
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        address = self.data[idx]["address"]
        labels = self.data[idx]["labels"]

        chars = list(address)
        label_ids = [label2id[label] for label in labels]

        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        label_ids_padded = [label2id["O"]] * self.max_len
        for i in range(min(len(label_ids), self.max_len - 2)):
            label_ids_padded[i + 1] = label_ids[i]

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids_padded, dtype=torch.long)
        }


# 模型定义
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


# 训练函数
def train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, device, num_epochs, model_save_path):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        if avg_loss < 0.3:
            return

    # 验证
    #eval_f1 = evaluate_model(model, dev_dataloader, device)
    #print(f"Epoch {epoch + 1}, Validation F1: {eval_f1:.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'label2id': label2id,
        'id2label': id2label
    }, os.path.join(model_save_path, f'checkpoint_epoch_{epoch + 1}.pt'))


# 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask)
            for pred, label, mask in zip(predictions, labels, attention_mask):
                valid_len = int(mask.sum().item())
                all_preds.extend(pred[1:valid_len - 1])
                all_labels.extend(label[1:valid_len - 1].cpu().numpy())

    return f1_score(all_labels, all_preds, average='micro')


# 主程序
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model_name = 'bert-base-chinese'
    max_len = 128
    batch_size = 16
    num_epochs = 100
    learning_rate = 3e-5
    model_save_path = './model_checkpoints'
    train_file = '../data/train1000.txt'  # 替换为你的 train.txt 路径
    dev_file = '../data/test.txt'  # 假设有验证集文件

    # 创建保存目录
    os.makedirs(model_save_path, exist_ok=True)

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # 准备数据
    train_dataset = AddressDataset(train_file, tokenizer, max_len)
    dev_dataset = AddressDataset(dev_file, tokenizer, max_len) if dev_file else train_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    # 初始化模型
    model = BertCRFModel(num_labels=len(label_list), bert_model_name=bert_model_name)
    model.to(device)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练
    train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, device, num_epochs, model_save_path)


if __name__ == "__main__":
    main()