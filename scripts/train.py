import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from models.bilstm_crf import BiLSTM_CRF
from utils.data_loader import load_data, build_vocab, save_vocab, load_vocab
import os
import time
from multiprocessing import cpu_count


class NERDataset(Dataset):
    def __init__(self, indexed_data):
        if not indexed_data:
            raise ValueError("indexed_data is empty")
        self.data = indexed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_data(training_data, char2idx, tag2idx):
    """预处理数据并返回索引格式"""
    if not training_data:
        raise ValueError("training_data is empty")
    indexed_data = []
    for sentence, tags in training_data:
        char_ids = [char2idx.get(char, char2idx["<UNK>"]) for char in sentence]
        tag_ids = [tag2idx.get(tag, tag2idx["<PAD>"]) for tag in tags]
        indexed_data.append((torch.tensor(char_ids, dtype=torch.long),
                             torch.tensor(tag_ids, dtype=torch.long)))
    return indexed_data


def collate_fn(batch):
    """动态填充到批次最大长度"""
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    return sentences_padded, tags_padded


def train():
    device = torch.device("cpu")
    torch.set_num_threads(2)
    print(f"Using {cpu_count()} CPU cores")

    data_file = "data/train1000.txt"
    char2idx_path = "data/char2idx.pkl.gz"
    tag2idx_path = "data/tag2idx.pkl.gz"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training file not found: {data_file}")

    print("Loading data...")
    start_time = time.time()
    training_data = load_data(data_file)
    print(f"Loaded {len(training_data)} samples in {time.time() - start_time:.2f}s")

    print("Building vocab...")
    char2idx, tag2idx, idx2tag = build_vocab(training_data)
    print(f"Vocab size: {len(char2idx)}, Tag size: {len(tag2idx)}")

    print("Saving vocab...")
    save_vocab(char2idx, tag2idx, char2idx_path, tag2idx_path)

    print("Preprocessing data...")
    indexed_data = preprocess_data(training_data, char2idx, tag2idx)
    print(f"Preprocessed {len(indexed_data)} samples in {time.time() - start_time:.2f}s")

    # 初始化数据集和 DataLoader
    dataset = NERDataset(indexed_data)
    num_workers = min(cpu_count(), 8)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn)

    # 初始化模型
    model = BiLSTM_CRF(len(char2idx), len(tag2idx), embedding_dim=100, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Vocab size: {len(char2idx)}, Tag size: {len(tag2idx)}")
    print(f"Model embedding size: {model.embedding.num_embeddings}")

    print("Starting training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (sentences, tags) in enumerate(dataloader):
            sentences, tags = sentences.to(device), tags.to(device)

            if i == 0 and epoch == 0:
                print(f"Batch 0: max sentence index: {sentences.max().item()}, max tag index: {tags.max().item()}")
                print(f"Batch 0 shape: sentences {sentences.shape}, tags {tags.shape}")

            optimizer.zero_grad(set_to_none=True)
            loss = model.neg_log_likelihood(sentences, tags)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    print("Saving model...")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    train()