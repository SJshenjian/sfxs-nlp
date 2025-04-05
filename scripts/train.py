import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from models.bilstm_crf import BiLSTM_CRF
from utils.data_loader import load_data_chunk, build_vocab_parallel, save_vocab
import os
import time
from multiprocessing import cpu_count
import gc
import psutil

class NERDataset(Dataset):
    def __init__(self, raw_data, char2idx, tag2idx):
        if not raw_data:
            raise ValueError("raw_data is empty")
        self.data = raw_data
        self.char2idx = char2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        char_ids = [self.char2idx.get(char, self.char2idx["<UNK>"]) for char in sentence]
        tag_ids = [self.tag2idx.get(tag, self.tag2idx["<PAD>"]) for tag in tags]
        return (torch.tensor(char_ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long))

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    return sentences_padded, tags_padded

def train_on_chunk(chunk_data, model, optimizer, device, char2idx, tag2idx, epoch_start=0):
    dataset = NERDataset(chunk_data, char2idx, tag2idx)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, collate_fn=collate_fn)
    for epoch in range(epoch_start, 10):
        model.train()
        total_loss = 0
        start_time = time.time()
        for sentences, tags in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
    return model

def train():
    device = torch.device("cpu")
    torch.set_num_threads(4)
    print(f"Using {cpu_count()} CPU cores")

    data_file = "data/train1000.txt"
    char2idx_path = "data/char2idx.pkl.gz"
    tag2idx_path = "data/tag2idx.pkl.gz"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training file not found: {data_file}")

    print("Building vocab...")
    start_time = time.time()
    char2idx, tag2idx, idx2tag = build_vocab_parallel(data_file, num_processes=4)
    print(f"Vocab size: {len(char2idx)}, Tag size: {len(tag2idx)}, Time: {time.time() - start_time:.2f}s")

    print("Saving vocab...")
    save_vocab(char2idx, tag2idx, char2idx_path, tag2idx_path)

    model = BiLSTM_CRF(len(char2idx), len(tag2idx), embedding_dim=100, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("Starting training...")
    chunk_size = 1000000
    for i, chunk in enumerate(load_data_chunk(data_file, chunk_size=chunk_size)):
        print(f"Training chunk {i + 1} with {len(chunk)} samples...")
        start_time = time.time()
        model = train_on_chunk(chunk, model, optimizer, device, char2idx, tag2idx)
        print(f"Chunk {i + 1} trained in {time.time() - start_time:.2f}s")
        del chunk
        gc.collect()
        print(f"Memory usage: {psutil.virtual_memory().percent}%")

    print("Saving model...")
    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    train()