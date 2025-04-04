import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from models.bilstm_crf import BiLSTM_CRF
from utils.data_loader import load_data, build_vocab, save_vocab
from functools import partial


class NERDataset(Dataset):
    def __init__(self, training_data, char2idx, tag2idx):
        self.data = []
        for sentence, tags in training_data:
            char_ids = [char2idx.get(char, char2idx["<UNK>"]) for char in sentence]
            tag_ids = [tag2idx.get(tag, tag2idx["<PAD>"]) for tag in tags]
            self.data.append((torch.tensor(char_ids), torch.tensor(tag_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, char2idx, tag2idx):
    sentences, tags = zip(*batch)
    sentences = pad_sequence(sentences, batch_first=True, padding_value=char2idx["<PAD>"])
    tags = pad_sequence(tags, batch_first=True, padding_value=tag2idx["<PAD>"])
    return sentences, tags


def train():
    print("Loading data...")
    training_data = load_data("data/train.txt")
    print(f"Loaded {len(training_data)} samples")

    print("Building vocab...")
    char2idx, tag2idx, idx2tag = build_vocab(training_data)
    if "<PAD>" not in char2idx:
        char2idx["<PAD>"] = len(char2idx)
    if "<UNK>" not in char2idx:
        char2idx["<UNK>"] = len(char2idx)
    if "<PAD>" not in tag2idx:
        tag2idx["<PAD>"] = len(tag2idx)

    print("Saving vocab...")
    save_vocab(char2idx, tag2idx, "data/char2idx.pkl.gz", "data/tag2idx.pkl.gz")

    print("Preprocessing data...")
    dataset = NERDataset(training_data, char2idx, tag2idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            collate_fn=partial(collate_fn, char2idx=char2idx, tag2idx=tag2idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(len(char2idx), len(tag2idx), embedding_dim=100, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Vocab size: {len(char2idx)}, Tag size: {len(tag2idx)}")
    print(f"Model embedding size: {model.embedding.num_embeddings}")

    print("Starting training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for i, (sentences, tags) in enumerate(dataloader):
            sentences, tags = sentences.to(device), tags.to(device)
            if i == 0 and epoch == 0:
                print(f"Batch 0: max sentence index: {sentences.max().item()}, max tag index: {tags.max().item()}")
                print(f"Batch 0 shape: sentences {sentences.shape}, tags {tags.shape}")
            model.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    train()