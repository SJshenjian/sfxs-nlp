import torch
from models.bilstm_crf import BiLSTM_CRF
from utils.data_loader import load_vocab
from utils.formatter import extract_fields, format_address

def predict(address, model_path="model.pt"):
    char2idx, tag2idx, idx2tag = load_vocab("data/char2idx.pkl.gz", "data/tag2idx.pkl.gz")
    model = BiLSTM_CRF(len(char2idx), len(tag2idx), embedding_dim=100, hidden_dim=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        char_ids = torch.tensor([char2idx.get(char, char2idx["<UNK>"]) for char in address])
        char_ids = char_ids.unsqueeze(0)  # 添加批维度: (1, seq_len)
        score, tags = model(char_ids)     # (1,), (1, seq_len)
        tags = tags[0].tolist()           # 取第一条数据: (seq_len,)
        tags = [idx2tag[tag] for tag in tags]

    tokens = []
    current_token = ""
    for char, tag in zip(address, tags):
        if tag.startswith("B"):
            if current_token:
                tokens.append((current_token, tags[len(tokens)*3].split("-")[1]))
            current_token = char
        elif tag.startswith("I"):
            current_token += char
        elif tag.startswith("E"):
            current_token += char
            tokens.append((current_token, tag.split("-")[1]))
            current_token = ""
    if current_token:
        tokens.append((current_token, tags[-1].split("-")[1]))

    structured = extract_fields(tokens)
    return format_address(structured)

if __name__ == "__main__":
    address = "锡山区锡北镇八士菜场"
    print(f"Address: {address}")
    result = predict(address)
    print(result)