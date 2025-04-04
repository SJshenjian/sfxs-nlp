# utils/data_loader.py
import json
from multiprocessing import Pool
import pickle
import gzip
from collections import Counter

# 分块加载数据
def load_data_chunk(file_path, chunk_size=10000):
    """
    分块加载训练数据，从文件中读取地址和对应的标签。

    Args:
        file_path (str): 训练数据文件路径。
        chunk_size (int): 每个数据块的大小。

    Yields:
        list: 包含 (sentence, tags) 元组的列表块。
    """
    with open(file_path, encoding="utf-8") as f:
        sentence, tags = [], []
        chunk = []
        for line in f:
            line = line.rstrip()  # 移除尾部换行符
            if not line:  # 空行表示一个地址结束
                if sentence:
                    chunk.append((sentence, tags))
                    sentence, tags = [], []
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            else:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    print(f"警告: 跳过格式错误的行: {line}")
                    continue
                char, tag = parts
                sentence.append(char)  # 保留空格字符
                tags.append(tag.strip())  # 去除标签中的空格
        if sentence:  # 处理最后一个地址
            chunk.append((sentence, tags))
        if chunk:
            yield chunk

# 原始加载函数（兼容性保留）
def load_data(file_path):
    """
    加载训练数据，从文件中读取地址和对应的标签。

    Args:
        file_path (str): 训练数据文件路径。

    Returns:
        list: 包含 (sentence, tags) 元组的列表。
    """
    training_data = []
    for chunk in load_data_chunk(file_path):
        training_data.extend(chunk)
    return training_data

# 并行处理数据块
def process_chunk(chunk):
    """
    处理单个数据块，返回字符集合。

    Args:
        chunk (list): 包含 (sentence, tags) 元组的列表。

    Returns:
        set: 字符集合。
    """
    char_set = set()
    for sentence, _ in chunk:
        char_set.update(sentence)
    return char_set

# 并行构建词汇表
def build_vocab_parallel(file_path, num_processes=4):
    """
    并行构建字符表和标签表。

    Args:
        file_path (str): 训练数据文件路径。
        num_processes (int): 并行进程数。

    Returns:
        tuple: (char2idx, tag2idx, idx2tag)
            - char2idx: 字符到索引的映射字典。
            - tag2idx: 标签到索引的映射字典。
            - idx2tag: 索引到标签的映射字典。
    """
    # 初始化字符表
    char2idx = {"<PAD>": 0, "<UNK>": 1}

    # 初始化标签表（固定标签集，无需并行计算）
    tag2idx = {
        "B-PROV": 0, "I-PROV": 1, "E-PROV": 2,
        "B-CITY": 3, "I-CITY": 4, "E-CITY": 5,
        "B-DISTRICT": 6, "I-DISTRICT": 7, "E-DISTRICT": 8,
        "B-TOWN": 9, "I-TOWN": 10, "E-TOWN": 11,
        "B-EXTRA": 12, "I-EXTRA": 13, "E-EXTRA": 14,
        "START": 15
    }
    idx2tag = {v: k for k, v in tag2idx.items()}

    # 并行构建字符表
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, load_data_chunk(file_path))

    # 合并所有字符集合
    all_chars = set().union(*results)
    for char in all_chars:
        if char not in char2idx:
            char2idx[char] = len(char2idx)

    return char2idx, tag2idx, idx2tag

# 原始构建词汇表（兼容性保留）
def build_vocab(training_data):
    """
    构建字符表和标签表。

    Args:
        training_data (list): 训练数据，包含 (sentence, tags) 元组的列表。

    Returns:
        tuple: (char2idx, tag2idx, idx2tag)
    """
    # 初始化字符表
    char2idx = {"<PAD>": 0, "<UNK>": 1}

    # 初始化标签表
    tag2idx = {
        "B-PROV": 0, "I-PROV": 1, "E-PROV": 2,
        "B-CITY": 3, "I-CITY": 4, "E-CITY": 5,
        "B-DISTRICT": 6, "I-DISTRICT": 7, "E-DISTRICT": 8,
        "B-TOWN": 9, "I-TOWN": 10, "E-TOWN": 11,
        "B-EXTRA": 12, "I-EXTRA": 13, "E-EXTRA": 14,
        "START": 15
    }
    idx2tag = {v: k for k, v in tag2idx.items()}

    # 构建字符表
    for sentence, _ in training_data:
        for char in sentence:
            if char not in char2idx:
                char2idx[char] = len(char2idx)

    return char2idx, tag2idx, idx2tag

# 保存函数（压缩格式）
def save_vocab(char2idx, tag2idx, char2idx_path, tag2idx_path):
    """
    保存字符表和标签表到压缩文件。

    Args:
        char2idx (dict): 字符到索引的映射字典。
        tag2idx (dict): 标签到索引的映射字典。
        char2idx_path (str): 字符表保存路径。
        tag2idx_path (str): 标签表保存路径。
    """
    # 分块保存 char2idx
    with gzip.open(char2idx_path, "wb", compresslevel=6) as f:
        pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_size = 10000
        items = list(char2idx.items())
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            pickler.dump(chunk)
            print(f"Saved chunk {i // chunk_size + 1} of char2idx")
        pickler.dump(None)  # 结束标记
        print("Saved all chunks of char2idx")

    # tag2idx 直接写入
    with gzip.open(tag2idx_path, "wb", compresslevel=6) as f:
        pickle.dump(tag2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

# 加载词汇表
def load_vocab(char2idx_path, tag2idx_path):
    """
    从压缩文件加载字符表和标签表。

    Args:
        char2idx_path (str): 字符表文件路径。
        tag2idx_path (str): 标签表文件路径。

    Returns:
        tuple: (char2idx, tag2idx, idx2tag)
    """
    char2idx = {}
    with gzip.open(char2idx_path, "rb") as f:
        unpickler = pickle.Unpickler(f)
        while True:
            chunk = unpickler.load()
            if chunk is None:
                break
            char2idx.update(chunk)
    with gzip.open(tag2idx_path, "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return char2idx, tag2idx, idx2tag

# 主流程示例
if __name__ == "__main__":
    file_path = "data/train.txt"
    char2idx_path = "data/char2idx.pkl.gz"
    tag2idx_path = "data/tag2idx.pkl.gz"

    print("Loading data...")
    training_data = load_data(file_path)
    print(f"Loaded {len(training_data)} samples")

    print("Building vocab (parallel)...")
    char2idx, tag2idx, idx2tag = build_vocab_parallel(file_path, num_processes=4)
    print(f"Vocab size: chars={len(char2idx)}, tags={len(tag2idx)}")

    print("Saving vocab...")
    save_vocab(char2idx, tag2idx, char2idx_path, tag2idx_path)
    print("Done!")