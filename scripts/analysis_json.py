import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BEE-spoke-data/cl100k_base")

# 1. 读取本地 JSON 文件
with open('/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/training_set/wiki.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Split the dataset based on the 'class' field
train_data = [entry for entry in data if entry.get('class') == 'train']
val_data = [entry for entry in data if entry.get('class') == 'valid']
test_data = [entry for entry in data if entry.get('class') == 'test']
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
# 2. 初始化统计容器
image_counts = []
text_lengths = []
text_page_lengths = []
blank_pages = 0
garbled_pages = 0
query_lengths = []
query_keywords = []

# 正则
math_formula_pattern = re.compile(r'(\$.*?\$|\\\(|\\\)|\\\[|\\\]|\\begin\{.*?\}|\\end\{.*?\})')
long_special_pattern = re.compile(r'[^\w\s]{10,}')  # 连续10个非字母数字空格符号

# 总乱码段落收集器
all_garbled_segments = []
garbled_samples = []

# 4. 遍历每一条数据
for item in tqdm(data, desc="Processing documents"):
    images = item.get("images", [])
    texts = item.get("texts", [])
    query = item.get("query", "")

    image_counts.append(len(images))
    doc_token_count = sum(len(tokenizer.encode(t)) for t in texts)
    text_lengths.append(doc_token_count)
    text_page_lengths.extend([len(tokenizer.encode(t)) for t in texts])
    blank_pages += sum(1 for t in texts if t.strip() == "")

    garbled_texts = []
    garbled_char_count = 0
    total_char_count = 0

    for t in texts:
        t_stripped = t.strip()
        t_tokenized = tokenizer.encode(t_stripped)
        t_len = len(t_tokenized)
        total_char_count += t_len

        if math_formula_pattern.search(t_stripped):
            continue

        garbled_spans = long_special_pattern.findall(t_stripped)
        substring_garbled_length = sum(len(s) for s in garbled_spans)

        if t_len > 0 and substring_garbled_length / t_len >= 0.5:
            garbled_texts.append(t)
            garbled_char_count += t_len

            all_garbled_segments.append({
                "query": query,
                "text": t,
                "substring_ratio": round(substring_garbled_length / t_len, 2)
            })

    if total_char_count > 0:
        garbled_ratio = garbled_char_count / total_char_count
        garbled_pages += len(garbled_texts)

        if garbled_ratio >= 0.5:
            garbled_doc = {
                "query": query,
                "garbled_ratio": garbled_ratio,
                "garbled_char_len": garbled_char_count,
                "total_char_len": total_char_count,
                "garbled_texts": garbled_texts,
                "all_texts": texts,
            }
            garbled_samples.append(garbled_doc)

    query_token_count = len(tokenizer.encode(query))
    query_lengths.append(query_token_count)

# 5. 打印统计摘要
print("=== 基本统计 ===")
print(f"总文档数：{len(data)}")
print(f"平均每个文档的图像数：{np.mean(image_counts):.2f}")
print(f"图像数量范围：{min(image_counts)} - {max(image_counts)}")
print(f"图片列表为空的文档数量：{sum(1 for c in image_counts if c == 0)}")
print(f"平均每个文档的总词数：{np.mean(text_lengths):.2f}")
print(f"平均每页的词数：{np.mean(text_page_lengths):.2f}")
print(f"空白页数量：{blank_pages}")
print(f"疑似乱码页数量：{garbled_pages}")
print(f"平均 Query 长度：{np.mean(query_lengths):.2f}")