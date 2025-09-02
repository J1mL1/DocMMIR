import json
from typing import List
import random

def read_json(file_path):
    """
    Read JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """
    Save JSON data to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def assign_splits(data: List[dict], val_size: int, test_size: int, random_seed: int = 42) -> List[dict]:
    """
    Assigns a 'class' field (train/valid/test) to each data entry in a single JSON file.

    @args:
        data (list): List of data entries.
        val_size (int): Number of entries for validation set.
        test_size (int): Number of entries for test set.
        random_seed (int): Random seed for reproducibility.

    @returns:
        list: The updated data with 'class' field.
    """
    random.seed(random_seed)
    random.shuffle(data)

    # Assign 'valid', 'test', and 'train' splits
    for i, entry in enumerate(data):
        if i < val_size:
            entry["class"] = "valid"
        elif i < val_size + test_size:
            entry["class"] = "test"
        else:
            entry["class"] = "train"

    return data
