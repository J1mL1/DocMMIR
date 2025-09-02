import random
from typing import List

from utils import read_json, save_json

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


if __name__ == "__main__":
    # Path to your JSON file
    input_file = "../data/normal_wiki_1M_remove_unreadable.json"
    output_file = "../data/normal_wiki_1M_remove_unreadable_add_split.json"

    val_size = 15000
    test_size = 15000

    # Load the data
    data = read_json(input_file)

    # Assign splits
    updated_data = assign_splits(data, val_size, test_size)

    # Save the updated data to a new file
    save_json(updated_data, output_file)

    print(f"Data successfully split into train, valid, and test sets!")
    print(f"Train: {sum(1 for d in updated_data if d['class'] == 'train')} entries")
    print(f"Valid: {sum(1 for d in updated_data if d['class'] == 'valid')} entries")
    print(f"Test: {sum(1 for d in updated_data if d['class'] == 'test')} entries")
