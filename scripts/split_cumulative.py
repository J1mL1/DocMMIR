"""
Split a JSON file into cumulative TRAIN subsets
and append the full VAL and TEST sets to each output.

Example:
    python split_cumulative.py \
        --input  data/domain/clean.json \
        --output_dir  data/domain/splits
"""

import os
import random
import argparse
from utils import read_json, save_json


def split_cumulative(
    input_path: str,
    output_dir: str,
    fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 42,
):
    data = read_json(input_path)

    train_data = [ex for ex in data if str(ex.get("class", "")).lower() == "train"]
    val_data   = [ex for ex in data if str(ex.get("class", "")).lower() == "valid"]
    test_data  = [ex for ex in data if str(ex.get("class", "")).lower() == "test"]

    if not train_data:
        raise ValueError("No training examples with class=='train' found!")

    print(f" Train {len(train_data)}  |  Val {len(val_data)}  |  Test {len(test_data)}")

    random.Random(seed).shuffle(train_data)

    os.makedirs(output_dir, exist_ok=True)

    for i, frac in enumerate(fractions, 1):
        k = int(round(frac * len(train_data)))
        subset = train_data[:k] + val_data + test_data      

        fname = f"split_{i}_{int(frac*100)}.json"
        save_json(subset, os.path.join(output_dir, fname))

        print(f"{fname}: {k} train + {len(val_data)} val + {len(test_data)} test "
              f"= {len(subset)} total  ({frac:.0%} train)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cleaned JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory for split files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_cumulative(args.input, args.output_dir, seed=args.seed)
