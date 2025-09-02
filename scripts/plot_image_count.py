from utils import *
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(args):
    docs = read_json(args.json_path)
    img_counts = []

    for doc in tqdm(docs, desc="Processing documents"):
        count = len(doc.get("images", []))
        img_counts.append(count)
    
    plt.figure(figsize=(8, 5))
    plt.hist(img_counts, bins=range(0, max(img_counts) + 2), edgecolor="black", align="left")
    plt.title("Number of Text per wiki Document")
    plt.xlabel("Image Count")
    plt.ylabel("Number of Documents")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"Saved histogram to {args.out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON or JSONL file")
    parser.add_argument("--out_path", type=str, default="image_count_hist.png", help="Output image path")
    main(parser.parse_args())
