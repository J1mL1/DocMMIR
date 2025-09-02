import torch
from tqdm import tqdm
import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Wikipages_scripts import utils
from src.models.clip_model import ImageEncoder


def preprocess_images(data, split_name, save_dir, encoder, batch_size=5000):
    """
    Preprocess images and save them as .pt files, while generating a simplified index mapping.

    @args:
        data (list): List of JSON objects containing image paths and other data.
        split_name (str): Dataset split name ('train', 'valid', 'test').
        save_dir (str): Directory to save preprocessed images and index mapping.
        encoder (ImageEncoder): Image encoder for preprocessing.
        batch_size (int): Number of images per saved .pt file.

    @returns:
        None
    """
    split_save_dir = os.path.join(save_dir, split_name)
    os.makedirs(split_save_dir, exist_ok=True)

    image_tensors = []
    index_mapping = []  # Records each entry's global image index range
    global_index = 0  # Tracks global image index
    batch_count = 0

    for entry_idx, entry in tqdm(enumerate(data), total=len(data), desc=f"Processing {split_name} images"):
        # Record the start of this entry's image indices
        entry_start_index = global_index

        for img_path in entry["images"]:
            try:
                # Preprocess image
                image_tensor = encoder.preprocess_image(img_path)
                image_tensors.append(image_tensor)

                global_index += 1  # Increment global index
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

            # Save batch when full
            if len(image_tensors) >= batch_size:
                save_batch(image_tensors, split_save_dir, split_name, batch_count)
                image_tensors = []
                batch_count += 1

        # Record the end of this entry's image indices, save 'query' for querying
        entry_end_index = global_index - 1
        index_mapping.append({"query": entry['query'], "start_index": entry_start_index, "end_index": entry_end_index})

    # Save remaining images
    if image_tensors:
        save_batch(image_tensors, split_save_dir, split_name, batch_count)

    # Save index mapping to JSON
    index_mapping_path = os.path.join(split_save_dir, f"{split_name}_index_mapping.json")
    utils.save_json(index_mapping, index_mapping_path)
    print(f"Saved index mapping for {split_name} to {index_mapping_path}")


def save_batch(image_tensors, save_dir, split_name, batch_count):
    """
    Save a batch of preprocessed images to a .pt file.

    @args:
        image_tensors (list): List of preprocessed image tensors.
        save_dir (str): Directory to save the .pt file.
        split_name (str): Dataset split name ('train', 'valid', 'test').
        batch_count (int): Batch number for file naming.

    @returns:
        None
    """
    save_path = os.path.join(save_dir, f"{split_name}_image_batch_{batch_count}.pt")
    torch.save(torch.cat(image_tensors), save_path)
    print(f"Saved {len(image_tensors)} tensors to {save_path}")


def load_split(json_path, data_class):
    """
    Load a specific split ('train', 'valid', or 'test') from a JSON file.

    @args:
        json_path (str): Path to the JSON file.
        data_class (str): Class of data to load ('train', 'valid', 'test').

    @returns:
        list: List of JSON objects belonging to the specified split.
    """
    data = utils.read_json(json_path)

    # Filter data by class
    split_data = [entry for entry in data if entry.get("class") == data_class]
    print(f"Loaded {len(split_data)} entries for split '{data_class}'.")
    return split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess images and save them as .pt files for train/valid/test splits")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON file containing image paths and split info")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the preprocessed images")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Model name for ImageEncoder")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="Pretrained weights for ImageEncoder")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for saving .pt files")

    args = parser.parse_args()

    # Load ImageEncoder
    encoder = ImageEncoder(model_name=args.model_name, pretrained=args.pretrained)

    # Load and preprocess splits
    for split in ["train", "valid", "test"]:
        split_data = load_split(args.json_path, split)
        preprocess_images(split_data, split, args.save_dir, encoder, batch_size=args.batch_size)
