import sys
from Wikipages_scripts.utils import *

import json
import os
import argparse


def read_txt(file_path):
    """Load the list of unreadable files from a text file."""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Unreadable files list not found at {file_path}")
        return []


def filter_json_data(data, unreadable_files):
    """Remove unreadable image paths from each entry and delete entries with empty 'images' lists."""
    filtered_data = []

    for item in data:

        removed_images = [img for img in item.get("images", []) if img in unreadable_files]

        # Remove unreadable images from the 'images' list
        item["images"] = [img for img in item.get("images", []) if img not in unreadable_files]

        if removed_images:
            print(f"Removed images: {removed_images} for query '{item.get('query', 'unknown')}'")

        # Only keep the entry if there are remaining images
        if item["images"]:
            filtered_data.append(item)
        else:
            # Log entry deletion if all images were removed
            print(f"Deleted entry with query '{item.get('query', 'unknown')}' due to empty 'images' list.")

    return filtered_data


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Filter JSON data to remove unreadable images.")
    parser.add_argument('--json_path', required=True, help="Path to the JSON file.")
    parser.add_argument('--unreadable_files_path', required=True, help="Path to the unreadable files list.")
    parser.add_argument('--output_path', required=True, help="Path for the output filtered JSON file.")

    args = parser.parse_args()

    # Load unreadable files and JSON data
    unreadable_files = read_txt(args.unreadable_files_path)
    data = read_json(args.json_path)

    # Filter JSON data
    filtered_data = filter_json_data(data, unreadable_files)

    # Save filtered JSON data
    save_json(filtered_data, args.output_path)

if __name__ == "__main__":
    main()
