import json
import os
import hashlib
import requests
from urllib.parse import urlparse
import cairosvg


def hash_string(input_string):
    """
    Generate a SHA-256 hash of the input string and return the first 10 characters.
    """
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()[:20]


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


def download_image(image_url, title, save_dir):
    """
    Download an image from a URL and save it to a directory with the hierarchy title as the filename.
    """
    try:
        headers = {'User-Agent': 'DocMMIR/0.0 (https://www.manchester.ac.uk/)'}

        # Extract image extension from URL
        parsed_url = urlparse(image_url)
        image_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
        image_ext = os.path.splitext(parsed_url.path)[1]

        # Store the hashed title
        hashed_title = hash_string(f'{title}_{image_name}')

        # Title for the filename
        image_name = f"{hashed_title}{image_ext}"

        image_path = os.path.join(save_dir, image_name)

        # Check if the file already exists
        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        # Download and save the image
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(image_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)

        print(f"Downloaded {image_url} to {image_path}")
        return image_path

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {image_url}: {e}")
        return None


def convert_svg_to_png(input_path, output_path):
    try:
        # Read the SVG data from the input file
        with open(input_path, 'rb') as svg_file:
            svg_data = svg_file.read()

        # Convert the SVG to PNG and save the new image
        cairosvg.svg2png(bytestring=svg_data, write_to=output_path)

    except EntitiesForbidden as e:
        print(f"Error: {e}. The SVG file {input_path} contains forbidden entities and cannot be processed.")

    except Exception as e:
        print(f"An error occurred during SVG conversion: {e}")
