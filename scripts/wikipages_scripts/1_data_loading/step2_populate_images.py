import os
import argparse
from utils import read_json, save_json, download_image
from tqdm import tqdm

def download_images_from_json(json_file, save_dir):
    """
    Download all images listed in the JSON file to the specified directory.
    """
    # Read JSON data
    data = read_json(json_file)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    total_images = sum(len(page.get('images', [])) for page in data)
    
    with tqdm(total=total_images, desc="Downloading images") as pbar:
        # Extract and download images
        for page in data:
            title = page.get('title', 'Untitled')
            for image_info in page.get('images', []):
                image_url = image_info.get('image_url')
                section_id = image_info.get('section_id', -1)
                # Find the section title using the section_id
                section_title = next(
                    (text['section_title'] for text in page.get('texts', []) if text['section_id'] == section_id),
                    'No_Section')
                file_title = f"{title}_{section_title}"
                if image_url:
                    image_path = download_image(image_url, file_title, save_dir)

                    if image_path:
                        image_info['image_path'] = image_path
                
                pbar.update(1)

    # Save the updated JSON data back to the file
    save_json(data, json_file)


def main(json_file, save_dir):
    # Download images
    download_images_from_json(json_file, save_dir)

    print(f"All images have been downloaded to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download images in wiki pages ')
    parser.add_argument('input_file', type=str, help='The input JSON file path')
    parser.add_argument('output_dir', type=str, help='The output directory for images')
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
