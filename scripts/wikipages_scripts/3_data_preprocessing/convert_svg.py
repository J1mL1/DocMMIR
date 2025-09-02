import os
import utils


def process_svg(data):
    """
    Process the JSON data to convert SVG images to PNG and update image paths.
    """
    for entry in data:
        if "images" in entry:
            for image in entry["images"]:
                image_path = image.get("image_path", "")
                if not image_path:
                    continue

                # Get the file extension in lowercase
                ext = os.path.splitext(image_path)[1].lower()

                if ext == '.svg':
                    # Create a new file path with .png extension
                    new_image_path = image_path.replace('.svg', '.png')

                    print(f"Converting {image_path} to {new_image_path}")

                    try:
                        # Convert the SVG to PNG and update the image path
                        utils.convert_svg_to_png(image_path, new_image_path)
                        image['image_path'] = new_image_path
                    except Exception as e:
                        print(f"Failed to convert {image_path}: {e}")
                        # If conversion fails, set the image_path to an empty string
                        image['image_path'] = ""

    return data


if __name__ == '__main__':
    json_file_path = "data/sample.json"

    # Read JSON data
    data = utils.read_json(json_file_path)

    # Process the JSON data (convert SVG to PNG)
    updated_data = process_svg(data)

    # Save the updated JSON data
    utils.save_json(updated_data, json_file_path)
