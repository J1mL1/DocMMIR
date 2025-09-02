from utils import read_json, save_json
import os


def change_image_dir(json_file):
    """
    Change the image directory of a slide deck.
    @args:
        - json_file: JSON file with slides.
    @return:
        - changed_json_file: JSON file with changed image directory.
    """
    # slide bath path
    base = "/mnt/data1_HDD_14TB/yizhi/SlidesShare_1M"
    json_data = json_file['vs']
    for slide_decks in json_data:
        slide_images = slide_decks['slide_images']
        for slide_image in slide_images:
            image_path = slide_image['image_path']
            real_path = os.path.relpath(image_path, base)
            slide_image['image_path'] = real_path

    return json_data


if __name__ == '__main__':
    input_path = '/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/slide_metadata.json'
    output_path = '/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/1_replace_image_path.json'

    json_file = read_json(input_path)
    changed_json_file = change_image_dir(json_file)
    save_json(changed_json_file, output_path)
    print('Done change image path')

