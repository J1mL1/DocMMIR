import json
import os
import argparse

def read_json_file(file_path):
    """
    Read a JSON file and return its content as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None

def calculate_text_length(texts):
    """
    Calculate the total length of the texts in the wiki page.
    """
    total_length = sum(len(text['content'][0]) for text in texts if text['content'])
    return total_length

def get_wiki_page_by_title(data, title):
    """
    Find and return the wiki page by title.
    """
    for page in data:
        if page.get('title') == title:
            return page
    return None

def process_screenshots_and_texts(json_data, screenshot_dir):
    results = []

    for filename in os.listdir(screenshot_dir):
        if filename.endswith('.png'):
            title = os.path.splitext(filename)[0]
            page = get_wiki_page_by_title(json_data, title)

            if page:
                texts = page.get('texts', [])
                total_text_length = calculate_text_length(texts)
                screenshot_path = os.path.join(screenshot_dir, filename)
                image_size = os.path.getsize(screenshot_path)

                results.append((title, total_text_length, image_size))
            else:
                print(f"Page with title {title} not found in JSON data.")

    return results

def main(json_file, screenshot_dir):
    json_data = read_json_file(json_file)

    if json_data is not None:
        results = process_screenshots_and_texts(json_data, screenshot_dir)

        for result in results:
            print(f"Title: {result[0]}, Text Length: {result[1]}, Screenshot Size: {result[2]} bytes")
    else:
        print("Failed to read data from the JSON file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare screenshot size with the total text length of the Wikipedia page.')
    parser.add_argument('json_file', type=str, help='The input JSON file path')
    parser.add_argument('screenshot_dir', type=str, help='The input screenshot directory path')
    args = parser.parse_args()

    main(args.json_file, args.screenshot_dir)
