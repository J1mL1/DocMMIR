import os
import time
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import read_json, save_json, hash_string
from tqdm import tqdm

def take_long_screenshot(url, output_path):
    """
    Take a long screenshot of the given URL and save it to the specified output path.
    """
    # Configure Selenium to use headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.set_page_load_timeout(60)
        driver.get(url)
        time.sleep(2)  # Let the page load

        content_width = driver.execute_script("return document.body.scrollWidth")
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(content_width, total_height)
        time.sleep(2)  # Let the resize take effect

        driver.save_screenshot(output_path)
        print(f"Screenshot saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error taking screenshot for {url}: {e}")
        return None
    finally:
        driver.quit()


def process_screenshot(data, screenshots_dir):
    """
    Generate screenshots for every wiki page and save the screenshot path back to the JSON data.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(
                take_long_screenshot,
                page.get('url'),
                os.path.join(screenshots_dir, f"{hash_string(page.get('title', 'Untitled'))}.png")
            ): page
            for page in data if page.get('url')
        }
        
        with tqdm(total=len(future_to_url), desc="Processing Screenshots", unit="screenshot") as pbar:
            for future in as_completed(future_to_url):
                page = future_to_url[future]
                try:
                    screenshot_path = future.result()
                    if screenshot_path:
                        page['screenshot_path'] = screenshot_path
                except Exception as e:
                    print(f"Error processing page {page.get('title', 'Untitled')}: {e}")
                finally:
                    # Update the progress bar
                    pbar.update(1)

    return data


def main(input_file, output_dir):
    # Read the JSON data
    data = read_json(input_file)

    if data is not None:
        # Ensure the screenshots directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        updated_data = process_screenshot(data, output_dir)

        # Save the updated data back to the JSON file
        save_json(updated_data,'./final_result.json')
    else:
        print("Failed to read data from the JSON file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download wiki pages into screenshot.')
    parser.add_argument('input_file', type=str, help='The input JSON file path')
    parser.add_argument('output_dir', type=str, help='The output directory for screenshots')
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
