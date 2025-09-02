from utils import read_json, save_json
import os

def merge_dataset(wiki_path, slide_path, arxiv_path):
    # Read JSON files for each domain
    wiki_json   = read_json(wiki_path)
    slides_json = read_json(slide_path)
    arxiv_json  = read_json(arxiv_path)

    def prefix_images(data_list, domain):
        """
        Prepend the domain name to each image filename in the 'images' list.
        - data_list: list of dicts, each dict may have an 'images' key
        - domain: string to prefix (e.g. 'wiki', 'slides', 'arxiv')
        """
        for item in data_list:
            # Retrieve the images list, defaulting to an empty list if missing
            images = item.get("images", [])
            new_images = []
            for img in images:
                # Join the domain folder with the image filename
                new_images.append(os.path.join(domain, img))
            # Update the item's images field with the prefixed paths
            item["images"] = new_images
            # Add a domain field
            item["domain"] = domain

    # Apply the prefixing function to each domain's data
    prefix_images(wiki_json,   "wiki")
    prefix_images(slides_json, "slides")
    prefix_images(arxiv_json,  "arxiv")

    # Concatenate all domain datasets into one list
    return wiki_json + slides_json + arxiv_json

if __name__ == "__main__":
    # Define paths to the domain-specific JSON files
    wiki_path   = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/wiki_json/6_clean_empty.json"
    slides_path = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/slide_json/7_delete_not_valid.json"
    arxiv_path  = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/arxiv_json/4_generated_query_split.json"

    # Merge datasets and adjust image paths
    full_list = merge_dataset(wiki_path, slides_path, arxiv_path)
    # Save the combined dataset to a new JSON file
    save_json(full_list, "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/full_set.json")
