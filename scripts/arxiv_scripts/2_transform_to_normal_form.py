from utils import read_json, save_json


def transform_data(input_json_path, output_json_path):
    """
    @return:
    [
      # one wiki page/ one slide
      {
        "images": ["image_path_1", "image_path_2"],
        "texts": ["text content 1", "text content 2"],
        "query": "query content",
        "screenshot": "screenshot_path"
      },
      ...
    ]
    """
    # Read input JSON data
    input_data = read_json(input_json_path)

    # Output data in desired format
    output_data = []
    count = 0
    # Transform the data
    for page in input_data:
        entry = {}

        # Extract images
        entry["images"] = page.get("imgs", "")
        if not entry["images"]:
            continue

        # Extract texts
        texts = page.get("texts", "")
        abstract = page.get("abstract", "")
        entry["texts"] = [abstract] + texts

        # Extract title as the query
        entry["query"] = page.get("title", "")
        if not entry["query"]:
            continue

        # Extract title as the query
        entry["id"] = page.get("id", "")
        # Append transformed entry to output list
        output_data.append(entry)

        count += 1
        if count % 2000 == 0:
            print(f'processing {count} entry')

    print(f'Total {count} entries processed')
    # Save transformed data to JSON file
    save_json(output_data, output_json_path)


def main():
    # Define the input file path
    input_file = '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/1_arxiv_data.json'
    output_path = '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/2_normalised_arxiv_data.json'
    # Read the JSON data
    transform_data(input_file, output_path)


if __name__ == "__main__":
    main()
