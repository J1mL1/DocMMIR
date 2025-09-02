from utils import read_json, save_json


def transform_to_normal_form(json_file):
    """
    Transforms raw slide JSON into a flat normal form.

    @args:
        - json_file: list of dicts, each with keys:
            - author
            - language
            - recognized_texts: list of { slide_idx, texts }
            - slide_images:      list of { slide_idx, image_path, height }
            - title
    @return:
        - new_list: list of dicts, each with keys:
            - title: str
            - images: list of image_path (sorted by slide_idx)
            - texts:  flat list of all text strings (in slide & line order)
    """
    new_list = []

    for entry in json_file:
        title = entry.get("title", "")

        # sort slides by index
        texts_entries = sorted(
            entry.get("recognized_texts", []),
            key=lambda x: int(x["slide_idx"])
        )
        image_entries = sorted(
            entry.get("slide_images", []),
            key=lambda x: int(x["slide_idx"])
        )

        # pull out image paths
        images_list = [i["image_path"] for i in image_entries]

        # flatten all text lines into one list
        texts_list = []
        for t in texts_entries:
            lines = t.get("texts", [])
            # append each line as its own entry
            for line in lines:
                texts_list.append(line.strip())

        new_list.append({
            "title":  title,
            "images": images_list,
            "texts":  texts_list
        })

    return new_list


if __name__ == "__main__":
    input_path = '/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/1_replace_image_path.json'
    output_path = '/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/2_normalised.json'

    json_data = read_json(input_path)
    normalized = transform_to_normal_form(json_data)

    print(f"Transformed {len(normalized)} numbers of slides")
    save_json(normalized, output_path)
