import json


def split_json_array(input_path, output_path1, output_path2):
    """
    Splits a JSON array from the input file into two halves and writes each half to separate files.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    half_index = len(data) // 2
    first_half, second_half = data[:half_index], data[half_index:]

    with open(output_path1, 'w', encoding='utf-8') as f:
        json.dump(first_half, f, indent=2, ensure_ascii=False)

    with open(output_path2, 'w', encoding='utf-8') as f:
        json.dump(second_half, f, indent=2, ensure_ascii=False)

    print(f"JSON array successfully split into '{output_path1}' and '{output_path2}'.")


def combine_json_arrays(input_paths, output_path):
    """
    Combines two JSON arrays from separate files into a single JSON array and writes it to a new file.
    """
    combined_data = []
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            part_json = json.load(f)
            combined_data += part_json

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"JSON arrays successfully combined into '{output_path}'.")


# Example usage:
if __name__ == "__main__":
    # split_json_array(
    #     '/map-vepfs/siwei/ziruili/DocMMIR/data/wiki_json/3_cleaned.json',
    #     '/map-vepfs/siwei/ziruili/DocMMIR/data/wiki_json/4_first_half.json',
    #     '/map-vepfs/siwei/ziruili/DocMMIR/data/wiki_json/4_second_half.json'
    # )

    combine_json_arrays(
        ['/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_1.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_2.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_3.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_4.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_5.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_part_6.json',
         '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query_final.json'],
        '/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/3_generated_query.json'
    )
