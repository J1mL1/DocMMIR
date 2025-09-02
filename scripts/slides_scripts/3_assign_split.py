from utils import assign_splits, read_json, save_json

if __name__ == "__main__":
    input_file = "/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/3_generated_query_all.json"
    output_file = "/map-vepfs/siwei/ziruili/DocMMIR/data/json/slide_json/3_generated_query_split.json"

    val_size = 1500
    test_size = 1500

    # Load the data
    data = read_json(input_file)

    # Assign splits
    updated_data = assign_splits(data, val_size, test_size)

    # Save the updated data to a new file
    save_json(updated_data, output_file)

    print(f"Data successfully split into train, valid, and test sets!")
    print(f"Train: {sum(1 for d in updated_data if d['class'] == 'train')} entries")
    print(f"Valid: {sum(1 for d in updated_data if d['class'] == 'valid')} entries")
    print(f"Test: {sum(1 for d in updated_data if d['class'] == 'test')} entries")
