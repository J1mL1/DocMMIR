import json
import random
from utils import read_json, save_json

def main():
    path = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/slide_json/6_perplexity_clean.json" 
    out_path = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/slide_json/7_delete_not_valid.json" 
    data = read_json(path)


    valid_docs = [doc for doc in data if doc.get("valid_doc", []) == ["yes"]]

    print(f"Total not valid docs: {len(valid_docs)}")
    train_data = [ex for ex in valid_docs if str(ex.get("class", "")).lower() == "train"]
    val_data   = [ex for ex in valid_docs if str(ex.get("class", "")).lower() == "valid"]
    test_data  = [ex for ex in valid_docs if str(ex.get("class", "")).lower() == "test"]

    if not train_data:
        raise ValueError("No training examples with class=='train' found!")

    print(f" Train {len(train_data)}  |  Val {len(val_data)}  |  Test {len(test_data)}")
    
    print("\nSample 10 'not valid' texts:")
    for doc in random.sample(valid_docs, min(10, len(valid_docs))):
        print("-" * 40)
        for t in doc.get("texts", []):
            print(f"• {t}")
        print()
    save_json(valid_docs, out_path)
if __name__ == "__main__":
    main()
