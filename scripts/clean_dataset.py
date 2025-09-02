from utils import *

def main() -> None:
    src  = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/wiki_json/5_generate_combined.json"
    dest = "/volume/ailab4sci/users/shuyue/swwu/docmmir/data/json/wiki_json/6_clean_empty.json"

    raw_data = read_json(src)
    cleaned_data = []
    total_blanks = 0
    total_removed_docs = 0

    for obj in raw_data:
        texts = obj.get("texts", [])

        if not isinstance(texts, list):
            texts = [texts]

        # strip + remove empty strings
        cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        total_blanks += len(texts) - len(cleaned)

        # if there's no valid text, skip this doc
        if not cleaned:
            total_removed_docs += 1
            continue

        obj["texts"] = cleaned
        cleaned_data.append(obj)

    print(f"Removed {total_blanks} blank strings.")
    print(f"Removed {total_removed_docs} docs with no valid text (from {len(raw_data)} total).")
    print(f"Saving cleaned {len(cleaned_data)} docs → {dest}")
    save_json(cleaned_data, dest)
    class_stats = {}

    for obj in cleaned_data:
        cls = obj.get("class", "unknown")
        if cls in class_stats:
            class_stats[cls] += 1
        else:
            class_stats[cls] = 1

    print("\nClass Distribution:")
    for cls in sorted(class_stats):
        print(f"  {cls}: {class_stats[cls]}")
    
if __name__ == "__main__":
    main()