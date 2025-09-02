import os
import re
import glob
import json
import pandas as pd

def extract_base_id_and_version(fname):
    """
    EXTRACT BASE ID & VERSION FROM file_name_index
    Example: "$2305.00009v2-Figure10-1.png" -> base_id="2305.00009", version=2
    """
    # Remove leading '$'
    fname = fname.lstrip('$')
    # Remove trailing '.png' or similar
    fname = re.sub(r'\.png$', '', fname, flags=re.IGNORECASE)
    # Split on '-' to isolate something like "2305.00009v2"
    parts = re.split(r'-', fname, maxsplit=1)
    main_part = parts[0]

    # Look for a pattern like "2305.00009v2"
    match = re.match(r'^(.+?)v(\d+)$', main_part)
    if match:
        base_id = match.group(1)       # e.g. "2305.00009"
        version_str = match.group(2)   # e.g. "2"
        version_num = int(version_str)
    else:
        # If there's no 'vN', default to version=1
        base_id = main_part
        version_num = 1

    return base_id, version_num

# -------------------------------------------------------------------------
# 1) PATHS AND SETUP
# -------------------------------------------------------------------------
parquet_dir = "/map-vepfs/siwei/ziruili/data/scimmir/data"  
arxiv_data_path = "/map-vepfs/siwei/ziruili/data/arxiv-metadata-oai-snapshot.json"
output_images_dir = "/map-vepfs/siwei/ziruili/data/arxiv-images/extracted"
final_json_path = "/map-vepfs/siwei/ziruili/DocMMIR/data/arxiv_json/arxiv_data.json"

os.makedirs(output_images_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 2) READ ARXIV METADATA ONCE
# -------------------------------------------------------------------------
arxiv_data = pd.read_json(arxiv_data_path, lines=True)
arxiv_data = arxiv_data[["id", "title", "abstract"]]
print("ArXiv DataFrame shape:", arxiv_data.shape)

# -------------------------------------------------------------------------
# 3) LOOP OVER PARQUET FILES & PROCESS IN CHUNKS
# -------------------------------------------------------------------------
doc_list = []  # we'll accumulate doc-level data for all Parquet files

parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
print(f"Found {len(parquet_files)} Parquet files in {parquet_dir}.")

for pf in parquet_files:
    print(f"\nProcessing {pf} ...")

    # ---------------------------------------------------------------------
    # 3a) READ THIS PARQUET FILE
    # ---------------------------------------------------------------------
    df = pd.read_parquet(pf, engine="pyarrow")
    print(f"Loaded DataFrame with shape: {df.shape}")

    # ---------------------------------------------------------------------
    # 3b) EXTRACT doc_id & version_num (FOR THIS FILE ONLY)
    # ---------------------------------------------------------------------
    df[["doc_id", "version_num"]] = df["file_name_index"].apply(
        lambda x: pd.Series(extract_base_id_and_version(x))
    )

    # ---------------------------------------------------------------------
    # 3c) KEEP ONLY LATEST VERSION *WITHIN THIS FILE*
    #     (WARNING: doesn't compare versions across multiple files!)
    # ---------------------------------------------------------------------
    max_version = df.groupby("doc_id")["version_num"].transform("max")
    df_latest = df[df["version_num"] == max_version].copy()
    print(f"Keeping {len(df_latest)} rows after filtering to latest version in this file.")

    # ---------------------------------------------------------------------
    # 3d) MERGE WITH ARXIV (LEFT JOIN)
    # ---------------------------------------------------------------------
    merged_df = df_latest.merge(arxiv_data, left_on="doc_id", right_on="id", how="left")
    print(f"Merged shape: {merged_df.shape}")

    # ---------------------------------------------------------------------
    # 3e) GROUP BY doc_id, SAVE IMAGES, BUILD DOC-LEVEL DATA
    # ---------------------------------------------------------------------
    grouped = merged_df.groupby("doc_id")

    for doc_id_value, group_df in grouped:
        texts = group_df["text"].tolist()
        local_img_paths = []

        for _, row in group_df.iterrows():
            image_dict = row["image"]
            if isinstance(image_dict, dict) and "bytes" in image_dict:
                raw_bytes = image_dict["bytes"]
            else:
                raw_bytes = None

            original_filename = row["file_name_index"]
            out_path = os.path.join(output_images_dir, original_filename)

            # Save image bytes to disk if present
            if raw_bytes:
                with open(out_path, "wb") as f:
                    f.write(raw_bytes)

            # We'll store the filename in the doc-level JSON
            local_img_paths.append(original_filename)

        # Grab ArXiv fields from the first row
        first_row = group_df.iloc[0]
        doc_data = {
            "id":       first_row["id"] if pd.notnull(first_row["id"]) else doc_id_value,
            "title":    first_row.get("title", ""),
            "abstract": first_row.get("abstract", ""),
            "texts":    texts,
            "imgs":     local_img_paths,
            "split":    first_row.get("split", "unknown")
        }

        # Append to our global doc_list
        doc_list.append(doc_data)

# -------------------------------------------------------------------------
# 4) WRITE OUT THE FINAL JSON (ALL FILES COMBINED)
# -------------------------------------------------------------------------
with open(final_json_path, "w", encoding="utf-8") as f:
    json.dump(doc_list, f, indent=4, ensure_ascii=False)

print(f"\nDone! Collected doc-level data from {len(parquet_files)} Parquet files.")
print(f"Total docs in doc_list: {len(doc_list)}")
print(f"Images saved to: {output_images_dir}")
print(f"Final JSON: {final_json_path}")
