import torch
from pathlib import Path

import torch
from pathlib import Path

def merge_tensor_chunks(root_dir: str):
    root_path = Path(root_dir)
    output_path = root_path / "merged_full"
    output_path.mkdir(parents=True, exist_ok=True)

    doc_offset = 0
    chunk_offset = 0
    new_chunk_id = 0
    new_doc_index_map = []
    new_query_embs = []
    new_query_to_doc = []

    for domain_dir in sorted(root_path.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name == "merged_full":
            continue

        print(f"🔗 Merging from {domain_dir.name}")
        chunks = sorted(domain_dir.glob("doc_chunk_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        
        # Copy and rename chunks
        for chunk in chunks:
            emb = torch.load(chunk)
            torch.save(emb, output_path / f"doc_chunk_{new_chunk_id}.pt")
            new_chunk_id += 1

        # Adjust index map
        index_map_file = domain_dir / "doc_index_map.pt"
        if index_map_file.exists():
            index_map = torch.load(index_map_file)
            adjusted_map = [(cid + chunk_offset, idx) for (cid, idx) in index_map]
            new_doc_index_map.extend(adjusted_map)
        chunk_offset += len(chunks)

        # Adjust query embeddings and query-to-doc index
        query_emb_file = domain_dir / "test_query_emb.pt"
        query_map_file = domain_dir / "query_to_doc_index.pt"
        if query_emb_file.exists() and query_map_file.exists():
            q_emb = torch.load(query_emb_file)
            q_idx = torch.load(query_map_file)
            new_query_embs.append(q_emb)
            adjusted_q_idx = [doc_offset + i for i in q_idx]
            new_query_to_doc.extend(adjusted_q_idx)

        # Count documents
        doc_count = sum(torch.load(f).size(0) for f in chunks)
        doc_offset += doc_count

    # Save merged output
    torch.save(new_doc_index_map, output_path / "doc_index_map.pt")
    if new_query_embs:
        torch.save(torch.cat(new_query_embs, dim=0), output_path / "test_query_emb.pt")
        torch.save(new_query_to_doc, output_path / "query_to_doc_index.pt")

    print(f"✅ Merged into {output_path}")
    print(f"   → {new_chunk_id} chunks")
    print(f"   → {len(new_doc_index_map)} documents")
    print(f"   → {len(new_query_to_doc)} queries")

if __name__ == "__main__":
    merge_tensor_chunks("/volume/ailab4sci/users/shuyue/swwu/docmmir/data/tensor_chunks/blip2_zs")
