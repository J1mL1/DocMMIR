import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from tqdm.auto import tqdm 

from dataset import TextImageDataset, custom_collate_fn
from encoders import get_text_encoder, get_image_encoder
from fusion_module import FusionModule
from retrieval_model import LightningModel


class DocEmbedWrapper(nn.Module):
    """Return fused **document** embedding."""

    def __init__(self, model: LightningModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        txt_e = self.model.text_encoder(texts)
        img_e = self.model.image_encoder(images)
        return self.model.fusion_module(img_e, txt_e)


class QueryEmbedWrapper(nn.Module):
    """Return **query** embedding (text only)."""

    def __init__(self, model: LightningModel):
        super().__init__()
        self.model = model

    def forward(self, queries: torch.Tensor):
        txt = self.model.text_encoder(queries)
        return self.model.fusion_module.text_projection(txt)


def dp_wrap(module: nn.Module, device_ids: List[int]):
    if torch.cuda.is_available():
        if len(device_ids) > 1 and torch.cuda.device_count() > 1:
            module = nn.DataParallel(module, device_ids=device_ids)
        primary = torch.device(f"cuda:{device_ids[0]}")
        module.to(primary)
    return module


def _existing_chunks(out_dir: Path):
    """Return sorted list of existing doc_chunk_i.pt files."""
    return sorted(
        [f for f in out_dir.glob("doc_chunk_*.pt") if f.is_file()],
        key=lambda p: int(p.stem.split("_")[-1]),
    )


def compute_doc_chunks(
    model: LightningModel,
    dataset: TextImageDataset,
    batch_size: int,
    image_dir: str,
    out_dir: Path,
    chunk_size: int,
    device_ids: List[int],
    resume: bool = False,
):
    """Encode **all** docs and save chunked embeddings.

    If *resume* is True, existing chunk files in *out_dir* are kept and
    the script continues from the next unfinished chunk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_docs = 0
    chunk_id = 0
    index_map: List = []

    if resume:
        chunks = _existing_chunks(out_dir)
        if chunks:
            # 统计已保存的向量数量
            for f in chunks:
                processed_docs += torch.load(f).size(0)
            chunk_id = int(chunks[-1].stem.split("_")[-1]) + 1
            # 预加载已有的索引映射
            idx_path = out_dir / "doc_index_map.pt"
            if idx_path.exists():
                index_map = torch.load(idx_path)
            print(f"[Resume] Detected {len(chunks)} chunks, {processed_docs} docs already processed.")

        if processed_docs > 0:
            total = len(dataset)
            if processed_docs >= total:
                print("[Resume] All documents have already been processed.")
                return
            dataset = Subset(dataset, list(range(processed_docs, total)))
            # 重置计数器，从 0 开始计数新加载的样本
            processed_docs = 0

    wrapper = dp_wrap(DocEmbedWrapper(model), device_ids)
    # wrapper = DocEmbedWrapper(model)
    wrapper.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        collate_fn=lambda b: custom_collate_fn(b, model.image_encoder, image_dir),
    )

    buffer: List[torch.Tensor] = []
    doc_counter = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Doc batches", unit="batch"):
            texts = batch["texts"]
            images = batch["images"]
            emb = wrapper(images, texts).cpu()

            for row in emb:
                doc_counter += 1
                buffer.append(row.unsqueeze(0))
                if len(buffer) == chunk_size:
                    pt = torch.cat(buffer, 0)
                    torch.save(pt, out_dir / f"doc_chunk_{chunk_id}.pt")
                    tqdm.write(f"Saved doc_chunk_{chunk_id}.pt  ({chunk_size} docs)")
                    index_map.extend([(chunk_id, i) for i in range(chunk_size)])
                    buffer.clear()
                    chunk_id += 1

    if buffer:
        pt = torch.cat(buffer, 0)
        torch.save(pt, out_dir / f"doc_chunk_{chunk_id}.pt")
        tqdm.write(f"Saved doc_chunk_{chunk_id}.pt  ({len(buffer)} docs)")
        index_map.extend([(chunk_id, i) for i in range(len(buffer))])

    torch.save(index_map, out_dir / "doc_index_map.pt")
    print("Doc index map saved (total docs =", len(index_map), ")")


def compute_query_embeddings(
    model: LightningModel,
    dataset: TextImageDataset,
    batch_size: int,
    image_dir: str,
    out_path: Path
):
    wrapper = QueryEmbedWrapper(model).to('cuda')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        collate_fn=lambda b: custom_collate_fn(b, model.image_encoder, image_dir),
    )

    q_emb = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Query batches", unit="batch"):
            queries = batch["queries"]
            emb = wrapper(queries).cpu()
            q_emb.append(emb)

    tensor = torch.cat(q_emb, 0)
    torch.save(tensor, out_path)
    print(f"Query emb {tensor.shape} saved {out_path}")


def main(args):
    pl.seed_everything(42)
    device_ids = [int(d) for d in args.devices.split(",")] if args.devices else [0]

    # Load / init LightningModel
    txt_enc = get_text_encoder(args.text_model, pretrained=args.pretrained_weights)
    img_enc = get_image_encoder(args.image_model, pretrained=args.pretrained_weights)
    txt_dim = txt_enc.get_embedding_size()
    img_dim = img_enc.get_embedding_size()
    proj_dim = min(txt_dim, img_dim)
    fusion = FusionModule(proj_dim, args.fusion_strategy, 0.5, img_dim, txt_dim)
    # print(fusion)
    lm_cfg = {"temperature": 0.07, "modality": args.modality}

    # print("===== Text Encoder Architecture =====")
    # print(txt_enc)
    # print("=====================================\n")

    # print("===== Image Encoder Architecture =====")
    # print(img_enc)
    # print("======================================\n")

    if args.zero_shot or not args.checkpoint_path:
        print("Zero-shot inference (pre-trained encoders).")
        model = LightningModel(txt_enc, img_enc, fusion, lm_cfg)
    else:
        print("Loading fine-tuned Lightning checkpoint …")
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            text_encoder=txt_enc,
            image_encoder=img_enc,
            fusion_module=fusion,
            config=lm_cfg,
            map_location=f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu",
        )

    model = model.to(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else model

    # Dataset splits
    train_ds, val_ds, test_ds = TextImageDataset.from_json(args.data_path)
    all_ds = TextImageDataset(train_ds.data + val_ds.data + test_ds.data, split_name="all")
    out_dir = Path(args.doc_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Doc emb
    compute_doc_chunks(model, all_ds, args.batch_size, args.image_dir, out_dir, args.chunk_size, device_ids, resume=True)

    # Query emb
    query_pt = out_dir / "test_query_emb.pt"
    compute_query_embeddings(model, test_ds, args.batch_size, args.image_dir, query_pt)

    # Mapping test-query to global doc idx
    offset = len(train_ds) + len(val_ds)
    mapping = [offset + i for i in range(len(test_ds))]
    torch.save(mapping, out_dir / "query_to_doc_index.pt")
    print("Saved query_to_doc_index.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Embed docs/queries")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--text_model", default="ViT-L-14")
    p.add_argument("--image_model", default="ViT-L-14")
    p.add_argument("--pretrained_weights", default="laion2b_s32b_b82k")
    p.add_argument("--fusion_strategy", default="weighted_sum")
    p.add_argument("--zero_shot", action="store_true")
    p.add_argument("--doc_out_dir", default="./doc_chunks")
    p.add_argument("--chunk_size", type=int, default=10000)
    p.add_argument("--devices", default="0", help="Comma-separated GPU ids, e.g. '0,1,2'")
    p.add_argument("--modality", default="both")
    
    
    main(p.parse_args())
