import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import random
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pytorch_lightning as pl


from dataset import TextImageDataset, custom_collate_fn
from encoders import get_text_encoder, get_image_encoder
from fusion_module import FusionModule
from retrieval_model import LightningModel


@torch.no_grad()
def compute_embeddings(model, dataset, image_dir, batch_size, device, out_dir):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda b: custom_collate_fn(b, model.image_encoder, image_dir),
    )

    model.eval()
    model.to(device)

    text_embs, image_embs, query_embs, doc_embs = [], [], [], []

    for batch in tqdm(loader, desc="Batch"):
        # CLIP text encoder expects list of strings (not tensor)
        texts = batch["texts"]
        queries = batch["queries"]

        # 保持图像为原始列表格式
        images = batch["images"]  # 应为 List[Tensor(C,H,W)]


        # Encode
        t = model.text_encoder(texts).cpu()
        i = model.image_encoder(images).cpu()  # 输入保持列表形式
        q = model.text_encoder(queries).cpu()
        d = model.fusion_module(i.to(device), t.to(device)).cpu()

        text_embs.append(t)
        image_embs.append(i)
        query_embs.append(q)
        doc_embs.append(d)

    torch.save(torch.cat(text_embs, dim=0), out_dir / "random_text.pt")
    torch.save(torch.cat(image_embs, dim=0), out_dir / "random_image.pt")
    torch.save(torch.cat(query_embs, dim=0), out_dir / "random_query.pt")
    torch.save(torch.cat(doc_embs, dim=0), out_dir / "random_doc.pt")

    print("✅ All 300 embeddings saved to", out_dir)

def main(args):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    text_encoder = get_text_encoder(args.text_model, pretrained=args.pretrained_weights)
    image_encoder = get_image_encoder(args.image_model, pretrained=args.pretrained_weights)
    fusion = FusionModule(fusion_strategy=args.fusion_strategy)
    config = {"temperature": 0.07, "modality": "both"}

    if args.zero_shot or not args.checkpoint_path:
        print("Using zero-shot pretrained encoders")
        model = LightningModel(text_encoder, image_encoder, fusion, config)
    else:
        print("Loading from fine-tuned checkpoint")
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            fusion_module=fusion,
            config=config,
            map_location=device,
        )

    # Load dataset and sample
    train_ds, val_ds, test_ds = TextImageDataset.from_json(args.data_path)
    all_data = train_ds.data + val_ds.data + test_ds.data
    print(f"Loaded {len(all_data)} examples from dataset.")

    random.seed(42)
    sample_indices = random.sample(range(len(all_data)), 300)
    sampled_ds = Subset(TextImageDataset(all_data, split_name="all"), sample_indices)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_embeddings(model, sampled_ds, args.image_dir, args.batch_size, device, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample and compute 300 embeddings from DocMMIR dataset")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--text_model", default="ViT-L-14")
    parser.add_argument("--image_model", default="ViT-L-14")
    parser.add_argument("--pretrained_weights", default="laion2b_s32b_b82k")
    parser.add_argument("--fusion_strategy", default="weighted_sum")
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--out_dir", default="./random_300_embeddings")

    main(parser.parse_args())
