import os
import argparse
from pathlib import Path
from compute_embeddings import main as run_embedding_main

def find_ckpt(model_dir: Path) -> str:
    ckpts = sorted(model_dir.glob("*.ckpt"), key=os.path.getmtime)
    print(ckpts)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt file found in {model_dir}")
    return str(ckpts[-1])  # use the latest checkpoint

def main(args):
    model_dir = Path(args.model_dir)
    json_path = Path(args.json_path)
    image_dir = args.image_dir
    out_root = Path(args.output_dir)
    dataset_name = args.dataset_name

    ckpt_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

    for ckpt_dir in ckpt_dirs:
        ckpt_base = ckpt_dir.name
        ckpt_path = find_ckpt(ckpt_dir)

        output_path = out_root / f"{ckpt_base}_{dataset_name}"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing checkpoint: {ckpt_base}")
        print(f"Using JSON:  {json_path}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Output to:   {output_path}")

        run_embedding_main(argparse.Namespace(
            data_path=str(json_path),
            image_dir=image_dir,
            checkpoint_path=str(ckpt_path),
            batch_size=16,
            text_model="ViT-L-14",
            image_model="ViT-L-14",
            pretrained_weights="laion2b_s32b_b82k",
            fusion_strategy="weighted_sum",
            zero_shot=False,
            doc_out_dir=str(output_path),
            chunk_size=10000,
            devices="0,1,2,3"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embed same dataset using all checkpoints")
    parser.add_argument("--json_path", required=True, help="Path to a single JSON file for embedding")
    parser.add_argument("--dataset_name", required=True, help="Name to suffix for output dir (e.g. arxivtest)")
    parser.add_argument("--model_dir", required=True, help="Folder with multiple checkpoint subdirs")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--output_dir", default="./embeddings", help="Where to save embeddings")
    main(parser.parse_args())
