import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import TextImageDataset, custom_collate_fn, GlobalQueryDataset
from encoders import get_image_encoder, get_text_encoder
from fusion_module import FusionModule
from retrieval_model import LightningModel


def test(args):
    # Set random seed
    pl.seed_everything(42)

    # Instantiate encoders
    text_encoder = get_text_encoder(model_name=args.text_model)
    image_encoder = get_image_encoder(model_name=args.image_model, pretrained=args.pretrained_weights)
    fusion_module = FusionModule(fusion_strategy=args.fusion_strategy)

    config = {
        'batch_size': args.batch_size,
        'is_global_test': args.is_global_test,
        'doc_dir': args.doc_dir,
    }

    if args.zero_shot:
        print("Initializing zero-shot classification...")
        model = LightningModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            fusion_module=fusion_module,
            config=config
        )
    else:
        print("Initializing fine-tuned classification...")
        # Load model from best checkpoint
        model = LightningModel.load_from_checkpoint(args.checkpoint_path,
                                                    text_encoder=text_encoder,
                                                    image_encoder=image_encoder,
                                                    fusion_module=fusion_module,
                                                    config=config
                                                    )

    if args.is_global_test:
        # Load global query dataset (precomputed query embeddings and global doc indices)
        test_query_dataset = GlobalQueryDataset(
            query_pt_path=os.path.join(args.doc_dir, "test_query_emb.pt"),
            global_index_path=os.path.join(args.doc_dir, "query_to_doc_index.pt")
        )

        test_loader = DataLoader(
            test_query_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    else:
        # Load test dataset
        _, _, test_dataset = TextImageDataset.from_json(
            json_path=args.data_path,
        )

        # Create DataLoader for testing
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, image_encoder, args.image_dir),
        )

    # Trainer for testing
    trainer = pl.Trainer(accelerator="cpu", devices=args.devices)

    # Run testing
    print("Starting model testing...")
    trainer.test(model=model, dataloaders=test_loader)
    print("Testing finished...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Script for DocMMIR Retrieval Model")

    # Model parameters
    parser.add_argument("--text_model", type=str, default="ViT-B-32", help="Text encoder model")
    parser.add_argument("--image_model", type=str, default="ViT-B-32", help="Image encoder model")
    parser.add_argument("--pretrained_weights", type=str, default="laion2b_s34b_b79k", help="Pretrained image model")
    parser.add_argument("--fusion_strategy", type=str, default="weighted_sum", help="Fusion strategy for combining text and image features")

    # Test parameters
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs='+', default=1, help="List of GPU devices to use")
    parser.add_argument("--checkpoint_path", type=str, help="Path to best checkpoint")
    parser.add_argument("--is_global_test", type=lambda x: x.lower() == 'true', default=False, help="Setting global test")
    parser.add_argument("--zero_shot", type=lambda x: x.lower() == 'true', default=False, help="Enable zero-shot evaluation mode")
    # Data parameters
    parser.add_argument("--image_dir", type=str, help="Path to image file")
    parser.add_argument("--data_path", type=str, help="Path to data JSON file")
    parser.add_argument("--doc_dir", type=str)
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")

    args = parser.parse_args()
    print(args)
    test(args)
