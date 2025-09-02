"""
Single‑run multimodal training script (pytorch‑lightning).

Example:
    export NEPTUNE_API_TOKEN=xxxxxxxxxxxxxxxx
    python train.py \
        --text_model bert-base-uncased \
        --image_model openai/clip-vit-base-patch32 \
        --data_path data/news/train.json \
        --image_dir /mnt/images \
        --experiment_name clip_news_full \
        --devices 8
"""

import os
import argparse
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from dataset import TextImageDataset, custom_collate_fn
from encoders import get_text_encoder, get_image_encoder
from fusion_module import FusionModule
from retrieval_model import LightningModel


def train(args):

    pl.seed_everything(args.seed, workers=True)

    text_enc  = get_text_encoder(args.text_model,  pretrained=args.pretrained_weights)
    image_enc = get_image_encoder(args.image_model, pretrained=args.pretrained_weights)
    fusion    = FusionModule(fusion_strategy=args.fusion_strategy, image_dim=768, text_dim=768)

    config = vars(args).copy()
    model  = LightningModel(text_enc, image_enc, fusion, config=config)

    train_ds, val_ds, _ = TextImageDataset.from_json(args.data_path, dry_run=False)

    collate = partial(custom_collate_fn,
                      image_encoder=image_enc,
                      image_dir=args.image_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{args.output_dir}/{args.experiment_name}",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=3, save_weights_only=True
    )
    early_cb = EarlyStopping(monitor="val_MRR10", patience=args.patience, mode="max")

    def build_logger(exp_name: str):
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            logger = NeptuneLogger(
                project="lord-jim/docmmir",
                api_key=os.getenv("NEPTUNE_API_TOKEN"),
                tags=[args.experiment_name],
                log_model_checkpoints=False,
            )
            logger.log_hyperparams(config)
            return logger
        return True  

    # logger
    logger = build_logger(args.experiment_name)

    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,             
        log_every_n_steps=args.log_steps,
        callbacks=[ckpt_cb, early_cb],
        profiler=SimpleProfiler() if args.profile else None,
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser("DocMMIR single‑run training")

    # --- encoders / fusion --- #
    p.add_argument("--text_model", required=True)
    p.add_argument("--image_model", required=True)
    p.add_argument("--pretrained_weights", default="")
    p.add_argument("--fusion_strategy", default="weighted_sum")

    # --- optimisation --- #
    p.add_argument("--lr_image", type=float, default=2e-5)
    p.add_argument("--lr_text",  type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=1e-8)

    # --- loss / modality --- #
    p.add_argument("--loss_type", choices=["infonce", "bce"], default="bce")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--modality", default="both")

    # --- schedule --- #
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)

    # --- data --- #
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)

    # --- logging / infra --- #
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--strategy", default="ddp_find_unused_parameters_true")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="32")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--profile", action="store_true")

    # --- misc --- #
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)

# import argparse
# import os

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import NeptuneLogger
# from torch.utils.data import DataLoader

# from dataset import TextImageDataset, custom_collate_fn
# from encoders import get_text_encoder, get_image_encoder
# from fusion_module import FusionModule
# from retrieval_model import LightningModel


# def train(args):
#     # Set up random seed for reproducibility
#     pl.seed_everything(42)

#     # Instantiate text and image encoders
#     text_encoder = get_text_encoder(model_name=args.text_model,pretrained=args.pretrained_weights)
#     image_encoder = get_image_encoder(model_name=args.image_model, pretrained=args.pretrained_weights)

#     # Instantiate fusion module
#     # Strategies: weighted_sum, mlp
#     fusion_module = FusionModule(fusion_strategy=args.fusion_strategy, image_dim=768,text_dim=768)

#     # Create configuration dictionary for model
#     modality = "both" if args.modality == "" else args.modality.lower()
#     config = {
#         "text_model_name": args.text_model,
#         "image_model_name": args.image_model,
#         "batch_size": args.batch_size,
#         "weight_decay": args.weight_decay,
#         "epsilon": args.epsilon,
#         "lr_image": args.lr_image,
#         "lr_text": args.lr_text,
#         "temperature": args.temperature,
#         "loss_type": args.loss_type.lower(),   # "infonce" | "bce"
#         "modality": modality,                  # "text" | "image" | "both"
#         "pretrained_weights": args.pretrained_weights,
#     }

#     # Create retrieval model with config
#     model = LightningModel(
#         text_encoder=text_encoder,
#         image_encoder=image_encoder,
#         fusion_module=fusion_module,
#         config=config
#     )

#     # Load train, validation, and test datasets
#     train_dataset, val_dataset, _ = TextImageDataset.from_json(
#         json_path=args.data_path,
#         dry_run=False
#     )

#     # Create DataLoaders for training and validation
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         collate_fn=lambda batch: custom_collate_fn(batch, image_encoder, args.image_dir),
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         pin_memory=True,
#         collate_fn=lambda batch: custom_collate_fn(batch, image_encoder, args.image_dir),
#     )

#     # Logger setup
#     logger = NeptuneLogger(
#         project="lord-jim/docmmir",
#         api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTljOWQ3NS05ZGJlLTRiNmUtYjcxMC1jNTNiMDY2NmZiNjEifQ==",
#         log_model_checkpoints=False,
#         tags=[args.experiment_name]
#     )
#     logger.log_hyperparams(params=config)

#     # Define checkpoint callback
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_loss",
#         dirpath=os.path.join(args.output_dir, f"checkpoints/{args.experiment_name}"),
#         filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
#         save_top_k=3,
#         mode="min",
#         save_weights_only=True,
#     )
    
#     # Create PyTorch Lightning Trainer
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         logger=logger,
#         strategy=args.strategy,
#         accelerator="gpu",
#         devices=args.devices,
#         callbacks=[checkpoint_callback],
#         precision=args.precision,
#     )

#     # Train the model
#     trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# if __name__ == "__main__":
#     p = argparse.ArgumentParser("DocMMIR training script (multimodal)")

#     # Encoders / fusion
#     p.add_argument("--text_model", required=True)
#     p.add_argument("--image_model", required=True)
#     p.add_argument("--pretrained_weights", type=str, default="")
#     p.add_argument("--fusion_strategy", default="weighted_sum")

#     # Optimisation
#     p.add_argument("--lr_image", type=float, default=2e-5)
#     p.add_argument("--lr_text", type=float, default=2e-5)
#     p.add_argument("--lr_Linner", type=float, default=1e-4)
#     p.add_argument("--weight_decay", type=float, default=0.01)
#     p.add_argument("--epsilon", type=float, default=1e-8)

#     # Loss / ablation
#     p.add_argument("--loss_type", choices=["infonce", "bce"], default="infonce")
#     p.add_argument("--temperature", type=float, default=0.07)
#     p.add_argument("--modality", type=str, default="", help="text | image | both (default both)")

#     # Training schedule
#     p.add_argument("--batch_size", type=int, default=32)
#     p.add_argument("--max_epochs", type=int, default=5)

#     # Data
#     p.add_argument("--data_path", required=True)
#     p.add_argument("--image_dir", default="/path/to/images")

#     # Logging / infra
#     p.add_argument("--experiment_name", required=True)
#     p.add_argument("--output_dir", default="./outputs")
#     p.add_argument("--strategy", default="ddp")
#     p.add_argument("--devices", type=int, default=1)
#     p.add_argument("--precision", type=str, default="bf16-mixed")

#     args = p.parse_args()
#     train(args)