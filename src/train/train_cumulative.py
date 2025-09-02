"""
Loop over cumulative splits within ONE domain and train the model for each split.

Run example:
    export NEPTUNE_API_TOKEN=xxxxxxxxxxxx
    python train_cumulative.py \
        --splits_dir data/news/splits \
        --base_exp_name clip_news \
        --text_model bert-base-uncased \
        --image_model openai/clip-vit-base-patch32 \
        --image_dir /mnt/images \
        --devices 8 \
        --batch_size 32
"""

import os
from glob import glob
from functools import partial
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from dataset import TextImageDataset, custom_collate_fn
from encoders import get_text_encoder, get_image_encoder
from fusion_module import FusionModule
from retrieval_model import LightningModel


# ---------- train once on one split ---------- #
def train_once(args, split_path: str, split_tag: str):
    pl.seed_everything(args.seed, workers=True)

    text_enc  = get_text_encoder(args.text_model,  pretrained=args.pretrained_weights)
    image_enc = get_image_encoder(args.image_model, pretrained=args.pretrained_weights)
    fusion    = FusionModule(fusion_strategy=args.fusion_strategy, image_dim=768, text_dim=768)

    config = vars(args).copy()
    config.update({"split_tag": split_tag})
    model = LightningModel(text_enc, image_enc, fusion, config=config)

    # dataset / dataloader
    train_ds, val_ds, _ = TextImageDataset.from_json(split_path, dry_run=False)
    collate = partial(custom_collate_fn, image_encoder=image_enc, image_dir=args.image_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.experiment_name),
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

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

    # trainer
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,  # fixed 5 epoch
        log_every_n_steps=args.log_steps,
        callbacks=[ckpt_cb, early_cb],
        profiler=SimpleProfiler() if args.profile else None,
        enable_progress_bar=True, 
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.strategy.teardown()       
    del trainer, model, text_enc, image_enc
    torch.cuda.empty_cache()            
    import gc; gc.collect()   


def main(args):
    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No .json files found in json_dir")

    for json_file in json_files:
        exp_tag = json_file.stem  # e.g. mydataset123.json → "mydataset123"
        print(f"\n===== Training on {exp_tag} =====")

        args.experiment_name = f"{args.base_exp_name}_{exp_tag}"
        train_once(args, str(json_file), exp_tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Train cumulatively on domain splits")

    # encoders / fusion
    p.add_argument("--text_model", required=True)
    p.add_argument("--image_model", required=True)
    p.add_argument("--pretrained_weights", default="")
    p.add_argument("--fusion_strategy", default="weighted_sum")

    # optimisation
    p.add_argument("--lr_image", type=float, default=2e-5)
    p.add_argument("--lr_text",  type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=1e-8)

    # loss / modality
    p.add_argument("--loss_type", choices=["infonce", "bce"], default="bce")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--modality", default="both")

    # schedule
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)

    # data
    p.add_argument("--json_dir", required=True)
    p.add_argument("--image_dir", required=True)

    # logging / infra
    p.add_argument("--base_exp_name", required=True)
    p.add_argument("--output_dir", default="./outputs/modality_test")
    p.add_argument("--strategy", default="ddp_find_unused_parameters_true")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="32")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_steps", type=int, default=1)
    p.add_argument("--profile", action="store_true")

    # misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)