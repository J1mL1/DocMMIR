import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import math

from metrics import (
    calculate_ranking,
    calculate_ranking_global,
    calculate_metrics,
)


class LightningModel(pl.LightningModule):
    """PyTorch‑Lightning module for multimodal document retrieval.

    Supports two contrastive objectives (InfoNCE | balanced‑BCE) **and**
    modality ablation (text‑only, image‑only, or both).
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        fusion_module: torch.nn.Module,
        config: Dict,
    ) -> None:
        super().__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_module = fusion_module

        self.config = config

        for enc, name in [
            (self.text_encoder, "text_encoder"),
            (self.image_encoder, "image_encoder"),
        ]:
            if not hasattr(enc, "get_embedding_size"):
                raise ValueError(f"{name} must implement get_embedding_size()")

        txt_dim = self.text_encoder.get_embedding_size()
        img_dim = self.image_encoder.get_embedding_size()
        proj_dim = min(txt_dim, img_dim)

        # Projection layers (kept even if dimensions match)
        self.query_projection = (
            nn.Linear(txt_dim, proj_dim) if txt_dim != proj_dim else nn.Identity()
        )
        self.text_projection = (
            nn.Linear(txt_dim, proj_dim) if txt_dim != proj_dim else nn.Identity()
        )
        self.image_projection = (
            nn.Linear(img_dim, proj_dim) if img_dim != proj_dim else nn.Identity()
        )

        # Training configuration
        self.loss_type = self.config.get("loss_type", "infonce").lower()  # infonce | bce
        self.modality = self.config.get("modality", "both").lower()      # text | image | both
        self.tau = float(self.config.get("temperature", 0.07))

        # Metric accumulators
        self.MRR, self.MRR10, self.Hit1, self.Hit3, self.Hit10, self.NDCG10 = [], [], [], [], [], []

    def _compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute row/column symmetric InfoNCE or balanced‑BCE."""
        B, device = logits.size(0), logits.device

        if self.loss_type == "infonce":
            labels = torch.arange(B, device=device)
            loss_row = F.cross_entropy(logits, labels)
            loss_col = F.cross_entropy(logits.T, labels)
            return 0.5 * (loss_row + loss_col)
        elif self.loss_type == "bce":
            targets = torch.eye(B, device=device)
            pos_weight = torch.full((B,), B - 1, device=device)
            loss_row = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none").mean()
            loss_col = F.binary_cross_entropy_with_logits(logits.T, targets, pos_weight=pos_weight, reduction="none").mean()
            return 0.5 * (loss_row + loss_col)
        else:
            raise ValueError("Unknown loss_type: " + self.loss_type)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scaled cosine‑similarity logits [B, B]."""
        q_raw = self.text_encoder(batch["queries"])
        q = F.normalize(self.query_projection(q_raw), dim=-1)

        # Document representation depending on modality setting
        if self.modality == "text":
            d_raw = self.text_encoder(batch["texts"])
            d = self.text_projection(d_raw)
        elif self.modality == "image":
            d_raw = self.image_encoder(batch["images"])
            d = self.image_projection(d_raw)
        elif self.modality == "both":
            txt_raw = self.text_encoder(batch["texts"])
            txt = self.text_projection(txt_raw)
            img_raw = self.image_encoder(batch["images"])
            img = self.image_projection(img_raw)
            fused = self.fusion_module(img, txt)
            d = fused
        else:
            raise ValueError("modality must be text | image | both, got " + self.modality)

        d = F.normalize(d, dim=-1)
        logits = torch.matmul(q, d.T) / self.tau
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logits = self(batch)
        loss = self._compute_loss(logits)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logits = self(batch)
        loss = self._compute_loss(logits)

        # Accumulate loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Ranking metrics
        target_rank, _ = calculate_ranking(logits)
        metrics = calculate_metrics(target_rank)

        self.MRR.append(metrics["mrr"])
        self.MRR10.append(metrics["mrr@10"])
        self.Hit1.append(metrics["hit@1"])
        self.Hit3.append(metrics["hit@3"])
        self.Hit10.append(metrics["hit@10"])
        self.NDCG10.append(metrics["ndcg@10"])

        
    
    def on_validation_epoch_end(self):
        avg = lambda lst: float(torch.tensor(lst).mean()) if lst else 0.0
        self.log_dict(
            {
                "val_MRR": avg(self.MRR),
                "val_MRR10": avg(self.MRR10),
                "val_Hit@1": avg(self.Hit1),
                "val_Hit@3": avg(self.Hit3),
                "val_Hit@10": avg(self.Hit10),
                "val_NDCG@10": avg(self.NDCG10),
            },
            prog_bar=True,
        )
        # clear
        self.MRR.clear(); 
        self.Hit1.clear(); 
        self.Hit3.clear(); 
        self.Hit10.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.config.get("is_global_test", False):
            return self.global_test_step(batch, batch_idx)
        
        logits = self(batch)
        loss = self._compute_loss(logits)

        # Accumulate loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Ranking metrics
        target_rank, _ = calculate_ranking(logits)
        metrics = calculate_metrics(target_rank)

        self.MRR.append(metrics["mrr"])
        self.MRR10.append(metrics["mrr@10"])
        self.Hit1.append(metrics["hit@1"])
        self.Hit3.append(metrics["hit@3"])
        self.Hit10.append(metrics["hit@10"])
        self.NDCG10.append(metrics["ndcg@10"])

    def on_test_epoch_end(self):
        avg = lambda lst: float(torch.tensor(lst).mean()) if lst else 0.0
        self.log_dict(
            {
                "test_MRR": avg(self.MRR),
                "test_MRR10": avg(self.MRR10),
                "test_Hit@1": avg(self.Hit1),
                "test_Hit@3": avg(self.Hit3),
                "test_Hit@10": avg(self.Hit10),
                "test_NDCG@10": avg(self.NDCG10),
            },
            prog_bar=True,
        )
        self.MRR.clear()
        self.MRR10.clear()
        self.Hit1.clear()
        self.Hit3.clear()
        self.Hit10.clear()
        self.NDCG10.clear()
        return 

    # global test
    def global_test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if "query_emb" not in batch or "global_index" not in batch:
            raise ValueError("Global test requires 'query_emb' and 'global_index'")

        query_emb = batch["query_emb"]
        correct = batch["global_index"]
        device = query_emb.device
        doc_dir = self.config["doc_dir"]

        chunk_files = sorted(
            [
                os.path.join(doc_dir, f)
                for f in os.listdir(doc_dir)
                if f.startswith("doc_chunk_") and f.endswith(".pt")
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        scores = []
        for ck in chunk_files:
            chunk = torch.load(ck, map_location=device)
            scores.append(torch.matmul(query_emb, chunk.T))
        full_scores = torch.cat(scores, dim=1)

        target_rank = calculate_ranking_global(full_scores, correct)
        metrics = calculate_metrics(target_rank)

        self.MRR.append(metrics["mrr"])
        self.MRR10.append(metrics["mrr@10"])
        self.Hit1.append(metrics["hit@1"])
        self.Hit3.append(metrics["hit@3"])
        self.Hit10.append(metrics["hit@10"])
        self.NDCG10.append(metrics["ndcg@10"])
        return 

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        wd = self.config["weight_decay"]
        lr_img = self.config["lr_image"]
        lr_txt = self.config["lr_text"]

        param_groups = [
            {
                "params": [p for n, p in self.image_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
                "lr": lr_img,
            },
            {
                "params": [p for n, p in self.image_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr_img,
            },
            {
                "params": [p for n, p in self.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
                "lr": lr_txt,
            },
            {
                "params": [p for n, p in self.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr_txt,
            },
        ]
        optimizer = AdamW(param_groups, eps=self.config["epsilon"])

        # Lightning provides trainer.estimated_stepping_batches after setup
        steps_per_epoch = self.trainer.estimated_stepping_batches
        total_steps = steps_per_epoch * self.trainer.max_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step"}
        ]
