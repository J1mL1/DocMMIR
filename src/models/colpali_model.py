import os
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

class ColPaliModel(nn.Module):
    """
    dual-encoder wrapper around vidore/colpali-v1.3.

    The public ColPali model yields per-token / per-patch embeddings (768-D).
    For parity with `E5VModel`, this class mean-pools those tokens so the
    `forward()` method still returns shape (B, D).

    Parameters
    ----------
    mode : {"image", "text"}
        Choose which encoder to expose in `forward()`.
    """

    EMB_DIM = 128  
    
    def __init__(self, mode: str = "image"):
        super().__init__()

        if mode not in {"image", "text"}:
            raise ValueError("mode must be 'image' or 'text'")

        self.mode = mode
        self.processor: ColPaliProcessor = ColPaliProcessor.from_pretrained(
            "/volume/pt-train/models/colpali-v1.3"
        )
        self.model: ColPali = ColPali.from_pretrained(
            "/volume/pt-train/models/colpali-v1.3",
            torch_dtype=torch.bfloat16,
        )

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk and convert to ColPali pixel tensor.
        Returns only `pixel_values` for symmetry with E5VModel.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} does not exist")

        img = Image.open(image_path).convert("RGB")
        batch = self.processor.process_images([img])
        return batch # pixel_value & input_ids

    def preprocess_text(self, texts: List[List[str]]) -> dict:
        """
        Concatenate sub-segments, then ColPali-tokenize.
        """
        flat_texts = [" ".join(t) for t in texts]
        return self.processor.process_queries(flat_texts)

    def forward(
        self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]
    ) -> torch.Tensor:
        """
        If mode=="text", `inputs` is List[List[str]] (sentences / sentence parts).
        If mode=="image", `inputs` is List[List[torch.Tensor]]
        where each inner list already contains pixel tensors
        (shape (1, 3, H, W)) for a *single* logical document.
        """
        device = next(self.parameters()).device
        if self.mode == "text":
            batch = self.preprocess_text(inputs)
            batch = {k: v.to(device) for k, v in batch.items()} 
            embeddings = self.model(**batch)  # (B, T, D)
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=1)  # pool tokens
            return embeddings  # (B, D)

        else:
            pooled: List[torch.Tensor] = []
            for batch in inputs:
                input_ids_list   = [d["input_ids"]   for d in batch]
                pixel_values_list = [d["pixel_values"] for d in batch]
                attention_mask_list = [d["attention_mask"] for d in batch]
                # concatenate the page images belonging to one doc
                tensor_list = {
                    "input_ids":torch.cat(input_ids_list,dim=0).to(device),       # (B, L)
                    "attention_mask":torch.cat(attention_mask_list,dim=0).to(device),  # (B, L)
                    "pixel_values":torch.cat(pixel_values_list,dim=0).to(device),    # (B, 3, H, W)
                }

                feats = self.model(**tensor_list)  # (N_img, Patches, D) 

                # Collapse over patch dimension and over images inside the doc
                if feats.dim() == 3:
                    feats = feats.mean(dim=1)  # mean over patches
                doc_emb = feats.mean(dim=0, keepdim=True)
                pooled.append(doc_emb)  # (1, D)

        return torch.cat(pooled, dim=0)  # (B, D)

    def get_embedding_size(self) -> int:
        return self.EMB_DIM