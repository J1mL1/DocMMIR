import os
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL.ImageFile import ImageFile
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True


class E5VModel(nn.Module):
    def __init__(self, mode: str = "image"):
        """
        E5-V based model compatible with ALIGN-style interface.
        mode: one of ["image", "text"]
        """
        super().__init__()
        self.processor = LlavaNextProcessor.from_pretrained(
            "/volume/pt-train/models/e5v",)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "/volume/pt-train/models/e5v", 
            torch_dtype=torch.float16
        )

        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prompt templates
        llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        self.img_template = llama3_template.format('<image>\nSummary above image in one word: ')
        self.txt_template = llama3_template.format('<sent>\nSummary above sentence in one word: ')

        if getattr(self.processor.image_processor, "patch_size", None) is None:
            self.processor.patch_size = 14
            self.processor.vision_feature_select_strategy = "default"

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and process one image (used inside batch construction).
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Invalid path: {image_path}")
        img = Image.open(image_path).convert("RGB")
        processed = self.processor([img], [self.img_template], return_tensors="pt", padding=True)
        return processed["pixel_values"]

    def preprocess_text(self, texts: List[List[str]]) -> dict:
        """
        Concatenate paragraphs and tokenize.
        """
        flat_texts = [" ".join(t) for t in texts]
        prompts = [self.txt_template.replace("<sent>", txt) for txt in flat_texts]
        return self.processor(prompts, return_tensors="pt", padding=True)

    def forward(self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Forward pass for text or image encoding depending on self.mode.
        Inputs are already batched (List of List of tensors or strings).
        """
        device = self.device
        if self.mode == "text":
            tokens = self.preprocess_text(inputs).to(device)
            out = self.model(**tokens, output_hidden_states=True, return_dict=True)
            return out.hidden_states[-1][:, -1, :]
        
        elif self.mode == "image":
            all_embeddings = []
            for image_tensor_list in inputs:  # List[Tensor], already (3, H, W)
                imgs = torch.cat(image_tensor_list, dim=0).to(device)  # (n_img, 3, H, W)
                prompts = [self.img_template] * imgs.size(0)

                inputs_img = {
                    "pixel_values": imgs,
                    "input_ids": self.processor.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device),
                    "attention_mask": self.processor.tokenizer(prompts, return_tensors="pt", padding=True).attention_mask.to(device),
                }

                out = self.model(**inputs_img, output_hidden_states=True, return_dict=True)
                emb = out.hidden_states[-1][:, -1, :]  # (n_img, D)
                avg_emb = emb.mean(dim=0, keepdim=True)  # (1, D)
                all_embeddings.append(avg_emb)

            return torch.cat(all_embeddings, dim=0)  # (B, D)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def get_embedding_size(self) -> int:
        return 4096
