import os
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image
from PIL.ImageFile import ImageFile
# from lavis.models import load_model_and_preprocess
from transformers import AutoProcessor, Blip2Model

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BLIPModel(nn.Module):
    def __init__(self, mode: str = "image"):
        """
        Initializes Blip model.

        @args:
            model (str)
            mode (str)
        """
        super(BLIPModel, self).__init__()

        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.mode = mode

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocesses a single image from a file path.

        @args:
            image_path (str): Path to the image file.

        @returns:
            torch.Tensor: Preprocessed image tensor with batch dimension.
        """
        if not os.path.isfile(image_path):
            raise ValueError(f"Invalid file path: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"]

    def preprocess_text(self, texts: Union[List[List[str]], List[str]]) -> dict:
        """
        Preprocesses a batch of text inputs.

        @args:
            texts (Union[List[List[str]], List[str]]): List of list of text descriptions / List of queries.

        @returns:
            dict: Tokenized text tensor ready for BLIP encoding.
        """
        if isinstance(texts, list) and all(isinstance(item, list) for item in texts):
            texts = [" ".join(paragraphs) for paragraphs in texts]  # Combine multiple paragraphs
        processed_tensors = self.processor(text=texts, padding=True, max_length=256, truncation=True, return_tensors="pt")
        # print("Debug: input_ids shape (batch_size, context_length):", processed_tensors["input_ids"].shape)
        # print("Max token id:", processed_tensors["input_ids"].max().item())
        return processed_tensors

    def forward(self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Encodes a batch of image tensors using BLIP.

        @args:
            inputs (Union[List[List[str]], torch.Tensor]):
                - A batch of text descriptions
                - A lists of preprocessed image tensors.

        @returns:
            torch.Tensor: Batch of averaged image/text embeddings, shape:(batch_size, embedding_dim).
        """
        try:
            # Read model device
            device = next(self.parameters()).device

            if self.mode == "image":
                all_image_embeddings = []

                # Process each sample (which can have multiple images)
                for image_tensors in inputs:
                    image_tensors = torch.cat(image_tensors, dim=0).to(device)
                    # Get image features from the model
                    image_features = self.model.get_qformer_features(image_tensors).last_hidden_state  # (num_images, patch_size, hidden_dim)
                    # Remove the patch dimention
                    avg_features = image_features.mean(dim=1)  # (num_images, hidden_dim)
                    # Average over the images for this sample (average along dim=0)
                    avg_features = avg_features.mean(dim=0, keepdim=True)# (1, hidden_dim)
                    all_image_embeddings.append(avg_features)
                return torch.cat(all_image_embeddings, dim=0)  # Shape: (batch_size, hidden_dim)

            elif self.mode == "text":
                text_tensors = self.preprocess_text(inputs).to(device)  # Preprocess text batch
                text_embeddings = self.model.get_text_features(**text_tensors, output_hidden_states=True).hidden_states[-1]
                avg_text_embedding = text_embeddings.mean(dim=1)
                return avg_text_embedding  # Shape: (batch_size, hidden_dim)

        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e

    def get_embedding_size(self) -> int:
        """
        Returns the size of the embeddings.
        """
        if self.mode == "image":
            return self.model.config.qformer_config.hidden_size
        elif self.mode == "text":
            return self.model.config.text_config.hidden_size