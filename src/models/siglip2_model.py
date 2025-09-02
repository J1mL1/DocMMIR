import os
from typing import Union, List

import torch
import torch.nn as nn
from PIL import Image
from PIL.ImageFile import ImageFile

from transformers import AutoModel, AutoTokenizer, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SigLIP2Model(nn.Module):

    def __init__(self, mode: str = "image"):
        """
        Initializes the SigLip2 model for both image and text encoding.

        @args:
            mode (str): Either "image" or "text" to specify the processing type.
        """
        super(SigLIP2Model, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
        self.processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384")
        self.mode = mode

    def preprocess_image(self, image_path: str) -> dict:
        """
        Preprocesses a single image from a file path using the SigLip2 processor.

        @args:
            image_path (str): Path to the image file.

        @returns:
            dict: Processed image inputs as expected by the model.
        """
        if not os.path.isfile(image_path):
            raise ValueError(f"Invalid file path: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"]

    def preprocess_text(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Preprocesses a list of text inputs using the SigLip2 processor.

        @args:
            texts (List[List[str]]): List of list of text descriptions.

        @returns:
            dict: Processed text inputs as expected by the model.
        """
        if isinstance(texts, list) and all(isinstance(item, list) for item in texts):
            texts = [" ".join(paragraphs) for paragraphs in texts]  # Combine multiple paragraphs
        return self.processor(text=texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

    def forward(self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Encodes a batch of inputs using SigLip2.

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
                for image_tensors in inputs:
                    # Concatenate the list of preprocessed image tensors along the batch dimension.
                    image_tensors = torch.cat(image_tensors, dim=0).to(device)
                    encoded_images = self.model.get_image_features(image_tensors)
                    # If multiple patches are present, average their embeddings.
                    avg_embedding = encoded_images.mean(dim=0, keepdim=True) if len(image_tensors) > 1 else encoded_images
                    all_image_embeddings.append(avg_embedding)
                return torch.cat(all_image_embeddings, dim=0)  # Shape: (batch_size, embedding_dim)
            elif self.mode == "text":
                tokens = self.preprocess_text(inputs).to(device)
                text_embeddings = self.model.get_text_features(**tokens)
                return text_embeddings  # Shape: (batch_size, embedding_dim)

        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e

    def get_embedding_size(self) -> int:
        """
        Returns the size of the embeddings.
        """
        try:
            return self.model.config.hidden_size
        except AttributeError:
            if hasattr(self.model.config, "vision_config"):
                return self.model.config.vision_config.hidden_size
            elif hasattr(self.model.config, "text_config"):
                return self.model.config.text_config.hidden_size
            else:
                raise ValueError("Unable to determine embedding size from model config.")