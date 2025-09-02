import os
from typing import List, Union

import open_clip
import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLIPModel(nn.Module):
    def __init__(self, model: str = 'ViT-L-14', pretrained: str = 'laion2b_s32b_b82k', mode: str = "image"):
        """
        Initializes the CLIP model for both image and text encoding.

        @args:
            model (str): The CLIP model name. Defaults to 'ViT-B-32'.
            pretrained (str): The pretrained weights to use. Defaults to 'laion2b_s34b_b79k'.
            mode (str): Either "image" or "text" to specify processing type.
        """
        super(CLIPModel, self).__init__()
        print(model,pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model)
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
        return self.preprocess(image).unsqueeze(0)

    def preprocess_text(self, texts: Union[List[List[str]], List[str]]) -> torch.Tensor:
        """
        Preprocesses a batch of text inputs.

        @args:
            texts (Union[List[List[str]], List[str]]): List of list of text descriptions / List of queries.

        @returns:
            torch.Tensor: Tokenized text tensor ready for CLIP encoding.
        """
        if isinstance(texts, list) and all(isinstance(item, list) for item in texts):
            texts = [" ".join(paragraphs) for paragraphs in texts]  # Combine multiple paragraphs
        tokens = open_clip.tokenize(texts)
        return tokens

    def forward(self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Encodes a batch of image tensors using CLIP.

        @args:
            inputs (Union[List[List[str]], torch.Tensor]):
                - A batch of text descriptions
                - A lists of preprocessed image tensors.

        @returns:
            torch.Tensor: Batch of averaged image/text embeddings, shape:(batch_size, embedding_dim).
        """
        device = next(self.parameters()).device
        if self.mode == "image":
            all_image_embeddings = []

            for image_tensors in inputs:
                image_tensors = torch.cat(image_tensors, dim=0).to(device)
                encoded_images = self.model.encode_image(image_tensors)
                avg_image_embedding = encoded_images.mean(dim=0, keepdim=True) if len(image_tensors) > 1 else encoded_images
                all_image_embeddings.append(avg_image_embedding)
            return torch.cat(all_image_embeddings, dim=0)  # Shape: (batch_size, embedding_dim)

        elif self.mode == "text":
            tokens = self.preprocess_text(inputs).to(device)  # Tokenize text batch
            text_embeddings = self.model.encode_text(tokens)
            return text_embeddings  # Shape: (batch_size, embedding_dim)

    def get_embedding_size(self) -> int:
        """
        Returns the size of the embeddings.
        """
        return 768