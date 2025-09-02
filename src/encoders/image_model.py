import torch
from torch import nn

from models import BLIPModel, CLIPModel, SigLIP2Model, VLM2VecModel, ALIGNModel, E5VModel, ColPaliModel


class ImageModel(nn.Module):
    def __init__(self, model_name: str, pretrained: str = ""):
        """
        Initializes the appropriate image encoder.

        @args:
            model_name (str): The model name (e.g., "ViT-B-32" for CLIP").
            pretrained (str): Pretrained weights source.
        """
        super(ImageModel, self).__init__()
        if "vit" in model_name.lower():
            self.model = CLIPModel(model_name, mode="image", pretrained=pretrained)
        elif "siglip2" in model_name.lower():
            self.model = SigLIP2Model(mode="image")
        elif "blip" in model_name.lower():
            self.model = BLIPModel(mode="image")
        elif "vlm2vec" in model_name.lower():
            self.model = VLM2VecModel(model_name, mode="image")
        elif "align" in model_name.lower():
            self.model = ALIGNModel(mode="image")
        elif "e5v" in model_name.lower():
            self.model = E5VModel(mode="image")
        elif "colpali" in model_name.lower():
            self.model = ColPaliModel(mode="image")

        else:
            raise ValueError(f"Unsupported image encoder: {model_name}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocesses a single image from a file path.

        @args:
            image_path (str): Path to the image file.

        @returns:
            torch.Tensor: Preprocessed image tensor with batch dimension.
        """
        return self.model.preprocess_image(image_path)

    def forward(self, images):
        """
        Calls the forward method of the encoder directly.

        @args:
            images (List[str]): List of image paths.

        @returns:
            torch.Tensor: Encoded image embeddings.
        """
        return self.model.forward(images)

    def get_embedding_size(self) -> int:
        """
        Returns the embedding size of the image encoder.

        @returns:
            int: The embedding size of the image encoder.
        """
        return self.model.get_embedding_size()


def get_image_encoder(model_name: str, pretrained: str = None):
    """
    Returns the appropriate image model.

    @args:
        model_name (str): Model name.
        pretrained (str): Pretrained weights.

    @returns:
        Initialized image model instance.
    """
    return ImageModel(model_name, pretrained)
