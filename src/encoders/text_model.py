from torch import nn

from models import BERTModel, BLIPModel, CLIPModel, SigLIP2Model, VLM2VecModel, ALIGNModel, E5VModel, ColPaliModel


class TextModel(nn.Module):
    def __init__(self, model_name: str, pretrained: str = ""):
        """
        Initializes the appropriate text encoder.

        @args:
            model_name (str): The model name (e.g., "bert-base-uncased" or "ViT-B-32" for CLIP).
        """
        super(TextModel, self).__init__()
        if "vit" in model_name.lower():
            self.model = CLIPModel(model_name, mode="text", pretrained=pretrained)
        elif "siglip2" in model_name.lower():
            self.model = SigLIP2Model(mode="text")
        elif "bert" in model_name.lower():
            self.model = BERTModel()
        elif "blip" in model_name.lower():
            self.model = BLIPModel(mode="text")
        elif "vlm2vec" in model_name.lower():
            self.model = VLM2VecModel(model_name, mode="text")
        elif "align" in model_name.lower():
            self.model = ALIGNModel(mode="text")
        elif "e5v" in model_name.lower():
            self.model = E5VModel(mode="text")
        elif "colpali" in model_name.lower():
            self.model = ColPaliModel(mode="text")
            
        else:
            raise ValueError(f"Unsupported text encoder: {model_name}")

    def forward(self, texts):
        """
        Calls the forward method of the encoder directly.

        @args:
            texts (List[str]): List of text inputs.

        @returns:
            torch.Tensor: Encoded text embeddings.
        """
        return self.model.forward(texts)

    def get_embedding_size(self) -> int:
        """
        Returns the embedding size of the text encoder.

        @returns:
            int: The embedding size of the text encoder.
        """
        return self.model.get_embedding_size()


def get_text_encoder(model_name: str, pretrained: str = ""):
    """
    Returns the appropriate text model.

    @args:
        model_name (str): Model name.
        pretrained (str): Pretrained weights.
    @returns:
        Initialized text model instance.
    """
    return TextModel(model_name, pretrained)
