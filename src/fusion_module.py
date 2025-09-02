import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self,
                 fusion_dim: int = 768,
                 fusion_strategy: str = 'mlp',
                 alpha: float = 0.5,
                 image_dim: int = 768,
                 text_dim: int = 768):
        """
        Initializes the FusionModule to combine text and image embeddings.

        Args:
            fusion_dim (int): The dimension of the output fused embeddings. Defaults to 512.
            fusion_strategy (str): The strategy for fusing text and image embeddings.
                                   Options: 'weighted_sum' or 'mlp'.
            alpha (float): The weight for combining embeddings if using weighted fusion.
            image_dim (int): The dimensionality of image embeddings (required for mlp fusion).
            text_dim (int): The dimensionality of text embeddings (required for mlp fusion).
        """
        super(FusionModule, self).__init__()
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.fusion_dim = fusion_dim

        if self.fusion_strategy == 'mlp':
            # Utilising image dimension and text dimension size to create input dimension
            if image_dim is None or text_dim is None:
                raise ValueError("For mlp fusion, image_dim and text_dim must be provided.")
            input_dim = image_dim + text_dim
            self.fusion_mlp = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        elif self.fusion_strategy == 'weighted_sum':
            self.text_projection = (
                nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
            )
            self.image_projection = (
                nn.Linear(image_dim, fusion_dim) if image_dim != fusion_dim else nn.Identity()
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if self.fusion_strategy == 'weighted_sum':
            img_proj = self.image_projection(image_embeddings)
            txt_proj = self.text_projection(text_embeddings)

            # Weighted sum
            fused_embedding = self.alpha * txt_proj + (1 - self.alpha) * img_proj 
        elif self.fusion_strategy == 'mlp':
            # Concatenate and then mlp
            combined = torch.cat([text_embeddings, image_embeddings], dim=-1)
            fused_embedding = self.fusion_mlp(combined)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return fused_embedding
