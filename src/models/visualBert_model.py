import torch
import torch.nn as nn
from transformers import AutoProcessor, BlipModel
from typing import List


class BlipModel(nn.Module):
    def __init__(self, model_name: str = "uclanlp/visualbert-vqa-coco-pre"):
        """
        Initializes the BertTextEncoder with a BERT tokenizer and model.

        @args:
            model_name (str): Name of the BERT model to use. Defaults to 'bert-base-uncased'.
        """
        super(BERTModel, self).__init__()
        self.model = VisualBertModel.from_pretrained(model_name)

    def forward(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Encodes a batch of image inputs.

        @args:
            texts (List[List[str]]): List of list of text descriptions.

        @returns:
            torch.Tensor: Text embeddings of shape (batch_size, embedding_dim).
        """
        # Combine multiple paragraphs into single strings
        processed_texts = [" ".join(paragraphs) for paragraphs in texts]

        # Tokenize text
        tokens = self.tokenizer(processed_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.model.device)
        attention_mask = tokens['attention_mask'].to(self.model.device)

        # Generate embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        return text_features  # Shape: (batch_size, embedding_dim)
