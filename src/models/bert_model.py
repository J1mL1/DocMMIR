from typing import List

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERTModel(nn.Module):
    def __init__(self, model_name: str = 'google-bert/bert-base-uncased'):
        """
        Initializes the BertTextEncoder with a BERT tokenizer and model.

        @args:
            model_name (str): Name of the BERT model to use. Defaults to 'bert-base-uncased'.
        """
        super(BERTModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Encodes a batch of text inputs.

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

    def get_embedding_size(self) -> int:
        """
        Returns the size of the embeddings.
        """
        return self.model.config.hidden_size