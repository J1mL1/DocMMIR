import os
from typing import List, Dict, Union, Tuple
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import utils


class TextImageDataset(Dataset):
    """
    Custom Dataset for multi-modal data.
    Each sample consists of multiple images and texts, along with a query.
    """

    def __init__(self, data: List[Dict[str, Union[str, List[str]]]], split_name: str):
        """
        Initializes the TextImageDataset.

        @args:
            data (List[Dict[str, Union[str, List[str]]]]): List of dictionaries with keys 'texts', 'query', 'images'.
            index_mapping_path (str): Path to the index mapping JSON file.
        """
        self.data = data
        self.split_name = split_name
        # Cache the currently loaded .pt file
        self.cached_batch = None
        self.cached_batch_idx = None

    # @classmethod
    # def from_huggingface(cls, dataset_name: str, split: str = "train"):
    #     """
    #     Loads the dataset from Hugging Face.
    #
    #     @args:
    #         dataset_name (str): The name of the dataset on Hugging Face.
    #         split (str): The split of the dataset to load (default is 'train').
    #
    #     @returns:
    #         TextImageDataset: A TextImageDataset object initialized with the loaded data.
    #     """
    #     dataset = load_dataset(dataset_name, split=split)
    #
    #     data = []
    #     for item in dataset:
    #         data.append({
    #             "images": item["images"],
    #             "texts": item["texts"],
    #             "query": item["query"],
    #             "screenshot": item.get("screenshot", "")  # Optional
    #         })
    #
    #     return cls(data)

    @classmethod
    def from_json(
            cls,
            json_path: str,
            load_all: bool = False,
            dry_run: bool = False
    ) -> [Tuple['TextImageDataset', 'TextImageDataset', 'TextImageDataset'] | 'TextImageDataset']:
        """
        Loads train, validation, and test datasets from a single JSON file.

        @args:
            json_path (str): Path to the combined JSON file containing all entries.
            load_all (bool): If True, returns a single dataset with all entries.
            dry_run (bool): If True and load_all is False, only 10% of the training data will be used.
                            This is useful for testing/training on a smaller subset.
        @returns:
            Tuple[TextImageDataset, TextImageDataset, TextImageDataset]: Train, validation, and test datasets.
        """
        # Load the full dataset from the JSON file
        data = utils.read_json(json_path)

        if not load_all:
            # Split the dataset based on the 'class' field
            train_data = [entry for entry in data if entry.get('class') == 'train']
            val_data = [entry for entry in data if entry.get('class') == 'valid']
            test_data = [entry for entry in data if entry.get('class') == 'test']

            # If dry_run is enabled, randomly select 10% of the training data
            if dry_run and train_data:
                original_count = len(train_data)
                train_data = train_data[:max(1, int(0.1 * original_count))]
                print(f"Dry run enabled: Reduced training samples from {original_count} to {len(train_data)}")

            print(
                f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

            # Return split datasets
            return (
                cls(train_data, 'train'),
                cls(val_data, 'valid'),
                cls(test_data, 'test')
            )

        else:
            # Return all datasets
            return cls(data, 'all')

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        @returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Returns the sample at the given index.

        @args:
            idx (int): Index of the sample.

        @returns:
            Dict[str, Union[str, List[str], torch.Tensor]]: A dictionary containing 'texts', 'query', and preprocessed images.
        """
        sample = self.data[idx]
        images = sample["images"]
        if len(images) > 10:
            images = random.sample(images, 10)
        texts  = sample["texts"]
        query  = sample["query"]
        return {"texts": texts, "query": query, "images": images}


def custom_collate_fn(batch: List[Dict[str, Union[str, List[str], List[torch.Tensor]]]], image_encoder: nn.Module,
                      image_dir: str) -> Dict[str, Union[List[str], List[List[torch.Tensor]]]]:
    """
    Custom collate function to handle variable-sized inputs in the batch, using precomputed image tensors.

    @args:
        batch (List[Dict[str, Union[str, List[str], List[torch.Tensor]]]]): List of dictionaries, where each dictionary contains
                                                                            'query', 'texts', and preloaded 'images'.

    @returns:
        Dict[str, Union[List[str], List[List[torch.Tensor]]]]: Batched inputs with precomputed images.
    """
    queries = [item['query'] for item in batch]  # Raw queries as strings
    texts = [item['texts'] for item in batch]  # List of lists of text descriptions
    images = [item['images'] for item in batch]
    try:
        # Process the images
        preprocessed_images = []
        for image_group in images:
            image_tensors = []
            for img_file in image_group:
                img_path = os.path.join(image_dir, img_file)
                image_tensor = image_encoder.preprocess_image(img_path)
                image_tensors.append(image_tensor)
            preprocessed_images.append(image_tensors)

    except Exception as e:
        raise e

    return {
        'queries': queries,
        'texts': texts,
        'images': preprocessed_images  # Keep images as List[List[Tensor]] for processing in the model
    }


class GlobalQueryDataset(Dataset):
    """
    A dataset for global test queries. It loads precomputed query embeddings from a .pt file
    and a global index mapping from another .pt file. The mapping file should be a list
    (of length equal to the number of queries) where each element is the correct global
    document index for the corresponding query.

    Each sample returned is a dictionary with keys:
      - "query_emb": the precomputed query embedding (tensor)
      - "global_index": the global document index (int)
    """

    def __init__(self, query_pt_path: str, global_index_path: str):
        """
        @args:
            query_pt_path (str): Path to the .pt file containing precomputed query embeddings.
                                  Expected shape: [num_queries, embedding_dim].
            global_index_path (str): Path to the .pt file containing the global index mapping.
                                     This should be a list of length num_queries.
        """
        self.query_embeddings = torch.load(query_pt_path)  # Tensor of shape [num_queries, D]
        self.global_indices = torch.load(global_index_path)

        if len(self.global_indices) != self.query_embeddings.size(0):
            raise ValueError(
                f"Length of global indices ({len(self.global_indices)}) does not match number "
                f"of query embeddings ({self.query_embeddings.size(0)})."
            )

    def __len__(self) -> int:
        return self.query_embeddings.size(0)

    def __getitem__(self, idx: int):
        return {
            "query_emb": self.query_embeddings[idx],
            "global_index": self.global_indices[idx]
        }