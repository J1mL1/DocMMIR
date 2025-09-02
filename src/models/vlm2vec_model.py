import os
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForCausalLM

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VLM2VecModel(nn.Module):
    def __init__(self, model_name: str = "TIGER-Lab/VLM2Vec-Full", mode: str = "image"):
        """
        Initializes the VLM2Vec model for both image and text encoding.

        @args:
            model_name (str): HuggingFace model name.
            mode (str): Either "image" or "text" to specify processing type.
        """
        super(VLM2VecModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def preprocess_image(self, image_input) -> torch.Tensor:
        """
        Accepts either a file path (str) or a PIL.Image object.
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise ValueError(f"Invalid file path: {image_input}")
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return image




    def preprocess_text(self, texts: Union[List[List[str]], List[str]]) -> List[str]:
        """
        Preprocesses text batch into flat string list.
        """
        if isinstance(texts, list) and all(isinstance(item, list) for item in texts):
            texts = [" ".join(paragraphs) for paragraphs in texts]
        return texts

    def forward(self, inputs: Union[List[List[str]], List[List[torch.Tensor]]]) -> torch.Tensor:
        if self.mode == "image":
            if isinstance(inputs[0], str):  
                inputs = [inputs] # Convert to batch with 1 image list

            all_image_embeddings = []

            for image_list in inputs:
                images = [self.preprocess_image(image_path) for image_path in image_list]

                texts = [f"<image{i}>" for i in range(len(images))]


                
                print(f"[DEBUG] images: {len(images)} | tags: {texts}")

                processor_inputs = self.processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    
                )

                processor_inputs = {k: v.to(self.device) for k, v in processor_inputs.items()}

                with torch.no_grad():
                    outputs = self.model.vision_model(**processor_inputs, output_hidden_states=True)
                    cls_tokens = outputs.hidden_states[-1][:, 0, :]
                avg_emb = cls_tokens.mean(dim=0, keepdim=True) if len(cls_tokens) > 1 else cls_tokens
                all_image_embeddings.append(avg_emb)

            return torch.cat(all_image_embeddings, dim=0)

        elif self.mode == "text":
            texts = self.preprocess_text(inputs)

            
            texts = [f"<image0> {text}" if not text.startswith("<image") else text for text in texts]

            print(f"[DEBUG TEXT] {texts}")

            processor_inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            processor_inputs = {k: v.to(self.device) for k, v in processor_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**processor_inputs, output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            return cls_embeddings