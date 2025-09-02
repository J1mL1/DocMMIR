import logging
import os
import argparse
from typing import List, NamedTuple, Optional
import ijson
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info

import utils
from utils import read_json, save_json


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[list[int]]
    image_data: list[Image]
    chat_template: Optional[str]


def load_qwen2_vl(
        llm: LLM,
        processor,
        doc_text: str,
        image_paths: List[str],
        image_dir: str,

) -> ModelRequestData:
    """
    Constructs the ModelRequestData using preloaded llm and processor.
    """
    # Create image placeholders by joining the image_dir with each image file name.
    placeholders = [
        {"type": "image", "image": os.path.join(image_dir, image_path)}
        for image_path in image_paths
    ]

    # Construct conversation messages:
    # - The system message sets the assistant's role.
    # - The user message includes image placeholders and an instruction prompt.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful natural language processing expert."
        },
        {
            "role": "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": (
                            "You are tasked with generating a thought-provoking question based on the given image-text data which is part of a document. "
                            "The question should reflect the overall theme or deeper meaning of the document rather than focus on specific details visible in the images. "
                            "It must be abstract, encouraging critical reflection and analysis, and should not have a straightforward answer from the context. "
                            "Avoid being overly generic—ensure the question resonates with the unique visual elements of the document."
                            "Begin your output with 'Q:' followed by the generated question. \n" + doc_text

                        # "Given the following image from Wikipedia document, analyze the content to identify key "
                        # "visual features and underlying themes in the text. Then, generate a single short title "
                        # "that requires synthesizing insights from both the image and the text. The summary should "
                        # "be multi-layered and thought-provoking, encouraging a deeper analysis of how the visual "
                        # "elements complement or contrast with the written narrative. focus on exploring the "
                        # "implicit relationships or potential implications suggested by the combined modalities. "
                        # "be short no more than 30 words"
                    )
                }
            ]
        }
    ]

    # Load the processor associated with the model to generate the final prompt.
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    stop_token_ids = None

    # Process vision information to prepare local images for model input.
    image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
        chat_template=None,
    )


def process_document(doc: dict, image_dir: str, llm: LLM, processor) -> dict:
    """
    Processes a single document from the dataset.
    """
    # Combine all text segments into one document string.
    doc_text = "\n".join(doc["texts"])
    image_paths = doc["images"][:6]

    # Create the model request data using the document text and image paths.
    request_data = load_qwen2_vl(llm, processor, doc_text, image_paths, image_dir)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        stop_token_ids=request_data.stop_token_ids,
        max_tokens=200
    )

    # Generate the QA pairs using the preloaded LLM instance.
    results = request_data.llm.generate(
        {
            "prompt": request_data.prompt,
            "multi_modal_data": {
                "image": request_data.image_data
            },
        },
        sampling_params=sampling_params
    )
    generated_text = ''
    for res in results:
        generated_text = res.outputs[0].text

    doc['query'] = generated_text
    return doc


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from multimodal document data using Qwen2-VL-7B-Instruct."
    )
    parser.add_argument(
        "--doc_file", type=str, required=True, help="Path to the JSON file containing the document dataset."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory path where images are stored."
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Path to the JSON file containing the document dataset."
    )

    args = parser.parse_args()

    # Pre-load the model and processor once.
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": 6},
        tensor_parallel_size=4
    )

    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_name, max_pixels=max_pixels
    )

    logging.basicConfig(
        filename="arxiv_qa.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    result = []
    doc_count = 0
    batch_size = 10000

    with open(args.doc_file, 'r', encoding='utf-8') as f_in:
        docs = ijson.items(f_in, 'item')
        for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
            try:
                processed_doc = process_document(doc, image_dir=args.image_dir, llm=llm, processor=processor)
                result.append(processed_doc)
                doc_count += 1

                if doc_count % batch_size == 0:
                    temp_out_file = f"{args.out_file.rstrip('.json')}_part_{doc_count // batch_size}.json"
                    utils.save_json(result, temp_out_file)
                    logging.info(f"Saved {doc_count} documents to {temp_out_file}")
                    result = []

            except Exception as e:
                logging.error(f"Error processing document index {idx} (query: {doc.get('query', 'unknown')}): {e}")
                continue

    # Write remaining documents after loop completes
    if result:
        temp_out_file = f"{args.out_file.rstrip('.json')}_final.json"
        utils.save_json(result, temp_out_file)
        logging.info(f"Saved remaining {len(result)} documents to {temp_out_file}")


if __name__ == "__main__":
    main()
