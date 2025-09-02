import os
import json
import argparse
import ijson
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import save_json


def build_prompt(doc_text: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a generous language quality classifier. "
                "Your task is to determine whether a given text segment, possibly extracted from an OCR-processed document, "
                "likely contains meaningful human-written content. "
                "You should accept text that is partially broken, informal, or noisy, as long as it seems intended to communicate something relevant. "
                "Accept marketing language, product descriptions, announcements, or technical explanations. "
                "Only reject text if it is purely noise, random symbols, or unreadable junk. /no_think"
            )
        },
        {
            "role": "user",
            "content": (
                "Below are some examples:\n\n"
                "Text: 'Figure 3: 0.233!!@@## 19982ab' → No\n"
                "Text: 'Explori enables survey management for licensed events.' → Yes\n"
                "Text: 'Chart axis: year, value, growth' → No\n"
                "Text: 'This document introduces a framework for multimodal IR tasks in scientific domains.' → Yes\n"
                "Text: 'http://bit.ly/xyz download summary' → No\n"
                "Text: 'Project overview and next steps: iterate, test, deploy' → Yes\n"
                "---\n\n"
                "Now classify the following:\n\n"
                f"Text: {doc_text}\n\n"
                "Is this meaningful human language? Respond with one word only: 'Yes' or 'No'."
            )
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def process_document(doc: dict, llm: LLM, tokenizer) -> dict:
    doc_text = "\n".join(doc.get("texts", []))
    prompt = build_prompt(doc_text, tokenizer)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1500
    )

    outputs = llm.generate(prompt, sampling_params)
    response = outputs[0].outputs[0].text.strip()

    lines = response.strip().splitlines()
    candidates = [line.strip().lower() for line in lines if line.strip().lower() in ("yes", "no")]
    if candidates:
        doc["valid_doc"] = candidates
    else:
        doc["valid_doc"] = "unknown"
    doc["llm_response"] = response
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()

    # Load model + tokenizer
    llm = LLM(model=args.model_path, tensor_parallel_size=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    from itertools import islice
    # Read and process data
    result = []
    with open(args.doc_file, "r", encoding="utf-8") as f:
        docs = ijson.items(f, "item")
        for idx, doc in enumerate(tqdm(docs, desc="Validating documents")):
            try:
                result.append(process_document(doc, llm, tokenizer))
            except Exception as e:
                print(f"Error at doc {idx}: {e}")

    save_json(result, args.out_file)


if __name__ == "__main__":
    main()
