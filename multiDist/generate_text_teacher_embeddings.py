import argparse
import json
from pathlib import Path
import sys
import os
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import re


SMALL_MODELS = [
    "Snowflake/snowflake-arctic-embed-s",
    "Snowflake/snowflake-arctic-embed-xs",
    "Snowflake/snowflake-arctic-embed-m",
    "Snowflake/snowflake-arctic-embed-l",
    # "BAAI/bge-base-en-v1.5",
    # "infgrad/stella-base-en-v2",
    # "intfloat/e5-large-v2",
    # "intfloat/multilingual-e5-small",
    # "sentence-transformers/sentence-t5-xl",
    # "sentence-transformers/sentence-t5-large",
    # "SmartComponents/bge-micro-v2",
    # "sentence-transformers/allenai-specter",
    # "sentence-transformers/average_word_embeddings_glove.6B.300d",
    # "sentence-transformers/average_word_embeddings_komninos",
    # "sentence-transformers/LaBSE",
    # "avsolatorio/GIST-Embedding-v0",
    # "Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    # "princeton-nlp/sup-simcse-bert-base-uncased",
    # "jinaai/jina-embedding-s-en-v1",
    # "sentence-transformers/msmarco-bert-co-condensor",
    # "sentence-transformers/gtr-t5-base",
    # "izhx/udever-bloom-560m",
    # "llmrails/ember-v1",
    # "jamesgpt1/sf_model_e5",
    # "thenlper/gte-large",
    # "TaylorAI/gte-tiny",
    # "sentence-transformers/gtr-t5-xl",
    # "intfloat/e5-small",
    # "sentence-transformers/gtr-t5-large",
    # "thenlper/gte-base",
    # "sentence-transformers/all-distilroberta-v1",
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "sentence-transformers/all-mpnet-base-v2",
    # "dunzhang/stella_en_400M_v5",
    # "dunzhang/stella_en_1.5B_v5",
    # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
]

LARGE_MODELS = [
    # "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    # "Alibaba-NLP/gte-Qwen2-7B-instruct",
    # "Salesforce/SFR-Embedding-2_R",
    # "croissantllm/base_5k",
    # "croissantllm/base_50k",
    # "croissantllm/base_100k",
    # "croissantllm/base_150k",
    # "croissantllm/CroissantCool",
    # "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    # "croissantllm/CroissantLLMBase",
    # "NousResearch/Llama-2-7b-hf",
    # "togethercomputer/LLaMA-2-7B-32K",
    # "google/gemma-7b",
    # "google/gemma-2b",
    # "google/gemma-7b-it",
    # "google/gemma-2b-it",
    # "WhereIsAI/UAE-Large-V1",
    # "Salesforce/SFR-Embedding-Mistral",
    # "GritLM/GritLM-7B",
    # "jspringer/echo-mistral-7b-instruct-lasttoken",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Icannos/distillation_training_gist_medi_mteb")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--no_float16", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset, split="train")

    try:
        model = SentenceTransformer(
            args.model,
            model_kwargs={
                "torch_dtype": torch.float16 if not args.no_float16 else torch.float32,
                "attn_implementation": (
                    "flash_attention_2" if args.flash_attn else None
                ),
            },
            device="cuda",
            trust_remote_code=True,
        )
    except Exception as e:
        model = SentenceTransformer(
            args.model,
            model_kwargs={
                "torch_dtype": torch.float16 if not args.no_float16 else torch.float32,
            },
            device="cuda",
            trust_remote_code=True,
        )

    if model.tokenizer.eos_token is not None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    output_dir = Path(args.output_dir) / args.model / args.dataset / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # find all files that fit the pattern
    # f"embeddings-{i}-{i+100000}.npy"
    # make regex pattern to retrieve start and end
    pattern = re.compile(r"embeddings-(\d+)-(\d+).npy")
    # find all files that fit the pattern
    files = list(output_dir.glob("embeddings-*.npy"))

    start = args.start
    if len(files) > 0:
        # extract start and end from the file names
        ends = [int(pattern.search(str(file)).group(2)) for file in files]
        # get largest end
        start = max(ends)

    # split dataset["text"] into chunk of 10000
    for i in range(start, len(dataset["text"]), 10000):
        texts = dataset["text"][i : i + 10000]
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float16)

        with open(output_dir / f"embeddings-{i}-{i+10000}.npy", "wb") as f:
            np.save(f, embeddings)

        # save text in jsonl
        with open(output_dir / f"inputs-{i}-{i+10000}.jsonl", "w") as f:
            for text in texts:
                f.write(json.dumps(text) + "\n")


if __name__ == "__main__":
    import sys

    print(sys.version)
    main()