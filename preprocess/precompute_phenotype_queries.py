import json
import os
import sys
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.text_encoder import TextEncoder
from pretraining.pml import load_balanced_phenotype_specs


@dataclass
class Arguments:
    stage: str = field(default="discover")
    pair_count_path: str = field(
        default="/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_pair_counts.csv"
    )
    phenotype_spec_output_path: str = field(
        default="/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_query_specs.json"
    )
    query_embedding_cache: str = field(
        default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/phenotype_query_knowledge_embeddings.pt"
    )
    knowledge_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt"
    )
    knowledge_encoder_base_model_path: str = field(
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    )
    query_max_length: int = field(default=128)
    query_embedding_batch_size: int = field(default=256)


def checkpoint_path(path):
    if os.path.isfile(path):
        return path
    for filename in ("model.safetensors", "pytorch_model.bin", "best.pt"):
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_state_dict(path):
    state = load_file(path) if path.endswith(".safetensors") else torch.load(
        path, map_location="cpu", weights_only=False
    )
    state = state.get("state_dict", state)
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def load_encoder(args, device):
    model = TextEncoder(args.knowledge_encoder_base_model_path)
    resolved = checkpoint_path(args.knowledge_encoder_path)
    if resolved:
        state = load_state_dict(resolved)
        matched = {
            key: value for key, value in state.items()
            if key in model.state_dict() and value.shape == model.state_dict()[key].shape
        }
        model.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)} knowledge-encoder tensors from {resolved}")
    return model.to(device).eval()


def discover(args, specs):
    os.makedirs(os.path.dirname(args.phenotype_spec_output_path), exist_ok=True)
    with open(args.phenotype_spec_output_path, "w", encoding="utf-8") as file:
        json.dump(specs, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(specs)} balanced phenotype query specs")


def encode(args, specs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.knowledge_encoder_base_model_path, use_fast=True)
    model = load_encoder(args, device)
    query_texts = {spec["key"]: spec["query_text"] for spec in specs}
    cached = {}
    if os.path.exists(args.query_embedding_cache):
        previous = torch.load(args.query_embedding_cache, map_location="cpu", weights_only=False)
        previous_texts = previous.get("query_texts", {})
        cached = {
            key: value.float() for key, value in previous.get("embeddings", {}).items()
            if previous_texts.get(key) == query_texts.get(key)
        }
    missing = [key for key in sorted(query_texts) if key not in cached]
    for start in tqdm(range(0, len(missing), args.query_embedding_batch_size)):
        keys = missing[start : start + args.query_embedding_batch_size]
        tokens = tokenizer(
            [query_texts[key] for key in keys],
            padding=True,
            truncation=True,
            max_length=args.query_max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            embeddings = model.encode_text(tokens).float().cpu()
        cached.update({key: embeddings[index] for index, key in enumerate(keys)})
    cached = {key: cached[key] for key in query_texts}
    os.makedirs(os.path.dirname(args.query_embedding_cache), exist_ok=True)
    torch.save(
        {
            "embeddings": cached,
            "text_dim": int(next(iter(cached.values())).numel()),
            "model_path": args.knowledge_encoder_path,
            "base_model_path": args.knowledge_encoder_base_model_path,
            "query_texts": query_texts,
        },
        args.query_embedding_cache,
    )
    print(f"Encoded {len(missing)} new queries; saved {len(cached)} embeddings")


def main():
    (args,) = HfArgumentParser((Arguments,)).parse_args_into_dataclasses()
    specs = load_balanced_phenotype_specs(args.pair_count_path)
    if args.stage == "discover":
        discover(args, specs)
    elif args.stage == "encode":
        discover(args, specs)
        encode(args, specs)
    else:
        raise ValueError("stage must be discover or encode")


if __name__ == "__main__":
    main()
