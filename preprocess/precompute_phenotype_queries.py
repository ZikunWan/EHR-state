import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
from transformers import HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from outdated.phenotype_metric_learning import (
    DataArguments,
    build_split_records,
    discover_numeric_query_specs,
    encode_knowledge_query_texts,
    load_query_specs,
)


@dataclass
class PrecomputeArguments:
    stage: str = field(default="discover")
    num_discovery_workers: int = field(default=32)
    phenotype_spec_output_path: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/phenotype_query_specs.json"
    )


def discover_queries(data_args: DataArguments, precompute_args: PrecomputeArguments) -> None:
    train_records, train_datasets = build_split_records(data_args, "train")
    query_specs = discover_numeric_query_specs(
        records=train_records,
        datasets=train_datasets,
        min_occurrence=data_args.min_phenotype_occurrence,
        max_phenotypes=data_args.max_auto_phenotypes,
        statistics=data_args.phenotype_statistics,
        time_windows=data_args.phenotype_time_windows,
        category_regex=data_args.phenotype_category_regex,
        num_workers=precompute_args.num_discovery_workers,
    )

    spec_dir = os.path.dirname(precompute_args.phenotype_spec_output_path)
    if spec_dir:
        os.makedirs(spec_dir, exist_ok=True)
    with open(precompute_args.phenotype_spec_output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(spec) for spec in query_specs], f, indent=2)

    print(f"Saved {len(query_specs)} phenotype query specs to {precompute_args.phenotype_spec_output_path}")


def encode_queries_distributed(data_args: DataArguments, precompute_args: PrecomputeArguments) -> None:
    if not os.path.exists(precompute_args.phenotype_spec_output_path):
        raise FileNotFoundError(
            f"Phenotype query spec not found: {precompute_args.phenotype_spec_output_path}. "
            "Run the discover stage first."
        )

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    query_specs = load_query_specs(precompute_args.phenotype_spec_output_path)
    query_texts = {spec.key: spec.query_text for spec in query_specs}
    cached_embeddings = {}
    if os.path.exists(data_args.query_embedding_cache):
        cache = torch.load(data_args.query_embedding_cache, map_location="cpu", weights_only=False)
        cached_embeddings.update(
            {key: value.float() for key, value in cache.get("embeddings", {}).items()}
        )

    missing_keys = sorted(set(query_texts) - set(cached_embeddings))
    local_keys = missing_keys[rank::world_size]
    local_embeddings = encode_knowledge_query_texts(
        query_texts={key: query_texts[key] for key in local_keys},
        model_path=data_args.knowledge_encoder_path,
        base_model_path=data_args.knowledge_encoder_base_model_path,
        max_length=data_args.query_max_length,
        batch_size=data_args.query_embedding_batch_size,
        device=device,
        show_progress=rank == 0,
    )

    part_dir = f"{data_args.query_embedding_cache}.parts"
    os.makedirs(part_dir, exist_ok=True)
    part_path = os.path.join(part_dir, f"rank_{rank}.pt")
    torch.save(local_embeddings, part_path)
    if distributed:
        dist.barrier()

    if rank == 0:
        embeddings = dict(cached_embeddings)
        for part_rank in range(world_size):
            rank_path = os.path.join(part_dir, f"rank_{part_rank}.pt")
            embeddings.update(torch.load(rank_path, map_location="cpu", weights_only=False))

        missing_after_merge = sorted(set(query_texts) - set(embeddings))
        if missing_after_merge:
            raise RuntimeError(f"Missing {len(missing_after_merge)} query embeddings after distributed encoding.")

        cache_dir = os.path.dirname(data_args.query_embedding_cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{data_args.query_embedding_cache}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        torch.save(
            {
                "embeddings": embeddings,
                "text_dim": int(next(iter(embeddings.values())).numel()),
                "model_path": data_args.knowledge_encoder_path,
                "base_model_path": data_args.knowledge_encoder_base_model_path,
            },
            tmp_path,
        )
        os.replace(tmp_path, data_args.query_embedding_cache)
        print(
            f"Encoded {len(missing_keys)} new queries across {world_size} ranks; "
            f"saved {len(query_texts)} query embeddings to {data_args.query_embedding_cache}"
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = HfArgumentParser((DataArguments, PrecomputeArguments))
    data_args, precompute_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")

    if precompute_args.stage == "discover":
        discover_queries(data_args, precompute_args)
    elif precompute_args.stage == "encode":
        encode_queries_distributed(data_args, precompute_args)
    else:
        raise ValueError("stage must be either 'discover' or 'encode'.")


if __name__ == "__main__":
    main()
