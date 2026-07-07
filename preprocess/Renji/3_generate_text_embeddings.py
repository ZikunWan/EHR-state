"""
Generate pre-computed text embeddings for unique table texts in Renji.

Usage:
    python preprocess/Renji/3_generate_text_embeddings.py --stage harvest

    torchrun --nproc_per_node=4 preprocess/Renji/3_generate_text_embeddings.py --stage encode \
        --model-path /data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT \
        --final-output /data/zikun_workspace/.cache/embeddings/renji/text_embeddings_clinicalbert.pt

    torchrun --nproc_per_node=4 preprocess/Renji/3_generate_text_embeddings.py --stage encode \
        --model-path /data/zikun_workspace/checkpoints/pretraining/knowledge_encode/epoch_100.pt \
        --final-output /data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage1.pt

    torchrun --nproc_per_node=4 preprocess/Renji/3_generate_text_embeddings.py --stage encode \
        --model-path /data/zikun_workspace/checkpoints/knowledge_encoder/clinicalBERT_after_stage2/epoch_5.pt \
        --final-output /data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dataset.renji.renji_dataset import RenjiDataset
from models.TableEncoder.text_encoder import TextEncoder


_HARVEST_DATASET = None


def collect_table_texts(df):
    unique_texts = set()
    if not df.empty:
        unique_texts.update(df["Item"].dropna().astype(str).unique())
        unique_texts.update(df["Unit"].dropna().astype(str).unique())
        value_texts = df["Value"].dropna().astype(str)
        numeric_values = pd.to_numeric(value_texts, errors="coerce")
        unique_texts.update(value_texts[pd.isna(numeric_values)].unique())
    return unique_texts


def init_harvest_worker(root_dir, split):
    global _HARVEST_DATASET
    _HARVEST_DATASET = RenjiDataset(
        root_dir=root_dir,
        split=split,
        shuffle=False,
    )


def harvest_worker(worker_args):
    part_idx, slice_list = worker_args
    dataset = _HARVEST_DATASET
    if dataset is None:
        raise RuntimeError("Harvest worker dataset was not initialized.")

    local_unique = set()
    for sample_idx in slice_list:
        sample = dataset[sample_idx]
        df = sample["measurement_table"]
        if df is not None:
            local_unique.update(collect_table_texts(df))
    return part_idx, local_unique


def harvest_part_path(parts_dir, split, part_idx):
    return os.path.join(parts_dir, f"{split}_part_{part_idx:06d}.pkl")


def save_harvest_part(parts_dir, split, part_idx, texts):
    with open(harvest_part_path(parts_dir, split, part_idx), "wb") as f:
        pickle.dump(texts, f)


def load_harvest_part(parts_dir, split, part_idx):
    with open(harvest_part_path(parts_dir, split, part_idx), "rb") as f:
        return pickle.load(f)


def cleanup_harvest_parts(parts_dir):
    if os.path.isdir(parts_dir):
        for name in os.listdir(parts_dir):
            if name.endswith(".pkl"):
                os.remove(os.path.join(parts_dir, name))
        os.rmdir(parts_dir)


def harvest_unique_texts(root_dir, splits, harvest_checkpoint, num_workers, num_harvest_chunks):
    os.makedirs(os.path.dirname(harvest_checkpoint), exist_ok=True)
    parts_dir = f"{harvest_checkpoint}.parts"
    os.makedirs(parts_dir, exist_ok=True)

    print("Phase 1: Harvesting unique texts from Renji tables...")
    split_part_counts = {}

    for split in splits:
        dataset = RenjiDataset(
            root_dir=root_dir,
            split=split,
            shuffle=False,
        )

        unique_slices = {}
        for i, sample_info in enumerate(dataset.samples):
            key = (sample_info["fname_key"], sample_info["cutoff_day"])
            if key not in unique_slices:
                unique_slices[key] = i

        slice_indices = list(unique_slices.values())
        print(
            f"{split}: total samples={len(dataset)}, "
            f"unique patients/windows={len(slice_indices)}"
        )

        num_chunks = max(1, min(len(slice_indices), num_harvest_chunks))
        split_part_counts[split] = num_chunks
        chunks = np.array_split(slice_indices, num_chunks)
        pending_args = [
            (part_idx, list(chunk))
            for part_idx, chunk in enumerate(chunks)
            if not os.path.exists(harvest_part_path(parts_dir, split, part_idx))
        ]

        print(f"{split}: harvest parts={num_chunks}, pending={len(pending_args)}")

        if pending_args:
            effective_workers = max(1, min(num_workers, len(pending_args)))
            imap_chunksize = max(1, len(pending_args) // (effective_workers * 4))
            with Pool(
                processes=effective_workers,
                initializer=init_harvest_worker,
                initargs=(root_dir, split),
            ) as pool:
                for part_idx, result in tqdm(
                    pool.imap_unordered(
                        harvest_worker,
                        pending_args,
                        chunksize=imap_chunksize,
                    ),
                    total=len(pending_args),
                    desc=f"{split} CPU Harvesting",
                ):
                    save_harvest_part(parts_dir, split, part_idx, result)

    all_unique = set()
    for split, num_chunks in split_part_counts.items():
        for part_idx in range(num_chunks):
            all_unique.update(load_harvest_part(parts_dir, split, part_idx))

    unique_texts = [str(text) for text in all_unique if text and str(text).strip()]
    print(f"Harvested {len(unique_texts)} unique strings. Saving checkpoint...")

    with open(harvest_checkpoint, "wb") as f:
        pickle.dump(unique_texts, f)
    cleanup_harvest_parts(parts_dir)
    print(f"Successfully saved to {harvest_checkpoint}")


def init_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def get_rank_info(distributed):
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        return rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def load_unique_texts(harvest_checkpoint):
    with open(harvest_checkpoint, "rb") as f:
        return pickle.load(f)


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def load_embedding_model(model_path, base_model_path, device):
    if model_path.endswith(".pt"):
        model = TextEncoder(base_model_path)
        state_dict = load_checkpoint_state_dict(model_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {model_path}")
        return model.to(device), "text_encoder"

    return AutoModel.from_pretrained(model_path).to(device), "auto_model"


def encode_batch(model, model_kind, tokens):
    with torch.no_grad():
        if model_kind == "text_encoder":
            return model.encode_text(tokens).cpu()

        out = model(**tokens)
        return out.last_hidden_state[:, 0, :].cpu()


def get_text_dim(model, model_kind):
    if model_kind == "text_encoder":
        return model.hidden_size
    return model.config.hidden_size


def encode_texts(
    model_path,
    base_model_path,
    cache_dir,
    harvest_checkpoint,
    final_output,
    rank,
    world_size,
    device,
    batch_size,
    max_token_len,
    distributed,
):
    unique_texts = load_unique_texts(harvest_checkpoint)

    print(f"Rank {rank}: Encoding phase starting on {device}...")
    tokenizer_path = base_model_path if model_path.endswith(".pt") else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model, model_kind = load_embedding_model(model_path, base_model_path, device)
    model.eval()

    my_shard = np.array_split(unique_texts, world_size)[rank]
    print(f"Rank {rank}: Processing {len(my_shard)} strings.")

    partial_checkpoint = os.path.join(cache_dir, f"partial_embs_rank_{rank}.pt")
    embeddings_dict = {}

    for i in tqdm(
        range(0, len(my_shard), batch_size),
        desc=f"Rank {rank} Encoding",
        disable=distributed and rank != 0,
    ):
        batch_texts = list(my_shard[i : i + batch_size])
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_token_len,
            return_tensors="pt",
        ).to(device)

        embs = encode_batch(model, model_kind, tokens)

        for j, text in enumerate(batch_texts):
            embeddings_dict[text] = embs[j]

        if (i // batch_size) % 50 == 0:
            torch.save(embeddings_dict, partial_checkpoint)

    torch.save(embeddings_dict, partial_checkpoint)

    if distributed:
        dist.barrier()

    if rank == 0:
        print("Phase 3: Merging partial results...")
        final_embeddings = {}
        for shard_rank in range(world_size):
            shard_path = os.path.join(cache_dir, f"partial_embs_rank_{shard_rank}.pt")
            shard_embs = torch.load(shard_path, weights_only=False)
            final_embeddings.update(shard_embs)

        text_dim = get_text_dim(model, model_kind)
        print(f"Merged {len(final_embeddings)} embeddings. Dimension: {text_dim}")
        torch.save(
            {
                "embeddings": final_embeddings,
                "text_dim": text_dim,
                "model_path": model_path,
                "base_model_path": base_model_path,
            },
            final_output,
        )

        for shard_rank in range(world_size):
            os.remove(os.path.join(cache_dir, f"partial_embs_rank_{shard_rank}.pt"))

        print(f"Successfully saved to {final_output}")
        print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Harvest Renji table texts and generate text embeddings."
    )
    parser.add_argument(
        "--stage",
        choices=["harvest", "encode", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/data/zikun_workspace/input/tables/renji/raw",
        help="Renji data root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to harvest from root_dir/index.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model path or text encoder .pt checkpoint.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT",
        help="Base HuggingFace model/tokenizer path used when --model-path is a .pt checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/zikun_workspace/.cache/embeddings/renji",
        help="Embedding cache dir.",
    )
    parser.add_argument(
        "--harvest-checkpoint",
        type=str,
        default="",
        help="Unique text checkpoint. Defaults to <cache-dir>/unique_texts_harvested.pkl.",
    )
    parser.add_argument(
        "--final-output",
        type=str,
        default="",
        help="Final embedding cache. Defaults to <cache-dir>/text_embeddings.pt.",
    )
    parser.add_argument("--num-workers", type=int, default=32, help="CPU workers for harvesting.")
    parser.add_argument("--num-harvest-chunks", type=int, default=1024, help="Harvest checkpoint chunks per split.")
    parser.add_argument("--batch-size", type=int, default=512, help="BERT encoding batch size.")
    parser.add_argument("--max-token-len", type=int, default=512, help="Tokenizer max length.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.harvest_checkpoint = args.harvest_checkpoint or os.path.join(
        args.cache_dir, "unique_texts_harvested.pkl"
    )
    args.final_output = args.final_output or os.path.join(args.cache_dir, "text_embeddings.pt")
    os.makedirs(args.cache_dir, exist_ok=True)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        init_distributed()
    rank, world_size, device = get_rank_info(distributed)

    if args.stage in ["harvest", "all"]:
        if rank == 0:
            harvest_unique_texts(
                root_dir=args.root_dir,
                splits=args.splits,
                harvest_checkpoint=args.harvest_checkpoint,
                num_workers=args.num_workers,
                num_harvest_chunks=args.num_harvest_chunks,
            )
        if distributed:
            dist.barrier()

    if args.stage in ["encode", "all"]:
        encode_texts(
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            cache_dir=args.cache_dir,
            harvest_checkpoint=args.harvest_checkpoint,
            final_output=args.final_output,
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=args.batch_size,
            max_token_len=args.max_token_len,
            distributed=distributed,
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
