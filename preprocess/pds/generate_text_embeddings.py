import argparse
import csv
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


TEXT_COLUMNS = ("Item", "Unit")


def is_numeric(value):
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def discover_patient_csvs(root_dir, trial_ids):
    patient_csvs = []
    for trial_id in trial_ids:
        patients_dir = os.path.join(root_dir, str(trial_id), "patients")
        for filename in sorted(os.listdir(patients_dir)):
            if filename.endswith(".csv"):
                patient_csvs.append(os.path.join(patients_dir, filename))
    return patient_csvs


def collect_texts_from_csv(csv_path):
    texts = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in TEXT_COLUMNS:
                text = str(row[column]).strip()
                if text:
                    texts.add(text)
            value = str(row["Value"]).strip()
            if value and not is_numeric(value):
                texts.add(value)
    return texts


def harvest_unique_texts(root_dir, trial_ids, harvest_checkpoint):
    print("Harvesting PDS table texts...")
    all_texts = set()
    patient_csvs = discover_patient_csvs(root_dir, trial_ids)
    for idx, csv_path in enumerate(patient_csvs, start=1):
        if idx == 1 or idx % 1000 == 0 or idx == len(patient_csvs):
            print(f"PDS harvest: {idx}/{len(patient_csvs)} patient files")
        all_texts.update(collect_texts_from_csv(csv_path))

    unique_texts = sorted(text for text in all_texts if text.strip())
    os.makedirs(os.path.dirname(harvest_checkpoint), exist_ok=True)
    with open(harvest_checkpoint, "wb") as f:
        pickle.dump(unique_texts, f)
    print(f"Saved {len(unique_texts)} unique texts to {harvest_checkpoint}")


def init_distributed():
    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def get_rank_info(distributed):
    import torch
    import torch.distributed as dist

    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def load_checkpoint_state_dict(checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def load_embedding_model(model_path, base_model_path, device):
    from transformers import AutoModel
    from models.TableEncoder.text_encoder import TextEncoder

    if model_path.endswith(".pt"):
        model = TextEncoder(base_model_path)
        state_dict = load_checkpoint_state_dict(model_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {model_path}")
        return model.to(device), "text_encoder"

    return AutoModel.from_pretrained(model_path).to(device), "auto_model"


def encode_batch(model, model_kind, tokens):
    import torch

    with torch.no_grad():
        if model_kind == "text_encoder":
            return model.encode_text(tokens).cpu()

        outputs = model(**tokens)
        return outputs.last_hidden_state[:, 0, :].cpu()


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
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoTokenizer

    with open(harvest_checkpoint, "rb") as f:
        unique_texts = pickle.load(f)

    tokenizer_path = base_model_path if model_path.endswith(".pt") else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model, model_kind = load_embedding_model(model_path, base_model_path, device)
    model.eval()

    shard = unique_texts[rank::world_size]
    partial_checkpoint = os.path.join(cache_dir, f"partial_embs_rank_{rank}.pt")
    embeddings = {}

    for start in tqdm(
        range(0, len(shard), batch_size),
        desc=f"Rank {rank} encoding",
        disable=distributed and rank != 0,
    ):
        batch_texts = shard[start:start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_token_len,
            return_tensors="pt",
        ).to(device)
        batch_embeddings = encode_batch(model, model_kind, tokens)
        for idx, text in enumerate(batch_texts):
            embeddings[text] = batch_embeddings[idx]

    torch.save(embeddings, partial_checkpoint)
    if distributed:
        dist.barrier()

    if rank == 0:
        final_embeddings = {}
        for shard_rank in range(world_size):
            shard_path = os.path.join(cache_dir, f"partial_embs_rank_{shard_rank}.pt")
            final_embeddings.update(torch.load(shard_path, map_location="cpu", weights_only=False))

        os.makedirs(os.path.dirname(final_output), exist_ok=True)
        torch.save(
            {
                "embeddings": final_embeddings,
                "text_dim": get_text_dim(model, model_kind),
                "model_path": model_path,
                "base_model_path": base_model_path,
            },
            final_output,
        )

        for shard_rank in range(world_size):
            os.remove(os.path.join(cache_dir, f"partial_embs_rank_{shard_rank}.pt"))
        print(f"Saved {len(final_embeddings)} embeddings to {final_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDS table text embeddings.")
    parser.add_argument("--stage", choices=["harvest", "encode", "all"], default="all")
    parser.add_argument("--root-dir", default="/data/zikun_workspace/input/tables/PDS")
    parser.add_argument(
        "--trial-ids",
        nargs="+",
        default=["102", "103", "105", "118", "119", "121", "122", "127", "128", "149"],
    )
    parser.add_argument(
        "--model-path",
        default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt",
    )
    parser.add_argument(
        "--base-model-path",
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT",
    )
    parser.add_argument("--cache-dir", default="/data/zikun_workspace/input/cache/embeddings/PDS")
    parser.add_argument("--harvest-checkpoint", default="")
    parser.add_argument("--final-output", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-token-len", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    args.harvest_checkpoint = args.harvest_checkpoint or os.path.join(
        args.cache_dir, "unique_texts_harvested.pkl"
    )
    args.final_output = args.final_output or os.path.join(args.cache_dir, "text_embeddings_stage2.pt")
    os.makedirs(args.cache_dir, exist_ok=True)

    needs_encode = args.stage in {"encode", "all"}
    distributed = needs_encode and int(os.environ.get("WORLD_SIZE", "1")) > 1
    dist = None
    if distributed:
        init_distributed()
        import torch.distributed as dist
    if needs_encode:
        rank, world_size, device = get_rank_info(distributed)
    else:
        rank, world_size, device = 0, 1, None

    if args.stage in {"harvest", "all"}:
        if rank == 0:
            harvest_unique_texts(
                root_dir=args.root_dir,
                trial_ids=args.trial_ids,
                harvest_checkpoint=args.harvest_checkpoint,
            )
        if distributed:
            dist.barrier()

    if args.stage in {"encode", "all"}:
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
