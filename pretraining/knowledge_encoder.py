from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import multiprocessing as mp
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.text_encoder import KnowledgeGraphEncoderForTrainer


RELATION_PROFILE_TOP_K = 3
_BUILD_CONCEPT_IDS: Optional[Set[str]] = None


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


@dataclass
class Args:
    output_dir: str = "/data/zikun_workspace/checkpoints/pretraining/text_encoder"
    model_name_or_path: str = "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 256
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 8
    freeze_bert: bool = False
    bf16: bool = True
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 5000
    save_total_limit: int = 1
    report_to: str = "wandb"
    wandb_project: str = ""
    wandb_run_name: str = ""
    deepspeed: str = ""
    cache_only: bool = False
    resume_from_checkpoint: str = ""
    concept_path: str = "/data/zikun_workspace/input/knowledge/omop/CONCEPT.csv"
    concept_relationship_path: str = "/data/zikun_workspace/input/knowledge/omop/CONCEPT_RELATIONSHIP.csv"
    triple_cache: str = "/data/zikun_workspace/input/knowledge/cache/triples_cache"
    kg_max_triples: Optional[int] = None
    kg_num_negatives: int = 4
    kg_margin: float = 1.0
    kg_distance_p: int = 2
    kg_relation_reg: float = 1e-4
    kg_eval_ratio: float = 0.0
    kg_build_workers: int = 1
    kg_build_chunksize: int = 1_000_000


class ConceptRelationshipDataset(Dataset):
    def __init__(
        self,
        triple_cache: str,
        concepts: Dict[str, str],
        concept_domains: Dict[str, str],
        concept_vocabularies: Dict[str, str],
        num_negatives: int,
        seed: int,
        split: str = "train",
    ):
        if split not in {"train", "eval"}:
            raise ValueError(f"Unknown split: {split}")
        self.concepts = concepts
        self.concept_domains = concept_domains
        self.concept_vocabularies = concept_vocabularies
        self.concept_ids = np.fromiter(
            (int(concept_id) for concept_id in concepts.keys()), dtype=np.int64
        )
        self.domain_to_ids = self._build_type_pools(concept_domains)
        self.vocabulary_to_ids = self._build_type_pools(concept_vocabularies)
        self.num_negatives = num_negatives
        self.seed = seed
        metadata_path = os.path.join(triple_cache, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.split = split
        self.num_triples = _metadata_num_triples(metadata, split)
        self.relation2id: Dict[str, int] = metadata["relation2id"]
        files = _metadata_files(metadata, split)
        if self.num_triples == 0:
            self.head_ids = np.empty(0, dtype=np.int64)
            self.tail_ids = np.empty(0, dtype=np.int64)
            self.relation_ids = np.empty(0, dtype=np.int64)
        else:
            self.head_ids = np.memmap(
                os.path.join(triple_cache, files["head_ids"]),
                dtype=np.int64,
                mode="r",
                shape=(self.num_triples,),
            )
            self.tail_ids = np.memmap(
                os.path.join(triple_cache, files["tail_ids"]),
                dtype=np.int64,
                mode="r",
                shape=(self.num_triples,),
            )
            self.relation_ids = np.memmap(
                os.path.join(triple_cache, files["relation_ids"]),
                dtype=np.int64,
                mode="r",
                shape=(self.num_triples,),
            )
        self.relation_head_domains: Dict[int, List[str]] = {}
        self.relation_tail_domains: Dict[int, List[str]] = {}
        self.relation_head_vocabularies: Dict[int, List[str]] = {}
        self.relation_tail_vocabularies: Dict[int, List[str]] = {}
        self._load_relation_type_profile(metadata)

    def __len__(self) -> int:
        return self.num_triples

    def _build_type_pools(
        self, concept_types: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        pools: DefaultDict[str, List[int]] = defaultdict(list)
        for concept_id in self.concepts:
            concept_type = concept_types.get(concept_id, "")
            if concept_type:
                pools[concept_type].append(int(concept_id))
        return {
            concept_type: np.asarray(concept_ids, dtype=np.int64)
            for concept_type, concept_ids in pools.items()
        }

    def _load_relation_type_profile(self, metadata: Dict[str, object]) -> None:
        profile = metadata.get("relation_type_profile")
        if not isinstance(profile, dict):
            raise ValueError(
                "Triple cache metadata has no relation_type_profile. "
                "Please rebuild the triple cache with the current knowledge_encoder.py."
            )

        for relation_id, values in profile.items():
            if not isinstance(values, dict):
                continue
            relation_id_int = int(relation_id)
            self.relation_head_domains[relation_id_int] = _profile_names(
                values.get("head_domains", [])
            )
            self.relation_tail_domains[relation_id_int] = _profile_names(
                values.get("tail_domains", [])
            )
            self.relation_head_vocabularies[relation_id_int] = _profile_names(
                values.get("head_vocabularies", [])
            )
            self.relation_tail_vocabularies[relation_id_int] = _profile_names(
                values.get("tail_vocabularies", [])
            )

    def _sample_from_pool(
        self, rng: random.Random, pool: Optional[np.ndarray], forbidden_ids: Set[int]
    ) -> Optional[int]:
        if pool is None or len(pool) == 0:
            return None
        for _ in range(50):
            negative_id = int(pool[rng.randrange(len(pool))])
            if negative_id not in forbidden_ids:
                return negative_id
        for negative_id in pool[:100]:
            negative_id_int = int(negative_id)
            if negative_id_int not in forbidden_ids:
                return negative_id_int
        return None

    def _candidate_pools(
        self, relation_id: int, corrupt_head: bool, original_id: int
    ) -> Iterable[np.ndarray]:
        if corrupt_head:
            relation_domains = self.relation_head_domains.get(relation_id, [])
            relation_vocabularies = self.relation_head_vocabularies.get(relation_id, [])
        else:
            relation_domains = self.relation_tail_domains.get(relation_id, [])
            relation_vocabularies = self.relation_tail_vocabularies.get(relation_id, [])

        used_domains = set()
        for domain in relation_domains:
            pool = self.domain_to_ids.get(domain)
            if pool is not None:
                used_domains.add(domain)
                yield pool

        original_id_str = str(original_id)
        original_domain = self.concept_domains.get(original_id_str, "")
        if original_domain and original_domain not in used_domains:
            pool = self.domain_to_ids.get(original_domain)
            if pool is not None:
                yield pool

        used_vocabularies = set()
        for vocabulary in relation_vocabularies:
            pool = self.vocabulary_to_ids.get(vocabulary)
            if pool is not None:
                used_vocabularies.add(vocabulary)
                yield pool

        original_vocabulary = self.concept_vocabularies.get(original_id_str, "")
        if original_vocabulary and original_vocabulary not in used_vocabularies:
            pool = self.vocabulary_to_ids.get(original_vocabulary)
            if pool is not None:
                yield pool

        yield self.concept_ids

    def _sample_negative(
        self, rng: random.Random, head_id: int, tail_id: int, relation_id: int
    ) -> Tuple[str, bool]:
        corrupt_head = rng.random() < 0.5
        original_id = head_id if corrupt_head else tail_id
        forbidden_ids = {head_id, tail_id}
        for pool in self._candidate_pools(relation_id, corrupt_head, original_id):
            negative_id = self._sample_from_pool(rng, pool, forbidden_ids)
            if negative_id is not None:
                return str(negative_id), corrupt_head
        return str(original_id), corrupt_head

    def __getitem__(self, idx: int) -> Dict[str, object]:
        head_id = int(self.head_ids[idx])
        tail_id = int(self.tail_ids[idx])
        relation_id = int(self.relation_ids[idx])
        rng = random.Random(self.seed + idx)
        negative_names = []
        negative_is_head = []
        for _ in range(self.num_negatives):
            negative_id, is_head = self._sample_negative(
                rng, head_id, tail_id, relation_id
            )
            negative_names.append(self.concepts[negative_id])
            negative_is_head.append(is_head)

        return {
            "head_name": self.concepts[str(head_id)],
            "tail_name": self.concepts[str(tail_id)],
            "relation_id": relation_id,
            "negative_names": negative_names,
            "negative_is_head": negative_is_head,
        }


def count_data_lines(path: str) -> int:
    lines = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            lines += block.count(b"\n")
    return max(0, lines - 1)


def count_file_chunks(path: str, chunksize: int) -> int:
    data_lines = count_data_lines(path)
    return max(1, (data_lines + chunksize - 1) // chunksize)


def _metadata_num_triples(metadata: Dict[str, object], split: str) -> int:
    split_key = f"{split}_num_triples"
    if split_key in metadata:
        return int(metadata[split_key])
    if split == "train" and "num_triples" in metadata:
        return int(metadata["num_triples"])
    return 0


def _metadata_files(metadata: Dict[str, object], split: str) -> Dict[str, str]:
    split_key = f"{split}_files"
    files = metadata.get(split_key)
    if isinstance(files, dict):
        return {
            "head_ids": str(files["head_ids"]),
            "tail_ids": str(files["tail_ids"]),
            "relation_ids": str(files["relation_ids"]),
        }
    if split == "train":
        files = metadata.get("files")
        if isinstance(files, dict):
            return {
                "head_ids": str(files["head_ids"]),
                "tail_ids": str(files["tail_ids"]),
                "relation_ids": str(files["relation_ids"]),
            }
    return {
        "head_ids": f"{split}_head_ids.int64.bin",
        "tail_ids": f"{split}_tail_ids.int64.bin",
        "relation_ids": f"{split}_relation_ids.int64.bin",
    }


def _profile_names(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item[0]) for item in values if isinstance(item, list) and item]


def _top_counter(counter: Counter[str]) -> List[List[object]]:
    return [
        [name, int(count)]
        for name, count in counter.most_common(RELATION_PROFILE_TOP_K)
        if name
    ]


def _serialize_relation_type_profile(
    head_domain_counts: Dict[int, Counter[str]],
    tail_domain_counts: Dict[int, Counter[str]],
    head_vocabulary_counts: Dict[int, Counter[str]],
    tail_vocabulary_counts: Dict[int, Counter[str]],
) -> Dict[str, Dict[str, List[List[object]]]]:
    relation_ids = (
        set(head_domain_counts)
        | set(tail_domain_counts)
        | set(head_vocabulary_counts)
        | set(tail_vocabulary_counts)
    )
    return {
        str(relation_id): {
            "head_domains": _top_counter(head_domain_counts[relation_id]),
            "tail_domains": _top_counter(tail_domain_counts[relation_id]),
            "head_vocabularies": _top_counter(head_vocabulary_counts[relation_id]),
            "tail_vocabularies": _top_counter(tail_vocabulary_counts[relation_id]),
        }
        for relation_id in sorted(relation_ids)
    }


def _update_profile_counts(
    profile_frame: pd.DataFrame,
    type_column: str,
    counters: DefaultDict[int, Counter[str]],
) -> None:
    counts = (
        profile_frame.groupby(["relation_id", type_column], dropna=True)
        .size()
        .reset_index(name="count")
    )
    for relation_id, concept_type, count in counts.itertuples(index=False, name=None):
        concept_type = str(concept_type).strip()
        if concept_type:
            counters[int(relation_id)][concept_type] += int(count)


def _process_relationship_chunk(
    chunk: pd.DataFrame,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if _BUILD_CONCEPT_IDS is None:
        raise RuntimeError("Build worker concept ids were not initialized.")
    raw_rows = len(chunk)
    chunk = chunk[
        chunk["concept_id_1"].ne("")
        & chunk["concept_id_2"].ne("")
        & chunk["relationship_id"].ne("")
        & chunk["concept_id_1"].ne(chunk["concept_id_2"])
        & chunk["concept_id_1"].isin(_BUILD_CONCEPT_IDS)
        & chunk["concept_id_2"].isin(_BUILD_CONCEPT_IDS)
    ]
    return (
        raw_rows,
        chunk["concept_id_1"].to_numpy(dtype=object),
        chunk["concept_id_2"].to_numpy(dtype=object),
        chunk["relationship_id"].to_numpy(dtype=object),
    )


def _iter_processed_relationship_chunks(
    reader: Iterable[pd.DataFrame], workers: int
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    if workers <= 1:
        for chunk in reader:
            yield _process_relationship_chunk(chunk)
        return

    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        pending = []
        for chunk in reader:
            pending.append(executor.submit(_process_relationship_chunk, chunk))
            if len(pending) >= workers:
                yield pending.pop(0).result()
        for future in pending:
            yield future.result()


def _write_triple_arrays(
    head_out,
    tail_out,
    relation_out,
    head_ids: np.ndarray,
    tail_ids: np.ndarray,
    relation_ids: np.ndarray,
) -> int:
    if len(head_ids) == 0:
        return 0
    head_ids.astype(np.int64).tofile(head_out)
    tail_ids.astype(np.int64).tofile(tail_out)
    relation_ids.astype(np.int64).tofile(relation_out)
    return len(head_ids)


def load_concept_names(
    concept_path: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    concepts: Dict[str, str] = {}
    concept_domains: Dict[str, str] = {}
    concept_vocabularies: Dict[str, str] = {}
    total_chunks = count_file_chunks(concept_path, 1_000_000)
    reader = pd.read_csv(
        concept_path,
        sep="\t",
        dtype=str,
        usecols=["concept_id", "concept_name", "domain_id", "vocabulary_id"],
        chunksize=1_000_000,
        keep_default_na=False,
    )
    for chunk in tqdm(
        reader,
        desc="Reading CONCEPT",
        total=total_chunks,
        unit="chunk",
        disable=not is_main_process(),
    ):
        for concept_id, concept_name, domain_id, vocabulary_id in chunk.itertuples(
            index=False, name=None
        ):
            concept_id = str(concept_id).strip()
            concept_name = str(concept_name).strip()
            if concept_id and concept_name:
                concepts[concept_id] = concept_name
                concept_domains[concept_id] = str(domain_id).strip()
                concept_vocabularies[concept_id] = str(vocabulary_id).strip()
    return concepts, concept_domains, concept_vocabularies


def build_triples(
    args: Args,
    triple_cache: str,
    concepts: Dict[str, str],
    concept_domains: Dict[str, str],
    concept_vocabularies: Dict[str, str],
) -> int:
    global _BUILD_CONCEPT_IDS
    os.makedirs(triple_cache, exist_ok=True)
    train_head_path = os.path.join(triple_cache, "train_head_ids.int64.bin")
    train_tail_path = os.path.join(triple_cache, "train_tail_ids.int64.bin")
    train_relation_path = os.path.join(triple_cache, "train_relation_ids.int64.bin")
    eval_head_path = os.path.join(triple_cache, "eval_head_ids.int64.bin")
    eval_tail_path = os.path.join(triple_cache, "eval_tail_ids.int64.bin")
    eval_relation_path = os.path.join(triple_cache, "eval_relation_ids.int64.bin")
    metadata_path = os.path.join(triple_cache, "metadata.json")
    concept_ids = set(concepts)
    _BUILD_CONCEPT_IDS = concept_ids
    relation2id: Dict[str, int] = {}
    head_domain_counts: DefaultDict[int, Counter[str]] = defaultdict(Counter)
    tail_domain_counts: DefaultDict[int, Counter[str]] = defaultdict(Counter)
    head_vocabulary_counts: DefaultDict[int, Counter[str]] = defaultdict(Counter)
    tail_vocabulary_counts: DefaultDict[int, Counter[str]] = defaultdict(Counter)
    train_written = 0
    eval_written = 0
    total_written = 0
    total_rows = count_data_lines(args.concept_relationship_path)
    eval_ratio = max(0.0, min(1.0, args.kg_eval_ratio))
    rng = np.random.default_rng(args.seed)

    reader = pd.read_csv(
        args.concept_relationship_path,
        sep="\t",
        dtype=str,
        usecols=[
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
        ],
        chunksize=args.kg_build_chunksize,
        keep_default_na=False,
    )
    with open(train_head_path, "wb") as train_head_out, open(
        train_tail_path, "wb"
    ) as train_tail_out, open(train_relation_path, "wb") as train_relation_out, open(
        eval_head_path, "wb"
    ) as eval_head_out, open(eval_tail_path, "wb") as eval_tail_out, open(
        eval_relation_path, "wb"
    ) as eval_relation_out, tqdm(
        total=total_rows,
        desc="Building triples",
        unit="row",
        disable=not is_main_process(),
    ) as progress:
        for raw_rows, head_ids_raw, tail_ids_raw, relation_names in _iter_processed_relationship_chunks(
            reader, args.kg_build_workers
        ):
            if args.kg_max_triples is not None and total_written >= args.kg_max_triples:
                progress.update(raw_rows)
                break

            if args.kg_max_triples is not None:
                remaining = args.kg_max_triples - total_written
                head_ids_raw = head_ids_raw[:remaining]
                tail_ids_raw = tail_ids_raw[:remaining]
                relation_names = relation_names[:remaining]

            if len(head_ids_raw) == 0:
                progress.update(raw_rows)
                progress.set_postfix(
                    {
                        "train": train_written,
                        "eval": eval_written,
                        "relations": len(relation2id),
                    },
                    refresh=False,
                )
                continue

            for relationship_id in pd.unique(relation_names):
                relationship_id = str(relationship_id)
                if relationship_id not in relation2id:
                    relation2id[relationship_id] = len(relation2id)

            head_ids = head_ids_raw.astype(np.int64)
            tail_ids = tail_ids_raw.astype(np.int64)
            relation_ids = np.fromiter(
                (relation2id[str(name)] for name in relation_names),
                dtype=np.int64,
                count=len(relation_names),
            )
            profile_frame = pd.DataFrame(
                {
                    "relation_id": relation_ids,
                    "head_domain": pd.Series(head_ids_raw).map(concept_domains),
                    "tail_domain": pd.Series(tail_ids_raw).map(concept_domains),
                    "head_vocabulary": pd.Series(head_ids_raw).map(concept_vocabularies),
                    "tail_vocabulary": pd.Series(tail_ids_raw).map(concept_vocabularies),
                }
            )
            _update_profile_counts(
                profile_frame,
                "head_domain",
                head_domain_counts,
            )
            _update_profile_counts(
                profile_frame,
                "tail_domain",
                tail_domain_counts,
            )
            _update_profile_counts(
                profile_frame,
                "head_vocabulary",
                head_vocabulary_counts,
            )
            _update_profile_counts(
                profile_frame,
                "tail_vocabulary",
                tail_vocabulary_counts,
            )
            if eval_ratio > 0:
                eval_mask = rng.random(len(head_ids)) < eval_ratio
            else:
                eval_mask = np.zeros(len(head_ids), dtype=bool)
            train_mask = ~eval_mask
            train_written += _write_triple_arrays(
                train_head_out,
                train_tail_out,
                train_relation_out,
                head_ids[train_mask],
                tail_ids[train_mask],
                relation_ids[train_mask],
            )
            eval_written += _write_triple_arrays(
                eval_head_out,
                eval_tail_out,
                eval_relation_out,
                head_ids[eval_mask],
                tail_ids[eval_mask],
                relation_ids[eval_mask],
            )
            total_written += len(head_ids)
            progress.update(raw_rows)
            progress.set_postfix(
                {
                    "train": train_written,
                    "eval": eval_written,
                    "relations": len(relation2id),
                },
                refresh=False,
            )
            if args.kg_max_triples is not None and total_written >= args.kg_max_triples:
                break

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_triples": train_written,
                "train_num_triples": train_written,
                "eval_num_triples": eval_written,
                "kg_eval_ratio": eval_ratio,
                "relation2id": relation2id,
                "relation_type_profile": _serialize_relation_type_profile(
                    head_domain_counts,
                    tail_domain_counts,
                    head_vocabulary_counts,
                    tail_vocabulary_counts,
                ),
                "format": "int64_memmap_v1",
                "files": {
                    "head_ids": os.path.basename(train_head_path),
                    "tail_ids": os.path.basename(train_tail_path),
                    "relation_ids": os.path.basename(train_relation_path),
                },
                "train_files": {
                    "head_ids": os.path.basename(train_head_path),
                    "tail_ids": os.path.basename(train_tail_path),
                    "relation_ids": os.path.basename(train_relation_path),
                },
                "eval_files": {
                    "head_ids": os.path.basename(eval_head_path),
                    "tail_ids": os.path.basename(eval_tail_path),
                    "relation_ids": os.path.basename(eval_relation_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rank0_print(
        f"Triple cache: {triple_cache} "
        f"(train={train_written}, eval={eval_written})"
    )
    rank0_print(f"Relations: {len(relation2id)}")
    return train_written


def make_kg_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        head_tokens = tokenizer(
            [str(item["head_name"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tail_tokens = tokenizer(
            [str(item["tail_name"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        num_negatives = len(batch[0]["negative_names"])
        flat_negative_names: List[str] = []
        negative_is_head: List[List[bool]] = []
        for item in batch:
            flat_negative_names.extend(str(name) for name in item["negative_names"])
            negative_is_head.append(list(item["negative_is_head"]))
        negative_tokens = tokenizer(
            flat_negative_names,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_size, negative_seq_len = len(batch), negative_tokens["input_ids"].size(1)

        output = {
            "head_input_ids": head_tokens["input_ids"],
            "head_attention_mask": head_tokens["attention_mask"],
            "tail_input_ids": tail_tokens["input_ids"],
            "tail_attention_mask": tail_tokens["attention_mask"],
            "relation_ids": torch.tensor(
                [int(item["relation_id"]) for item in batch], dtype=torch.long
            ),
            "negative_input_ids": negative_tokens["input_ids"].view(
                batch_size, num_negatives, negative_seq_len
            ),
            "negative_attention_mask": negative_tokens["attention_mask"].view(
                batch_size, num_negatives, negative_seq_len
            ),
            "negative_is_head": torch.tensor(negative_is_head, dtype=torch.bool),
        }
        if "token_type_ids" in head_tokens:
            output["head_token_type_ids"] = head_tokens["token_type_ids"]
        if "token_type_ids" in tail_tokens:
            output["tail_token_type_ids"] = tail_tokens["token_type_ids"]
        if "token_type_ids" in negative_tokens:
            output["negative_token_type_ids"] = negative_tokens["token_type_ids"].view(
                batch_size, num_negatives, negative_seq_len
            )
        return output

    return collate


def configure_wandb(args: Args) -> None:
    if args.report_to != "wandb":
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

def run_training(args: Args) -> None:
    random.seed(args.seed)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.kg_num_negatives <= 0:
        raise ValueError("kg_num_negatives must be greater than 0.")

    concepts, concept_domains, concept_vocabularies = load_concept_names(
        args.concept_path
    )
    triple_cache = args.triple_cache
    if not os.path.exists(os.path.join(triple_cache, "metadata.json")):
        build_triples(
            args,
            triple_cache,
            concepts,
            concept_domains,
            concept_vocabularies,
        )
    if args.cache_only:
        rank0_print(f"Triple cache is ready: {triple_cache}")
        return

    train_dataset = ConceptRelationshipDataset(
        triple_cache=triple_cache,
        concepts=concepts,
        concept_domains=concept_domains,
        concept_vocabularies=concept_vocabularies,
        num_negatives=args.kg_num_negatives,
        seed=args.seed,
        split="train",
    )
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty after filtering concept relationships.")
    eval_dataset = None
    if args.kg_eval_ratio > 0:
        eval_dataset = ConceptRelationshipDataset(
            triple_cache=triple_cache,
            concepts=concepts,
            concept_domains=concept_domains,
            concept_vocabularies=concept_vocabularies,
            num_negatives=args.kg_num_negatives,
            seed=args.seed + 1_000_000,
            split="eval",
        )
        if len(eval_dataset) == 0:
            eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate_fn = make_kg_collate_fn(tokenizer, args.max_length)
    model = KnowledgeGraphEncoderForTrainer(
        args.model_name_or_path,
        num_relations=len(train_dataset.relation2id),
        margin=args.kg_margin,
        distance_p=args.kg_distance_p,
        relation_reg=args.kg_relation_reg,
        freeze_bert=args.freeze_bert,
    )
    configure_wandb(args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        prediction_loss_only=True,
        bf16=args.bf16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.wandb_run_name or None,
        deepspeed=args.deepspeed or None,
        log_on_each_node=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )
    rank0_print(
        f"Knowledge graph triples: train={len(train_dataset)}, "
        f"eval={0 if eval_dataset is None else len(eval_dataset)}, "
        f"relations={len(train_dataset.relation2id)}"
    )
    rank0_print(f"Concept text: concept_name")
    rank0_print(f"Model: {args.model_name_or_path}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(
        {
            "state_dict": trainer.model.state_dict(),
            "model_name_or_path": args.model_name_or_path,
            "args": asdict(args),
            "relation2id": train_dataset.relation2id,
        },
        os.path.join(args.output_dir, "best.pt"),
    )
    rank0_print(f"Saved checkpoint: {os.path.join(args.output_dir, 'best.pt')}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Knowledge graph encoder pretraining")
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_type = type(default)
        if arg_type is bool:
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
        elif field_name == "kg_max_triples":
            parser.add_argument(f"--{field_name}", type=str, default=default)
        else:
            parser.add_argument(f"--{field_name}", type=arg_type, default=default)
    parser.add_argument("--local_rank", type=int, default=-1)
    values = vars(parser.parse_args())
    values.pop("local_rank", None)
    if isinstance(values["kg_max_triples"], str):
        raw_value = values["kg_max_triples"].strip()
        values["kg_max_triples"] = (
            None if raw_value.lower() == "none" else int(raw_value)
        )
    return Args(**values)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
