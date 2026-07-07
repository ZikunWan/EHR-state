import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info as get_mimic_iv_cdm_task_info
from dataset.pds.pds_dataset import PDSDataset
from dataset.pds.task_info import get_task_info as get_pds_task_info
from dataset.renji.renji_dataset import RenjiDataset
from dataset.renji.task_info import get_task_info as get_renji_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import EncoderClassifierModel
from outdated import task_query_classification as tqc
from train.classification.train_encoder_classifier import (
    RENJI_ACTIVE_POINTS,
    build_query_tensor,
    build_query_texts,
    build_renji_query_texts,
)
from utils.collate import create_multi_query_classifier_collate_fn, create_query_classifier_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import compute_classification_metrics
from utils.weight_loader import load_encoder_and_query_head_weights, load_task_model_weights


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_name: str = field(default="eicu")
    data_dir: str = field(default="")
    processed_dir: Optional[str] = field(default=None)
    sample_info_path: Optional[str] = field(default=None)
    sample_info_test_path: Optional[str] = field(default=None)
    split_info_path: Optional[str] = field(default=None)
    checkpoint_dir: str = field(default="")
    task_name: str = field(default="")
    embedding_cache: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    max_table_len: Optional[int] = field(default=None)
    batch_size: int = field(default=64)
    max_eval_samples: Optional[int] = field(default=None)
    seed: int = field(default=42)
    split: str = field(default="test")
    trial_id: Optional[str] = field(default=None)
    patient_split_path: Optional[str] = field(default=None)


def get_dataset_task_info(dataset_name: str):
    if dataset_name == "eicu":
        return get_eicu_task_info()
    if dataset_name == "ehrshot":
        return get_ehrshot_task_info()
    if dataset_name == "mimic_iv_cdm":
        return get_mimic_iv_cdm_task_info()
    if dataset_name == "ehr_bench":
        return get_mimic_task_info()
    if dataset_name == "renji":
        return get_renji_task_info()
    if dataset_name == "pds":
        return get_pds_task_info()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def resolve_eval_path(data_args: DataArguments):
    return data_args.sample_info_path or data_args.sample_info_test_path or data_args.split_info_path


def build_eval_dataset(data_args: DataArguments):
    eval_path = resolve_eval_path(data_args)
    if data_args.dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.data_dir,
            processed_dir=data_args.processed_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "mimic_iv_cdm":
        return MIMICIVCDM(
            root_dir=data_args.data_dir,
            split=data_args.split,
            task_name=data_args.task_name,
            lazy_mode=False,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            index_dir=data_args.processed_dir,
        )
    if data_args.dataset_name == "ehr_bench":
        return MIMICIV(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            use_table_length_cache=False,
        )
    if data_args.dataset_name == "renji":
        return RenjiDataset(
            root_dir=data_args.data_dir,
            split=data_args.split,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            target_prediction_points=RENJI_ACTIVE_POINTS,
        )
    if data_args.dataset_name == "pds":
        if data_args.patient_split_path is None or not str(data_args.patient_split_path).strip():
            raise ValueError("--patient_split_path is required when --dataset_name pds")
        trial_ids = [item.strip() for item in str(data_args.trial_id).split(",") if item.strip()]
        return PDSDataset(
            root_dir=data_args.data_dir,
            task_name=data_args.task_name,
            trial_ids=trial_ids,
            split=data_args.split,
            patient_split_path=data_args.patient_split_path,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def move_tensors_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def probabilities_from_logits(logits: torch.Tensor, binary: bool) -> np.ndarray:
    logits = logits.float()
    if binary:
        probs_yes = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - probs_yes, probs_yes], dim=-1).cpu().numpy()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.seed)

    is_multi_query_dataset = data_args.dataset_name == "renji"
    if is_multi_query_dataset:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name or "candidate_metric_prediction"]
        query_texts = build_renji_query_texts()
        query_key = f"{data_args.dataset_name}:{data_args.task_name or 'candidate_metric_prediction'}"
    else:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name]
        if data_args.dataset_name == "pds":
            task_info = dict(task_info)
            task_info["instruction"] = PDSDataset.task_instruction(data_args.task_name, data_args.trial_id)
        query_key = f"{data_args.dataset_name}:{data_args.task_name}"
        if data_args.dataset_name == "pds" and data_args.trial_id:
            query_key = f"{query_key}:trial:{data_args.trial_id}"
        query_texts = build_query_texts(query_key, task_info)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    with open(data_args.type_vocab_file, "r") as f:
        type_vocab = json.load(f)

    embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=query_texts,
        cache_path=data_args.query_embedding_cache,
        max_length=data_args.query_max_length,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    if not is_multi_query_dataset:
        query_tensor, label_map = build_query_tensor(query_key, task_info, embeddings_by_text)
    else:
        label_map = None

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=1 if task_info.get("task_type") == "binary_classification" else int(task_info.get("num_classes", 1)),
        problem_type="single_label_classification",
    )
    model = EncoderClassifierModel(config=config, embedding_matrix=embedding_matrix, query_dim=query_dim)
    if model_args.pretrained_path:
        model = load_encoder_and_query_head_weights(model, model_args.pretrained_path)
    model = load_task_model_weights(model, data_args.checkpoint_dir)

    dataset = build_eval_dataset(data_args)
    if is_multi_query_dataset:
        collator = create_multi_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=embeddings_by_text,
            include_metadata=True,
        )
        binary_output = True
    else:
        collator = create_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeds=query_tensor,
            label_map=label_map,
        )
        binary_output = task_info.get("task_type") == "binary_classification"

    dataloader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_logits = []
    all_probs = []
    all_metadata = []
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata", None)
            labels = batch["labels"].cpu().numpy()
            batch = move_tensors_to_device(batch, device)
            outputs = model(**batch)
            if is_multi_query_dataset:
                logits = outputs.logits.float().cpu().numpy()
                probs_yes = 1.0 / (1.0 + np.exp(-logits))
                for batch_idx, sample_metadata in enumerate(metadata):
                    for query_idx, item in enumerate(sample_metadata):
                        if labels[batch_idx, query_idx] == -100:
                            continue
                        all_labels.append(int(labels[batch_idx, query_idx]))
                        all_logits.append(float(logits[batch_idx, query_idx]))
                        yes = float(probs_yes[batch_idx, query_idx])
                        all_probs.append([1.0 - yes, yes])
                        all_metadata.append(item)
            else:
                probs = probabilities_from_logits(outputs.logits, binary=binary_output)
                all_labels.extend(labels.reshape(-1).tolist())
                if binary_output:
                    all_logits.extend(outputs.logits.float().reshape(-1).cpu().numpy().tolist())
                else:
                    all_logits.extend(outputs.logits.float().cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

    eval_pred = type("EvalPred", (), {"predictions": np.asarray(all_logits), "label_ids": np.asarray(all_labels)})
    metrics = compute_classification_metrics(eval_pred)
    print(f"\n=== Encoder Classifier Evaluation: {data_args.dataset_name}/{data_args.task_name} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    output_task_name = data_args.task_name or data_args.dataset_name
    output_file = os.path.join(data_args.checkpoint_dir, f"test_results_{output_task_name}.csv")
    probs = np.asarray(all_probs)
    preds = probs.argmax(axis=-1)
    rows = []
    for idx, label in enumerate(all_labels):
        row = {"label": int(label), "prediction": int(preds[idx])}
        if all_metadata:
            row.update(all_metadata[idx])
        for class_idx in range(probs.shape[-1]):
            row[f"prob_{class_idx}"] = float(probs[idx, class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Raw predictions saved to {output_file}")


if __name__ == "__main__":
    main()
