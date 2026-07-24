import ast

import torch
from typing import List, Dict, Any, Optional
import pandas as pd
from .load_embedding import get_embedding, get_pad_embedding


def build_table_token_tensors(
    tables_list: List[Any],
    text_to_idx: Dict[str, int],
    pad_idx: int,
    row_block_ids_list: Optional[List[List[Any]]] = None,
    kept_block_ids_list: Optional[List[List[Any]]] = None,
    type_vocab: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
    """
    Build table token tensors for TableEncoder from a list of measurement tables.

    Args:
        tables_list: List of per-sample measurement tables. Each non-empty table
            is expected to be a pandas DataFrame with Item, Value, Unit, and
            Category columns, and optionally a Time column.
        text_to_idx: Mapping from cached table text strings to token ids.
        pad_idx: Token id used when a text cell is missing or absent from
            text_to_idx, and for padded positions.
        row_block_ids_list: Optional per-sample row block ids aligned with table
            rows. Used only when filtering rows by kept_block_ids_list.
        kept_block_ids_list: Optional per-sample block ids to keep. When present
            with matching row_block_ids_list, rows outside these blocks are
            dropped before tensorization.
        type_vocab: Optional mapping from Category strings to type ids.

    Returns:
        A dict of padded tensors with shape [batch_size, max_table_len]:
            item_ids: token ids for Item strings.
            unit_ids: token ids for Unit strings, or pad_idx for missing cells.
            value_text_ids: token ids for non-numeric Value strings, or pad_idx
                for numeric/missing values.
            times: relative hours from the first valid row time, starting at 1;
                0 marks missing or padded times.
            numeric_values: parsed numeric Value values; 0 for non-numeric,
                missing, or padded values.
            numeric_mask: 1 where Value is numeric, otherwise 0.
            seq_mask: 1 for real table rows, 0 for padding or empty tables.
            type_ids: Category ids.
    """
    if row_block_ids_list is None:
        row_block_ids_list = [None] * len(tables_list)
    if kept_block_ids_list is None:
        kept_block_ids_list = [None] * len(tables_list)

    all_item_ids, all_unit_ids, all_value_text_ids = [], [], []
    all_times, all_numeric_values, all_numeric_masks = [], [], []
    all_type_ids = []
    seq_lens = []

    for b_idx, t in enumerate(tables_list):
        # Optional row filtering by kept block ids (for text truncation alignment).
        if isinstance(t, pd.DataFrame) and len(t) > 0:
            row_block_ids = row_block_ids_list[b_idx]
            kept_block_ids = kept_block_ids_list[b_idx]
            if isinstance(row_block_ids, list) and isinstance(kept_block_ids, list) and len(row_block_ids) == len(t):
                kept_set = set(str(x) for x in kept_block_ids)
                if len(kept_set) > 0:
                    keep_mask = [str(bid) in kept_set for bid in row_block_ids]
                    t = t.loc[keep_mask].reset_index(drop=True)

        if isinstance(t, pd.DataFrame) and len(t) > 0:
            sl = len(t)
            seq_lens.append(sl)

            item_ids = [
                text_to_idx.get(str(item), pad_idx) if pd.notna(item) else pad_idx
                for item in t["Item"]
            ]

            unit_ids = [
                text_to_idx.get(str(unit), pad_idx) if pd.notna(unit) else pad_idx
                for unit in t["Unit"]
            ]

            val_series = t["Value"]
            numeric_series = pd.to_numeric(val_series, errors="coerce")
            is_numeric = numeric_series.notna()
            num_mask = is_numeric.astype(float).tolist()
            num_vals = numeric_series.fillna(0.0).tolist()
            value_text_ids_list = [
                text_to_idx.get(str(value), pad_idx)
                if (not is_num and pd.notna(value))
                else pad_idx
                for value, is_num in zip(val_series, is_numeric)
            ]

            all_item_ids.append(item_ids)
            all_unit_ids.append(unit_ids)
            all_value_text_ids.append(value_text_ids_list)
            all_numeric_values.append(num_vals)
            all_numeric_masks.append(num_mask)

            if "Time" in t.columns:
                if not pd.api.types.is_datetime64_any_dtype(t["Time"]):
                    time_col = pd.to_datetime(t["Time"], errors="coerce")
                else:
                    time_col = t["Time"]
                if len(time_col) > 0:
                    first_date = time_col.iloc[0]
                    delta_hours = (time_col - first_date).dt.total_seconds() / 3600 + 1
                    delta_hours = delta_hours.fillna(0.0).tolist()
                else:
                    delta_hours = [0.0] * sl
            else:
                delta_hours = [0.0] * sl
            all_times.append(delta_hours)

            type_ids = [type_vocab.get(str(c), 0) for c in t["Category"]]
            all_type_ids.append(type_ids)
        else:
            seq_lens.append(0)
            all_item_ids.append([])
            all_unit_ids.append([])
            all_value_text_ids.append([])
            all_times.append([])
            all_numeric_values.append([])
            all_numeric_masks.append([])
            all_type_ids.append([])

    bs = len(tables_list)
    max_len = max(max(seq_lens), 1)

    item_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    unit_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    value_text_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    times_t = torch.zeros(bs, max_len, dtype=torch.float)
    numeric_values_t = torch.zeros(bs, max_len, dtype=torch.float)
    numeric_mask_t = torch.zeros(bs, max_len, dtype=torch.float)
    seq_mask_t = torch.zeros(bs, max_len, dtype=torch.float)
    type_ids_t = torch.zeros(bs, max_len, dtype=torch.long)

    for i in range(bs):
        sl = seq_lens[i]
        item_ids_t[i, :sl] = torch.tensor(all_item_ids[i], dtype=torch.long)
        unit_ids_t[i, :sl] = torch.tensor(all_unit_ids[i], dtype=torch.long)
        value_text_ids_t[i, :sl] = torch.tensor(all_value_text_ids[i], dtype=torch.long)
        times_t[i, :sl] = torch.tensor(all_times[i], dtype=torch.float)
        numeric_values_t[i, :sl] = torch.tensor(all_numeric_values[i], dtype=torch.float)
        numeric_mask_t[i, :sl] = torch.tensor(all_numeric_masks[i], dtype=torch.float)
        seq_mask_t[i, :sl] = 1.0
        type_ids_t[i, :sl] = torch.tensor(all_type_ids[i], dtype=torch.long)

    return {
        "item_ids": item_ids_t,
        "unit_ids": unit_ids_t,
        "value_text_ids": value_text_ids_t,
        "times": times_t,
        "numeric_values": numeric_values_t,
        "numeric_mask": numeric_mask_t,
        "seq_mask": seq_mask_t,
        "type_ids": type_ids_t,
    }


def create_collate_fn(type_vocab=None, label_map=None, max_table_len: Optional[int] = None):
    """
    Requires dataset __getitem__ to return a dictionary with:
    - `label`: An integer, float, or tensor representing the classification target.
    - `measurement_table`: A pandas DataFrame containing columns ['Item', 'Value', 'Unit', 'Category'] and optional column ['Time'].
    """
    if type_vocab is None: type_vocab = {}

    def lookup_or_pad(value, pad_emb):
        if pd.isna(value):
            return pad_emb
        text = str(value)
        if not text.strip():
            return pad_emb
        emb = get_embedding(text)
        return emb if emb is not None else pad_emb
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [
            sample for sample in batch
            if sample.get("measurement_table") is not None and not sample["measurement_table"].empty
        ]
        if len(batch) == 0:
            raise ValueError("All samples in this batch have empty measurement_table.")

        bs = len(batch)
        all_item_embs, all_unit_embs, all_value_embs = [], [], []
        all_times, all_num_vals, all_num_masks, all_type_ids = [], [], [], []
        labels_list, seq_lens = [], []
        pad_emb = get_pad_embedding()
        
        for sample in batch:
            # 1. Parse Label
            raw_label = sample['output']
            if isinstance(raw_label, torch.Tensor):
                labels_list.append(raw_label.long())
            elif isinstance(raw_label, str):
                norm_label = raw_label.strip().strip('"').strip("'").strip()
                norm_label_lower = norm_label.lower()
                if norm_label_lower == 'yes':
                    labels_list.append(1)
                elif norm_label_lower == 'no':
                    labels_list.append(0)
                elif label_map is not None and norm_label in label_map:
                    labels_list.append(label_map[norm_label])
                elif label_map is not None and norm_label_lower in label_map:
                    labels_list.append(label_map[norm_label_lower])
                else:
                    labels_list.append(int(float(norm_label)))
            else:
                labels_list.append(int(raw_label))
            
            # 2. Parse Table
            df = sample.get('measurement_table')
            if max_table_len is not None:
                df = df.tail(max_table_len).reset_index(drop=True)

            item_embs = [
                lookup_or_pad(item, pad_emb)
                for item in df['Item']
            ]
            unit_embs = [
                lookup_or_pad(unit, pad_emb)
                for unit in df['Unit']
            ]

            num_series = pd.to_numeric(df['Value'], errors="coerce")
            num_mask = num_series.notna().astype(float).tolist()
            num_vals = num_series.fillna(0.0).tolist()

            value_embs = [
                pad_emb if m else lookup_or_pad(v, pad_emb)
                for v, m in zip(df['Value'], num_mask)
            ]

            # Assume dataset already converted time to relative days float. E.g. 0.0 for MIMIC
            time_vals = pd.to_numeric(df['Time'], errors="coerce").fillna(0.0).tolist() if 'Time' in df.columns else [0.0] * len(item_embs)

            type_ids = [type_vocab.get(str(c), 0) for c in df["Category"]]

            seq_lens.append(len(item_embs))
            all_item_embs.append(torch.stack(item_embs))
            all_unit_embs.append(torch.stack(unit_embs))
            all_value_embs.append(torch.stack(value_embs))
            all_times.append(time_vals)
            all_num_vals.append(num_vals)
            all_num_masks.append(num_mask)
            all_type_ids.append(type_ids)
        
        # 3. Padding tensors to max length
        max_len = max(seq_lens)
        for i in range(bs):
            p = max_len - seq_lens[i]
            if p > 0:
                pad_t = pad_emb.unsqueeze(0).repeat(p, 1)
                all_item_embs[i] = torch.cat([all_item_embs[i], pad_t])
                all_unit_embs[i] = torch.cat([all_unit_embs[i], pad_t])
                all_value_embs[i] = torch.cat([all_value_embs[i], pad_t])
                all_times[i].extend([0.0]*p)
                all_num_vals[i].extend([0.0]*p)
                all_num_masks[i].extend([0.0]*p) # padding values should have 0 mask
                all_type_ids[i].extend([0]*p)
        
        seq_mask = torch.zeros(bs, max_len)
        for i, sl in enumerate(seq_lens): seq_mask[i, :sl] = 1.0
            
        labels_tensor = torch.stack(labels_list) if isinstance(labels_list[0], torch.Tensor) else torch.tensor(labels_list)
        
        return {
            "item_emb": torch.stack(all_item_embs),
            "unit_emb": torch.stack(all_unit_embs),
            "value_emb": torch.stack(all_value_embs),
            "times": torch.tensor(all_times, dtype=torch.float32),
            "numeric_values": torch.tensor(all_num_vals, dtype=torch.float32),
            "numeric_mask": torch.tensor(all_num_masks, dtype=torch.float32),
            "type_ids": torch.tensor(all_type_ids, dtype=torch.long),
            "seq_mask": seq_mask,
            "labels": labels_tensor
        }
    return collate_fn


def create_query_collate_fn(
    type_vocab=None,
    label_map=None,
    max_table_len: Optional[int] = None,
    text_to_idx: Optional[Dict[str, int]] = None,
    pad_idx: int = 0,
    query_embed: Optional[torch.Tensor] = None,
    query_embeddings_by_text: Optional[Dict[str, torch.Tensor]] = None,
):
    if type_vocab is None:
        type_vocab = {}

    def parse_label(raw_label):
        if isinstance(raw_label, torch.Tensor):
            return raw_label
        if isinstance(raw_label, str):
            norm_label = raw_label.strip().strip('"').strip("'").strip()
            norm_label_lower = norm_label.lower()
            if norm_label_lower == "yes":
                return 1
            if norm_label_lower == "no":
                return 0
            if label_map is not None and norm_label in label_map:
                return label_map[norm_label]
            if label_map is not None and norm_label_lower in label_map:
                return label_map[norm_label_lower]
            return int(float(norm_label))
        return int(raw_label)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [
            sample for sample in batch
            if sample.get("measurement_table") is not None and not sample["measurement_table"].empty
        ]
        if len(batch) == 0:
            raise ValueError("All samples in this batch have empty measurement_table.")

        tables = []
        labels = []
        query_embeds = []
        for sample in batch:
            df = sample["measurement_table"]
            if max_table_len is not None:
                df = df.tail(max_table_len).reset_index(drop=True)
            tables.append(df)
            labels.append(parse_label(sample["output"]))

            if query_embeddings_by_text is not None:
                query_embeds.append(query_embeddings_by_text[str(sample["instruction"])])
            else:
                query_embeds.append(query_embed)

        tensors = build_table_token_tensors(
            tables,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            type_vocab=type_vocab,
        )
        tensors["query_embeds"] = torch.stack(query_embeds)
        tensors["labels"] = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        return tensors

    return collate_fn


def create_multi_query_classifier_collate_fn(
    type_vocab=None,
    max_table_len: Optional[int] = None,
    text_to_idx: Optional[Dict[str, int]] = None,
    pad_idx: int = 0,
    query_embeddings_by_text: Optional[Dict[str, torch.Tensor]] = None,
    include_metadata: bool = False,
):
    if type_vocab is None:
        type_vocab = {}

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        tables = []
        query_embed_rows = []
        label_rows = []
        metadata = []

        for sample in batch:
            table = sample.get("measurement_table")
            if table is None or table.empty:
                continue
            if max_table_len is not None:
                table = table.tail(max_table_len).reset_index(drop=True)
            tasks = sample.get("candidate_tasks", [])
            if not tasks:
                continue

            tables.append(table)
            sample_query_embeds = []
            sample_labels = []
            sample_metadata = []
            for task in tasks:
                sample_query_embeds.append(query_embeddings_by_text[str(task["query"])].float())
                sample_labels.append(int(task["label"]))
                if include_metadata:
                    sample_metadata.append(
                        {
                            "point": task.get("point"),
                            "point_key": task.get("point_key"),
                            "metric": task.get("metric"),
                            "window": task.get("window"),
                        }
                    )
            query_embed_rows.append(sample_query_embeds)
            label_rows.append(sample_labels)
            if include_metadata:
                metadata.append(sample_metadata)

        if len(tables) == 0:
            raise ValueError("All samples in this batch have empty candidate tasks.")

        tensors = build_table_token_tensors(
            tables,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            type_vocab=type_vocab,
        )
        batch_size = len(query_embed_rows)
        max_queries = max(len(row) for row in query_embed_rows)
        query_dim = int(query_embed_rows[0][0].numel())

        query_embeds = torch.zeros(batch_size, max_queries, query_dim)
        labels = torch.full((batch_size, max_queries), -100, dtype=torch.float)
        query_mask = torch.zeros(batch_size, max_queries, dtype=torch.float)
        for batch_idx, task_queries in enumerate(query_embed_rows):
            for query_idx, query_embed in enumerate(task_queries):
                query_embeds[batch_idx, query_idx] = query_embed
                labels[batch_idx, query_idx] = float(label_rows[batch_idx][query_idx])
                query_mask[batch_idx, query_idx] = 1.0

        tensors["query_embeds"] = query_embeds
        tensors["query_mask"] = query_mask
        tensors["labels"] = labels
        if include_metadata:
            tensors["metadata"] = metadata
        return tensors

    return collate_fn


def parse_classification_label(raw_label, label_map=None):
    if isinstance(raw_label, str) and raw_label.strip().startswith("["):
        raw_label = ast.literal_eval(raw_label)
    if isinstance(raw_label, (list, tuple, set)):
        if label_map is None:
            return torch.as_tensor(list(raw_label), dtype=torch.long)
        target = torch.zeros(len(label_map), dtype=torch.float)
        for value in raw_label:
            if value in label_map:
                class_index = label_map[value]
            elif str(value) in label_map:
                class_index = label_map[str(value)]
            else:
                class_index = int(value)
            target[int(class_index)] = 1.0
        return target
    if isinstance(raw_label, torch.Tensor):
        if raw_label.numel() == 1:
            return int(raw_label.item())
        return raw_label.long()
    if isinstance(raw_label, str):
        norm_label = raw_label.strip().strip('"').strip("'").strip()
        norm_label_lower = norm_label.lower()
        if label_map is not None and norm_label in label_map:
            return int(label_map[norm_label])
        if label_map is not None and norm_label_lower in label_map:
            return int(label_map[norm_label_lower])
        return int(float(norm_label))
    if label_map is not None and raw_label in label_map:
        return int(label_map[raw_label])
    return int(raw_label)


def create_query_classifier_collate_fn(
    type_vocab=None,
    max_table_len: Optional[int] = None,
    text_to_idx: Optional[Dict[str, int]] = None,
    pad_idx: int = 0,
    query_embeds: Optional[torch.Tensor] = None,
    query_embeddings_by_text: Optional[Dict[str, torch.Tensor]] = None,
    label_map: Optional[Dict[str, int]] = None,
    include_metadata: bool = False,
):
    if type_vocab is None:
        type_vocab = {}

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [
            sample
            for sample in batch
            if sample.get("measurement_table") is not None and not sample["measurement_table"].empty
        ]
        if len(batch) == 0:
            raise ValueError("All samples in this batch have empty measurement_table.")

        tables = []
        labels = []
        sample_query_embeds = []
        metadata = []
        for sample in batch:
            table = sample["measurement_table"]
            if max_table_len is not None:
                table = table.tail(max_table_len).reset_index(drop=True)
            tables.append(table)
            labels.append(parse_classification_label(sample["output"], label_map=label_map))
            if include_metadata:
                metadata.append({"prediction_point": sample.get("prediction_point")})
            if query_embeddings_by_text is not None:
                sample_query_embeds.append(query_embeddings_by_text[str(sample["instruction"])].float())

        tensors = build_table_token_tensors(
            tables,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            type_vocab=type_vocab,
        )
        batch_size = len(tables)
        if query_embeddings_by_text is not None:
            tensors["query_embeds"] = torch.stack(sample_query_embeds)
        else:
            query_tensor = query_embeds.float()
            if query_tensor.dim() == 1:
                tensors["query_embeds"] = query_tensor.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                tensors["query_embeds"] = query_tensor.unsqueeze(0).expand(batch_size, -1, -1).clone()
                tensors["query_mask"] = torch.ones(batch_size, query_tensor.size(0), dtype=torch.float32)
        if isinstance(labels[0], torch.Tensor):
            tensors["labels"] = torch.stack(labels)
        else:
            tensors["labels"] = torch.tensor(labels, dtype=torch.long)
        if include_metadata:
            tensors["metadata"] = metadata
        return tensors

    return collate_fn


def create_survival_query_collate_fn(
    type_vocab=None,
    max_table_len: Optional[int] = None,
    text_to_idx: Optional[Dict[str, int]] = None,
    pad_idx: int = 0,
    query_embeddings_by_text: Optional[Dict[str, torch.Tensor]] = None,
):
    if type_vocab is None:
        type_vocab = {}

    def build_survival_label(target):
        if target is None:
            return None
        time_to_event = float(target["time_to_event"])
        event_observed = bool(target["event_observed"])
        num_bins = int(target["num_bins"])
        max_bins = int(target.get("max_bins", num_bins))
        stage_end_horizon = float(target["stage_end_horizon"])
        stage_id = int(target["stage_id"])

        observed_time = min(time_to_event, float(num_bins))
        exposure = torch.zeros(max_bins, dtype=torch.float32)
        event_bins = torch.zeros(max_bins, dtype=torch.float32)
        stage_mask = torch.zeros(max_bins, dtype=torch.float32)
        metadata = torch.zeros(max_bins, dtype=torch.float32)
        stage_mask[:num_bins] = 1.0

        full_bins = min(int(observed_time // 1), num_bins)
        if full_bins:
            exposure[:full_bins] = 1.0
        if full_bins < num_bins:
            exposure[full_bins] = float(observed_time - full_bins)

        if event_observed and 0.0 < observed_time <= num_bins:
            event_bin = min(int(torch.ceil(torch.tensor(observed_time)).item()) - 1, num_bins - 1)
            event_bins[event_bin] = 1.0

        metadata[0] = stage_end_horizon
        if max_bins > 1:
            metadata[1] = float(stage_id)
        return torch.stack([exposure, event_bins, stage_mask, metadata])

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [
            sample
            for sample in batch
            if sample.get("measurement_table") is not None
            and not sample["measurement_table"].empty
        ]
        if not batch:
            raise ValueError("All samples in this batch have empty measurement_table.")

        tables = []
        for sample in batch:
            table = sample["measurement_table"]
            if max_table_len is not None:
                table = table.tail(max_table_len).reset_index(drop=True)
            tables.append(table)

        tensors = build_table_token_tensors(
            tables,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            type_vocab=type_vocab,
        )
        tensors["query_embeds"] = torch.stack(
            [
                query_embeddings_by_text[str(sample["instruction"])]
                for sample in batch
            ]
        )
        tensors["stage_ids"] = torch.tensor(
            [sample["stage_id"] for sample in batch],
            dtype=torch.long,
        )
        labels = []
        for sample in batch:
            if "output" in sample:
                labels.append(sample["output"])
            else:
                labels.append(build_survival_label(sample["survival_target"]))
        tensors["labels"] = torch.stack(labels)
        return tensors

    return collate_fn
