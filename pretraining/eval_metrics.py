from collections import defaultdict
import re
import warnings

import numpy as np
import torch.distributed as dist
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from pretraining.sft import (
    TASK_TYPE_BINARY,
    TASK_TYPE_CANDIDATE_DIAGNOSIS,
    TASK_TYPE_MULTICLASS,
    TASK_TYPE_MULTILABEL,
    TASK_TYPE_TTE,
)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _mean_valid_class_metric(labels, scores, metric):
    values = []
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size < 2:
            continue
        values.append(metric(labels[:, column], scores[:, column]))
    return float(np.mean(values)) if values else None


def _concordance_index(times, events, risks):
    concordant = 0.0
    comparable = 0
    for index in np.flatnonzero(events):
        mask = times > times[index]
        comparable += int(mask.sum())
        concordant += float((risks[index] > risks[mask]).sum())
        concordant += 0.5 * float((risks[index] == risks[mask]).sum())
    return concordant / comparable if comparable else None


def _integrated_brier_score(times, events, survival):
    sample_count, horizon = survival.shape
    censor_times = np.sort(np.unique(times[~events]))
    censor_survival = []
    current = 1.0
    for value in censor_times:
        at_risk = int((times >= value).sum())
        censored = int(((times == value) & ~events).sum())
        if at_risk:
            current *= 1.0 - censored / at_risk
        censor_survival.append(current)
    censor_survival = np.asarray(censor_survival, dtype=np.float64)

    def censor_probability(value, include_equal):
        side = "right" if include_equal else "left"
        position = np.searchsorted(censor_times, value, side=side) - 1
        if position < 0:
            return 1.0
        return max(float(censor_survival[position]), 1e-6)

    scores = []
    for bin_index in range(horizon):
        time = float(bin_index + 1)
        alive = times > time
        failed = events & (times <= time)
        weights = np.zeros(sample_count, dtype=np.float64)
        weights[alive] = 1.0 / censor_probability(time, include_equal=True)
        for index in np.flatnonzero(failed):
            weights[index] = 1.0 / censor_probability(
                times[index], include_equal=False
            )
        targets = alive.astype(np.float64)
        scores.append(float(np.sum(weights * (targets - survival[:, bin_index]) ** 2) / sample_count))
    return float(np.mean(scores)) if scores else None


class EvaluationAccumulator:
    def __init__(self):
        self.scalar_sums = defaultdict(float)
        self.scalar_counts = defaultdict(float)
        self.pml_predictions = []
        self.pml_targets = []
        self.sft_records = []

    def update(self, payload):
        objective = payload["objective"]
        if objective == "ntp":
            for name, value in payload["sums"].items():
                self.scalar_sums[name] += float(value.detach().float().cpu())
            for name, value in payload["counts"].items():
                self.scalar_counts[name] += float(value.detach().float().cpu())
            return
        if objective == "pml":
            self.pml_predictions.extend(
                payload["predictions"].detach().float().cpu().tolist()
            )
            self.pml_targets.extend(
                payload["targets"].detach().float().cpu().tolist()
            )
            return

        task_ids = payload["task_ids"].detach().cpu().tolist()
        task_types = payload["task_type_ids"].detach().cpu().tolist()
        for row, (task_id, task_type) in enumerate(zip(task_ids, task_types)):
            record = {"task_id": int(task_id), "task_type": int(task_type)}
            if task_type == TASK_TYPE_BINARY:
                record.update(
                    label=float(payload["labels"][row].detach().cpu()),
                    score=float(payload["binary_logits"][row].detach().float().cpu()),
                )
            elif task_type == TASK_TYPE_MULTICLASS:
                record.update(
                    label=int(payload["labels"][row].detach().cpu()),
                    scores=payload["classification_logits"][row].detach().float().cpu().numpy(),
                )
            elif task_type == TASK_TYPE_MULTILABEL:
                record.update(
                    labels=payload["multilabel_labels"][row].detach().float().cpu().numpy(),
                    scores=payload["classification_logits"][row].detach().float().cpu().numpy(),
                )
            elif task_type == TASK_TYPE_CANDIDATE_DIAGNOSIS:
                valid = payload["candidate_mask"][row].bool()
                record.update(
                    labels=payload["candidate_labels"][row][valid].detach().float().cpu().numpy(),
                    scores=payload["candidate_logits"][row][valid].detach().float().cpu().numpy(),
                )
            elif task_type == TASK_TYPE_TTE:
                horizon = int(payload["survival_labels"][row, 2].sum().item())
                record.update(
                    labels=payload["survival_labels"][row, :, :horizon].detach().float().cpu().numpy(),
                    scores=payload["survival_logits"][row, :horizon].detach().float().cpu().numpy(),
                )
            self.sft_records.append(record)

    def _state(self):
        return {
            "scalar_sums": dict(self.scalar_sums),
            "scalar_counts": dict(self.scalar_counts),
            "pml_predictions": self.pml_predictions,
            "pml_targets": self.pml_targets,
            "sft_records": self.sft_records,
        }

    def _gather(self):
        local = self._state()
        if not (dist.is_available() and dist.is_initialized()):
            return [local]
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        return gathered

    def compute(self, prefix, task_names, task_infos):
        states = self._gather()
        sums = defaultdict(float)
        counts = defaultdict(float)
        predictions = []
        targets = []
        records = []
        for state in states:
            for name, value in state["scalar_sums"].items():
                sums[name] += value
            for name, value in state["scalar_counts"].items():
                counts[name] += value
            predictions.extend(state["pml_predictions"])
            targets.extend(state["pml_targets"])
            records.extend(state["sft_records"])

        metrics = {}
        for name, value in sums.items():
            count_name = name.removesuffix("_sum") + "_count"
            if counts[count_name]:
                metrics[f"{prefix}_{name.removesuffix('_sum')}"] = value / counts[count_name]
        if predictions:
            pred = np.asarray(predictions, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            error = pred - target
            metrics[f"{prefix}_delta_mae"] = float(np.abs(error).mean())
            metrics[f"{prefix}_delta_rmse"] = float(np.sqrt(np.square(error).mean()))
            metrics[f"{prefix}_delta_sign_accuracy"] = float(
                (np.sign(pred) == np.sign(target)).mean()
            )
            if np.std(pred) > 0 and np.std(target) > 0:
                metrics[f"{prefix}_delta_pearson"] = float(np.corrcoef(pred, target)[0, 1])
                metrics[f"{prefix}_delta_spearman"] = float(spearmanr(pred, target).statistic)
        if records:
            metrics.update(self._compute_sft(prefix, records, task_names, task_infos))
        return metrics

    @staticmethod
    def _compute_sft(prefix, records, task_names, task_infos):
        metrics = {}
        grouped = defaultdict(list)
        for record in records:
            grouped[record["task_id"]].append(record)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for task_id, task_records in grouped.items():
                name = f"{prefix}_{_safe_name(task_names[task_id])}"
                task_type = task_records[0]["task_type"]
                info = task_infos[task_id]
                if task_type == TASK_TYPE_BINARY:
                    labels = np.asarray([row["label"] for row in task_records])
                    scores = np.asarray([row["score"] for row in task_records])
                    probabilities = 1.0 / (1.0 + np.exp(-scores))
                    if np.unique(labels).size == 2:
                        metrics[f"{name}_auroc"] = float(roc_auc_score(labels, probabilities))
                        metrics[f"{name}_auprc"] = float(average_precision_score(labels, probabilities))
                elif task_type == TASK_TYPE_MULTICLASS:
                    class_count = int(info["num_classes"])
                    labels = np.asarray([row["label"] for row in task_records])
                    predicted = np.stack([row["scores"][2 : 2 + class_count] for row in task_records]).argmax(axis=1)
                    metrics[f"{name}_accuracy"] = float(accuracy_score(labels, predicted))
                    metrics[f"{name}_macro_f1"] = float(f1_score(labels, predicted, average="macro"))
                elif task_type == TASK_TYPE_MULTILABEL:
                    class_count = int(info["num_classes"])
                    labels = np.stack([row["labels"][:class_count] for row in task_records])
                    logits = np.stack([row["scores"][2 : 2 + class_count] for row in task_records])
                    probabilities = 1.0 / (1.0 + np.exp(-logits))
                    predicted = probabilities >= 0.5
                    if np.unique(labels).size == 2:
                        metrics[f"{name}_micro_auroc"] = float(roc_auc_score(labels, probabilities, average="micro"))
                        metrics[f"{name}_micro_auprc"] = float(average_precision_score(labels, probabilities, average="micro"))
                    macro_auroc = _mean_valid_class_metric(labels, probabilities, roc_auc_score)
                    macro_auprc = _mean_valid_class_metric(labels, probabilities, average_precision_score)
                    if macro_auroc is not None:
                        metrics[f"{name}_macro_auroc"] = macro_auroc
                    if macro_auprc is not None:
                        metrics[f"{name}_macro_auprc"] = macro_auprc
                    metrics[f"{name}_micro_f1"] = float(f1_score(labels, predicted, average="micro", zero_division=0))
                    metrics[f"{name}_macro_f1"] = float(f1_score(labels, predicted, average="macro", zero_division=0))
                elif task_type == TASK_TYPE_CANDIDATE_DIAGNOSIS:
                    reciprocal_ranks = []
                    recalls = {1: [], 5: [], 10: []}
                    for row in task_records:
                        positive_count = float(row["labels"].sum())
                        if positive_count <= 0:
                            continue
                        order = np.argsort(-row["scores"])
                        ranked = row["labels"][order]
                        first = int(np.flatnonzero(ranked)[0]) + 1
                        reciprocal_ranks.append(1.0 / first)
                        for k in recalls:
                            recalls[k].append(float(ranked[:k].sum() / positive_count))
                    if reciprocal_ranks:
                        metrics[f"{name}_mrr"] = float(np.mean(reciprocal_ranks))
                        for k, values in recalls.items():
                            metrics[f"{name}_recall_at_{k}"] = float(np.mean(values))
                elif task_type == TASK_TYPE_TTE:
                    horizon = min(row["scores"].size for row in task_records)
                    labels = np.stack([row["labels"][:, :horizon] for row in task_records])
                    logits = np.stack([row["scores"][:horizon] for row in task_records])
                    hazards = np.logaddexp(0.0, logits)
                    survival = np.exp(-np.cumsum(hazards, axis=1))
                    times = labels[:, 0].sum(axis=1)
                    events = labels[:, 1].sum(axis=1) > 0
                    risks = -survival.sum(axis=1)
                    c_index = _concordance_index(times, events, risks)
                    if c_index is not None:
                        metrics[f"{name}_c_index"] = float(c_index)
                    metrics[f"{name}_integrated_brier_score"] = _integrated_brier_score(times, events, survival)
        return metrics


__all__ = ["EvaluationAccumulator"]
