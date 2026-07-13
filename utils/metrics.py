from functools import partial

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def calc_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calc_auroc(y_true, y_prob, *, multi_class=None, average=None):
    try:
        kwargs = {}
        if multi_class is not None:
            kwargs["multi_class"] = multi_class
        if average is not None:
            kwargs["average"] = average
        return roc_auc_score(y_true, y_prob, **kwargs)
    except ValueError:
        return 0.5


def calc_auprc(y_true, y_prob, *, average=None):
    try:
        kwargs = {}
        if average is not None:
            kwargs["average"] = average
        return average_precision_score(y_true, y_prob, **kwargs)
    except ValueError:
        return 0.0


def calc_f1(y_true, y_pred, *, average="binary"):
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calc_recall(y_true, y_pred, *, average="binary"):
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_classification_metrics(eval_pred):
    """
    Unified metric function for HuggingFace Trainer.

    Returns:
        {
            "auroc": float,
            "auprc": float,
            "accuracy": float,
            "f1": float,
            "recall": float,
        }
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    y_true = labels.reshape(-1).astype(int)

    # Binary classification: logits shape [N] or [N, 1]
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        probs = _sigmoid(logits.reshape(-1))
        preds = (probs > 0.5).astype(int)

        auroc = calc_auroc(y_true, probs)
        auprc = calc_auprc(y_true, probs)
        acc = calc_accuracy(y_true, preds)
        f1 = calc_f1(y_true, preds, average="binary")
        recall = calc_recall(y_true, preds, average="binary")
        return {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc,
            "f1": f1,
            "recall": recall,
        }

    # Binary classification: two-logit format [N, 2]
    if logits.ndim == 2 and logits.shape[-1] == 2 and labels.ndim == 1:
        probs_2d = _softmax(logits, axis=-1)
        probs = probs_2d[:, 1]
        preds = np.argmax(probs_2d, axis=-1).astype(int)

        # In case labels are not exactly {0,1}, remap to {0,1} by sorted order.
        unique_labels = np.unique(y_true)
        if unique_labels.size == 2 and not np.array_equal(unique_labels, np.array([0, 1])):
            remap = {int(unique_labels[0]): 0, int(unique_labels[1]): 1}
            y_true = np.array([remap[int(v)] for v in y_true], dtype=int)

        auroc = calc_auroc(y_true, probs)
        auprc = calc_auprc(y_true, probs)
        acc = calc_accuracy(y_true, preds)
        f1 = calc_f1(y_true, preds, average="binary")
        recall = calc_recall(y_true, preds, average="binary")
        return {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc,
            "f1": f1,
            "recall": recall,
        }

    # Multi-label classification: logits and labels both [N, C]
    if labels.ndim == logits.ndim and labels.shape == logits.shape and logits.ndim == 2:
        probs = _sigmoid(logits)
        preds = (probs > 0.5).astype(int)
        mask = labels != -100
        if not np.any(mask):
            return {
                "auroc": 0.5,
                "auprc": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0,
            }

        y_true = labels[mask].astype(int)
        y_prob = probs[mask]
        y_pred = preds[mask]

        auroc = calc_auroc(y_true, y_prob)
        auprc = calc_auprc(y_true, y_prob)
        acc = calc_accuracy(y_true, y_pred)
        f1 = calc_f1(y_true, y_pred, average="binary")
        recall = calc_recall(y_true, y_pred, average="binary")
        return {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc,
            "f1": f1,
            "recall": recall,
        }

    # Multi-class classification: logits [N, C], labels [N]
    probs = _softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=-1)

    auroc = calc_auroc(y_true, probs, multi_class="ovr")
    auprc = calc_auprc(y_true, probs, average="macro")
    acc = calc_accuracy(y_true, preds)
    f1 = calc_f1(y_true, preds, average="macro")
    recall = calc_recall(y_true, preds, average="macro")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
    }


def compute_metrics(eval_pred):
    return compute_classification_metrics(eval_pred)


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def cumulative_hazard_at_times(hazards, times):
    hazards = np.asarray(hazards, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 0:
        times = np.full(len(hazards), float(times))
    bin_left = np.arange(hazards.shape[1], dtype=np.float64)
    exposure = np.clip(times[:, None] - bin_left, 0.0, 1.0)
    return np.sum(hazards * exposure, axis=1)


def risk_at_time(hazards, time):
    return 1.0 - np.exp(-cumulative_hazard_at_times(hazards, time))


def risk_at_times(hazards, times):
    return np.column_stack([risk_at_time(hazards, time) for time in times])


def survival_at_times(hazards, times):
    return 1.0 - risk_at_times(hazards, times)


def average_hazard_over_interval(hazards, start, end):
    if end <= start:
        raise ValueError(f"Invalid averaging interval: {start} to {end}")
    h_start = cumulative_hazard_at_times(hazards, start)
    h_end = cumulative_hazard_at_times(hazards, end)
    return (h_end - h_start) / (end - start)


def harrell_c_index(times, events, risk_scores):
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    concordant = 0.0
    comparable = 0
    for i in range(len(times)):
        if not events[i]:
            continue
        comparators = np.flatnonzero(
            (times > times[i])
            | ((times == times[i]) & (~events))
        )
        if comparators.size == 0:
            continue
        differences = risk_scores[i] - risk_scores[comparators]
        concordant += np.sum(differences > 1e-8)
        concordant += 0.5 * np.sum(np.abs(differences) <= 1e-8)
        comparable += comparators.size
    return concordant / comparable if comparable else np.nan


def _km_curve(times, events):
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    unique_times = np.unique(times)
    survival = 1.0
    values = []
    for time in unique_times:
        at_risk = np.sum(times >= time)
        event_count = np.sum((times == time) & events)
        if at_risk:
            survival *= 1.0 - event_count / at_risk
        values.append(survival)
    return unique_times, np.asarray(values, dtype=np.float64)


def _step_lookup(query_times, curve_times, curve_values):
    query_times = np.asarray(query_times, dtype=np.float64)
    indices = np.searchsorted(curve_times, query_times, side="right") - 1
    output = np.ones(query_times.shape, dtype=np.float64)
    valid = indices >= 0
    output[valid] = curve_values[indices[valid]]
    return output


def _censoring_survival(train_times, train_events, query_times):
    train_times = np.asarray(train_times, dtype=np.float64)
    train_events = np.asarray(train_events, dtype=bool)
    curve_times = np.unique(train_times)
    survival = 1.0
    curve_values = []
    for time in curve_times:
        failed = np.sum((train_times == time) & train_events)
        censored = np.sum((train_times == time) & (~train_events))
        at_risk = np.sum(train_times >= time) - failed
        if at_risk > 0:
            survival *= 1.0 - censored / at_risk
        curve_values.append(survival)
    curve_values = np.asarray(curve_values, dtype=np.float64)
    return _step_lookup(query_times, curve_times, curve_values)


def _event_survival(times, events, query_times):
    curve_times, curve_values = _km_curve(times, events)
    return _step_lookup(query_times, curve_times, curve_values)


def cumulative_dynamic_auc(
    train_times,
    train_events,
    eval_times_observed,
    eval_events,
    risk_matrix,
    eval_times,
):
    eval_times_observed = np.asarray(eval_times_observed, dtype=np.float64)
    eval_events = np.asarray(eval_events, dtype=bool)
    risk_matrix = np.asarray(risk_matrix, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)
    g_event = _censoring_survival(
        train_times, train_events, eval_times_observed
    )
    ipcw = np.zeros(len(eval_times_observed), dtype=np.float64)
    valid_event = eval_events & (g_event > 0)
    ipcw[valid_event] = 1.0 / g_event[valid_event]

    aucs = np.full(len(eval_times), np.nan, dtype=np.float64)
    for time_index, time in enumerate(eval_times):
        cases = (eval_times_observed <= time) & eval_events & (ipcw > 0)
        controls = eval_times_observed > time
        if not np.any(cases) or not np.any(controls):
            continue
        case_risk = risk_matrix[cases, time_index][:, None]
        control_risk = risk_matrix[controls, time_index][None, :]
        differences = case_risk - control_risk
        concordance = (differences > 1e-8).astype(np.float64)
        concordance += 0.5 * (np.abs(differences) <= 1e-8)
        weights = ipcw[cases, None]
        aucs[time_index] = np.sum(concordance * weights) / (
            np.sum(weights) * np.sum(controls)
        )

    finite = np.isfinite(aucs)
    if not np.any(finite):
        return aucs, np.nan
    survival = _event_survival(
        eval_times_observed, eval_events, eval_times[finite]
    )
    increments = -np.diff(np.r_[1.0, survival])
    denominator = 1.0 - survival[-1]
    mean_auc = (
        np.sum(aucs[finite] * increments) / denominator
        if denominator > 0
        else np.nan
    )
    return aucs, float(mean_auc)


def integrated_brier_score(
    train_times,
    train_events,
    eval_times_observed,
    eval_events,
    survival_matrix,
    eval_times,
):
    eval_times_observed = np.asarray(eval_times_observed, dtype=np.float64)
    eval_events = np.asarray(eval_events, dtype=bool)
    survival_matrix = np.asarray(survival_matrix, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)
    g_observed = _censoring_survival(
        train_times, train_events, eval_times_observed
    )
    g_times = _censoring_survival(train_times, train_events, eval_times)
    scores = np.zeros(len(eval_times), dtype=np.float64)

    for time_index, time in enumerate(eval_times):
        estimate = survival_matrix[:, time_index]
        event_before = (eval_times_observed <= time) & eval_events
        still_at_risk = eval_times_observed > time
        event_weight = np.zeros(len(estimate), dtype=np.float64)
        valid_event = event_before & (g_observed > 0)
        event_weight[valid_event] = 1.0 / g_observed[valid_event]
        at_risk_weight = (
            still_at_risk.astype(np.float64) / g_times[time_index]
            if g_times[time_index] > 0
            else np.zeros(len(estimate), dtype=np.float64)
        )
        scores[time_index] = np.mean(
            estimate**2 * event_weight
            + (1.0 - estimate) ** 2 * at_risk_weight
        )

    interval = eval_times[-1] - eval_times[0]
    if interval <= 0:
        return np.nan
    return float(np.trapezoid(scores, eval_times) / interval)


def km_survival_at_time(times, events, time):
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    if len(times) == 0:
        return np.nan
    if np.max(times) < time and not np.any(events & (times <= time)):
        return np.nan
    return float(_event_survival(times, events, np.asarray([time]))[0])


def nd_calibration_metric(
    times,
    events,
    predicted_survival,
    time,
    n_bins=10,
    eps=1e-6,
):
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    predicted_survival = np.clip(
        np.asarray(predicted_survival, dtype=np.float64),
        eps,
        1.0 - eps,
    )
    valid = np.isfinite(times) & np.isfinite(predicted_survival)
    times = times[valid]
    events = events[valid]
    predicted_survival = predicted_survival[valid]
    if len(times) < 2:
        return np.nan, 0

    groups = np.array_split(
        np.argsort(predicted_survival),
        max(2, min(int(n_bins), len(times))),
    )
    statistic = 0.0
    used_bins = 0
    for group in groups:
        if len(group) == 0:
            continue
        predicted = float(
            np.clip(np.mean(predicted_survival[group]), eps, 1.0 - eps)
        )
        observed = km_survival_at_time(
            times[group], events[group], time
        )
        if not np.isfinite(observed):
            continue
        statistic += (
            (observed - predicted) ** 2 / (predicted * (1.0 - predicted))
        )
        used_bins += 1
    return (
        (float(statistic), used_bins)
        if used_bins
        else (np.nan, 0)
    )


def get_stage_eval_times(event_times, n_grid=256):
    event_times = np.asarray(event_times, dtype=np.float64)
    event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
    if len(event_times) == 0:
        return None
    start = float(np.percentile(event_times, 10.0))
    end = float(np.percentile(event_times, 90.0))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    return np.linspace(start, end, int(n_grid), dtype=np.float64)


def build_survival_reference(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset.samples))
    return {
        "stage_id": np.asarray(
            [dataset.samples[index]["stage_id"] for index in indices],
            dtype=np.int64,
        ),
        "time": np.asarray(
            [dataset.samples[index]["time_to_event"] for index in indices],
            dtype=np.float64,
        ),
        "event": np.asarray(
            [dataset.samples[index]["event_observed"] for index in indices],
            dtype=bool,
        ),
        "stage_end_horizon": np.asarray(
            [
                dataset.samples[index]["stage_end_horizon"]
                for index in indices
            ],
            dtype=np.float64,
        ),
    }


def _stage_metrics(
    logits,
    labels,
    train_reference,
    stage_id,
    stage_bins,
    n_eval_grid,
    nd_bins,
):
    num_bins = stage_bins[stage_id]
    hazards_all = softplus(logits[:, :num_bins])
    times_all = np.sum(labels[:, 0, :num_bins], axis=1)
    events_all = np.sum(labels[:, 1, :num_bins], axis=1) > 0
    stage_end_all = labels[:, 3, 0]
    event_times = times_all[events_all]
    eval_times = get_stage_eval_times(event_times, n_grid=n_eval_grid)
    base = {
        "samples_total": float(len(times_all)),
        "events_total": float(np.sum(events_all)),
    }
    if eval_times is None:
        return base

    eval_tmax = float(eval_times[-1])
    nd_time = float(np.median(event_times))
    eval_keep = stage_end_all >= eval_tmax
    train_stage = train_reference["stage_id"] == stage_id
    train_keep = train_stage & (
        train_reference["stage_end_horizon"] >= eval_tmax
    )
    times = times_all[eval_keep]
    events = events_all[eval_keep]
    hazards = hazards_all[eval_keep]
    train_times = train_reference["time"][train_keep]
    train_events = train_reference["event"][train_keep]
    if len(times) < 2 or len(train_times) < 2:
        return base

    max_observed = min(float(np.max(train_times)), float(np.max(times)))
    if eval_tmax >= max_observed:
        eval_times = eval_times[eval_times < max_observed]
        if len(eval_times) < 2:
            return base
    eval_tmin = float(eval_times[0])
    eval_tmax = float(eval_times[-1])

    risks = risk_at_times(hazards, eval_times)
    _, time_dependent_c = cumulative_dynamic_auc(
        train_times,
        train_events,
        times,
        events,
        risks,
        eval_times,
    )
    harrell_risk = average_hazard_over_interval(
        hazards, eval_tmin, eval_tmax
    )
    harrell = harrell_c_index(times, events, harrell_risk)
    predicted_survival = 1.0 - risk_at_time(hazards, nd_time)
    nd_statistic, nd_used_bins = nd_calibration_metric(
        times,
        events,
        predicted_survival,
        time=nd_time,
        n_bins=nd_bins,
    )
    ibs = integrated_brier_score(
        train_times,
        train_events,
        times,
        events,
        survival_at_times(hazards, eval_times),
        eval_times,
    )
    return {
        **base,
        "samples_used": float(len(times)),
        "eval_time_min_days": eval_tmin,
        "eval_time_max_days": eval_tmax,
        "time_dependent_c": time_dependent_c,
        "harrell_c_index": harrell,
        "nd_calibration": nd_statistic,
        "nd_calibration_bins": float(nd_used_bins),
        "nd_calibration_time_days": nd_time,
        "ibs": ibs,
    }


def compute_piecewise_survival_metrics(
    eval_pred,
    train_reference,
    stage_bins=(31, 150, 185),
    n_eval_grid=256,
    nd_bins=10,
):
    stage_bins = tuple(int(value) for value in stage_bins)
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(eval_pred.label_ids)
    if labels.shape[1] < 4:
        raise ValueError(
            "Survival labels must include stage_end_horizon metadata"
        )
    horizons = np.sum(labels[:, 2, :], axis=1).astype(int)
    label_stage_ids = None
    if labels.shape[2] > 1 and np.any(labels[:, 3, 1] > 0):
        label_stage_ids = labels[:, 3, 1].astype(int)

    output = {}
    stage_results = []
    for stage_id, horizon in enumerate(stage_bins):
        mask = (
            label_stage_ids == stage_id
            if label_stage_ids is not None
            else horizons == horizon
        )
        if not np.any(mask):
            continue
        metrics = _stage_metrics(
            logits[mask],
            labels[mask],
            train_reference,
            stage_id,
            stage_bins,
            n_eval_grid,
            nd_bins,
        )
        stage_results.append(metrics)
        for name, value in metrics.items():
            output[f"stage_{stage_id}_{name}"] = float(value)

    for name in (
        "time_dependent_c",
        "harrell_c_index",
        "nd_calibration",
        "ibs",
    ):
        values = [
            metrics[name]
            for metrics in stage_results
            if name in metrics and np.isfinite(metrics[name])
        ]
        output[name] = float(np.mean(values)) if values else 0.0
    output["nam_dagostino"] = output["nd_calibration"]
    return output


def create_piecewise_survival_metrics(
    train_reference,
    stage_bins=(31, 150, 185),
    n_eval_grid=256,
    nd_bins=10,
):
    return partial(
        compute_piecewise_survival_metrics,
        train_reference=train_reference,
        stage_bins=stage_bins,
        n_eval_grid=n_eval_grid,
        nd_bins=nd_bins,
    )
