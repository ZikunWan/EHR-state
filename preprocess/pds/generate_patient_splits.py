import argparse
import csv
import json
import os
import random
from collections import defaultdict


TASK_LABEL_FILES = {
    "severe_outcome": "severe_outcome.csv",
    "adverse_event_next_visit": "adverse_event_next_visit.csv",
}


def normalize_patient_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def parse_csv_list(value):
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_trials(root_dir):
    return sorted(
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name, "labels"))
    )


def read_patient_label_counts(label_path):
    patient_label_counts = defaultdict(lambda: {0: 0, 1: 0})
    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = normalize_patient_id(row["patient_id"])
            patient_label_counts[patient_id][int(row["label"])] += 1
    return dict(patient_label_counts)


def split_trial_patients(patient_label_counts, seed, ratios, num_attempts=20, local_search_swaps=2000):
    patient_ids = sorted(patient_label_counts)
    n_total = len(patient_ids)
    if n_total == 0:
        raise ValueError("Cannot split an empty patient set.")

    total_label_counts = {
        0: sum(counts[0] for counts in patient_label_counts.values()),
        1: sum(counts[1] for counts in patient_label_counts.values()),
    }
    if total_label_counts[0] == 0 or total_label_counts[1] == 0:
        raise ValueError("Patient split requires both positive and negative labels.")

    n_train = round(n_total * ratios[0])
    n_val = round(n_total * ratios[1])
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    target_patient_counts = {
        "train": n_train,
        "val": n_val,
        "test": n_total - n_train - n_val,
    }
    target_label_counts = {
        split: {
            0: total_label_counts[0] * ratios[idx],
            1: total_label_counts[1] * ratios[idx],
        }
        for idx, split in enumerate(("train", "val", "test"))
    }

    best_patients = None
    best_label_counts = None
    best_score = float("inf")

    for attempt in range(max(1, num_attempts)):
        rng = random.Random(f"{seed}:{attempt}")
        ordered_patients = list(patient_ids)
        rng.shuffle(ordered_patients)
        ordered_patients.sort(
            key=lambda patient_id: (
                sum(patient_label_counts[patient_id].values()) * (1.0 + 0.01 * rng.random()),
                abs(patient_label_counts[patient_id][1] - patient_label_counts[patient_id][0]),
            ),
            reverse=True,
        )

        assigned_patients = {"train": [], "val": [], "test": []}
        assigned_label_counts = {
            "train": {0: 0, 1: 0},
            "val": {0: 0, 1: 0},
            "test": {0: 0, 1: 0},
        }

        for patient_id in ordered_patients:
            patient_counts = patient_label_counts[patient_id]
            candidate_splits = [
                split
                for split in ("train", "val", "test")
                if len(assigned_patients[split]) < target_patient_counts[split]
            ]
            best_split = min(
                candidate_splits,
                key=lambda split: assignment_score_after_add(
                    assigned_label_counts,
                    split,
                    patient_counts,
                    target_label_counts,
                ),
            )
            assigned_patients[best_split].append(patient_id)
            assigned_label_counts[best_split][0] += patient_counts[0]
            assigned_label_counts[best_split][1] += patient_counts[1]

        assigned_patients, assigned_label_counts = improve_by_swapping_patients(
            assigned_patients,
            assigned_label_counts,
            patient_label_counts,
            target_label_counts,
            rng,
            local_search_swaps,
        )
        score = label_balance_score(assigned_label_counts, target_label_counts)
        if score < best_score:
            best_score = score
            best_patients = assigned_patients
            best_label_counts = assigned_label_counts

    return {
        split: sorted(patient_ids)
        for split, patient_ids in best_patients.items()
    }, best_label_counts


def assignment_score_after_add(all_counts, split_to_add, patient_counts, target_label_counts):
    next_counts = {
        split: {0: counts[0], 1: counts[1]}
        for split, counts in all_counts.items()
    }
    next_counts[split_to_add][0] += patient_counts[0]
    next_counts[split_to_add][1] += patient_counts[1]
    return label_balance_score(next_counts, target_label_counts)


def label_balance_score(label_counts, target_label_counts):
    score = 0.0
    for split in ("train", "val", "test"):
        for label in (0, 1):
            target = max(float(target_label_counts[split][label]), 1.0)
            diff = label_counts[split][label] - target
            score += (diff / target) ** 2
    return score


def improve_by_swapping_patients(
    assigned_patients,
    assigned_label_counts,
    patient_label_counts,
    target_label_counts,
    rng,
    max_swaps,
):
    splits = ("train", "val", "test")
    current_score = label_balance_score(assigned_label_counts, target_label_counts)
    for _ in range(max(0, max_swaps)):
        left_split, right_split = rng.sample(splits, 2)
        if not assigned_patients[left_split] or not assigned_patients[right_split]:
            continue
        left_patient = rng.choice(assigned_patients[left_split])
        right_patient = rng.choice(assigned_patients[right_split])
        left_counts = patient_label_counts[left_patient]
        right_counts = patient_label_counts[right_patient]

        next_counts = {
            split: {0: counts[0], 1: counts[1]}
            for split, counts in assigned_label_counts.items()
        }
        for label in (0, 1):
            next_counts[left_split][label] += right_counts[label] - left_counts[label]
            next_counts[right_split][label] += left_counts[label] - right_counts[label]

        next_score = label_balance_score(next_counts, target_label_counts)
        if next_score < current_score:
            assigned_patients[left_split].remove(left_patient)
            assigned_patients[right_split].remove(right_patient)
            assigned_patients[left_split].append(right_patient)
            assigned_patients[right_split].append(left_patient)
            assigned_label_counts = next_counts
            current_score = next_score

    return assigned_patients, assigned_label_counts


def split_ratio_score(current_counts, patient_counts, target_counts):
    return sum(
        abs(current_counts[label] + patient_counts[label] - target_counts[label])
        / target_counts[label]
        for label in (0, 1)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate PDS patient-level train/val/test splits.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--trial_ids", default=None, help="Comma-separated trial IDs. Defaults to all trials.")
    parser.add_argument(
        "--tasks",
        default="severe_outcome,adverse_event_next_visit",
        help="Comma-separated task names.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_attempts", type=int, default=20)
    parser.add_argument("--local_search_swaps", type=int, default=2000)
    args = parser.parse_args()

    trial_ids = parse_csv_list(args.trial_ids) or discover_trials(args.root_dir)
    tasks = parse_csv_list(args.tasks)
    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] <= 0:
        raise ValueError("train/val/test ratios must all be positive.")

    output = {}
    for task_name in tasks:
        label_file = TASK_LABEL_FILES[task_name]
        output[task_name] = {}
        for trial_id in trial_ids:
            label_path = os.path.join(args.root_dir, trial_id, "labels", label_file)
            if not os.path.exists(label_path):
                continue

            patient_label_counts = read_patient_label_counts(label_path)
            split_patients, split_label_counts = split_trial_patients(
                patient_label_counts,
                seed=f"{args.seed}:{task_name}:{trial_id}",
                ratios=ratios,
                num_attempts=args.num_attempts,
                local_search_swaps=args.local_search_swaps,
            )
            output[task_name][trial_id] = split_patients
            counts_text = " ".join(
                f"{split}=patients:{len(split_patients[split])},labels:{split_label_counts[split]}"
                for split in ("train", "val", "test")
            )
            print(f"{task_name} trial {trial_id}: {counts_text}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
