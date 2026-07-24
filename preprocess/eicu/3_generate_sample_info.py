"""
==============================================================================
STEP 3 of 5: Generate Sample Info and Train/Val/Test Splits
==============================================================================

Purpose:
    Create individual samples for each (ICU stay, task) combination.
    Split data into training, validation, and test sets by patient ID.
    Save sample metadata in JSON format for dataset loading.

Input:
    - config.yaml: Configuration parameters
    - data/eicu-crd/processed/labeled_cohorts.csv: From Step 2

Output:
    - data/eicu-crd/processed/sample_info_train.json: Training samples
    - data/eicu-crd/processed/sample_info_val.json: Validation samples
    - data/eicu-crd/processed/sample_info_test.json: Test samples
    - data/eicu-crd/processed/sample_info_all.csv: All samples (CSV format)
    - <classification-index-dir>/<split>/<task>.csv: Per-task classification indices

Usage:
    python 3_generate_sample_info.py --config config.yaml

Prerequisites:
    Must run 1_build_cohorts.py and 2_prepare_tasks.py first

Next Step:
    1) Run 4_partition_patients.py to build per-patient table shards
    2) Run 5_generate_embeddings.py to build text embedding cache

==============================================================================
"""
import os
import ast
import argparse
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from collections import Counter


def write_classification_indices(samples, output_dir):
    index_df = pd.DataFrame(samples)
    for split in ("train", "val", "test"):
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = index_df[index_df["split"] == split]
        for task_name, task_df in split_df.groupby("task_name", sort=True):
            output_path = split_dir / f"{task_name}.csv"
            task_df.to_csv(output_path, index=False)
            print(f"  Saved {len(task_df)} samples to {output_path}")


def write_pretraining_indices(labeled_cohorts, patient_split, output_dir):
    samples_by_split = {split: [] for split in ("train", "val", "test")}
    for _, row in labeled_cohorts.iterrows():
        patient_id = str(row["uniquepid"])
        split = patient_split[patient_id]
        context_end = int(np.ceil(float(row["OUTTIME"])))
        if context_end <= 0:
            continue
        icustay_id = int(row["patientunitstayid"])
        samples_by_split[split].append(
            {
                "dataset": "eicu",
                "sample_id": f"eicu|{patient_id}|{icustay_id}|0|{context_end}",
                "patient_id": patient_id,
                "icustay_id": icustay_id,
                "context_begin": 0,
                "context_end": context_end,
                "obs_hours": int(np.ceil(context_end / 60.0)),
                "split": split,
                "task": "pretraining_context",
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    for split, samples in samples_by_split.items():
        json_path = os.path.join(output_dir, f"sample_info_{split}.json")
        csv_path = os.path.join(output_dir, f"sample_info_{split}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        pd.DataFrame(samples).to_csv(csv_path, index=False)
        print(f"  Saved {len(samples)} pretraining samples to {json_path}")


def generate_sample_info(
    config,
    labeled_cohorts,
    classification_index_dir=None,
    pretraining_index_dir=None,
):
    """
    Generate sample info for each (patient, task) combination
    
    Output format:
    {
        "icustay_id": int,
        "patient_id": str,
        "task_name": str,
        "label": int/str/list,
        "split": str,
        "obs_hours": int
    }
    """
    print("=" * 80)
    print("eICU Sample Info Generation")
    print("=" * 80)
    
    output_dir = config['output_dir']
    enabled_tasks = config.get('tasks', [])
    obs_hours = config.get('obs_size', 12)
    gap_hours = config.get('gap_size', 12)
    pred_hours = config.get('pred_size', 24)
    
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    
    print(f"\nEnabled tasks: {', '.join(enabled_tasks)}")
    print(f"Observation window: {obs_hours} hours")
    print(f"Gap window: {gap_hours} hours")
    print(f"Prediction window: {pred_hours} hours")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Assign splits based on patient ID (to avoid data leakage)
    print("\nAssigning train/val/test splits...")
    unique_patients = labeled_cohorts['uniquepid'].to_numpy(copy=True)
    np.random.RandomState(config.get('random_seed', 42)).shuffle(unique_patients)
    
    n_train = int(len(unique_patients) * train_ratio)
    n_val = int(len(unique_patients) * val_ratio)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]
    
    # Create patient to split mapping
    patient_split = {}
    for p in train_patients:
        patient_split[p] = 'train'
    for p in val_patients:
        patient_split[p] = 'val'
    for p in test_patients:
        patient_split[p] = 'test'
    
    labeled_cohorts['split'] = labeled_cohorts['uniquepid'].map(patient_split)
    
    split_counts = labeled_cohorts['split'].value_counts()
    print(f"  Train: {split_counts.get('train', 0)} stays from {len(train_patients)} patients")
    print(f"  Val: {split_counts.get('val', 0)} stays from {len(val_patients)} patients")
    print(f"  Test: {split_counts.get('test', 0)} stays from {len(test_patients)} patients")
    
    # Generate samples
    print("\nGenerating samples...")
    all_samples = []
    
    for _, row in labeled_cohorts.iterrows():
        icustay_id = int(row['patientunitstayid'])
        patient_id = str(row['uniquepid'])
        split = row['split']
        
        for task in enabled_tasks:
            if task not in labeled_cohorts.columns:
                continue
            
            label = row[task]
            
            # Skip NaN labels
            if pd.isna(label):
                continue
            
            # Convert label to appropriate type
            if task in ['mortality', 'long_term_mortality', 'readmission', 'los_3day', 'los_7day']:
                label = int(label)
            elif task in ['creatinine', 'bilirubin', 'platelets', 'wbc', 'final_acuity', 'imminent_discharge']:
                label = int(label)
            elif task == 'diagnosis':
                if isinstance(label, str):
                    label = ast.literal_eval(label) if label.startswith('[') else [label]
                if not label:
                    continue
            
            sample = {
                'icustay_id': icustay_id,
                'patient_id': patient_id,
                'task_name': task,
                'label': label,
                'split': split,
                'obs_hours': obs_hours,
                'gap_hours': gap_hours,
                'pred_hours': (
                    config.get('mortality_pred_size', 48)
                    if task == 'mortality'
                    else config.get('long_term_pred_size', 336)
                    if task == 'long_term_mortality'
                    else config.get('imminent_discharge_pred_size', 48)
                    if task == 'imminent_discharge'
                    else pred_hours
                )
            }
            
            all_samples.append(sample)
    
    print(f"Generated {len(all_samples)} samples")
    
    # Split into train/val/test
    train_samples = [s for s in all_samples if s['split'] == 'train']
    val_samples = [s for s in all_samples if s['split'] == 'val']
    test_samples = [s for s in all_samples if s['split'] == 'test']
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    # Save to JSON
    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        json_path = os.path.join(output_dir, f'sample_info_{split_name}.json')
        with open(json_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"  Saved {len(samples)} samples to {json_path}")
    
    # Also save combined CSV for reference
    df = pd.DataFrame(all_samples)
    csv_path = os.path.join(output_dir, 'sample_info_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved combined CSV to {csv_path}")
    if classification_index_dir:
        write_classification_indices(all_samples, classification_index_dir)
    if pretraining_index_dir:
        write_pretraining_indices(labeled_cohorts, patient_split, pretraining_index_dir)
    
    # Print task distribution
    print("\n" + "=" * 80)
    print("TASK DISTRIBUTION")
    print("=" * 80)
    
    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        print(f"\n{split_name.upper()} SET:")
        task_counts = Counter([s['task_name'] for s in samples])
        for task, count in task_counts.most_common():
            print(f"  {task}: {count}")
    
    print("\n" + "=" * 80)
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Generate eICU sample info')
    parser.add_argument('--config', type=str, default='preprocess/eicu/config.yaml',
                        help='Path to config file')
    parser.add_argument('--labeled_cohorts', type=str, help='Path to labeled_cohorts.csv (overrides config)')
    parser.add_argument(
        '--classification-index-dir',
        type=str,
        help='Output directory for split/task classification CSV files',
    )
    parser.add_argument(
        '--pretraining-index-dir',
        type=str,
        help='Output directory for unlabeled pretraining context indices',
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load labeled cohorts
    if args.labeled_cohorts:
        cohorts_path = args.labeled_cohorts
    else:
        cohorts_path = os.path.join(config['output_dir'], 'labeled_cohorts.csv')
    
    print(f"Loading labeled cohorts from {cohorts_path}...")
    labeled_cohorts = pd.read_csv(cohorts_path)
    print(f"Loaded {len(labeled_cohorts)} labeled cohorts")
    
    # Generate sample info
    samples = generate_sample_info(
        config,
        labeled_cohorts,
        classification_index_dir=args.classification_index_dir,
        pretraining_index_dir=args.pretraining_index_dir,
    )
    
    print("\n✓ Sample info generation completed successfully!")


if __name__ == "__main__":
    main()
