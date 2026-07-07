"""
Generate train/val/test splits for Renji dataset using labels.csv.
Uses multilabel stratified splitting based on label distribution.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


DEFAULT_DATA_DIR = "/data/zikun_workspace/input/tables/renji/raw"
DEFAULT_SAVE_DIR = f"{DEFAULT_DATA_DIR}/index"

LABEL_WINDOWS = ["0-30d", "30-180d", "180-365d", "365d+"]

# Target label names as they appear in labels.csv.
TARGET_LABEL_METRICS = [
    "ALB",
    "ALP",
    "CR",
    "血糖",
    "HB",
    "INR",
    "N(%)",
    "PLT",
    "PT",
    "TP",
    "WBC",
    "尿酸",
]
SURVIVAL_TARGETS = [
    "death_survival",
    "tacrolimus_abnormal_survival",
]

DEV_SAMPLE_SIZE = None  # Set to integer for quick testing


def get_label_fingerprints(labels_df):
    """
    Generate label fingerprints for stratification.
    Use all individual [window]_[metric] columns as the fingerprint.
    """
    target_cols = []
    
    # Identify all valid target columns that exist in the dataframe
    for m in TARGET_LABEL_METRICS:
        for w in LABEL_WINDOWS:
            col_name = f"{w}_{m}"
            if col_name in labels_df.columns:
                target_cols.append(col_name)
    for col_name in SURVIVAL_TARGETS:
        if col_name in labels_df.columns:
            target_cols.append(col_name)
    
    print(f"Using {len(target_cols)} individual label columns for stratification.")
    
    # Extract the binary matrix
    # Fill NaN with 0 for stratification purposes (treating missing as negative/ignore)
    # in this context, we just want to balance the positive labels we DO have.
    y = labels_df[target_cols].fillna(0).astype(int).values
    
    return y, target_cols


def random_train_val_test_split(num_samples, test_size, val_size, random_state):
    indices = np.arange(num_samples)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    test_count = int(round(num_samples * test_size))
    val_count = int(round(num_samples * val_size))
    test_count = min(max(test_count, 1), num_samples - 2)
    val_count = min(max(val_count, 1), num_samples - test_count - 1)

    test_idx = indices[:test_count]
    val_idx = indices[test_count:test_count + val_count]
    train_idx = indices[test_count + val_count:]
    return train_idx, val_idx, test_idx


def stratified_train_val_test_split(indices, y, test_size, val_size, random_state):
    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(msss_test.split(indices, y))

    relative_val_size = val_size / (1.0 - test_size)
    relative_val_size = min(max(relative_val_size, 1.0 / len(train_val_idx)), 0.5)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state,
    )
    train_rel_idx, val_rel_idx = next(
        msss_val.split(train_val_idx, y[train_val_idx])
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Generate Renji dataset train/val/test splits.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, 
                        help="Directory containing labels.csv")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, 
                        help="Directory to save split JSON files.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Validation set ratio over all patients (default: 0.1)")
    parser.add_argument("--dev_sample", type=int, default=DEV_SAMPLE_SIZE, 
                        help="Size of dev set for testing (optional).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    # 1. Load labels.csv
    labels_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"Error: labels.csv not found at {labels_path}")
        return
    
    labels_df = pd.read_csv(labels_path, encoding='utf-8-sig')
    print(f"Loaded labels.csv: {labels_df.shape[0]} patients, {labels_df.shape[1]} columns")
    
    if 'filename' not in labels_df.columns:
        print("Error: 'filename' column not found in labels.csv")
        return
    
    filenames = labels_df['filename'].tolist()
    
    # 2. Generate Fingerprints for stratification
    print("Generating label fingerprints for stratification...")
    y, target_cols = get_label_fingerprints(labels_df)
    indices = np.arange(len(filenames))
    
    # Handle edge case: if all fingerprints are zero, just do random split
    if y.sum() == 0:
        print("Warning: No positive labels found, using random split")
    else:
        # Downsampling for dev if requested
        if args.dev_sample is not None and args.dev_sample < len(filenames):
            print(f"Downsampling dataset to {args.dev_sample} samples for development...")
            remaining_size = len(filenames) - args.dev_sample
            
            msss_dev = MultilabelStratifiedShuffleSplit(
                n_splits=1, 
                train_size=args.dev_sample, 
                test_size=remaining_size, 
                random_state=args.random_state
            )
            dev_idx, _ = next(msss_dev.split(indices, y))
            
            indices = indices[dev_idx]
            y = y[dev_idx]
            filenames = [filenames[i] for i in dev_idx]
            indices = np.arange(len(filenames))  # Reset indices
            
            print(f"Development set size: {len(filenames)}")

        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            indices,
            y,
            args.test_size,
            args.val_size,
            args.random_state,
        )
    
    if y.sum() == 0:
        train_idx, val_idx, test_idx = random_train_val_test_split(
            len(filenames),
            args.test_size,
            args.val_size,
            args.random_state,
        )
    
    print(f"Split Result: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # 4. Save Results
    train_files = [filenames[i] for i in train_idx]
    val_files = [filenames[i] for i in val_idx]
    test_files = [filenames[i] for i in test_idx]
    
    # 4.1. all_valid_renji.json (Complete dictionary)
    split_data = {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "total_patients": len(filenames),
        "val_size": args.val_size,
        "test_size": args.test_size,
        "random_state": args.random_state
    }
    
    save_path_all = os.path.join(args.save_dir, "all_valid_renji.json")
    with open(save_path_all, "w", encoding='utf-8') as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_all}")

    # 4.2. train_renji.json
    save_path_train = os.path.join(args.save_dir, "train_renji.json")
    with open(save_path_train, "w", encoding='utf-8') as f:
        json.dump(train_files, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_train}")

    # 4.3. val_renji.json
    save_path_val = os.path.join(args.save_dir, "val_renji.json")
    with open(save_path_val, "w", encoding='utf-8') as f:
        json.dump(val_files, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_val}")

    # 4.4. test_renji.json
    save_path_test = os.path.join(args.save_dir, "test_renji.json")
    with open(save_path_test, "w", encoding='utf-8') as f:
        json.dump(test_files, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_test}")
    
    # Print label distribution stats
    print("\n--- Label Distribution ---")
    train_y = y[train_idx]
    val_y = y[val_idx]
    test_y = y[test_idx]
    
    for i, metric in enumerate(target_cols[:10]):  # Show first 10
        train_pos = train_y[:, i].sum()
        val_pos = val_y[:, i].sum()
        test_pos = test_y[:, i].sum()
        print(f"  {metric}: Train={train_pos}, Val={val_pos}, Test={test_pos}")
    if len(target_cols) > 10:
        print(f"  ... and {len(target_cols) - 10} more labels")


if __name__ == "__main__":
    main()
