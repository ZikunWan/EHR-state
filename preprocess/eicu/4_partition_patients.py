"""
==============================================================================
STEP 4 of 5: Partition Raw eICU Tables by ICU Stay
==============================================================================

Purpose:
    Split large raw tables into per-patientunitstayid CSV files under
    `processed/patients/{patientunitstayid}/`.

Input:
    - patient.csv(.gz)
    - lab.csv(.gz)
    - medication.csv(.gz)
    - infusionDrug.csv(.gz)

Output:
    - data/eicu-crd/2.0/processed/patients/{id}/{table}.csv

Usage:
    python 4_partition_patients.py

Prerequisites:
    Recommended to complete Steps 1-3 first so partitions align with generated
    sample info and downstream training/evaluation.

Next Step:
    Run 5_generate_embeddings.py

==============================================================================
"""
import argparse
import os
import multiprocessing as mp

import pandas as pd
import yaml
from tqdm import tqdm

def write_patient_group(args):
    """Worker function to write a single patient's dataframe to disk"""
    pid, group_df, out_dir, table_name = args
    pid_dir = os.path.join(out_dir, str(pid))
    os.makedirs(pid_dir, exist_ok=True)
    out_file = os.path.join(pid_dir, f"{table_name}.csv")
    group_df.to_csv(out_file, index=False)
    return 1

def process_table(table_path, out_dir, num_workers=8):
    """Loads a full table into memory, groups by patient, and writes files fast"""
    if not os.path.exists(table_path):
        if os.path.exists(table_path + ".gz"):
            table_path = table_path + ".gz"
        else:
            print(f"Skipping {table_path}, file not found.")
            return

    table_name = os.path.basename(table_path).split('.')[0]
    print(f"\n[{table_name}] Loading entire file into memory...")
    
    # Load entire dataset into memory to avoid chunking append-mode overhead
    df = pd.read_csv(table_path, low_memory=False)
    
    if 'patientunitstayid' not in df.columns:
        print(f"[{table_name}] No patientunitstayid column. Skipping.")
        return
        
    print(f"[{table_name}] Grouping by patientunitstayid...")
    grouped = df.groupby('patientunitstayid')
    
    # Prepare arguments for multiprocessing write
    # We use multiprocessing here just to speed up the disk I/O of writing thousands of small files
    args_list = [(pid, group, out_dir, table_name) for pid, group in grouped]
    
    print(f"[{table_name}] Writing {len(args_list)} patients to disk using {num_workers} processes...")
    
    with mp.Pool(num_workers) as pool:
        # Use imap_unordered with tqdm for progress bar
        list(tqdm(pool.imap_unordered(write_patient_group, args_list, chunksize=50), 
                  total=len(args_list), 
                  desc=f"{table_name} Write Progress"))
                  
    print(f"[{table_name}] Finished successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="preprocess/eicu/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data_dir"]
    out_dir = os.path.join(config["output_dir"], "patients")
    os.makedirs(out_dir, exist_ok=True)
    
    tables = ["patient.csv", "lab.csv", "medication.csv", "infusionDrug.csv"]
    
    print(f"Partitioning eICU data from {raw_dir} into {out_dir}")
    print(f"Strategy: Load full table -> In-memory GroupBy -> Multiprocessing Write")
    
    # Determine safe number of workers (leave some cores free)
    num_workers = max(1, mp.cpu_count() - 2)
    
    # Process each table sequentially to avoid out-of-memory from loading multiple huge tables
    for table in tables:
        process_table(os.path.join(raw_dir, table), out_dir, num_workers)
        
    print("\nAll partitioning complete!")

if __name__ == "__main__":
    main()
