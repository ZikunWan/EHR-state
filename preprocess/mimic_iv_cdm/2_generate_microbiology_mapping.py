import argparse
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIMIC-IV-CDM microbiology test mapping.")
    parser.add_argument("--micro_events_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Reading {args.micro_events_path}...")
    
    # Read relevant columns to save memory
    cols = ['test_itemid', 'test_name']
    df = pd.read_csv(args.micro_events_path, usecols=cols)
    
    # Create mappings
    print("Generating mappings...")
    mapping = {}
    
    # test_itemid -> test_name
    test_map = df[['test_itemid', 'test_name']].dropna().drop_duplicates()
    for _, row in test_map.iterrows():
        mapping[int(row['test_itemid'])] = row['test_name']
     
    print(f"Total mapping items: {len(mapping)}")
    
    # Save
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(mapping, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
