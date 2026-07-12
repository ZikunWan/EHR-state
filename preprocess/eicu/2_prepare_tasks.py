"""
==============================================================================
STEP 2 of 5: Prepare Task Labels for All Prediction Tasks
==============================================================================

Purpose:
    Generate labels for all 12 prediction tasks:
    - Basic: mortality, long_term_mortality, readmission, los_3day, los_7day
    - Discharge: final_acuity, imminent_discharge
    - Diagnosis: diagnosis (multi-label CCS categories)
    - Lab values: creatinine, bilirubin, platelets, wbc (SOFA-based severity)

Input:
    - config.yaml: Configuration parameters
    - data/eicu-crd/processed/cohorts.csv: From Step 1
    - data/eicu-crd/2.0/diagnosis.csv: For diagnosis task
    - data/eicu-crd/2.0/lab.csv: For clinical lab tasks
    - data/eicu-crd/2.0/intakeOutput.csv: For dialysis filtering (creatinine)
    - preprocess/eicu/ccs_multi_dx_tool_2015.csv: CCS mapping (diagnosis task)
    - preprocess/eicu/icd10cmtoicd9gem.csv: ICD-10 to ICD-9 mapping (diagnosis task)

Output:
    - data/eicu-crd/processed/labeled_cohorts.csv: Cohorts with all task labels
    - data/eicu-crd/processed/final_acuity_classes.txt: Category mapping
    - data/eicu-crd/processed/imminent_discharge_classes.txt: Category mapping

Usage:
    python 2_prepare_tasks.py --config config.yaml

Prerequisites:
    Must run 1_build_cohorts.py first

Next Step:
    Run 3_generate_sample_info.py to create train/val/test splits

==============================================================================
"""
import os
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Import task utilities
from diagnosis_utils import (
    load_ccs_mapping,
    load_icd10_to_icd9_gem,
    create_diagnosis_mapping,
    process_diagnosis_labels
)
from lab_task_utils import process_clinical_lab_task


def prepare_tasks(config, cohorts):
    """
    Prepare task labels for all enabled tasks
    
    Tasks:
    - mortality, long_term_mortality
    - los_3day, los_7day
    - readmission (already labeled in cohorts)
    - final_acuity, imminent_discharge
    - diagnosis (requires diagnosis.csv + CCS mapping)
    - creatinine, bilirubin, platelets, wbc (requires lab.csv + thresholds)
    """
    print("=" * 80)
    print("eICU Task Label Generation")
    print("=" * 80)
    
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    enabled_tasks = config.get('tasks', [])
    
    obs_size = config.get('obs_size', 12)  # hours
    gap_size = config.get('gap_size', 12)
    pred_size = config.get('pred_size', 24)
    mortality_pred_size = config.get('mortality_pred_size', 48)
    imminent_discharge_pred_size = config.get('imminent_discharge_pred_size', 48)
    long_term_pred_size = config.get('long_term_pred_size', 336)
    
    print(f"\nEnabled tasks: {', '.join(enabled_tasks)}")
    print(f"\nTime windows:")
    print(f"  Observation: {obs_size} hours")
    print(f"  Gap: {gap_size} hours")
    print(f"  Prediction: {pred_size} hours")
    print(f"  Long-term prediction: {long_term_pred_size} hours")
    
    labeled_cohorts = cohorts[[
        'patientunitstayid',
        'patienthealthsystemstayid',
        'uniquepid',
        'readmission',
        'LOS',
        'INTIME',
        'OUTTIME',
        'DISCHTIME',
        'IN_ICU_MORTALITY',
        'HOS_DISCHARGE_LOCATION'
    ]].copy()
    
    # Mortality prediction
    if 'mortality' in enabled_tasks:
        print("\n→ Labeling mortality task...")
        # Death within pred_size after obs_size + gap_size
        labeled_cohorts['mortality'] = (
            (
                (labeled_cohorts['IN_ICU_MORTALITY'] == True) |
                (labeled_cohorts['HOS_DISCHARGE_LOCATION'] == 'Death')
            ) &
            (obs_size * 60 + gap_size * 60 < labeled_cohorts['DISCHTIME']) &
            (labeled_cohorts['DISCHTIME'] <= (obs_size + gap_size + mortality_pred_size) * 60)
        ).astype(int)
        
        mortality_dist = labeled_cohorts['mortality'].value_counts()
        print(f"  Mortality distribution: {dict(mortality_dist)}")
    
    # Long-term mortality
    if 'long_term_mortality' in enabled_tasks:
        print("\n→ Labeling long-term mortality task...")
        labeled_cohorts['long_term_mortality'] = (
            (
                (labeled_cohorts['IN_ICU_MORTALITY'] == True) |
                (labeled_cohorts['HOS_DISCHARGE_LOCATION'] == 'Death')
            ) &
            (obs_size * 60 + gap_size * 60 < labeled_cohorts['DISCHTIME']) &
            (labeled_cohorts['DISCHTIME'] <= (obs_size + gap_size + long_term_pred_size) * 60)
        ).astype(int)
        
        ltm_dist = labeled_cohorts['long_term_mortality'].value_counts()
        print(f"  Long-term mortality distribution: {dict(ltm_dist)}")
    
    # Length of stay
    if 'los_3day' in enabled_tasks:
        print("\n→ Labeling los_3day task...")
        labeled_cohorts['los_3day'] = (labeled_cohorts['LOS'] > 3).astype(int)
        los3_dist = labeled_cohorts['los_3day'].value_counts()
        print(f"  LOS > 3 days distribution: {dict(los3_dist)}")
    
    if 'los_7day' in enabled_tasks:
        print("\n→ Labeling los_7day task...")
        labeled_cohorts['los_7day'] = (labeled_cohorts['LOS'] > 7).astype(int)
        los7_dist = labeled_cohorts['los_7day'].value_counts()
        print(f"  LOS > 7 days distribution: {dict(los7_dist)}")
    
    # Final acuity and imminent discharge
    if 'final_acuity' in enabled_tasks or 'imminent_discharge' in enabled_tasks:
        print("\n→ Labeling acuity/discharge tasks...")
        
        # IN_HOSPITAL_MORTALITY (died in hospital but not in ICU)
        labeled_cohorts['IN_HOSPITAL_MORTALITY'] = (
            (~labeled_cohorts['IN_ICU_MORTALITY']) &
            (labeled_cohorts['HOS_DISCHARGE_LOCATION'] == 'Death')
        ).astype(int)
        
        if 'final_acuity' in enabled_tasks:
            labeled_cohorts['final_acuity'] = labeled_cohorts['HOS_DISCHARGE_LOCATION']
            labeled_cohorts.loc[
                labeled_cohorts['IN_ICU_MORTALITY'] == True,
                'final_acuity'
            ] = 'IN_ICU_MORTALITY'
            labeled_cohorts.loc[
                labeled_cohorts['IN_HOSPITAL_MORTALITY'] == 1,
                'final_acuity'
            ] = 'IN_HOSPITAL_MORTALITY'
            
            # Convert to category codes
            labeled_cohorts['final_acuity'] = labeled_cohorts['final_acuity'].astype('category')
            
            # Save category mapping
            categories = labeled_cohorts['final_acuity'].cat.categories
            with open(os.path.join(output_dir, 'final_acuity_classes.txt'), 'w') as f:
                for i, cat in enumerate(categories):
                    f.write(f"{i}\t{cat}\n")
            
            labeled_cohorts['final_acuity'] = labeled_cohorts['final_acuity'].cat.codes
            print(f"  Final acuity categories: {len(categories)}")
        
        if 'imminent_discharge' in enabled_tasks:
            is_discharged = (
                (obs_size * 60 + gap_size * 60 <= labeled_cohorts['DISCHTIME']) &
                (labeled_cohorts['DISCHTIME'] <= (obs_size + gap_size + imminent_discharge_pred_size) * 60)
            )
            
            labeled_cohorts['imminent_discharge'] = 'No Discharge'
            labeled_cohorts.loc[is_discharged, 'imminent_discharge'] = labeled_cohorts.loc[
                is_discharged, 'HOS_DISCHARGE_LOCATION'
            ]
            labeled_cohorts.loc[
                is_discharged & (
                    (labeled_cohorts['IN_ICU_MORTALITY'] == True) |
                    (labeled_cohorts['IN_HOSPITAL_MORTALITY'] == 1)
                ),
                'imminent_discharge'
            ] = 'Death'
            
            # Convert to category codes
            labeled_cohorts['imminent_discharge'] = labeled_cohorts['imminent_discharge'].astype('category')
            
            # Save category mapping
            categories = labeled_cohorts['imminent_discharge'].cat.categories
            with open(os.path.join(output_dir, 'imminent_discharge_classes.txt'), 'w') as f:
                for i, cat in enumerate(categories):
                    f.write(f"{i}\t{cat}\n")
            
            labeled_cohorts['imminent_discharge'] = labeled_cohorts['imminent_discharge'].cat.codes
            print(f"  Imminent discharge categories: {len(categories)}")
        
        # Drop temporary column
        labeled_cohorts = labeled_cohorts.drop(columns=['IN_HOSPITAL_MORTALITY'])
    
    # Diagnosis task (requires diagnosis.csv and CCS mapping)
    if 'diagnosis' in enabled_tasks:
        print("\n→ Labeling diagnosis task...")
        
        # Check for required files
        diagnosis_path = os.path.join(data_dir, "diagnosis.csv")
        if not os.path.exists(diagnosis_path):
            diagnosis_path = os.path.join(data_dir, "diagnosis.csv.gz")
        
        ccs_path = config.get('ccs_path', 'ccs_multi_dx_tool_2015.csv')
        gem_path = config.get('gem_path', 'icd10cmtoicd9gem.csv')
        
        if not os.path.exists(ccs_path):
            print(f"  Warning: CCS file not found: {ccs_path}")
            print("  Skipping diagnosis task. Please download from:")
            print("  https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip")
        elif not os.path.exists(gem_path):
            print(f"  Warning: ICD GEM file not found: {gem_path}")
            print("  Skipping diagnosis task. Please download from:")
            print("  https://data.nber.org/gem/icd10cmtoicd9gem.csv")
        elif not os.path.exists(diagnosis_path):
            print(f"  Warning: Diagnosis file not found: {diagnosis_path}")
            print("  Skipping diagnosis task.")
        else:
            try:
                # Load CCS and GEM mappings
                icd2cat = load_ccs_mapping(ccs_path)
                icd10_to_icd9 = load_icd10_to_icd9_gem(gem_path)
                
                # Load diagnosis data
                diagnosis_df = pd.read_csv(diagnosis_path, low_memory=False)
                
                # Create diagnosis mapping
                str2cat = create_diagnosis_mapping(diagnosis_df, icd2cat, icd10_to_icd9)
                
                # Process labels
                labeled_cohorts = process_diagnosis_labels(
                    labeled_cohorts, diagnosis_path, str2cat
                )
                
                print("  ✓ Diagnosis task completed")
            except Exception as e:
                print(f"  Error processing diagnosis task: {e}")
                import traceback
                traceback.print_exc()
    
    # Clinical lab tasks (requires lab.csv and threshold-based labeling)
    lab_tasks = [t for t in ['creatinine', 'bilirubin', 'platelets', 'wbc'] if t in enabled_tasks]
    if lab_tasks:
        print(f"\n→ Processing clinical lab tasks: {', '.join(lab_tasks)}")
        
        # Check if lab.csv exists
        lab_path = os.path.join(data_dir, "lab.csv")
        if not os.path.exists(lab_path):
            lab_path = os.path.join(data_dir, "lab.csv.gz")
        
        if not os.path.exists(lab_path):
            print(f"  Warning: Lab file not found: {lab_path}")
            print("  Skipping lab tasks.")
        else:
            try:
                # Process each lab task
                for task in lab_tasks:
                    labeled_cohorts = process_clinical_lab_task(
                        data_dir=data_dir,
                        cohorts=labeled_cohorts,
                        task_name=task,
                        obs_hours=obs_size,
                        gap_hours=gap_size,
                        pred_hours=pred_size
                    )
                
                print("  ✓ All lab tasks completed")
            except Exception as e:
                print(f"  Error processing lab tasks: {e}")
                import traceback
                traceback.print_exc()
    
    # Clean up unnecessary columns
    labeled_cohorts = labeled_cohorts.drop(
        columns=['LOS', 'IN_ICU_MORTALITY', 'DISCHTIME', 'HOS_DISCHARGE_LOCATION'],
        errors='ignore'
    )
    
    # Save labeled cohorts
    output_path = os.path.join(output_dir, "labeled_cohorts.csv")
    labeled_cohorts.to_csv(output_path, index=False)
    print(f"\n✓ Saved labeled cohorts to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TASK LABEL SUMMARY")
    print("=" * 80)
    print(f"Total cohorts: {len(labeled_cohorts)}")
    
    for task in enabled_tasks:
        if task in labeled_cohorts.columns:
            print(f"\n{task}:")
            if labeled_cohorts[task].dtype in ['int64', 'float64']:
                dist = labeled_cohorts[task].value_counts().sort_index()
                for val, count in dist.items():
                    print(f"  {val}: {count} ({count/len(labeled_cohorts)*100:.1f}%)")
            else:
                print(f"  Non-numeric labels (multi-label or categorical)")
    
    print("\n" + "=" * 80)
    
    return labeled_cohorts


def main():
    parser = argparse.ArgumentParser(description='Prepare eICU task labels')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--cohorts', type=str, help='Path to cohorts.csv (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load cohorts
    if args.cohorts:
        cohorts_path = args.cohorts
    else:
        cohorts_path = os.path.join(config['output_dir'], 'cohorts.csv')
    
    print(f"Loading cohorts from {cohorts_path}...")
    cohorts = pd.read_csv(cohorts_path)
    print(f"Loaded {len(cohorts)} cohorts")
    
    # Prepare tasks
    labeled_cohorts = prepare_tasks(config, cohorts)
    
    print("\n✓ Task preparation completed successfully!")


if __name__ == "__main__":
    main()
