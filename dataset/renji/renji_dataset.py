import os
import sys
import json
import math
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.renji.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key


def _is_main_process():
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0

    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank is None or int(local_rank) == 0


def _rank0_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)


class RenjiDataset(Dataset):
    """
    Renji Pediatric Liver Transplant Dataset.
    
    Predicts outcomes at 3 fixed time points per patient:
    - Day 30: Predict 30-180d labels
    - Day 180: Predict 180-365d labels
    - Day 365: Predict 365d+ labels
    """
    
    # Prediction points: (cutoff_day, label_prefix, readable_name)
    PREDICTION_POINTS = {
        'day30': (30, '30-180d', 'Day 30'),
        'day180': (180, '180-365d', 'Day 180'),
        'day365': (365, '365d+', 'Day 365'),
    }

    DRUG_CONC_MED_COLS = {
        'Tacrolimus_Conc': ['Tacrolimus_ER', 'Tacrolimus_Saifukai', 'Tacrolimus_Prograf'],
        'CsA_Trough': ['Cyclosporine', 'Sandimmun'],
        'CsA_Peak': ['Cyclosporine', 'Sandimmun'],
    }

    DRUG_CONC_RANGES = {
        'Tacrolimus_Conc': [
            (0, 31, 8.0, 12.0),
            (31, 181, 7.0, 10.0),
            (181, 366, 5.0, 8.0),
            (366, float('inf'), 4.0, 6.0),
        ],
        'CsA_Trough': [
            (0, 31, 150.0, 200.0),
            (31, 181, 120.0, 150.0),
            (181, 366, 100.0, 120.0),
            (366, float('inf'), 80.0, 120.0),
        ],
        'CsA_Peak': [
            (0, 31, 1000.0, 1200.0),
            (31, 181, 800.0, 1000.0),
            (181, 366, 500.0, 800.0),
            (366, float('inf'), 400.0, 600.0),
        ],
    }
    
    # Static features from patient info: {column_name: readable_name}
    STATIC_FEATURES = {
        'recipient_gender': 'Recipient Gender',
        'recipient_weight': 'Recipient Weight (kg)',
        'donor_liver_weight': 'Donor Liver Weight (g)',
        'GRWR cl': 'GRWR',
        'operation_type_code': 'Operation Type',
        'recipient_blood_type': 'Recipient Blood Type',
        'recipient_cyp3a5_genotype': 'Recipient CYP3A5 Genotype',
        'donor_gender': 'Donor Gender',
        'donor_age': 'Donor Age',
        'donor_blood_type': 'Donor Blood Type',
        'donor_cyp3a5_genotype': 'Donor CYP3A5 Genotype',
        'ABO_compatibility': 'ABO Compatibility',
    }
    
    # Balanced metrics for multi-task classification.
    # Selected over day30/day180/day365 with:
    # 0.2 <= positive_rate <= 0.8 and minority_count >= 150 at every time point.
    ALL_METRICS = sorted([
        'ALB', 'ALP', 'PT', 'INR', 'TP',                           # Graft injury
        'WBC', 'N_Percent', 'HB', 'PLT',                           # Immune/Infection
        'CR', 'Glucose', 'Uric_Acid',                              # Metabolic/Renal
    ])
    
    # Combined list of all prediction points (sorted by day)
    ALL_POINTS = ['day30', 'day180', 'day365']
    TASK_INFO = get_task_info()
    TASK_ALL_METRICS = ALL_METRICS
    TASK_ALL_POINTS = ALL_POINTS
    TASK_PREDICTION_POINTS = PREDICTION_POINTS

    def __init__(self, 
                 root_dir, 
                 split='train', 
                 max_samples=None,
                 target_metrics=None,
                 target_prediction_points=None,
                 shuffle=False,
                 ):
        """
        Initialize RenjiDataset.
        
        Args:
            root_dir: Root directory containing the data
            split: Data split ('train', 'val', 'test')
            max_samples: Maximum number of samples to use
            target_metrics: List of metric names to filter (e.g., ['ALT', 'AST', 'TB'])
                          If None, use all metrics
            target_prediction_points: List of prediction point keys to filter
                          (e.g., ['day30', 'day180', 'day365']). If None, use all.
            shuffle: Whether to shuffle the dataset
        """
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.target_metrics = target_metrics
        self.target_prediction_points = target_prediction_points
        self.active_points = list(target_prediction_points) if target_prediction_points is not None else list(self.ALL_POINTS)
        self.shuffle = shuffle
        self.task_schema = get_task_info()
        
        self.followup_dir = os.path.join(self.root_dir, 'follow_ups')
        self.index_dir = os.path.join(self.root_dir, 'index')
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")
        
        self._init_configs()
        self._load_auxiliary_data()
        
        # Load split file list
        split_path = os.path.join(self.index_dir, f'{split}_renji.json')
        with open(split_path, 'r', encoding='utf-8') as f:
            self.filenames = json.load(f)
        
        self._valid_followup_cache = {}
        self.samples = self._build_index()

    def _init_configs(self):
        """Initialize feature configurations."""
        
        # Columns to exclude from follow-up data (prediction targets, not inputs)
        self.EXCLUDED_COLS = {
            '免疫抑制剂浓度',
            'HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb',
            '细菌真菌感染', '排斥', 'CMV感染', 'EBV感染', 'HBV感染'
        }
    
        # Medication columns
        self.MED_COLS_ZH = {
            '他克莫司缓释胶囊', '他克莫司(赛福开)', '他克莫司(普乐可复)', 
            '环孢素', '新山地明', '吗替麦考酚酯', '骁悉', '赛可平', 
            '米芙', '雷帕鸣', '醋酸泼尼松', '甲泼尼龙片', '美卓乐'
        }

        # Chinese to English mapping
        self.ZH_TO_EN = {
            # Medications
            '他克莫司缓释胶囊': 'Tacrolimus_ER',
            '他克莫司(赛福开)': 'Tacrolimus_Saifukai',
            '他克莫司(普乐可复)': 'Tacrolimus_Prograf',
            '环孢素': 'Cyclosporine',
            '新山地明': 'Sandimmun',
            '吗替麦考酚酯': 'MMF',
            '骁悉': 'CellCept',
            '赛可平': 'Saikeping',
            '米芙': 'Mifu',
            '雷帕鸣': 'Rapamune',
            '醋酸泼尼松': 'Prednisone',
            '甲泼尼龙片': 'Methylprednisolone',
            '美卓乐': 'Medrol',

            # Labs & Vitals
            '身高': 'Height',
            '体重': 'Weight',
            'WBC': 'WBC',
            'N(%)': 'N_Percent',
            '淋巴细胞绝对值': 'Lymphocyte_Abs',
            '嗜酸性粒细胞百分比': 'Eosinophil_Percent',
            'HB': 'HB',
            'PLT': 'PLT',
            'TP': 'TP',
            'ALB': 'ALB',
            'ALT': 'ALT',
            'AST': 'AST',
            'ALP': 'ALP',
            'γ-GT': 'GGT',
            'DB': 'DB',
            'TB': 'TB',
            '胆汁酸': 'Bile_Acid',
            'CR': 'CR',
            '血糖': 'Glucose',
            '甘油三脂': 'Triglyceride', 
            '总胆固醇': 'Cholesterol',
            '尿酸': 'Uric_Acid',
            'PT': 'PT',
            'INR': 'INR',
            '雷帕浓度': 'Rapa_Conc',
            '血氨': 'Blood_Ammonia',
            '他克莫司浓度': 'Tacrolimus_Conc',
            '环孢素谷浓度': 'CsA_Trough',
            '环孢素峰浓度': 'CsA_Peak',
            'CMV-DNA': 'CMV_DNA',
            'EBV-DNA': 'EBV_DNA',
            'HBV-DNA': 'HBV_DNA',
            'HBsAg': 'HBsAg',
            'HBsAb': 'HBsAb',
            'HBeAg': 'HBeAg',
            'HBeAb': 'HBeAb',
            'HBcAb': 'HBcAb'
        }
        
        self.TASK_GROUPS = {
            "graft_injury": {
                "name": "Graft Injury Assessment",
                "features": ['ALT', 'AST', 'ALP', 'GGT', 'TB', 'DB', 'Bile_Acid', 'TP', 'ALB', 'PT', 'INR']
            },
            "drug_conc": {
                "name": "Drug Concentration Monitoring",
                "features": ['Tacrolimus_Conc', 'CsA_Trough', 'CsA_Peak']
            },
            "immune_infection": {
                "name": "Immune & Infection Assessment",
                "features": ['WBC', 'N_Percent', 'Lymphocyte_Abs', 'HB', 'PLT']
            },
            "metabolic_renal": {
                "name": "Metabolic & Renal Function",
                "features": ['CR', 'Glucose', 'Uric_Acid', 'Triglyceride', 'Cholesterol']
            },
            "virus_activation": {
                "name": "Virus Activation Monitoring",
                "features": ['CMV_DNA', 'EBV_DNA', 'HBV_DNA']
            }
        }

        self.med_cols_en = {self.ZH_TO_EN[c] for c in self.MED_COLS_ZH}

        # Medication metadata
        self.MED_META = {
            'Tacrolimus_ER':      {'str': '24 HR tacrolimus 0.5 MG Extended Release Oral Capsule', 'unit': 'mg'},
            'Tacrolimus_Saifukai':{'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg'},
            'Tacrolimus_Prograf': {'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg'},
            'Tacrolimus':         {'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg'},
            'Cyclosporine':       {'str': 'cycloSPORINE 25 MG Oral Capsule', 'unit': 'mg'},
            'Sandimmun':          {'str': 'cycloSPORINE 25 MG Oral Capsule', 'unit': 'mg'},
            'Rapamune':           {'str': 'sirolimus 1 MG Oral Tablet', 'unit': 'mg'},
            'Prednisone':         {'str': 'predniSONE 5 MG Oral Tablet', 'unit': 'mg'},
            'Methylprednisolone': {'str': 'methylPREDNISolone 4 MG Oral Tablet', 'unit': 'mg'},
            'Medrol':             {'str': 'methylPREDNISolone 4 MG Oral Tablet', 'unit': 'mg'},
            'Mifu':               {'str': 'mycophenolic acid 180 MG Delayed Release Oral Tablet', 'unit': 'mg'},
            'MMF':                {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g'},
            'CellCept':           {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g'},
            'Saikeping':          {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g'},
        }

        # Value mapping for static features
        self.ZH_VALUE_MAP = {
            # Gender
            '男': 'Male',
            '女': 'Female',
            # Operation Type
            '活体': 'Living Donor', 
            '劈离': 'Split Liver',
            '全肝': 'Whole Liver',
            # ABO Compatibility
            '不相容': 'Incompatible',
            '相容': 'Compatible',
            '相同': 'Identical',
        }
        
        # Lab reference ranges (based on pediatric liver transplant standards)
        # Age in years, ranges list: [(age_min, age_max, gender, low, high), ...]
        # gender: 'M'=male, 'F'=female, None=both
        # Age ranges: 0-0.5 (28天-6月), 0.5-1, 1-3, 3-6, 6-12, 12-18, 18+
        self.LAB_META = {            
            'WBC': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 4.3, 14.2), (0.5, 1, None, 4.8, 14.6),
                    (1, 2, None, 5.1, 14.1), (2, 6, None, 4.4, 11.9),
                    (6, 13, None, 4.3, 11.3), (13, 99, None, 4.1, 11.0)
                ],
                "LONIC": "Leukocytes [#/volume] in Blood by Automated count"
            },
            'N_Percent': {
                'unit': '%',
                'ranges': [
                    (0, 0.5, None, 7, 56), (0.5, 1, None, 9, 57),
                    (1, 2, None, 13, 55), (2, 6, None, 22, 65),
                    (6, 13, None, 31, 70), (13, 99, None, 37, 77)
                ],
                "LONIC": "Neutrophils/Leukocytes in Blood by Automated count"
            },
            'Lymphocyte_Abs': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 2.4, 9.5), (0.5, 1, None, 2.5, 9.0),
                    (1, 2, None, 2.4, 8.7), (2, 6, None, 1.8, 6.3),
                    (6, 13, None, 1.5, 4.6), (13, 99, None, 1.2, 3.8)
                ],
                "LONIC": "Lymphocytes [#/volume] in Blood by Automated count"
            },
            'Eosinophil_Percent': {'unit': '%',
                                   'ranges': [(0, 2, None, 0.0, 0.1), (2, 99, None, 0.0, 0.07)],
                                   "LONIC": "Eosinophils/Leukocytes in Blood by Automated count"},
            'HB': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 97, 183), (0.5, 1, None, 97, 141),
                    (1, 2, None, 107, 141), (2, 6, None, 112, 149),
                    (6, 13, None, 118, 156),
                    (13, 99, 'M', 129, 172), (13, 99, 'F', 114, 154)
                ],
                "LONIC": "Hemoglobin [Mass/volume] in Central venous blood by calculation"
            },
            'PLT': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 183, 614), (0.5, 1, None, 190, 445),
                    (1, 2, None, 190, 472), (2, 6, None, 188, 472),
                    (6, 13, None, 167, 453), (13, 99, None, 150, 407),
                ],
                "LONIC": "Platelets [#/volume] in Blood by Automated count"
            },
            'TP': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 49, 71), (0.5, 1, None, 55, 75),
                    (1, 2, None, 58, 76), (2, 6, None, 61, 79),
                    (6, 13, None, 65, 84), (13, 99, None, 68, 88)
                ],
                "LONIC": "Protein [Mass/volume] in Serum or Plasma"
            },
            'ALB': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 35, 50), (0.5, 13, None, 39, 54),
                    (13, 99, None, 42, 56),
                ],
                "LONIC": "Albumin in serum - albumin in pleural fluid [Mass concentration difference]"
            },
            'ALT': {
                'unit': 'U/L',
                'ranges': [
                    (0, 1, None, 8, 71), (1, 2, None, 8, 42),
                    (2, 13, None, 7, 30),
                    (13, 99, 'M', 7, 43), (13, 99, 'F', 6, 29)
                ],
                "LONIC": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma"
            },
            'AST': {
                'unit': 'U/L',
                'ranges': [
                    (0, 1, None, 21, 80), (1, 2, None, 22, 59),
                    (2, 13, None, 14, 44),
                    (13, 99, 'M', 12, 37), (13, 99, 'F', 10, 31)
                ],
                "LONIC": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma"
            },
            'ALP': {
                'unit': 'U/L',
                'ranges': [
                    (0, 0.5, None, 98, 532), (0.5, 1, None, 106, 420),
                    (1, 2, None, 128, 432), (2, 9, None, 143, 406),
                    (9, 12, None, 146, 500),
                    (12, 14, 'M', 160, 610), (12, 14, 'F', 81, 454),
                    (14, 15, 'M', 82, 603), (14, 15, 'F', 63, 327),
                    (15, 17, 'M', 64, 443), (15, 17, 'F', 52, 215),
                    (17, 99, 'M', 51, 202), (17, 99, 'F', 43, 130),
                ],
                "LONIC": "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma"
            },
            'GGT': {
                'unit': 'U/L',
                'ranges': [
                    (0, 0.5, None, 9, 150), (0.5, 1, None, 6, 31),
                    (1, 13, None, 5, 19), (13, 99, 'M', 8, 40), (13, 99, 'F', 6, 26)
                ],
                "LONIC": "Gamma glutamyl transferase [Enzymatic activity/volume] in Serum or Plasma"
            },
            'DB': {'unit': 'μmol/L',
                   'ranges': [(0, 99, None, 0, 6.84)],
                   "LONIC": "Bilirubin.direct [Moles/volume] in Serum or Plasma"},
            'TB': {
                'unit': 'μmol/L',
                'ranges': [(0, 99, None, 0, 23)],
                "LONIC": "Bilirubin.total [Moles/volume] in Serum or Plasma"
            },
            'Bile_Acid': {'unit': 'μmol/L',
                          'ranges': [(0, 99, None, 0.01, 10)],
                          "LONIC": "Bile acid [Moles/volume] in Serum or Plasma"},
            'CR': {
                'unit': 'μmol/L',
                'ranges': [
                    (0, 2, None, 13, 33), (2, 6, None, 19, 44),
                    (6, 13, None, 27, 66),
                    (13, 16, 'M', 37, 93), (13, 16, 'F', 33, 75),
                    (16, 99, 'M', 52, 101), (16, 99, 'F', 39, 76),
                ],
                "LONIC": "Creatinine [Moles/volume] in Blood"
            },
            'Glucose': {
                'unit': 'mmol/L',
                'ranges': [(0, 99, None, 3.9, 6.1)],
                "LONIC": "Glucose [Moles/volume] in Capillary blood by Glucometer"
            },
            'Triglyceride': {'unit': 'mmol/L',
                             'ranges': [(0, 99, None, 0, 1.7)],
                             'severity_bands': [
                                {'label': 'Normal', 'range': (0, 99, None, 0, 1.7)},
                                {'label': 'Elevated', 'range': (0, 99, None, 1.7, 2.3)},
                                {'label': 'Very High', 'range': (0, 99, None, 2.3, float('inf'))},
                            ],
                             "LONIC": "Triglyceride [Moles/volume] in Serum or Plasma"},
            'Cholesterol': {'unit': 'mmol/L',
                            'ranges': [(0, 99, None, 0, 5.2)],
                            'severity_bands': [
                                {'label': 'Normal', 'range': (0, 99, None, 0, 5.2)},
                                {'label': 'Elevated', 'range': (0, 99, None, 5.2, 6.2)},
                                {'label': 'Very High', 'range': (0, 99, None, 6.2, float('inf'))},
                            ],
                            "LONIC": "Cholesterol [Moles/volume] in Serum or Plasma"},
            'Uric_Acid': {
                'unit': 'μmol/L',
                'ranges': [
                    (0, 99, None, 155, 428),
                ],
                "LONIC": "Urate [Mass/volume] in Serum or Plasma"
            },
            'PT': {'unit': 's',
                   'ranges': [(0, 99, None, 9.4, 12.5)],
                   "LONIC": "Prothrombin time (PT) in Blood by Coagulation assay"},
            'INR': {'unit': '',
                    'ranges': [(0, 99, None, 0.8, 1.15)],
                    "LONIC": "INR in Blood by Coagulation assay"},
            'Blood_Ammonia': {'unit': 'μmol/L',
                              'ranges': [(0, 99, None, 9, 30)],
                              "LONIC": "Ammonia [Moles/volume] in Plasma"},
            'Tacrolimus_Conc': {'unit': 'ng/mL',
                                'ranges': [],
                                "LONIC": "Tacrolimus [Mass/volume] in Serum or Plasma"},
            'CsA_Trough': {'unit': 'ng/mL',
                           'ranges': [],
                           "LONIC": "cycloSPORINE [Mass/volume] in Blood"},
            'CsA_Peak': {'unit': 'ng/mL',
                         'ranges': [],
                         "LONIC": "cycloSPORINE [Mass/volume] in Blood --2 hours post dose"},
            'Rapa_Conc': {'unit': 'ng/mL',
                          'ranges': [(0, 99, None, 5, 20)],
                          "LONIC": "Sirolimus [Mass/volume] in Blood"},
            'CMV_DNA': {'unit': 'copies/mL',
                        'ranges': [(0, 99, None, 0, 400)],
                        "LONIC": "Cytomegalovirus DNA [#/volume] (viral load) in Blood by NAA with probe detection"},
            'EBV_DNA': {'unit': 'copies/mL',
                        'ranges': [(0, 99, None, 0, 400)],
                        "LONIC": "Epstein Barr virus DNA [#/volume] (viral load) in Specimen by NAA with probe detection"},
            'HBV_DNA': {'unit': 'IU/mL',
                        'ranges': [(0, 99, None, 0, 20)],
                        "LONIC": "Hepatitis B virus DNA [Units/volume] (viral load) in Serum or Plasma by NAA with probe detection"},
            'HBsAg': {'unit': 'COI',
                      'ranges': [(0, 99, None, 0, 1)],
                      "LONIC": "Hepatitis B virus surface Ag [Presence] in Serum or Plasma by Immunoassay"},
            'HBsAb': {'unit': 'mIU/mL',
                      'ranges': [(0, 99, None, 0, 10)],
                      "LONIC": "Hepatitis B virus surface Ab [Units/volume] in Serum or Plasma by Immunoassay"},
            'HBeAg': {'unit': 'COI',
                      'ranges': [(0, 99, None, 0, 1)],
                      "LONIC": "Hepatitis B virus e Ag [Presence] in Serum or Plasma by Immunoassay"},
            'HBeAb': {'unit': 'COI', 'ranges': [(0, 99, None, 1, float('inf'))],
                      "LONIC": "Hepatitis B virus e Ab [Presence] in Serum or Plasma by Immunoassay"},
            'HBcAb': {'unit': 'COI', 'ranges': [(0, 99, None, 1, float('inf'))],
                      "LONIC": "Hepatitis B virus core Ab [Presence] in Serum or Plasma by Immunoassay"},
        }

    def _get_reference_range(self, lab_item, age_years, gender=None, postop_day=None, first_drug_days=None):
        """
        Get reference range for a lab item based on age and gender.
        
        Args:
            lab_item: Lab item name (English)
            age_years: Patient age in years (float)
            gender: 'M' for male, 'F' for female, or None
        
        Returns:
            (low, high, unit, range_str) or (None, None, unit, '-') if not found
        """
        meta = self.LAB_META[lab_item]
        unit = meta['unit']

        if lab_item in self.DRUG_CONC_RANGES:
            return self._get_drug_concentration_reference_range(
                lab_item, unit, postop_day, first_drug_days
            )

        ranges = meta['ranges']
        
        # Find matching range
        for age_min, age_max, range_gender, low, high in ranges:
            if age_min <= age_years < age_max:
                # Check gender match
                if range_gender is None or range_gender == gender:
                    range_str = f"{low} - {high}" if high != float('inf') else f"> {low}"
                    return low, high, unit, range_str
        
        for age_min, age_max, range_gender, low, high in ranges:
            if age_min <= age_years < age_max and range_gender is None:
                range_str = f"{low} - {high}" if high != float('inf') else f"> {low}"
                return low, high, unit, range_str

        raise ValueError(f"No reference range for {lab_item} at age={age_years}, gender={gender}")

    def _numeric_values(self, value):
        value_str = str(value).strip()
        return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', value_str)]

    def _postop_day_value(self, value):
        value_str = str(value).strip()
        if not value_str:
            raise ValueError("Empty postoperative day value")
        match = re.search(r'-?\d+(?:\.\d+)?', value_str)
        if match is None:
            raise ValueError(f"No numeric postoperative day in value: {value}")
        return float(match.group(0))

    def _get_first_drug_days(self, df_followup):
        first_drug_days = {}
        for lab_item, med_cols in self.DRUG_CONC_MED_COLS.items():
            first_drug_day = None
            for _, row in df_followup.iterrows():
                day = self._postop_day_value(row['术后天数'])

                has_drug = any(
                    any(v > 0 for v in self._numeric_values(row[med_col]))
                    for med_col in med_cols
                    if med_col in df_followup.columns
                )
                if has_drug:
                    first_drug_day = day
                    break

            first_conc_day = None
            if lab_item in df_followup.columns:
                for _, row in df_followup.iterrows():
                    value = row[lab_item]
                    if pd.notna(value) and str(value).strip() != '':
                        first_conc_day = self._postop_day_value(row['术后天数'])
                        break

            first_days = [day for day in [first_drug_day, first_conc_day] if day is not None]
            first_drug_days[lab_item] = min(first_days) if first_days else None

        return first_drug_days

    def _get_drug_concentration_reference_range(self, lab_item, unit, postop_day, first_drug_days):
        first_day = first_drug_days.get(lab_item)
        current_day = self._postop_day_value(postop_day)
        if first_day is None:
            raise ValueError(f"No first drug day for {lab_item}")

        elapsed_days = current_day - first_day

        for day_min, day_max, low, high in self.DRUG_CONC_RANGES[lab_item]:
            if day_min <= elapsed_days < day_max:
                return low, high, unit, f"{low} - {high}"

        raise ValueError(f"No drug concentration range for {lab_item} at elapsed_days={elapsed_days}")

    def _get_severity_flag(self, lab_item, value, age_years, gender=None):
        """
        Get a finer-grained severity flag for lab items that define severity_bands.

        Returns:
            A severity label such as "Normal", "Elevated", or "Very High",
            or None if no matching severity band is defined.
        """
        meta = self.LAB_META[lab_item]
        severity_bands = meta.get('severity_bands')
        if not severity_bands:
            return None

        numeric_value = float(value)

        for band in severity_bands:
            band_range = band.get('range')
            label = band.get('label')
            if not band_range or not label or len(band_range) != 5:
                continue

            age_min, age_max, range_gender, low, high = band_range
            if not (age_min <= age_years < age_max):
                continue
            if range_gender is not None and range_gender != gender:
                continue

            # Severity bands use left-inclusive / right-exclusive intervals so
            # shared boundaries like 1.7 or 5.2 fall into the higher category.
            if high == float('inf'):
                in_range = numeric_value >= low
            else:
                in_range = low <= numeric_value < high

            if in_range:
                return label

        return None

    def _split_multivalue_parts(self, value):
        """Split a value like '100-200' or '阴性-<400' into meaningful parts."""
        value_str = str(value).strip()
        if not value_str or '-' not in value_str:
            return [value_str]

        raw_parts = [part.strip() for part in value_str.split('-') if part.strip()]
        if len(raw_parts) < 2:
            return [value_str]

        def _is_meaningful_part(part):
            if part in {"阴性", "阳性"}:
                return True
            if "<" in part or ">" in part:
                return True
            return re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', part) is not None

        if all(_is_meaningful_part(part) for part in raw_parts):
            return raw_parts
        return [value_str]

    def _describe_lab_value(self, lab_item, value, age_years, gender=None, postop_day=None, first_drug_days=None):
        """
        Build display-facing lab value information for text/table rendering.

        Returns:
            dict with keys: value, unit, flag, low_str, high_str
        """
        low_limit, high_limit, unit, _ = self._get_reference_range(
            lab_item,
            age_years,
            gender,
            postop_day=postop_day,
            first_drug_days=first_drug_days,
        )
        low_str = str(low_limit) if low_limit is not None else "-"
        high_str = str(high_limit) if high_limit is not None and high_limit != float('inf') else "-"

        raw_value = str(value).strip()
        raw_lower = raw_value.lower()
        display_value = raw_value
        display_unit = unit

        if raw_value in {"阴性"} or raw_lower in {"negative"}:
            return {
                "value": "Normal",
                "unit": "",
                "flag": "Normal",
                "low_str": "-",
                "high_str": "-",
            }

        if raw_value in {"阳性"} or raw_lower in {"positive"}:
            return {
                "value": "Abnormal",
                "unit": "",
                "flag": "Abnormal",
                "low_str": "-",
                "high_str": "-",
            }

        if "<" in raw_value:
            return {
                "value": display_value,
                "unit": display_unit,
                "flag": "Normal",
                "low_str": low_str,
                "high_str": high_str,
            }

        if ">" in raw_value:
            return {
                "value": display_value,
                "unit": display_unit,
                "flag": "Abnormal",
                "low_str": low_str,
                "high_str": high_str,
            }

        numeric_value = float(raw_value)
        severity_flag = self._get_severity_flag(
            lab_item,
            numeric_value,
            age_years,
            gender,
        )
        if severity_flag is not None:
            flag = severity_flag
        else:
            flag = "Abnormal" if (numeric_value < low_limit or numeric_value > high_limit) else "Normal"

        return {
            "value": display_value,
            "unit": display_unit,
            "flag": flag,
            "low_str": low_str,
            "high_str": high_str,
        }

    def _load_auxiliary_data(self):
        """Load labels and patient info."""
        labels_path = os.path.join(self.root_dir, 'labels.csv')
        self.labels_df = pd.read_csv(labels_path, encoding='utf-8-sig')

        # Translate Chinese column names to English
        # Column format: "{window}_{metric}" e.g. "0-30d_胆汁酸" -> "0-30d_Bile_Acid"
        new_columns = {}
        for col in self.labels_df.columns:
            if '_' in col and col != 'filename':
                parts = col.split('_', 1)
                if len(parts) == 2:
                    window, metric = parts
                    metric_en = self.ZH_TO_EN[metric]
                    new_columns[col] = f"{window}_{metric_en}"
        self.labels_df.rename(columns=new_columns, inplace=True)
        self.labels_df.set_index('filename', inplace=True)

        patient_info_path = os.path.join(self.root_dir, '患儿基本信息总表251023_含免疫事件.csv')
        self.patient_info_df = pd.read_csv(patient_info_path, encoding='utf-8-sig')
        self.patient_info_map = {}
        for _, row in self.patient_info_df.iterrows():
            key = os.path.splitext(str(row['file_name']))[0]
            self.patient_info_map[key] = row
        _rank0_print(f"Loaded patient info: {len(self.patient_info_map)} patients")

    def _build_index(self):
        """
        Build sample index.
        - One sample per (patient, prediction_point) with point-specific multi-label supervision.
        """
        _rank0_print(f"[{self.split}] Building sample index (mode=candidate_metric)...")
        
        # Determine which prediction points to use
        active_points = self.active_points
        
        samples = []
        
        for fname in tqdm(self.filenames, desc=f"[{self.split}] Indexing", disable=not _is_main_process()):
            # Get filename without extension
            fname_key = os.path.splitext(fname)[0] if fname.endswith(('.xlsx')) else fname
            
            patient_labels = self.labels_df.loc[fname_key]
            patient_info = self.patient_info_map[fname_key]
            dob = pd.to_datetime(patient_info['date_of_birth'], errors='coerce')
            if pd.isna(dob):
                continue
            
            # For each prediction point
            for point_key in active_points:
                cutoff_day, label_prefix, readable_point = self.PREDICTION_POINTS[point_key]
                sample_base = {
                    'fname': fname,
                    'fname_key': fname_key,
                    'prediction_point': point_key,
                    'cutoff_day': cutoff_day,
                    'label_prefix': label_prefix,
                }
                sample = {
                    **sample_base,
                    'metric': 'all',
                    'label_col': 'all',
                    'label_val': -100,
                }
                if not self._has_valid_followup_after_birth(sample, dob):
                    continue
                if not self._has_any_valid_multilabel_target(patient_labels, point_key):
                    continue
                samples.append(sample)
                        
        # Apply max_samples
        if self.max_samples and len(samples) > self.max_samples:
            samples = self._balanced_sample(samples, self.max_samples)
            np.random.shuffle(samples)
            samples = samples[:self.max_samples]
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(samples)
        
        # Print statistics
        self._print_stats(samples)
        
        return samples

    def _balanced_sample(self, samples, target_count):
        """Sample with task balancing."""
        from collections import defaultdict
        
        # Group by (prediction_point, metric)
        task_buckets = defaultdict(list)
        for s in samples:
            key = (s['prediction_point'], s['metric'])
            task_buckets[key].append(s)
        
        all_tasks = list(task_buckets.keys())
        sorted_tasks = sorted(all_tasks, key=lambda t: len(task_buckets[t]))
        
        final_samples = []
        remaining_quota = target_count
        remaining_tasks_count = len(all_tasks)
        
        for task in sorted_tasks:
            bucket = task_buckets[task]
            fair_share = remaining_quota // remaining_tasks_count
            take_count = min(len(bucket), fair_share)
            indices = np.random.choice(len(bucket), min(take_count, len(bucket)), replace=False)
            selected = [bucket[i] for i in indices]
            
            final_samples.extend(selected)
            remaining_quota -= len(selected)
            remaining_tasks_count -= 1
        
        return final_samples

    def _print_stats(self, samples):
        """Print dataset statistics."""
        from collections import Counter
        
        _rank0_print(f"\n[{self.split}] === Dataset Statistics ===")
        _rank0_print(f"Total samples: {len(samples)}")
        
        # By prediction point
        point_dist = Counter(s['prediction_point'] for s in samples)
        _rank0_print(f"\nBy prediction point:")
        for point, count in sorted(point_dist.items()):
            _rank0_print(f"  {point}: {count}")
        
        # By metric (top 10) - checks if 'metric' exists
        if samples and 'metric' in samples[0]:
            metric_dist = Counter(s['metric'] for s in samples)
            _rank0_print(f"\nTop 10 metrics:")
            for metric, count in metric_dist.most_common(10):
                _rank0_print(f"  {metric}: {count}")
        
        # Label distribution is only meaningful for single-label samples.
        if samples and 'label_val' in samples[0] and any(s['label_val'] in {0, 1} for s in samples):
            label_0 = sum(1 for s in samples if s['label_val'] == 0)
            label_1 = sum(1 for s in samples if s['label_val'] == 1)
            if label_0 > 0:
                _rank0_print(f"\nLabel distribution: 0={label_0}, 1={label_1}, ratio={label_1/(label_0):.2f}")
            else:
                _rank0_print(f"\nLabel distribution: 0={label_0}, 1={label_1}")

    def _load_followup_data(self, sample):
        """Load and filter follow-up data for a sample."""
        fname = sample['fname']
        cutoff_day = sample['cutoff_day']

        fpath = os.path.join(self.followup_dir, fname if fname.endswith('.csv') else f"{fname}.csv")
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        df['报告日期'] = pd.to_datetime(df['报告日期'])
        df['术后天数'] = pd.to_numeric(df['术后天数'])
        df = df.sort_values('报告日期').reset_index(drop=True)
        df_filtered = df[df['术后天数'] <= cutoff_day].copy()
        
        # Remove excluded and label columns
        cols_to_keep = []
        for c in df_filtered.columns:
            if c in self.EXCLUDED_COLS:
                continue
            if c.endswith('_label'):
                continue
            cols_to_keep.append(c)
        
        df_clean = df_filtered[cols_to_keep].copy()
        
        # Rename columns to English
        new_columns = [
            c if c in {'报告日期', '术后天数'} else self.ZH_TO_EN[c]
            for c in df_clean.columns
        ]
        df_clean.columns = new_columns
        
        return df_clean

    def _has_valid_followup_after_birth(self, sample, dob):
        cache_key = (sample['fname'], sample['cutoff_day'])
        if cache_key in self._valid_followup_cache:
            return self._valid_followup_cache[cache_key]

        df_followup = self._load_followup_data(sample)
        has_valid_followup = bool((df_followup['报告日期'] >= dob).any())
        self._valid_followup_cache[cache_key] = has_valid_followup
        return has_valid_followup

    def _has_any_valid_multilabel_target(self, patient_labels, point_key):
        _, prefix, _ = self.PREDICTION_POINTS[point_key]
        for metric in self.ALL_METRICS:
            col_name = f"{prefix}_{metric}"
            if col_name in patient_labels and pd.notna(patient_labels[col_name]):
                return True
        return False

    def _get_static_features(self, fname_key):
        """Get static patient features as a dict with readable names."""
        row = self.patient_info_map[fname_key]
        features = {}  # {readable_name: value}
        for col_name, readable_name in self.STATIC_FEATURES.items():
            val = row.get(col_name)
            if pd.notna(val):
                features[readable_name] = val
        return features

    def _build_static_table(self, features, surgery_date=None):
        """Convert static features to DataFrame table."""
        if not features:
            return None
        
        # Use provided surgery_date or NaT (not empty string)
        # Ensure it's treated as a timestamp compatible with other tables
        time_val = surgery_date if surgery_date is not None else pd.NaT
        
        data = []
        for k, v in features.items():
            # Map value if it's Chinese
            v_str = str(v).strip()
            if v_str in self.ZH_VALUE_MAP:
                v_str = self.ZH_VALUE_MAP[v_str]
            
            item_name = k
            unit = "" # Default to empty string instead of "-"
            
            # Handle special static features with units
            if 'Weight (kg)' in item_name:
                item_name = item_name.replace(' (kg)', '')
                unit = 'kg'
            elif 'Weight (g)' in item_name:
                item_name = item_name.replace(' (g)', '')
                unit = 'g'
            
            data.append({
                'Time': time_val,
                'Item': item_name,
                'Value': v_str,
                'Unit': unit,
                'Category': 'person'
            })
        return pd.DataFrame(data)

    def structed_EHR_input_process(self, static_features, df_followup, surgery_date=None, age_years=None, gender=None):
        measurement_tables = {}
        first_drug_days = self._get_first_drug_days(df_followup)

        static_table = self._build_static_table(static_features, surgery_date)
        if static_table is not None:
            measurement_tables['static'] = static_table

        if '报告日期' in df_followup.columns:
            med_cols = [c for c in df_followup.columns if c in self.med_cols_en or c == '报告日期']
            if len(med_cols) > 1:
                med_df = df_followup[med_cols].melt(id_vars=['报告日期'], var_name='Item', value_name='Value')
                med_df = med_df.dropna(subset=['Value'])
                if len(med_df) > 0:
                    med_df = med_df.rename(columns={'报告日期': 'Time'})
                    med_df['Category'] = 'drug_exposure'
                    med_df = self._expand_multivalue_rows(med_df)
                    med_df['Unit'] = med_df['Item'].apply(lambda x: self.MED_META[x]['unit'])
                    med_df['Item'] = med_df['Item'].apply(lambda x: self.MED_META[x]['str'])
                    med_df = med_df[['Time', 'Item', 'Value', 'Unit', 'Category']].sort_values('Time')
                    measurement_tables['medication'] = med_df

            lab_id_vars = ['报告日期']
            if '术后天数' in df_followup.columns:
                lab_id_vars.append('术后天数')
            lab_value_cols = [
                c for c in df_followup.columns
                if c in self.LAB_META
            ]
            if lab_value_cols:
                lab_df = df_followup[lab_id_vars + lab_value_cols].melt(id_vars=lab_id_vars, var_name='Item', value_name='Value')
                lab_df = lab_df.dropna(subset=['Value'])
                if len(lab_df) > 0:
                    lab_df = lab_df.rename(columns={'报告日期': 'Time'})
                    lab_df['Category'] = 'measurement'
                    lab_df = self._expand_multivalue_rows(lab_df)
                    desc_df = lab_df.apply(
                        lambda row: pd.Series(
                            self._describe_lab_value(
                                row['Item'],
                                row['Value'],
                                age_years,
                                gender,
                                postop_day=row['术后天数'],
                                first_drug_days=first_drug_days,
                            )
                        ),
                        axis=1,
                    )
                    lab_df['Value'] = desc_df['value']
                    lab_df['Unit'] = desc_df['unit']
                    lab_df['Item'] = lab_df['Item'].apply(lambda x: self.LAB_META[x]['LONIC'])
                    lab_df = lab_df[['Time', 'Item', 'Value', 'Unit', 'Category']].sort_values('Time')
                    measurement_tables['laboratory'] = lab_df

        dynamic_dfs = []
        if 'medication' in measurement_tables:
            dynamic_dfs.append(measurement_tables['medication'])
        if 'laboratory' in measurement_tables:
            dynamic_dfs.append(measurement_tables['laboratory'])

        if dynamic_dfs:
            final_dynamic = pd.concat(dynamic_dfs, ignore_index=True)
            final_dynamic['Time'] = pd.to_datetime(final_dynamic['Time'])
            final_dynamic = final_dynamic.sort_values('Time')
        else:
            final_dynamic = pd.DataFrame()

        all_dfs = []
        if 'static' in measurement_tables:
            all_dfs.append(measurement_tables['static'])
        if not final_dynamic.empty:
            all_dfs.append(final_dynamic)

        final_table = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if not final_table.empty and 'Time' in final_table.columns:
            sort_keys = pd.to_datetime(final_table['Time'])
            final_table = final_table.loc[sort_keys.sort_values(na_position='first').index].reset_index(drop=True)

        return final_table

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data
        fname = sample['fname']
        fname_key = sample['fname_key']
        prediction_point = sample['prediction_point']
        cutoff_day, label_prefix, readable_point = self.PREDICTION_POINTS[prediction_point]
        
        # Load follow-up data
        df_followup = self._load_followup_data(sample)
        
        # Load static features
        static_features = self._get_static_features(fname_key)
        
        # Get patient age and gender for reference range lookup
        patient_info = self.patient_info_map[fname_key]
        recipient_gender = patient_info['recipient_gender']
        gender = 'M' if str(recipient_gender).upper() in ['M', 'MALE', '男'] else 'F'
        dob = pd.to_datetime(patient_info['date_of_birth'], errors='coerce')
        df_followup = df_followup[df_followup['报告日期'] >= dob].reset_index(drop=True)

        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row['报告日期']) - pd.Timedelta(days=float(first_row['术后天数']))

        report_date = pd.to_datetime(df_followup['报告日期'].iloc[0])
        age_years = (report_date - dob).days / 365.25
        
        # PREPARE TARGETS
        labels_matrix = torch.full((len(self.active_points), len(self.ALL_METRICS)), -100, dtype=torch.float32)

        patient_labels = self.labels_df.loc[sample['fname_key']]
        target_point_idx = self.active_points.index(prediction_point)
        _, target_prefix, _ = self.PREDICTION_POINTS[prediction_point]
        for m_idx, met in enumerate(self.ALL_METRICS):
            col_name = f"{target_prefix}_{met}"
            if col_name in patient_labels and pd.notna(patient_labels[col_name]):
                labels_matrix[target_point_idx, m_idx] = float(patient_labels[col_name])

        candidate_tasks = []
        for m_idx, met in enumerate(self.ALL_METRICS):
            label_value = labels_matrix[target_point_idx, m_idx]
            if label_value == -100:
                continue
            query = (
                f"Task: predict future {met} abnormality during {label_prefix} "
                f"after liver transplantation using clinical history up to "
                f"{readable_point} post-transplant."
            )
            candidate_tasks.append(
                {
                    "query": query,
                    "candidates": ["no", "yes"],
                    "label": int(label_value.item()),
                    "point": readable_point,
                    "point_key": prediction_point,
                    "point_index": target_point_idx,
                    "metric": met,
                    "metric_index": m_idx,
                    "window": label_prefix,
                }
            )

        output_label = labels_matrix
        task_info = deepcopy(self.task_schema["candidate_metric_prediction"])
        instruction = task_info["instruction_template"].format(
            prediction_point=f"{readable_point} post-transplant",
        )
        task_info.update(
            {
                "task": "candidate_metric",
                "prediction_point": readable_point,
                "label_window": label_prefix,
                "window": label_prefix,
                "instruction": instruction,
            }
        )
        final_table = self._cached_measurement_table(
            ["candidate_metric", fname_key, prediction_point, cutoff_day],
            static_features,
            df_followup,
            surgery_date,
            age_years,
            gender,
        )
        

        output_sample = {
            "idx": idx,
            "instruction": instruction,
            "input": "",
            "output": output_label if isinstance(output_label, torch.Tensor) else str(output_label),
            "task_info": task_info,
            "measurement_table": final_table,
            "candidate_tasks": candidate_tasks,
        }
        
        output_sample["candidates"] = ["no", "yes"]

        return output_sample

    def _cached_measurement_table(
        self,
        cache_key_parts,
        static_features,
        df_followup,
        surgery_date=None,
        age_years=None,
        gender=None,
    ):
        cache_key = stable_cache_key(self.__class__.__name__, self.split, *cache_key_parts)
        return get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(
                static_features=static_features,
                df_followup=df_followup,
                surgery_date=surgery_date,
                age_years=age_years,
                gender=gender,
            ),
        )

    def _expand_multivalue_rows(self, df):
        """
        Split rows where Value contains multiple meaningful parts (e.g., '100-200' or '阴性-<400').
        Distribute them evenly across the day.
        """
        if df.empty or 'Value' not in df.columns:
            return df
            
        new_rows = []
        has_changes = False

        for idx in range(len(df)):
            row = df.iloc[idx]
            parts = self._split_multivalue_parts(row['Value'])

            if len(parts) >= 2:
                has_changes = True
                base_time = pd.to_datetime(row['Time'])
                interval_hours = 24.0 / len(parts)

                for i, part in enumerate(parts):
                    new_row_dict = row.to_dict()
                    new_row_dict['Value'] = str(part).strip()

                    if pd.notna(base_time):
                        delta = pd.Timedelta(hours=i * interval_hours)
                        new_row_dict['Time'] = base_time + delta
                    else:
                        new_row_dict['Time'] = row['Time']

                    new_rows.append(new_row_dict)
            else:
                new_rows.append(row.to_dict())
                
        if has_changes:
            return pd.DataFrame(new_rows)
        return df

    def __len__(self):
        return len(self.samples)


STAGE_SPECS = (
    {"stage_id": 0, "start_day": 0.0, "end_day": 31.0, "num_bins": 31},
    {"stage_id": 1, "start_day": 31.0, "end_day": 181.0, "num_bins": 150},
    {"stage_id": 2, "start_day": 181.0, "end_day": 366.0, "num_bins": 185},
)
MAX_SURVIVAL_BINS = 185
DEATH_SURVIVAL_HORIZON_DAYS = 1825
DEATH_STAGE_SPECS = (
    {
        "stage_id": 0,
        "start_day": 0.0,
        "end_day": float(DEATH_SURVIVAL_HORIZON_DAYS),
        "prediction_day": 0.0,
        "num_bins": DEATH_SURVIVAL_HORIZON_DAYS,
    },
)
MAX_DEATH_SURVIVAL_BINS = DEATH_SURVIVAL_HORIZON_DAYS


def load_patient_subset(patient_subset_path):
    if patient_subset_path is None:
        return None
    with open(patient_subset_path, "r", encoding="utf-8") as file:
        patient_files = json.load(file)
    if not isinstance(patient_files, list):
        raise ValueError("patient_subset_path must contain a JSON list")
    return {
        os.path.splitext(os.path.basename(str(patient_file)))[0]
        for patient_file in patient_files
    }


def build_survival_instruction(task_schema, sample):
    stage_window = (
        f"[{int(sample['stage_start_day'])}, {int(sample['stage_end_day'])}) "
        "days post-transplant"
    )
    instruction = task_schema["instruction_template"].format(
        prediction_day=f"{sample['prediction_day']:g}",
        stage_window=stage_window,
    )
    return instruction, stage_window


def build_piecewise_survival_target(
    time_to_event: float,
    event_observed: bool,
    num_bins: int,
    max_bins: int = MAX_SURVIVAL_BINS,
):
    """Encode follow-up time using one-day intervals (k, k + 1]."""
    if time_to_event < 0:
        raise ValueError("time_to_event must be non-negative")
    if not 0 < num_bins <= max_bins:
        raise ValueError("num_bins must be in [1, max_bins]")

    observed_time = min(float(time_to_event), float(num_bins))
    bin_exposure = np.zeros(max_bins, dtype=np.float32)
    event_bins = np.zeros(max_bins, dtype=np.float32)
    stage_mask = np.zeros(max_bins, dtype=np.float32)
    stage_mask[:num_bins] = 1.0

    full_bins = min(int(math.floor(observed_time)), num_bins)
    if full_bins:
        bin_exposure[:full_bins] = 1.0
    if full_bins < num_bins:
        bin_exposure[full_bins] = observed_time - full_bins

    if event_observed and 0.0 < observed_time <= num_bins:
        event_bin = min(int(math.ceil(observed_time) - 1), num_bins - 1)
        event_bins[event_bin] = 1.0

    return bin_exposure, event_bins, stage_mask


class RenjiTacrolimusSurvivalDataset(RenjiDataset):
    """One time-to-event sample per patient and postoperative stage."""

    STAGE_SPECS = STAGE_SPECS
    MAX_SURVIVAL_BINS = MAX_SURVIVAL_BINS

    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=None,
        shuffle=False,
        patient_subset_path=None,
        tte_index_dir=None,
    ):
        self.patient_subset_path = patient_subset_path
        self.patient_subset = load_patient_subset(patient_subset_path)
        self.tte_index_dir = tte_index_dir
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.target_metrics = None
        self.target_prediction_points = []
        self.active_points = []
        self.shuffle = shuffle
        self.task_schema = deepcopy(self.TASK_INFO)
        self.followup_dir = os.path.join(self.root_dir, "follow_ups")
        self.index_dir = os.path.join(self.root_dir, "index")
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")
        self._init_configs()
        self._load_auxiliary_data()
        self._valid_followup_cache = {}
        self.samples = self._build_index()

    def _index_path(self):
        index_dir = self.tte_index_dir or self.index_dir
        return os.path.join(index_dir, f"tacrolimus_tte_{self.split}.csv")

    def _load_index_rows(self):
        index_path = self._index_path()
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Tacrolimus TTE index not found: {index_path}. "
                "Run preprocess/Renji/generate_tte_task_index.py first."
            )
        _rank0_print(f"[{self.split}] Loading tacrolimus survival index: {index_path}")
        index_df = pd.read_csv(index_path)
        rows = index_df.to_dict(orient="records")
        if self.patient_subset is not None:
            rows = [
                row
                for row in rows
                if str(row.get("fname_key") or os.path.splitext(str(row.get("file_name")))[0])
                in self.patient_subset
            ]
            _rank0_print(
                f"[{self.split}] Patient subset filter: "
                f"{len(rows)}/{len(index_df)} index rows retained "
                f"from {self.patient_subset_path}"
            )
            if not rows:
                raise ValueError(
                    f"No {self.split} patients matched {self.patient_subset_path}"
                )
        return rows

    def _build_index_from_rows(self, rows):
        samples = []
        for row in rows:
            fname = str(row["file_name"])
            fname_key = str(row.get("fname_key") or os.path.splitext(fname)[0])
            if fname_key not in self.patient_info_map:
                continue
            time_to_event = float(row["time_to_event"])
            event_observed = bool(int(row["event_observed"]))
            num_bins = int(row["num_bins"])
            exposure, event_bins, stage_mask = build_piecewise_survival_target(
                time_to_event=time_to_event,
                event_observed=event_observed,
                num_bins=num_bins,
            )
            samples.append(
                {
                    "fname": fname,
                    "fname_key": fname_key,
                    "stage_id": int(row["stage_id"]),
                    "stage_start_day": float(row["stage_start_day"]),
                    "stage_end_day": float(row["stage_end_day"]),
                    "num_bins": num_bins,
                    "prediction_day": float(row["prediction_day"]),
                    "cutoff_day": float(row["cutoff_day"]),
                    "observed_day": float(row["observed_day"]),
                    "time_to_event": time_to_event,
                    "event_observed": event_observed,
                    "stage_end_horizon": float(row["stage_end_horizon"]),
                    "bin_exposure": exposure,
                    "event_bins": event_bins,
                    "stage_mask": stage_mask,
                }
            )
        return samples

    def _build_index(self):
        index_rows = self._load_index_rows()
        samples = self._build_index_from_rows(index_rows)
        if self.max_samples and len(samples) > self.max_samples:
            indices = np.random.choice(len(samples), self.max_samples, replace=False)
            samples = [samples[index] for index in indices]
        if self.shuffle:
            np.random.shuffle(samples)

        events = sum(sample["event_observed"] for sample in samples)
        _rank0_print(
            f"[{self.split}] Tacrolimus survival samples={len(samples)}, "
            f"events={events}, censored={len(samples) - events}"
        )
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df_followup = self._load_followup_data(sample)
        fname_key = sample["fname_key"]
        static_features = self._get_static_features(fname_key)
        patient_info = self.patient_info_map[fname_key]
        recipient_gender = patient_info["recipient_gender"]
        gender = "M" if str(recipient_gender).upper() in {"M", "MALE", "男"} else "F"
        dob = pd.to_datetime(patient_info["date_of_birth"], errors="coerce")
        if df_followup.empty:
            raise ValueError(f"No valid follow-up context for {fname_key}")

        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row["报告日期"]) - pd.Timedelta(
            days=float(first_row["术后天数"])
        )
        age_years = (pd.to_datetime(first_row["报告日期"]) - dob).days / 365.25
        if not np.isfinite(age_years):
            age_years = 0.0
        age_years = max(0.0, age_years)
        task_info = deepcopy(self.task_schema["tacrolimus_abnormal_survival"])
        instruction, stage_window = build_survival_instruction(
            task_info,
            sample,
        )
        task_info.update(
            {
                "task": "tacrolimus_abnormal_survival",
                "stage_id": sample["stage_id"],
                "stage_window": stage_window,
                "prediction_day": sample["prediction_day"],
                "observed_day": sample["observed_day"],
                "time_to_event": sample["time_to_event"],
                "event_observed": sample["event_observed"],
                "stage_end_horizon": sample["stage_end_horizon"],
            }
        )

        final_table = self._cached_measurement_table(
            [
                "tacrolimus_survival",
                sample["fname_key"],
                sample["stage_id"],
                sample["prediction_day"],
                sample["observed_day"],
            ],
            static_features,
            df_followup,
            surgery_date,
            age_years,
            gender,
        )
        survival_metadata = np.zeros(self.MAX_SURVIVAL_BINS, dtype=np.float32)
        survival_metadata[0] = sample["stage_end_horizon"]
        survival_metadata[1] = sample["stage_id"]
        labels = torch.tensor(
            np.stack(
                [
                    sample["bin_exposure"],
                    sample["event_bins"],
                    sample["stage_mask"],
                    survival_metadata,
                ]
            ),
            dtype=torch.float32,
        )
        return {
            "idx": idx,
            "instruction": instruction,
            "input": "",
            "output": labels,
            "stage_id": sample["stage_id"],
            "task_info": task_info,
            "measurement_table": final_table,
        }


class RenjiDeathSurvivalDataset(RenjiDataset):
    """Death time-to-event samples at fixed postoperative prediction points."""

    STAGE_SPECS = DEATH_STAGE_SPECS
    MAX_SURVIVAL_BINS = MAX_DEATH_SURVIVAL_BINS

    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=None,
        shuffle=False,
        patient_subset_path=None,
        tte_index_dir=None,
    ):
        self.patient_subset_path = patient_subset_path
        self.patient_subset = load_patient_subset(patient_subset_path)
        self.tte_index_dir = tte_index_dir
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.target_metrics = None
        self.target_prediction_points = []
        self.active_points = []
        self.shuffle = shuffle
        self.task_schema = deepcopy(self.TASK_INFO)
        self.followup_dir = os.path.join(self.root_dir, "follow_ups")
        self.index_dir = os.path.join(self.root_dir, "index")
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")
        self._init_configs()
        self._load_patient_info_data()
        self._valid_followup_cache = {}
        self.samples = self._build_index()

    def _load_patient_info_data(self):
        patient_info_path = os.path.join(
            self.root_dir,
            "患儿基本信息总表251023_含免疫事件.csv",
        )
        self.patient_info_df = pd.read_csv(patient_info_path, encoding="utf-8-sig")
        self.patient_info_map = {}
        for _, row in self.patient_info_df.iterrows():
            key = os.path.splitext(str(row["file_name"]))[0]
            self.patient_info_map[key] = row
        _rank0_print(f"Loaded patient info: {len(self.patient_info_map)} patients")

    def _index_path(self):
        index_dir = self.tte_index_dir or self.index_dir
        return os.path.join(index_dir, f"death_tte_{self.split}.csv")

    def _build_index(self):
        index_path = self._index_path()
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Death TTE index not found: {index_path}. "
                "Run preprocess/Renji/generate_tte_task_index.py first."
            )
        _rank0_print(f"[{self.split}] Loading death survival index: {index_path}")
        index_df = pd.read_csv(index_path)
        samples = []
        rows = index_df.to_dict(orient="records")
        if self.patient_subset is not None:
            rows = [
                row
                for row in rows
                if str(row.get("fname_key") or os.path.splitext(str(row.get("file_name")))[0])
                in self.patient_subset
            ]
            _rank0_print(
                f"[{self.split}] Patient subset filter: "
                f"{len(rows)}/{len(index_df)} index rows retained "
                f"from {self.patient_subset_path}"
            )
            if not rows:
                raise ValueError(
                    f"No {self.split} patients matched {self.patient_subset_path}"
                )

        for row in rows:
            fname = str(row["file_name"])
            fname_key = str(row.get("fname_key") or os.path.splitext(fname)[0])
            if fname_key not in self.patient_info_map:
                continue
            samples.append(
                {
                    "fname": fname,
                    "fname_key": fname_key,
                    "stage_id": int(row["stage_id"]),
                    "stage_start_day": float(row["stage_start_day"]),
                    "stage_end_day": float(row["stage_end_day"]),
                    "num_bins": int(row["num_bins"]),
                    "prediction_day": float(row["prediction_day"]),
                    "cutoff_day": float(row["cutoff_day"]),
                    "observed_day": float(row["observed_day"]),
                    "time_to_event": float(row["time_to_event"]),
                    "event_observed": bool(int(row["event_observed"])),
                    "stage_end_horizon": float(row["stage_end_horizon"]),
                }
            )

        if self.max_samples and len(samples) > self.max_samples:
            indices = np.random.choice(len(samples), self.max_samples, replace=False)
            samples = [samples[index] for index in indices]
        if self.shuffle:
            np.random.shuffle(samples)

        events = sum(sample["event_observed"] for sample in samples)
        _rank0_print(
            f"[{self.split}] Death survival samples={len(samples)}, events={events}, "
            f"censored={len(samples) - events}"
        )
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df_followup = self._load_followup_data(sample)
        fname_key = sample["fname_key"]
        static_features = self._get_static_features(fname_key)
        patient_info = self.patient_info_map[fname_key]
        recipient_gender = patient_info["recipient_gender"]
        gender = "M" if str(recipient_gender).upper() in {"M", "MALE", "男"} else "F"
        dob = pd.to_datetime(patient_info["date_of_birth"], errors="coerce")
        if df_followup.empty:
            raise ValueError(f"No valid follow-up context for {fname_key}")

        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row["报告日期"]) - pd.Timedelta(
            days=float(first_row["术后天数"])
        )
        age_years = (pd.to_datetime(first_row["报告日期"]) - dob).days / 365.25
        if not np.isfinite(age_years):
            age_years = 0.0
        age_years = max(0.0, age_years)
        task_info = deepcopy(self.task_schema["death_survival"])
        instruction, stage_window = build_survival_instruction(
            task_info,
            sample,
        )
        task_info.update(
            {
                "task": "death_survival",
                "stage_id": sample["stage_id"],
                "stage_window": stage_window,
                "prediction_day": sample["prediction_day"],
                "observed_day": sample["observed_day"],
                "time_to_event": sample["time_to_event"],
                "event_observed": sample["event_observed"],
                "stage_end_horizon": sample["stage_end_horizon"],
            }
        )

        final_table = self._cached_measurement_table(
            [
                "death_survival",
                sample["fname_key"],
                sample["stage_id"],
                sample["prediction_day"],
                sample["observed_day"],
            ],
            static_features,
            df_followup,
            surgery_date,
            age_years,
            gender,
        )
        return {
            "idx": idx,
            "instruction": instruction,
            "input": "",
            "stage_id": sample["stage_id"],
            "survival_target": {
                "time_to_event": sample["time_to_event"],
                "event_observed": sample["event_observed"],
                "num_bins": sample["num_bins"],
                "max_bins": self.MAX_SURVIVAL_BINS,
                "stage_end_horizon": sample["stage_end_horizon"],
                "stage_id": sample["stage_id"],
            },
            "task_info": task_info,
            "measurement_table": final_table,
        }
