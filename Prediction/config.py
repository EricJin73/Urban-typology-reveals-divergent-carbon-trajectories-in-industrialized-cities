"""
config.py
Global configuration for hyperparameters, paths, and model settings.
"""
import os
import numpy as np
import torch

class Config:
    # --- Paths ---
    CITY_DATA_PATH = r"city.csv"
    SSP_DATA_PATH = r"ssp.csv"
    OUT_DIR_ROOT = "output"

    # --- Scope ---
    COUNTRY_CODES = [
        'AUS', 'AUT', 'BEL', 'BGR', 'BLR', 'CAN', 'CHE', 'CYP', 'CZE',
        'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV',
        'HUN', 'IRL', 'ISL', 'ITA', 'JPN', 'LIE', 'LTU', 'LUX', 'LVA',
        'MCO', 'MLT', 'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'ROU', 'RUS',
        'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'USA'
    ]
    SCENARIOS = ['BAU', 'SSP1', 'SSP5']

    # --- Temporal Settings ---
    HIST_START, HIST_END = 2001, 2020
    FUTURE_START, FUTURE_END = 2021, 2050

    # --- Features ---
    VARIABLES = [
        'gdp', 'pop', 'builtup_area', 'cropland_area', 'forest_area'
    ]

    # --- Data Loading ---
    SEQ_LEN = 12         # Lookback window
    PRED_LEN_TRAIN = 4   # Training horizon

    # --- Training ---
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 8e-4
    WEIGHT_DECAY = 1e-5
    CLIP_NORM = 1.0

    # --- Model Architecture ---
    EMB_DIM = 12
    ENC_HIDDEN = 128
    DEC_HIDDEN = 128
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # --- Uncertainty (MC Dropout) ---
    USE_MC = True
    MC_SAMPLES = 20
    MC_Q_LO = 0.05
    MC_Q_HI = 0.95

    # --- System ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 0

    @classmethod
    def init_dirs(cls):
        """Create output directory structure."""
        cls.OUT_DIR_CITY = os.path.join(cls.OUT_DIR_ROOT, "city")
        cls.OUT_DIR_SUMMARY = os.path.join(cls.OUT_DIR_ROOT, "summary")
        cls.OUT_DIR_LOSS = os.path.join(cls.OUT_DIR_ROOT, "loss")
        
        os.makedirs(cls.OUT_DIR_CITY, exist_ok=True)
        os.makedirs(cls.OUT_DIR_SUMMARY, exist_ok=True)
        os.makedirs(cls.OUT_DIR_LOSS, exist_ok=True)

    @classmethod
    def set_seed(cls):
        """Reproducibility setup."""
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
