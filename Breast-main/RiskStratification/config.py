"""
Breast Cancer Risk Stratification Model - Config Profile
"""

import os
from pathlib import Path

BASE_DIR = 'D:/Desktop'

DATA_PATHS = {
    'excel_path': 'D:/Desktop/breast cancer contrast.xlsx', 
    'dicom_root_dir': 'D:/Data_noncontrast',  
    'segmentation_root_dir': 'D:/Desktop/Data_noncontrast_segmentation',  
}

OUTPUT_DIR = os.path.join(BASE_DIR, "risk_output_noncontrast")

MODEL_PARAMS = {
    # Machine Learning Model Parameters
    'ml_model': {
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    },
    
    # Deep Learning Model Parameters
    'dl_model': {
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.0001,
        'dropout_rate': 0.4,
        'early_stopping_patience': 15,
        # NEW: Add fusion strategy parameters
        'fusion_strategy': 'attention',  # Options: 'concat', 'attention', 'cross_attention', 'gated'
        'backbone': 'densenet',   # Options: 'efficientnet', 'densenet', 'resnet'
    },
    # Model Ensemble Parameters
    'ensemble': {
        'ml_weight': 0.7,  # Machine Learning Model Weights
    }
}

# Feature extraction parameters
FEATURE_EXTRACTION = {
    'output_dir': 'output/processed_images',
    'target_size': (224, 224),
    # Set max_slices to None to process all slices
    'max_slices': None,
    'save_debug_images': True,
    'normalization': 'clahe',
    'augmentation': True,
    # Added a new parameter to specify whether to focus only on glandular_tissue
    'focus_on_glandular': True
}

# Data partitioning parameters
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
}

# Log Configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(OUTPUT_DIR, 'app.log'),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
