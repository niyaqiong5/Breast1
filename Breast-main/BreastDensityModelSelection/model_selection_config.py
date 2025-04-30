# model_selection_config.py

import torch.optim as optim

BASE_CONFIG = {
    "data_root": "D:/Dataset_only_breast",
    "segmentation_root": "D:/Dataset_only_breast_segmentation",
    "excel_path": "D:/Desktop/breast cancer.xlsx",
    "output_dir": "D:/Desktop/model_selection_results",
    "rare_classes": [0, 1, 3],  # A, B, D
    "use_class_weights": True,
    "use_mixed_precision": True,
    "use_balanced_sampling": True,
    "generate_synthetic": True,
}

# Quick Evaluation Configuration - for quickly comparing multiple models
QUICK_EVALUATION = {
    **BASE_CONFIG,
    "models": ["LightweightBreastDensityNet", "BreastDensity3DNet"],  # Prioritize lightweight models
    "epochs": 20,
    "batch_size": 4,
    "learning_rate": 0.0005,  # A slightly higher learning rate speeds up convergence
    "early_stopping_patience": 5,
    "output_dir": "D:/Desktop/quick_model_selection",
}

# Full evaluation configuration - for detailed comparison of all models
FULL_EVALUATION = {
    **BASE_CONFIG,
    "models": [
        "BEiTWithCNNBreastDensityNet",     # 混合CNN-BEiT模型
        "BEiTBreastDensityNet",            # 改进的BEiT模型
        "BreastDensity3DNet",              # 原有模型
        "LightweightBreastDensityNet", 
        "DeepBreastDensityNet",
        "ResNetBreastDensityNet"
    ],
    "epochs": 50,
    "batch_size": 4,
    "learning_rate": 0.0001,  # 针对CNN模型的学习率
    "early_stopping_patience": 10,
    "output_dir": "D:/Desktop/full_model_selection",
}

# Production deployment configuration - Focused on balancing accuracy and speed
PRODUCTION_DEPLOYMENT = {
    **BASE_CONFIG,
    "models": ["LightweightBreastDensityNet", "ResNetBreastDensityNet", "BEiTBreastDensityNet"],  # 增加BEiT模型评估
    "epochs": 40,
    "batch_size": 8,
    "learning_rate": 0.0002,
    "early_stopping_patience": 8,
    "output_dir": "D:/Desktop/production_model_selection",
}

# Rare Class Optimization Configuration - Special focus on rare class performance
RARE_CLASS_OPTIMIZATION = {
    **BASE_CONFIG,
    "models": ["DeepBreastDensityNet", "ResNetBreastDensityNet", "BEiTBreastDensityNet"],  # 增加BEiT模型评估
    "epochs": 60,
    "batch_size": 4,
    "learning_rate": 0.00008,  # 使用更低的学习率避免过拟合
    "early_stopping_patience": 15,
    "generate_synthetic": True,  # 确保生成合成样本
    "output_dir": "D:/Desktop/rare_class_optimization",
}

# BEiT Model Optimization Configuration - Specialized optimization of BEiT model
BEIT_OPTIMIZATION = {
    **BASE_CONFIG,
    "models": ["BEiTBreastDensityNet"],
    "epochs": 45,
    "batch_size": 4,
    "learning_rate": 1e-5, 
    "early_stopping_patience": 12,
    "output_dir": "D:/Desktop/beit_optimization",
}

# improved BEiT model configuration
IMPROVED_BEIT_CONFIG = {
    **BASE_CONFIG, 
    "models": ["ImprovedBEiTBreastDensityNet"],
    "epochs": 60,
    "batch_size": 4,
    "learning_rate": 5e-6,  # 使用非常小的学习率
    "early_stopping_patience": 15,  
    "generate_synthetic": True,  
    "use_class_weights": True,  # 使用类别权重
    "output_dir": "D:/Desktop/improved_beit_results",
}

# Hybrid CNN-BEiT model configuration
HYBRID_BEIT_CONFIG = {
    **BASE_CONFIG,
    "models": ["BEiTWithCNNBreastDensityNet"],
    "epochs": 50,
    "batch_size": 8,  # Hybrid models can use larger batch sizes
    "learning_rate": 1e-5,
    "early_stopping_patience": 12,
    "output_dir": "D:/Desktop/hybrid_beit_results",
}
# Preset Optimizer Configuration
OPTIMIZER_CONFIG = {
    "BreastDensity3DNet": {
        "optimizer_class": optim.Adam,
        "optimizer_params": {"lr": 0.0001, "weight_decay": 2e-5},
        "scheduler_class": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "scheduler_params": {"T_0": 15, "T_mult": 2, "eta_min": 1e-6}
    },
    "LightweightBreastDensityNet": {
        "optimizer_class": optim.Adam,
        "optimizer_params": {"lr": 0.0002, "weight_decay": 1e-5},  
        "scheduler_class": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "scheduler_params": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6}
    },
    "DeepBreastDensityNet": {
        "optimizer_class": optim.Adam,
        "optimizer_params": {"lr": 0.00008, "weight_decay": 3e-5},  
        "scheduler_class": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "scheduler_params": {"T_0": 20, "T_mult": 2, "eta_min": 1e-6}
    },
    "ResNetBreastDensityNet": {
        "optimizer_class": optim.Adam,
        "optimizer_params": {"lr": 0.00008, "weight_decay": 3e-5},
        "scheduler_class": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "scheduler_params": {"T_0": 20, "T_mult": 2, "eta_min": 1e-6}
    },
    "ImprovedBEiTBreastDensityNet": {
        "optimizer_class": optim.AdamW,
        "optimizer_params": {
            "lr": 5e-6,  
            "weight_decay": 1e-2, 
            "betas": (0.9, 0.999),
            "eps": 1e-8
        },
        "scheduler_class": optim.lr_scheduler.OneCycleLR,  # Using the OneCycleLR scheduler
        "scheduler_params": {
            "max_lr": 1e-5,
            "pct_start": 0.3, 
            "anneal_strategy": "cos",
            "div_factor": 25.0,
            "final_div_factor": 10000.0,
            "total_steps": None 
        }
    },
    "BEiTWithCNNBreastDensityNet": {
        "optimizer_class": optim.AdamW,
        "optimizer_params": {
            "lr": 1e-5,
            "weight_decay": 5e-3, 
            "betas": (0.9, 0.999)
        },
        "scheduler_class": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "scheduler_params": {
            "T_0": 10, 
            "T_mult": 2, 
            "eta_min": 1e-7
        }
    }
}


def get_config(config_name="FULL_EVALUATION"):
    config_map = {
        "QUICK_EVALUATION": QUICK_EVALUATION,
        "FULL_EVALUATION": FULL_EVALUATION,
        "PRODUCTION_DEPLOYMENT": PRODUCTION_DEPLOYMENT,
        "RARE_CLASS_OPTIMIZATION": RARE_CLASS_OPTIMIZATION,
        "BEIT_OPTIMIZATION": BEIT_OPTIMIZATION,  # 添加新的BEiT优化配置
        "IMPROVED_BEIT_CONFIG": IMPROVED_BEIT_CONFIG,  # 添加改进的BEiT配置
        "HYBRID_BEIT_CONFIG": HYBRID_BEIT_CONFIG  # 添加混合CNN-BEiT模型配置
    }
    
    if config_name in config_map:
        return config_map[config_name]
    else:
        print(f"警告: 配置 '{config_name}' 不存在，使用默认的完整评估配置")
        return FULL_EVALUATION


def apply_config_to_args(args, config_name="FULL_EVALUATION"):
    """Apply the configuration to the args object"""
    config = get_config(config_name)
    
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    if hasattr(args, "models") and isinstance(config.get("models"), list):
        args.models = ",".join(config["models"])
    
    return args

    
