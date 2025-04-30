# Model selection run script 


import os
import sys
from pathlib import Path
from improved_bd_pipeline_model_selection import BreastDensityPipeline
from model_selection_config import get_config, OPTIMIZER_CONFIG

def run_model_selection(config_name="FULL_EVALUATION"):
    """
   Run model selection using the specified configuration
    """
    config = get_config(config_name)
    print(f"Use Configuration: {config_name}")
    print("\nConfiguration details:")
    for key, value in config.items():
        if key != "rare_classes":  # Rare categories have been treated separately
            print(f"  {key}: {value}")
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    pipeline = BreastDensityPipeline(
        data_root=config["data_root"],
        segmentation_root=config["segmentation_root"],
        excel_path=config["excel_path"],
        output_dir=config["output_dir"]
    )
    
    pipeline.load_patient_data()
    
    pipeline.extract_features_from_segmentation()
    
    # Get a list of models to compare
    model_names = config.get("models", None)
    
    rare_classes = config.get("rare_classes", [0, 1, 3]) # A,B,D
    print(f"The rare classes are set to: {rare_classes}")

    print("\n=== Start model selection ===")
    best_model_name, best_model, results = pipeline.train_with_model_selection(
        model_names=model_names,
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 0.0001),
        use_class_weights=config.get("use_class_weights", True),
        early_stopping_patience=config.get("early_stopping_patience", 5),
        use_mixed_precision=config.get("use_mixed_precision", True),
        use_balanced_sampling=config.get("use_balanced_sampling", True),
        generate_synthetic=config.get("generate_synthetic", True),
        rare_classes=rare_classes
    )

    if best_model_name:
        print(f"\nModel selection complete! Best model: {best_model_name} (Accuracy: {results['accuracy']:.4f})")
        
        # Detail result
        print("\nDetailed evaluation results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Best validation loss: {results['best_val_loss']:.4f}")
        print(f"  Training time: {results['training_time']/60:.2f} minutes")
        
        # for rare classes
        if 'rare_class_metrics' in results:
            print("\nRare classes indicators:")
            for cls, metrics in results['rare_class_metrics'].items():
                print(f"  Class {cls}:")
                print(f"    F1 score: {metrics.get('f1', 0):.4f}")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall: {metrics.get('recall', 0):.4f}")
        
        model_path = os.path.join(config["output_dir"], "best_selected_model.pth")
    
    return best_model_name, best_model, results

def main():
    default_config = "FULL_EVALUATION"
    
    # Parse command line parameters to select configuration
    if len(sys.argv) > 1:
        config_name = sys.argv[1].upper()
    else:
        config_name = default_config
    
    run_model_selection(config_name)

if __name__ == "__main__":
    main()
