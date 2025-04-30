import os
import argparse
from pathlib import Path
from improved_bd_pipeline_model_selection import BreastDensityPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Breast Density Classification Pipeline')

    parser.add_argument('--data_root', type=str, default="D:/Dataset_only_breast")
    parser.add_argument('--segmentation_root', type=str, default="D:/Dataset_only_breast_segmentation")
    parser.add_argument('--excel_path', type=str, default="D:/Desktop/breast cancer.xlsx")
    parser.add_argument('--output_dir', type=str, default="D:/Desktop/22results_improved")
    
    parser.add_argument('--mode', type=str, default="all",
                        choices=['train_dl', 'model_selection', 'predict', 'visualize', 'analysis', 'all']')
    

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_stratified', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--use_mixed_precision', action='store_true')
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True)
    parser.add_argument('--generate_synthetic', action='store_true', default=True)
    parser.add_argument('--rare_classes', type=str, default="0,1,3")
    
    parser.add_argument('--models', type=str, default=None)
    
    parser.add_argument('--predict_only', action='store_true')
    
    # Individual patient IDs (for prediction or visualization)
    parser.add_argument('--patient_id', type=str, default=None)
    
    # Model used for prediction
    parser.add_argument('--predict_model', type=str, default=None)
                        
    return parser.parse_args()

class Args:
    def __init__(self):
        self.data_root = "D:/Dataset_only_breast"
        self.segmentation_root = "D:/Dataset_only_breast_segmentation"
        self.excel_path = "D:/Desktop/breast cancer.xlsx"
        self.output_dir = "D:/Desktop/11results_improved"
        self.mode = "all"
        self.epochs = 50
        self.batch_size = 2
        self.learning_rate = 0.0001
        self.use_class_weights = True
        self.use_stratified = True
        self.early_stopping_patience = 5
        self.use_mixed_precision = True
        self.skip_feature_extraction = False
        self.predict_only = False
        self.patient_id = None
        self.use_balanced_sampling = True
        self.generate_synthetic = True
        self.rare_classes = "0,1,3"
        self.models = None
        self.predict_model = None


    def parse_args():
        return Args()

def main():
    args = Args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = BreastDensityPipeline(
        data_root=args.data_root,
        segmentation_root=args.segmentation_root,
        excel_path=args.excel_path,
        output_dir=args.output_dir
    )

    pipeline.load_patient_data()
    
    # Define rare categories - convert from string parameter to list
    isinstance(args.rare_classes, str):
    rare_classes = [int(cls.strip()) for cls in args.rare_classes.split(',') if cls.strip()]
    
    # Model selection mode
    if args.mode in ["model_selection", "all"] and not args.predict_only:
        
        # Parse the list of models to compare
        model_names = None
        if args.models:
            model_names = [name.strip() for name in args.models.split(',') if name.strip()]
            print(f"The following models will be compared: {', '.join(model_names)}")
            
        best_model_name, best_model, results = pipeline.train_with_model_selection(
            model_names=model_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_class_weights=args.use_class_weights,
            early_stopping_patience=args.early_stopping_patience,
            use_mixed_precision=args.use_mixed_precision,
            use_balanced_sampling=args.use_balanced_sampling,
            generate_synthetic=args.generate_synthetic,
            rare_classes=rare_classes
        )
        
        if best_model_name:
            print(f"\nBest model: {best_model_name} (Accuracy: {results['accuracy']:.4f})")

            if 'rare_class_metrics' in results:
                print("Rare classes indicators:")
                for cls, metrics in results['rare_class_metrics'].items():
                    print(f"  Class {cls}: F1 = {metrics.get('f1', 0):.4f}, Precision = {metrics.get('precision', 0):.4f}, Recall = {metrics.get('recall', 0):.4f}")
        else:
            print("Model selection failed and no valid results were obtained")
    
    # train
    if args.mode in ["train_dl", "all"] and not args.predict_only:
        print("\n=== Training 3D deep learning models (enhanced handling of rare categories）===")
        cv_results = pipeline.train_with_cross_validation(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_class_weights=args.use_class_weights,
            n_folds=5,  
            early_stopping_patience=args.early_stopping_patience,
            use_mixed_precision=args.use_mixed_precision,
            generate_synthetic=args.generate_synthetic,
            rare_classes=rare_classes
        )

        if cv_results:
            print("\n=== Cross validation results ===")
            print(f"Best validation accuracy: {cv_results['val_acc']:.4f}")
            print("Accuracy of each class:")
            for cls, acc in cv_results['class_acc'].items():
                cls_name = ["A", "B", "C", "D"][int(cls)]
                print(f"  Class {cls_name} (Class {cls}): {acc:.4f}")
            
            print(f"Average cross validation accuracy: {cv_results['cv_results']['avg_acc']:.4f} ± {cv_results['cv_results']['std_acc']:.4f}")
            print(f"Accuracy of each fold: {', '.join([f'{acc:.4f}' for acc in cv_results['cv_results']['fold_accs']])}")
        else:
            print("Cross-validation training failed and no valid results were obtained")
    
    # Predictions for a single patient
    if args.mode == "predict":
        print(f"\n=== Predicting patients {args.patient_id} ===")
        dl_result = pipeline.predict_with_selected_model(args.patient_id, model_name=args.predict_model)

        print("\nPrediction results:")
        if dl_result:
            model_name = dl_result.get('model_name', '3D Deep Learning Models')
            print(f"Model used: {model_name}")
            print(f"Prediction Class: {dl_result['predicted_class']} (Confidence: {max(dl_result['class_probabilities'].values()):.4f})")
            print(f"Probability of each class: {dl_result['class_probabilities']}")
            print(f"Density ratio: {dl_result['stats']['density_ratio']:.4f}")
            
            # Check if it is a rare category
            density_class = dl_result['predicted_class']
            density_encoded = None
            for i, cls in enumerate(['A', 'B', 'C', 'D']):
                if cls == density_class:
                    density_encoded = i
                    break
    
    # Predict all unlabeled patients
    if args.mode in ["predict", "all"]:
        print("\n=== Predict all unlabeled patients ===")
        results_df = pipeline.predict_all_unlabeled_with_3d()
        if results_df is not None:
            # Calculate the predicted proportion of rare classes
            if 'predicted_density' in results_df.columns:
                rare_count = results_df['predicted_density'].isin(['A','B', 'D']).sum()
                rare_percentage = (rare_count / len(results_df)) * 100
                print(f"Successfully predicted breast density for {len(results_df)} patients")
                print(f"{rare_count} of these patients ({rare_percentage:.1f}%) were predicted to be of the rare class (A, B, or D)")
    
    # Feature analysis
    if args.mode in ["analysis", "all"]:
        print("\nAnalysis of the relationship between features and breast density (with a special focus on rare categories)")
        pipeline.visual_feature_analysis(rare_classes=rare_classes)
    
    # Visualizing a single patient
    if args.mode == "visualize":
        print(f"\n Visualize the patient {args.patient_id} ")
        result = pipeline.visualize_predictions(args.patient_id, rare_classes=rare_classes)
        if result:
            is_rare = result.get('is_rare_class', False)
            true_label = result.get('true_label')
            print(f"The visualization results have been saved to {args.output_dir}/patient_visualizations/")

if __name__ == "__main__":
    main()
