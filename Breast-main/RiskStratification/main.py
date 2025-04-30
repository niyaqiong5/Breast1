"""
Breast cancer risk stratification model 
Realize the training and evaluation of multimodal deep learning architecture
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical

def set_seeds(seed=42):
    """Set all random seeds to ensure reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("breast_cancer_risk.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from data_loader import ClinicalDataLoader, DicomLoader, SegmentationLoader
from clinical_features import ClinicalFeatureExtractor
from image_features import ImprovedImageProcessor
from feature_selector import FeatureSelector
from dl_model import ImprovedDLModel
from ml_model import MLModel
from ensemble_model import EnsembleModel
from visualization import VisualizationTools

class Configuration:
    """Configuration class, centralized management of various parameters"""
    
    def __init__(self):
        from config import DATA_PATHS, OUTPUT_DIR, MODEL_PARAMS, FEATURE_EXTRACTION, TRAIN_TEST_SPLIT
    
        self.DATA_PATHS = DATA_PATHS
        
        self.OUTPUT_DIR = OUTPUT_DIR
        
        self.FEATURE_EXTRACTION = {
            'output_dir': os.path.join(self.OUTPUT_DIR, 'processed_images'),
            'target_size': (224, 224),
            'max_slices': 10,
            'save_debug_images': True,
            'normalization': 'minmax',
            'augmentation': True
        }
        # Update values ​​imported from config.py
        if FEATURE_EXTRACTION:
            self.FEATURE_EXTRACTION.update(FEATURE_EXTRACTION)
        
        # Deep Learning Model Parameters - Merge default values ​​with imported configuration
        self.DL_MODEL_PARAMS = {
            'backbone': 'efficientnet',
            'use_pretrained': True,
            'clinical_hidden_layers': [128, 64],
            'fusion_hidden_layers': [256, 128],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 100,
            'early_stopping_patience': 15
        }
        # If there is a dl_model key in MODEL_PARAMS, update the deep learning model parameters
        if MODEL_PARAMS and 'dl_model' in MODEL_PARAMS:
            self.DL_MODEL_PARAMS.update(MODEL_PARAMS['dl_model'])
        
        self.ML_MODEL_PARAMS = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        if MODEL_PARAMS and 'ml_model' in MODEL_PARAMS:
            self.ML_MODEL_PARAMS.update(MODEL_PARAMS['ml_model'])
        
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT if TRAIN_TEST_SPLIT else {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }
        
        # GPU
        self.GPU_CONFIG = {
            'use_gpu': True,
            'memory_limit': 4096  # MB
        }
        
        # Create Output Directory
        for subdir in ['models', 'features', 'results', 'plots']:
            path = os.path.join(self.OUTPUT_DIR, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def configure_gpu(self):
        """Configure GPU usage"""
        if self.GPU_CONFIG['use_gpu']:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for all GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit if specified
                    if self.GPU_CONFIG['memory_limit'] > 0:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=self.GPU_CONFIG['memory_limit']
                            )]
                        )
                    logger.info(f"Using GPU: {len(gpus)} available devices")
                except RuntimeError as e:
                    logger.error(f"GPU configuration failed: {e}")
            else:
                logger.warning("No GPU detected, using CPU")
        else:
            # Force CPU usage
            tf.config.experimental.set_visible_devices([], 'GPU')
            logger.info("GPU disabled, using CPU")


def load_and_process_data(config):
    """
    Load and process data
    """
    clinical_loader = ClinicalDataLoader(config.DATA_PATHS['excel_path'])
    clinical_df = clinical_loader.load_data()
    clinical_summary = clinical_loader.get_data_summary()
    
    # Extract clinical features
    clinical_extractor = ClinicalFeatureExtractor()
    clinical_features_df = clinical_extractor.process_batch(clinical_df)
    
    # Preservation of clinical features
    clinical_features_path = os.path.join(config.OUTPUT_DIR, 'features', 'clinical_features.csv')
    clinical_features_df.to_csv(clinical_features_path, index=False)
    
    # Loading DICOM and segmentation data
    dicom_loader = DicomLoader(config.DATA_PATHS['dicom_root_dir'])
    segmentation_loader = SegmentationLoader(config.DATA_PATHS['segmentation_root_dir'])
    
    # Creating an Image Processor
    image_processor = ImprovedImageProcessor(config.FEATURE_EXTRACTION)
    
    # Collect data for each patient
    patient_data = {}
    
    for pid in tqdm(clinical_df['PID'].unique(), desc="Loading patient data"):
        dicom_data = dicom_loader.load_patient_dicom(pid)
        segmentation_data = segmentation_loader.load_patient_segmentation_with_glandular(pid)
        
        if dicom_data is not None and segmentation_data is not None:
            patient_data[pid] = (dicom_data, segmentation_data)
    
    # Batch processing of image data
    processed_image_data = image_processor.batch_process(patient_data)
    
    # Save the processed image data
    processed_data_dir = image_processor.save_processed_data(processed_image_data)
    
    # Extract all 2D slices for training
    all_slices, all_masks, slice_to_patient = image_processor.extract_2d_slices(processed_image_data)
    
    # Linking it to clinical data
    image_features = {}
    for i, pid in enumerate(slice_to_patient):
        if pid not in image_features:
            image_features[pid] = []
        image_features[pid].append(all_slices[i])
    
    # Building a multimodal dataset
    multimodal_data = {
        'clinical_features': clinical_features_df,
        'image_features': image_features,
        'all_slices': all_slices,
        'all_masks': all_masks,
        'slice_to_patient': slice_to_patient,
        'patient_data': patient_data
    }
    
    return multimodal_data

def prepare_train_test_data_all_slices(multimodal_data, config):
    """
    Prepare training and test data, using all slices
    """ 
    clinical_df = multimodal_data['clinical_features']
    image_features = multimodal_data['image_features']
    
    # Get the pid with complete data
    common_pids = list(set(clinical_df['PID']).intersection(set(image_features.keys())))
    
    # Filter patients with complete data only
    filtered_clinical_df = clinical_df[clinical_df['PID'].isin(common_pids)]

    y = filtered_clinical_df['risk_numeric'].values
    
    # Split into training, validation, and test sets
    test_size = config.TRAIN_TEST_SPLIT['test_size']
    val_size = config.TRAIN_TEST_SPLIT['val_size'] / (1 - test_size)
    
    # First divide into training + validation and testing
    train_val_pids, test_pids, _, _ = train_test_split(
        common_pids, y, 
        test_size=test_size, 
        random_state=config.TRAIN_TEST_SPLIT['random_state'],
        stratify=y
    )
    
    # Re-dividing training and validation
    train_pids, val_pids = train_test_split(
        train_val_pids,
        test_size=val_size,
        random_state=config.TRAIN_TEST_SPLIT['random_state']
    )
    
    logger.info(f"训练集: {len(train_pids)}个患者")
    logger.info(f"验证集: {len(val_pids)}个患者")
    logger.info(f"测试集: {len(test_pids)}个患者")
    
    # Split the dataset by ID
    def filter_by_pids(df, pids):
        return df[df['PID'].isin(pids)]
    
    # Clinical Data
    train_clinical = filter_by_pids(clinical_df, train_pids)
    val_clinical = filter_by_pids(clinical_df, val_pids)
    test_clinical = filter_by_pids(clinical_df, test_pids)
    
    # Extract relevant columns as features
    clinical_feature_cols = [col for col in train_clinical.columns 
                           if col not in ['PID', 'label', 'risk', 'risk_numeric']]
    
    # Image data
    train_images = {pid: image_features[pid] for pid in train_pids if pid in image_features}
    val_images = {pid: image_features[pid] for pid in val_pids if pid in image_features}
    test_images = {pid: image_features[pid] for pid in test_pids if pid in image_features}
    
    train_labels = train_clinical['risk_numeric'].values
    val_labels = val_clinical['risk_numeric'].values
    test_labels = test_clinical['risk_numeric'].values

    def get_all_slice_data(image_dict, clinical_df):
        all_images = []
        all_pids = []
        all_slice_indices = []
        
        for pid, img_list in image_dict.items():
            if img_list:
                # Add all images
                for slice_idx, img in enumerate(img_list):
                    all_images.append(img)
                    all_pids.append(pid)
                    all_slice_indices.append(slice_idx)

        all_clinical = []
        all_labels = []
        
        for pid in all_pids:
            patient_data = clinical_df[clinical_df['PID'] == pid].iloc[0]
            all_clinical.append(patient_data[clinical_feature_cols].values)
            all_labels.append(patient_data['risk_numeric'])
        
        return np.array(all_images), np.array(all_clinical), np.array(all_labels), all_pids, all_slice_indices
    
    # Get all slice data
    train_images_array, train_clinical_array, train_labels_array, train_image_pids, train_slice_indices = get_all_slice_data(train_images, train_clinical)
    val_images_array, val_clinical_array, val_labels_array, val_image_pids, val_slice_indices = get_all_slice_data(val_images, val_clinical)
    test_images_array, test_clinical_array, test_labels_array, test_image_pids, test_slice_indices = get_all_slice_data(test_images, test_clinical)
    
    # Feature selection for clinical features
    feature_selector = FeatureSelector(os.path.join(config.OUTPUT_DIR, 'features'))
    
    # Use training set data for feature selection (each patient’s clinical data is used only once)
    selected_features = feature_selector.ensemble_feature_selection(
        pd.DataFrame(train_clinical[clinical_feature_cols]), 
        train_labels, 
        k=min(20, len(clinical_feature_cols))
    )
    
    # Get the selected feature index
    selected_indices = [clinical_feature_cols.index(feature) for feature in selected_features]
    
    # Apply feature selection to all slices of clinical data
    train_clinical_selected = train_clinical_array[:, selected_indices]
    val_clinical_selected = val_clinical_array[:, selected_indices]
    test_clinical_selected = test_clinical_array[:, selected_indices]

    dataset = {
        'train': {
            'clinical': train_clinical_selected,
            'images': train_images_array,
            'labels': train_labels_array,
            'pids': train_image_pids,
            'slice_indices': train_slice_indices
        },
        'val': {
            'clinical': val_clinical_selected,
            'images': val_images_array,
            'labels': val_labels_array,
            'pids': val_image_pids,
            'slice_indices': val_slice_indices
        },
        'test': {
            'clinical': test_clinical_selected,
            'images': test_images_array,
            'labels': test_labels_array,
            'pids': test_image_pids,
            'slice_indices': test_slice_indices
        },
        'feature_cols': selected_features,
        'all_feature_cols': clinical_feature_cols
    }
    
    return dataset, feature_selector

def train_multimodal_model(dataset, config):

    model_dir = os.path.join(config.OUTPUT_DIR, 'models', 'multimodal_dl')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # init
    multimodal_model = ImprovedDLModel(
        config.DL_MODEL_PARAMS,
        model_dir
    )
    
    # Get the dataset
    train_data = dataset['train']
    val_data = dataset['val']

    #train model
    history = multimodal_model.train(
        train_data['clinical'], 
        train_data['images'], 
        train_data['labels'],
        val_data['clinical'],
        val_data['images'],
        val_data['labels']
    )
    
    # evaluate
    test_data = dataset['test']
    metrics = multimodal_model.evaluate(
        test_data['clinical'],
        test_data['images'],
        test_data['labels'],
        test_data['pids']
    )

    return multimodal_model

def analyze_class_distribution(dataset):
    """Analyze the distribution of categories in the dataset"""
    train_labels = dataset['train']['labels']
    val_labels = dataset['val']['labels']
    test_labels = dataset['test']['labels']
    
    # Get all unique categories
    all_unique_labels = set(np.unique(train_labels))
    all_unique_labels.update(np.unique(val_labels))
    all_unique_labels.update(np.unique(test_labels))
    all_unique_labels = sorted(list(all_unique_labels))
    
    # Training set
    train_counts = np.bincount(train_labels.astype(int), minlength=len(all_unique_labels))
    train_percentages = train_counts / len(train_labels) * 100
    
    # Validation set
    val_counts = np.bincount(val_labels.astype(int), minlength=len(all_unique_labels))
    val_percentages = val_counts / len(val_labels) * 100
    
    # Test Set
    test_counts = np.bincount(test_labels.astype(int), minlength=len(all_unique_labels))
    test_percentages = test_counts / len(test_labels) * 100
    
    # Draw a distribution map
    plt.figure(figsize=(10, 6))
    labels = ['Low risk', 'Medium risk', 'High risk'][:len(all_unique_labels)]
    x = np.arange(len(labels))
    width = 0.25
    
    # Make sure all percentage arrays have the same length
    plt.bar(x - width, train_percentages[:len(labels)], width, label='Training set')
    plt.bar(x, val_percentages[:len(labels)], width, label='Validation set')
    plt.bar(x + width, test_percentages[:len(labels)], width, label='Test Set')
    
    plt.ylabel('percentage')
    plt.title('Class distribution')
    plt.xticks(x, labels)
    plt.legend()
    
    from config import OUTPUT_DIR
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.close()


def evaluate_patient_level(model, dataset, config):
    """
 Evaluating model performance at the patient level
    """
    
    # Preparing test data
    test_data = dataset['test']
    test_pids = test_data['pids']
    
    # Get a unique pid
    unique_pids = list(set(test_pids))
 
    patient_predictions = {}
    patient_true_labels = {}
    
    # Predict each slice
    y_pred_proba = model.predict_proba(test_data['clinical'], test_data['images'])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Group prediction results by patient ID
    for i, pid in enumerate(test_pids):
        if pid not in patient_predictions:
            patient_predictions[pid] = []
            patient_true_labels[pid] = test_data['labels'][i] 
        
        patient_predictions[pid].append(y_pred_proba[i])
    
    # Final prediction at patient level (mean probability)
    patient_final_pred = {}
    patient_final_proba = {}
    for pid, preds in patient_predictions.items():
    # Average the predicted probabilities for all slices
        avg_proba = np.mean(preds, axis=0)
        final_pred = np.argmax(avg_proba)
        patient_final_pred[pid] = final_pred
        patient_final_proba[pid] = avg_proba
    
    # Prepare for Assessment
    y_true_patient = [patient_true_labels[pid] for pid in unique_pids]
    y_pred_patient = [patient_final_pred[pid] for pid in unique_pids]
    y_proba_patient = np.array([patient_final_proba[pid] for pid in unique_pids])
    
    # Calculating evaluation metrics
    accuracy = np.mean(np.array(y_true_patient) == np.array(y_pred_patient))

    report = classification_report(y_true_patient, y_pred_patient, output_dict=True)
    
    cm = confusion_matrix(y_true_patient, y_pred_patient)
    
    logger.info(f"Patient-level accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['low', 'medium', 'high'],
               yticklabels=['low', 'medium', 'high'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True label')
    plt.title('Patient-level confusion matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'plots', 'patient_level_confusion_matrix.png'))
    
    # Calculate ROC curve and AUC
    n_classes = len(np.unique(y_true_patient))
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(y_true_patient) == i).astype(int), y_proba_patient[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color, label in zip(range(n_classes), colors, ['low', 'medium', 'high']):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{label} (AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-level ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'plots', 'patient_level_roc_curve.png'))
 
    macro_roc_auc = np.mean(list(roc_auc.values()))
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'auc': roc_auc,
        'macro_auc': macro_roc_auc
    }





def main():
    # Initialize configuration
    config = Configuration()
    
    # Update configuration
    if args.data_path:
        config.DATA_PATHS = {
            'excel_path': os.path.join(args.data_path, 'clinical_data.xlsx'),
            'dicom_root_dir': os.path.join(args.data_path, 'dicom'),
            'segmentation_root_dir': os.path.join(args.data_path, 'segmentation')
        }
    
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        # Update output subdirectories
        for subdir in ['models', 'features', 'results', 'plots']:
            path = os.path.join(config.OUTPUT_DIR, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    # Update image processing configuration
    config.FEATURE_EXTRACTION['max_slices'] = None if args.use_all_slices else config.FEATURE_EXTRACTION.get('max_slices', 10)
    config.FEATURE_EXTRACTION['focus_on_glandular'] = args.focus_glandular
    
    # Configure GPU
    config.GPU_CONFIG['use_gpu'] = args.use_gpu
    config.GPU_CONFIG['memory_limit'] = args.gpu_memory_limit
    config.configure_gpu()
    
    set_seeds(42)

    # Execute operations based on mode
    if args.mode == 'train':
        # Training mode
        logger.info("=== Running mode: Train ===")
        
        # Load and process data
        multimodal_data = load_and_process_data(config)
        
        # Prepare training and test data
        dataset, feature_selector = prepare_train_test_data_all_slices(multimodal_data, config)

        analyze_class_distribution(dataset)
        
        # Train multimodal deep learning model
        multimodal_model = train_multimodal_model(dataset, config)
        
        # Evaluate patient-level performance and output key metrics
        patient_metrics = evaluate_patient_level(multimodal_model, dataset, config)
        
        logger.info("============== Deep Learning Model Results ==============")
        logger.info(f"Accuracy on slice level: {multimodal_model.evaluate(dataset['test']['clinical'], dataset['test']['images'], dataset['test']['labels'])['accuracy']:.4f}")
        logger.info(f"Accuracy on patient level: {patient_metrics['accuracy']:.4f}")
        logger.info(f"宏平均AUC on patient level: {patient_metrics.get('macro_auc', 'N/A'):.4f}")
        
    elif args.mode == 'compare':
        # Comparative mode
        logger.info("=== Running mode: Compare ===")
        
        # Load and process data
        multimodal_data = load_and_process_data(config)
        
        # Run comparative experiments
        experiment_results = run_comparative_experiments(multimodal_data, config)
        
        # Find best model based on accuracy
        best_experiment = max(experiment_results.keys(), 
                             key=lambda exp: experiment_results[exp]['metrics']['accuracy'])
        
        logger.info(f"Comparative experiments complete. Best model: {best_experiment}")
        logger.info(f"Best model accuracy: {experiment_results[best_experiment]['metrics']['accuracy']:.4f}")
        
    elif args.mode == 'evaluate':
        # Evaluation mode
        logger.info("=== Running mode: Evaluate ===")
        
        if not args.model_path:
            logger.error("Evaluation mode requires --model-path")
            return 1
        
        # Load and process data
        multimodal_data = load_and_process_data(config)
        
        # Prepare evaluation data
        dataset, _ = prepare_train_test_data_all_slices(multimodal_data, config)
        
        # Load model
        model_dir = args.model_path
        multimodal_model = ImprovedDLModel.load_model(
            os.path.join(model_dir, 'best_model.h5'),
            model_dir
        )
        
        # Evaluate model
        metrics = multimodal_model.evaluate(
            dataset['test']['clinical'],
            dataset['test']['images'],
            dataset['test']['labels']
        )
        
        # 评估患者级别性能
        patient_metrics = evaluate_patient_level(multimodal_model, dataset, config)
        
        logger.info("============== Deep learning model evaluation results ==============")
        logger.info(f"Slice level accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Patient-level accuracy: {patient_metrics['accuracy']:.4f}")
        logger.info(f"Patient-level macro-average AUC: {patient_metrics.get('macro_auc', 'N/A'):.4f}")
        
    elif args.mode == 'predict':
        # Prediction mode
        logger.info("=== Running mode: Predict ===")
        
        if not args.model_path:
            logger.error("Prediction mode requires --model-path")
            return 1
        
        # Load and process data
        multimodal_data = load_and_process_data(config)
        
        # Prepare prediction data
        dataset, _ = prepare_train_test_data_all_slices(multimodal_data, config)
        
        # Load model
        model_dir = args.model_path
        multimodal_model = ImprovedDLModel.load_model(
            os.path.join(model_dir, 'best_model.h5'),
            model_dir
        )
        
        # Make predictions
        test_data = dataset['test']
        y_pred = multimodal_model.predict(test_data['clinical'], test_data['images'])
        y_pred_proba = multimodal_model.predict_proba(test_data['clinical'], test_data['images'])
        
        # Create prediction results dataframe
        results_df = pd.DataFrame({
            'PID': test_data['pids'],
            'True_Risk': test_data['labels'],
            'Predicted_Risk': y_pred
        })
        
        # Add probability columns
        for i in range(y_pred_proba.shape[1]):
            results_df[f'Prob_Class_{i}'] = y_pred_proba[:, i]
        
        # Save predictions
        results_path = os.path.join(config.OUTPUT_DIR, 'results', 'predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Predictions saved to {results_path}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
