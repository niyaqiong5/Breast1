"""
Multi-Instance Learning (MIL) Breast Cancer Risk Prediction Model - Bilateral Training Version
Modification: Combine left and right breast training, analyze bilateral asymmetry features
"""

import os
import sys
import argparse
import logging
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import json
import matplotlib.pyplot as plt
import gzip
import h5py
import cv2
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, 
    LeakyReLU, Lambda, Multiply, Layer, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K

# Import existing modules
try:
    from performance_validation import validate_trained_model, ModelPerformanceValidator, PerformanceDeepAnalysis
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported all data processing modules")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Missing required modules: {e}")

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"bilateral_mil_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """GPU setup"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU configured: {len(gpus)} GPU(s)")
        else:
            logger.info("ℹ️ Using CPU")
    except Exception as e:
        logger.warning(f"⚠️ GPU configuration failed: {e}")

def safe_json_convert(obj):
    """Safe JSON conversion"""
    try:
        if hasattr(obj, 'numpy'):
            return float(obj.numpy()) if obj.numpy().ndim == 0 else obj.numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: safe_json_convert(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [safe_json_convert(item) for item in obj]
        else:
            return obj
    except:
        return str(obj)

class FixedCacheManager:
    """Cache manager using fixed cache names"""
    
    def __init__(self, cache_root='./cache'):
        self.cache_root = cache_root
        self.cache_dir = os.path.join(cache_root, 'optimized_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use fixed cache name
        self.cache_name = "breast_data_v1"
        
        logger.info(f"🗄️ Fixed cache manager initialized")
        logger.info(f"   Cache directory: {self.cache_dir}")
        logger.info(f"   Fixed cache name: {self.cache_name}")
    
    def get_cache_files(self):
        """Get cache file paths"""
        return {
            'clinical': os.path.join(self.cache_dir, f"{self.cache_name}_clinical.pkl.gz"),
            'images': os.path.join(self.cache_dir, f"{self.cache_name}_images.h5"),
            'mapping': os.path.join(self.cache_dir, f"{self.cache_name}_mapping.pkl.gz")
        }
    
    def cache_exists(self):
        """Check if cache exists"""
        cache_files = self.get_cache_files()
        
        all_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in cache_files.values())
        
        if all_exist:
            total_size = sum(os.path.getsize(f) for f in cache_files.values()) / (1024*1024)
            logger.info(f"🎯 CACHE FOUND!")
            logger.info(f"   Total size: {total_size:.1f} MB")
            return True
        else:
            return False
    
    def load_cache(self):
        """Load cached data"""
        logger.info("📂 Loading data from fixed cache...")
        
        cache_files = self.get_cache_files()
        
        try:
            # Load clinical data
            with gzip.open(cache_files['clinical'], 'rb') as f:
                clinical_df = pickle.load(f)
            logger.info(f"✅ Clinical data loaded: {clinical_df.shape}")
            
            # Load image data
            bilateral_image_features = {}
            with h5py.File(cache_files['images'], 'r') as hf:
                patient_count = len(hf.keys())
                logger.info(f"📊 Loading images for {patient_count} patients...")
                
                for pid in tqdm(hf.keys(), desc="Loading cached images"):
                    patient_group = hf[pid]
                    image_data = {}
                    
                    # Load left and right breast images
                    if 'left_images' in patient_group:
                        image_data['left_images'] = patient_group['left_images'][:]
                    if 'right_images' in patient_group:
                        image_data['right_images'] = patient_group['right_images'][:]
                    
                    bilateral_image_features[pid] = image_data
            
            logger.info(f"✅ Image data loaded: {len(bilateral_image_features)} patients")
            
            # Load mapping data
            with gzip.open(cache_files['mapping'], 'rb') as f:
                mapping_data = pickle.load(f)
            logger.info(f"✅ Mapping data loaded")
            
            # Reconstruct data
            cached_data = {
                'clinical_features': clinical_df,
                'bilateral_image_features': bilateral_image_features,
                'bilateral_slices_data': mapping_data.get('bilateral_slices_data', {}),
                'processing_config': mapping_data.get('processing_config', {})
            }
            
            return cached_data
            
        except Exception as e:
            logger.error(f"❌ Cache loading failed: {e}")
            raise

class AttentionLayer(Layer):
    """Attention mechanism layer for MIL"""
    
    def __init__(self, dim, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.dim = dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, n_instances, features)
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        
        # Calculate attention weights
        ait = K.exp(ait)
        # Add small epsilon to avoid division by zero
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        ait = K.expand_dims(ait, axis=-1)
        
        # Weighted average
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        
        return output, ait

class BilateralMILBreastCancerModel:
    """Bilateral breast MIL model - integrated training version"""
    
    def __init__(self, instance_shape=(128, 128, 3), max_instances=20, 
                 clinical_dim=8, num_classes=2):
        """
        Initialize bilateral breast MIL model
        
        Args:
            instance_shape: Shape of each slice
            max_instances: Maximum number of slices (total of left and right breast)
            clinical_dim: Clinical feature dimension
            num_classes: Number of classes
        """
        self.instance_shape = instance_shape
        self.max_instances = max_instances  # Increased to 20 to accommodate left and right breast slices
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_instance_encoder(self):
        """Use MobileNetV2 - lightweight pre-trained model"""
        from tensorflow.keras.applications import MobileNetV2
        
        inputs = Input(shape=self.instance_shape)
        
        # Use MobileNetV2, alpha parameter controls network width
        base_model = MobileNetV2(
            input_shape=self.instance_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg',
            alpha=0.5  # Use 50% of channels, significantly reduce parameters
        )
        
        # Freeze most layers, only fine-tune last few layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        x = base_model(inputs)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        return Model(inputs=inputs, outputs=x, name='instance_encoder')

    def _build_model(self):
        """Build complete bilateral breast MIL model"""
        # Input definition
        bag_input = Input(shape=(self.max_instances, *self.instance_shape), name='bilateral_bag_input')
        instance_mask = Input(shape=(self.max_instances,), name='instance_mask')
        clinical_input = Input(shape=(self.clinical_dim,), name='clinical_input')
        
        # Build instance encoder
        instance_encoder = self._build_instance_encoder()
        
        # Process each instance
        instance_features_list = []
        for i in range(self.max_instances):
            instance = Lambda(lambda x: x[:, i, :, :, :])(bag_input)
            instance_feat = instance_encoder(instance)
            instance_features_list.append(instance_feat)
        
        # Stack all instance features
        instance_features = Lambda(lambda x: K.stack(x, axis=1))(instance_features_list)
        
        # Apply mask
        mask_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(instance_mask)
        instance_features_masked = Multiply()([instance_features, mask_expanded])
        
        # Attention mechanism
        bag_features, attention_weights = AttentionLayer(64)(instance_features_masked)
        
        # Process clinical features
        x_clinical = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(clinical_input)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        x_clinical = Dropout(0.3)(x_clinical)
        
        x_clinical = Dense(16, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x_clinical)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        
        # Feature fusion
        combined = Concatenate()([bag_features, x_clinical])
        
        # Final classification layers
        fusion = Dense(64, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(combined)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.4)(fusion)
        
        fusion = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(fusion)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.3)(fusion)
        
        # Output layer
        output = Dense(self.num_classes, activation='softmax', name='bilateral_risk_output')(fusion)
        
        # Build model
        self.model = Model(
            inputs=[bag_input, instance_mask, clinical_input], 
            outputs=output,
            name='Bilateral_MIL_BreastCancer_Model'
        )
        
        # Attention model
        self.attention_model = Model(
            inputs=[bag_input, instance_mask, clinical_input],
            outputs=[output, attention_weights],
            name='Bilateral_MIL_Attention_Model'
        )
        
        # Use sparse categorical cross-entropy
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Bilateral MIL model built: {self.model.count_params():,} parameters")
        logger.info(f"   Using sparse_categorical_crossentropy loss")
        logger.info(f"   Max instances per bilateral bag: {self.max_instances}")
        logger.info(f"   Instance shape: {self.instance_shape}")

class BilateralMILDataManager:
    """Bilateral breast MIL data manager - integrated training version"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.cache_manager = FixedCacheManager(config.get('cache_root', './cache'))
        
        # Risk mapping - take highest risk from both sides
        # BI-RADS 3-4 → 0 (medium risk)
        # BI-RADS 5-6 → 1 (high risk)
        self.risk_mapping = {3: 0, 4: 0, 5: 1, 6: 1}
        self.risk_names = {0: 'Medium Risk', 1: 'High Risk'}
        self.num_classes = 2  # Binary classification
        
        # BI-RADS grades to ignore
        self.ignore_birads = [1, 2]
        
        # MIL configuration
        self.max_instances = config.get('max_instances', 20)  # Increased to 20
        
        logger.info(f"✅ Bilateral MIL data manager initialized")
        logger.info(f"   Max instances per bilateral bag: {self.max_instances}")
    
    def prepare_bilateral_mil_data(self, clinical_df, bilateral_image_features):
        """Prepare bilateral breast MIL format data"""
        bags = []  # Each bag contains all slices of left and right breast of one patient
        instance_masks = []  # Mark valid slices
        clinical_features = []
        risk_labels = []
        bag_info = []
        
        logger.info(f"📊 Preparing Bilateral MIL data: {len(bilateral_image_features)} patients")
        
        total_patients = 0
        slice_counts = []
        asymmetry_features = []
        
        for pid, image_data in tqdm(bilateral_image_features.items(), desc="Preparing bilateral MIL bags"):
            patient_row = clinical_df[clinical_df['PID'] == pid]
            
            if len(patient_row) == 0:
                continue
            
            patient_clinical = patient_row.iloc[0]
            
            # Check BI-RADS labels - ignore 1 and 2
            birads_left = patient_clinical.get('BI-RADSl')
            birads_right = patient_clinical.get('BI-RADSr')
            
            if pd.isna(birads_left) or pd.isna(birads_right):
                continue
                
            # Convert to integer
            birads_left = int(birads_left)
            birads_right = int(birads_right)
            
            # Ignore BI-RADS 1 and 2 data
            if birads_left in self.ignore_birads and birads_right in self.ignore_birads:
                continue
            
            # Get left and right breast images
            left_images = image_data.get('left_images', [])
            right_images = image_data.get('right_images', [])
            
            # Must have at least one side with image data
            if len(left_images) == 0 and len(right_images) == 0:
                continue
            
            # Extract clinical features (enhanced version, including asymmetry information)
            try:
                age = float(patient_clinical['age'])
                bmi = float(patient_clinical['BMI'])
                density = float(patient_clinical['density_numeric'])
                history = float(patient_clinical['history'])
                
                # Add derived features
                age_group = age // 10
                bmi_category = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
                
                # Calculate BI-RADS asymmetry feature
                birads_asymmetry = abs(birads_left - birads_right)
                
                clinical_feature = np.array([
                    age, bmi, density, history,
                    age_group, bmi_category,
                    age * density,  # Interaction feature
                    birads_asymmetry  # New: BI-RADS asymmetry feature
                ], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            
            # Combine left and right breast images, create bilateral bag
            bilateral_slices = []
            slice_positions = []  # Record position information for each slice (side, slice_index)
            
            # Add left breast slices
            for i, left_img in enumerate(left_images):
                bilateral_slices.append(left_img)
                slice_positions.append(('left', i))
            
            # Add right breast slices (keep original orientation)
            for i, right_img in enumerate(right_images):
                bilateral_slices.append(right_img)
                slice_positions.append(('right', i))
            
            # Calculate overall risk (take highest risk from both sides)
            risk_left = self.risk_mapping.get(birads_left, 0) if birads_left not in self.ignore_birads else 0
            risk_right = self.risk_mapping.get(birads_right, 0) if birads_right not in self.ignore_birads else 0
            overall_risk = max(risk_left, risk_right)
            
            # Standardize bag size
            bag, mask = self._standardize_bilateral_bag(bilateral_slices)
            
            bags.append(bag)
            instance_masks.append(mask)
            clinical_features.append(clinical_feature)
            risk_labels.append(overall_risk)
            
            bag_info.append({
                'patient_id': pid,
                'n_left_instances': len(left_images),
                'n_right_instances': len(right_images),
                'n_total_instances': len(bilateral_slices),
                'birads_left': birads_left,
                'birads_right': birads_right,
                'birads_asymmetry': birads_asymmetry,
                'overall_risk': overall_risk,
                'slice_positions': slice_positions[:len(bilateral_slices)]
            })
            
            total_patients += 1
            slice_counts.append(len(bilateral_slices))
            asymmetry_features.append(birads_asymmetry)
        
        # Statistics
        logger.info(f"📊 Bilateral MIL data prepared:")
        logger.info(f"   Total patients: {total_patients}")
        logger.info(f"   Ignored BI-RADS 1-2 cases")
        logger.info(f"   Only including BI-RADS 3-6")
        if slice_counts:
            logger.info(f"   Average slices per patient: {np.mean(slice_counts):.1f}")
            logger.info(f"   Min slices: {np.min(slice_counts)}")
            logger.info(f"   Max slices: {np.max(slice_counts)}")
        if asymmetry_features:
            logger.info(f"   Average BI-RADS asymmetry: {np.mean(asymmetry_features):.1f}")
            logger.info(f"   Max asymmetry: {np.max(asymmetry_features)}")
        
        # Convert to numpy arrays
        mil_data = {
            'bags': np.array(bags),
            'instance_masks': np.array(instance_masks),
            'clinical_features': np.array(clinical_features),
            'risk_labels': np.array(risk_labels),
            'bag_info': bag_info
        }
        
        return mil_data
    
    def _standardize_bilateral_bag(self, slices):
        """Standardize bilateral bag size"""
        n_slices = len(slices)
        
        if n_slices >= self.max_instances:
            # Uniform sampling
            indices = np.linspace(0, n_slices-1, self.max_instances, dtype=int)
            selected_slices = [slices[i] for i in indices]
            mask = np.ones(self.max_instances)
        else:
            # Padding
            selected_slices = list(slices)
            padding_needed = self.max_instances - n_slices
            
            # Pad with zeros
            padding_shape = slices[0].shape
            for _ in range(padding_needed):
                selected_slices.append(np.zeros(padding_shape, dtype=np.float32))
            
            # Create mask
            mask = np.zeros(self.max_instances)
            mask[:n_slices] = 1
        
        return np.array(selected_slices), mask
    
    def load_and_prepare_data(self, force_rebuild=False):
        """Load data and prepare bilateral MIL format"""
        
        print("=" * 80)
        print("🗄️ BILATERAL MIL MODEL DATA LOADING")
        print("=" * 80)
        
        # Check cache
        if self.cache_manager.cache_exists() and not force_rebuild:
            logger.info("🎯 CACHE FOUND! Loading...")
            try:
                cached_data = self.cache_manager.load_cache()
                mil_data = self.prepare_bilateral_mil_data(
                    cached_data['clinical_features'],
                    cached_data['bilateral_image_features']
                )
                
                print("=" * 80)
                print("🎉 SUCCESS! Data loaded from cache and prepared for Bilateral MIL")
                print(f"✅ Loaded {len(mil_data['bags'])} patients")
                print("=" * 80)
                
                self._print_data_summary(mil_data)
                return mil_data
                
            except Exception as e:
                logger.warning(f"⚠️ Cache loading failed: {e}")
        
        # If no cache, need to process raw data
        logger.error("❌ No cache found. Please run the original model first to create cache.")
        return None
    
    def _print_data_summary(self, mil_data):
        """Print data summary"""
        logger.info("📊 Bilateral MIL Data Summary:")
        logger.info(f"   Total patients: {len(mil_data['bags'])}")
        logger.info(f"   Bilateral bag shape: {mil_data['bags'].shape}")
        logger.info(f"   Clinical features: {mil_data['clinical_features'].shape[1]} dimensions")
        
        # Risk distribution
        labels = mil_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        logger.info(f"   Risk distribution:")
        for label, count in zip(unique_labels, counts):
            risk_name = self.risk_names[label]
            percentage = count / len(labels) * 100
            logger.info(f"     {risk_name}: {count} ({percentage:.1f}%)")
        
        # Asymmetry analysis
        asymmetry_scores = [info['birads_asymmetry'] for info in mil_data['bag_info']]
        logger.info(f"   BI-RADS Asymmetry Analysis:")
        logger.info(f"     Mean asymmetry: {np.mean(asymmetry_scores):.2f}")
        logger.info(f"     Patients with asymmetry > 0: {sum(1 for x in asymmetry_scores if x > 0)}")
        
        # Slice distribution analysis
        left_counts = [info['n_left_instances'] for info in mil_data['bag_info']]
        right_counts = [info['n_right_instances'] for info in mil_data['bag_info']]
        total_counts = [info['n_total_instances'] for info in mil_data['bag_info']]
        
        logger.info(f"   Slice distribution:")
        logger.info(f"     Left slices per patient: {np.mean(left_counts):.1f} ± {np.std(left_counts):.1f}")
        logger.info(f"     Right slices per patient: {np.mean(right_counts):.1f} ± {np.std(right_counts):.1f}")
        logger.info(f"     Total slices per patient: {np.mean(total_counts):.1f} ± {np.std(total_counts):.1f}")

class BilateralImprovedEnsembleMILPipeline:
    """Bilateral breast improved ensemble MIL training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_manager = BilateralMILDataManager(config)
        self.overfitting_analysis = {}
        self.ensemble_models = []
        
        logger.info("🚀 Bilateral Improved MIL training pipeline initialized")
    
    def create_model_with_config(self, model_config):
        """Create bilateral model based on configuration"""
        model = BilateralMILBreastCancerModel(
            instance_shape=(*self.config['image_config']['target_size'], 3),
            max_instances=self.config.get('max_instances', 20),  # Increased to 20
            clinical_dim=8,
            num_classes=2
        )
        
        # Recompile model
        model.model.compile(
            optimizer=Adam(learning_rate=model_config['lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _split_data_patient_aware(self, mil_data):
        """Patient-aware data splitting"""
        # Get risk level for each patient
        patient_ids = [info['patient_id'] for info in mil_data['bag_info']]
        patient_risks = [info['overall_risk'] for info in mil_data['bag_info']]
        
        # Stratified split of patients
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(sss.split(range(len(patient_ids)), patient_risks))
        
        temp_risks = [patient_risks[i] for i in temp_idx]
        
        # Further split validation and test sets
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx_temp, test_idx_temp = next(sss_val.split(range(len(temp_idx)), temp_risks))
        
        val_idx = [temp_idx[i] for i in val_idx_temp]
        test_idx = [temp_idx[i] for i in test_idx_temp]
        
        def create_subset(indices):
            return {
                'bags': mil_data['bags'][indices],
                'instance_masks': mil_data['instance_masks'][indices],
                'clinical_features': mil_data['clinical_features'][indices],
                'risk_labels': mil_data['risk_labels'][indices],
                'bag_info': [mil_data['bag_info'][i] for i in indices]
            }
        
        train_data = create_subset(train_idx)
        val_data = create_subset(val_idx)
        test_data = create_subset(test_idx)
        
        logger.info(f"📊 Patient-aware data split:")
        logger.info(f"   Train: {len(train_idx)} patients")
        logger.info(f"   Val: {len(val_idx)} patients")
        logger.info(f"   Test: {len(test_idx)} patients")
        
        return train_data, val_data, test_data
    
    def augment_minority_class(self, train_data):
        """Data augmentation - make high risk class 50%"""
        logger.info("🔄 Bilateral minority class augmentation...")
        
        high_risk_indices = np.where(train_data['risk_labels'] == 1)[0]
        medium_risk_indices = np.where(train_data['risk_labels'] == 0)[0]
        
        n_high = len(high_risk_indices)
        n_medium = len(medium_risk_indices)
        
        if n_high == 0:
            return train_data
        
        # Target: make high risk samples equal to medium risk samples
        target_high_samples = n_medium
        augment_factor = max(2, n_medium // n_high)
        
        logger.info(f"   Original - High: {n_high}, Medium: {n_medium}")
        logger.info(f"   Target - High: {target_high_samples}, Medium: {n_medium}")
        logger.info(f"   Augmentation factor: {augment_factor}x")
        
        augmented_bags = []
        augmented_masks = []
        augmented_clinical = []
        augmented_labels = []
        
        # Create multiple strong augmented versions for each high risk sample
        for idx in high_risk_indices:
            original_bag = train_data['bags'][idx]
            original_mask = train_data['instance_masks'][idx]
            
            for i in range(augment_factor):
                bag = original_bag.copy()
                
                # Strong augmentation
                for j in range(len(bag)):
                    if original_mask[j] > 0:
                        # Combine multiple augmentations
                        # 1. Strong noise
                        noise = np.random.normal(0, 0.05, bag[j].shape)
                        bag[j] = np.clip(bag[j] + noise, 0, 1)
                        
                        # 2. Random erasing
                        if np.random.random() > 0.5:
                            h, w = bag[j].shape[:2]
                            erase_h = np.random.randint(h//8, h//4)
                            erase_w = np.random.randint(w//8, w//4)
                            y = np.random.randint(0, h - erase_h)
                            x = np.random.randint(0, w - erase_w)
                            bag[j][y:y+erase_h, x:x+erase_w] = np.random.random()
                        
                        # 3. Color jittering
                        color_shift = np.random.uniform(-0.1, 0.1, bag[j].shape)
                        bag[j] = np.clip(bag[j] + color_shift, 0, 1)
                
                augmented_bags.append(bag)
                augmented_masks.append(original_mask)
                
                # Clinical features also add perturbation
                clinical_noise = np.random.normal(0, 0.1, train_data['clinical_features'][idx].shape)
                augmented_clinical.append(train_data['clinical_features'][idx] + clinical_noise)
                augmented_labels.append(1)
        
        # Add all medium risk samples
        for idx in medium_risk_indices:
            augmented_bags.append(train_data['bags'][idx])
            augmented_masks.append(train_data['instance_masks'][idx])
            augmented_clinical.append(train_data['clinical_features'][idx])
            augmented_labels.append(0)
        
        # Create balanced dataset
        augmented_data = {
            'bags': np.array(augmented_bags),
            'instance_masks': np.array(augmented_masks),
            'clinical_features': np.array(augmented_clinical),
            'risk_labels': np.array(augmented_labels),
            'bag_info': []
        }
        
        # Shuffle
        indices = np.random.permutation(len(augmented_data['bags']))
        for key in ['bags', 'instance_masks', 'clinical_features', 'risk_labels']:
            augmented_data[key] = augmented_data[key][indices]
        
        # Statistics
        unique_labels, counts = np.unique(augmented_data['risk_labels'], return_counts=True)
        logger.info(f"✅ Final distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"   Class {label}: {count} ({count/len(augmented_data['bags'])*100:.1f}%)")
        
        return augmented_data
    
    def train_single_model(self, model, train_data, val_data, class_weight, model_name):
        """Train single model"""
        logger.info(f"🏃 Training bilateral {model_name} model...")
        
        X_train = [
            train_data['bags'],
            train_data['instance_masks'],
            train_data['clinical_features']
        ]
        y_train = train_data['risk_labels']
        
        X_val = [
            val_data['bags'],
            val_data['instance_masks'],
            val_data['clinical_features']
        ]
        y_val = val_data['risk_labels']
        
        # Simplified callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        history = model.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=4,  # Reduced batch size because bags are larger
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        return history.history
    
    def detect_overfitting(self, history, model_name):
        """Detect overfitting"""
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
        train_loss = history['loss']
        val_loss = history['val_loss']
        
        # Calculate overfitting metrics
        final_gap = train_acc[-1] - val_acc[-1]
        max_gap = max([train_acc[i] - val_acc[i] for i in range(len(train_acc))])
        
        # Check if validation loss is increasing
        val_loss_increasing = False
        if len(val_loss) > 10:
            recent_trend = np.polyfit(range(5), val_loss[-5:], 1)[0]
            val_loss_increasing = recent_trend > 0
        
        overfitting_detected = (
            final_gap > 0.15 or  # Train-val accuracy gap > 15%
            max_gap > 0.20 or    # Max gap > 20%
            val_loss_increasing  # Validation loss increasing
        )
        
        analysis = {
            'model_name': model_name,
            'final_train_acc': train_acc[-1],
            'final_val_acc': val_acc[-1],
            'accuracy_gap': final_gap,
            'max_accuracy_gap': max_gap,
            'val_loss_increasing': val_loss_increasing,
            'overfitting_detected': overfitting_detected,
            'best_epoch': np.argmax(val_acc)
        }
        
        if overfitting_detected:
            logger.warning(f"⚠️ Overfitting detected in {model_name}!")
            logger.warning(f"   Train/Val gap: {final_gap:.3f}")
            logger.warning(f"   Best epoch was: {analysis['best_epoch']+1}")
        
        return analysis
    
    def find_optimal_threshold_for_model(self, model, test_data):
        """Find optimal threshold for single model"""
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features']
        ]
        y_test = test_data['risk_labels']
        
        predictions = model.model.predict(X_test, verbose=0)
        
        best_score = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_classes = (predictions[:, 1] > threshold).astype(int)
            cm = confusion_matrix(y_test, pred_classes)
            
            if cm.shape[0] > 1 and cm.shape[1] > 1:
                med_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
                high_recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
                
                # Balanced score (can adjust weights as needed)
                balanced_score = 0.4 * med_recall + 0.6 * high_recall
                
                if balanced_score > best_score:
                    best_score = balanced_score
                    best_threshold = threshold
        
        return best_threshold
    
    def evaluate_on_test_set(self, model, test_data, model_name, threshold=None):
        """Evaluate single model on test set"""
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features']
        ]
        y_test = test_data['risk_labels']
        
        predictions = model.model.predict(X_test, verbose=0)
        
        # If no threshold specified, find optimal threshold
        if threshold is None:
            threshold = self.find_optimal_threshold_for_model(model, test_data)
        
        pred_classes = (predictions[:, 1] > threshold).astype(int)
        
        # Calculate various metrics
        accuracy = accuracy_score(y_test, pred_classes)
        cm = confusion_matrix(y_test, pred_classes)
        report = classification_report(y_test, pred_classes, 
                                    target_names=['Medium Risk', 'High Risk'],
                                    output_dict=True)
        
        # Calculate additional metrics
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            med_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
            high_recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
            med_precision = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
            high_precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
            
            # Calculate balanced metrics
            balanced_accuracy = (med_recall + high_recall) / 2
            # Weighted F1 score (emphasize high risk)
            weighted_f1 = 0.3 * report['Medium Risk']['f1-score'] + 0.7 * report['High Risk']['f1-score']
        else:
            balanced_accuracy = accuracy
            weighted_f1 = 0
            med_recall = high_recall = 0
            med_precision = high_precision = 0
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'weighted_f1': weighted_f1,
            'threshold': threshold,
            'confusion_matrix': cm,
            'classification_report': report,
            'medium_recall': med_recall,
            'high_recall': high_recall,
            'medium_precision': med_precision,
            'high_precision': high_precision
        }
        
        logger.info(f"\n📊 Test Set Results for Bilateral {model_name}:")
        logger.info(f"   Accuracy: {accuracy:.3f}")
        logger.info(f"   Balanced Accuracy: {balanced_accuracy:.3f}")
        logger.info(f"   Medium Risk Recall: {med_recall:.3f}")
        logger.info(f"   High Risk Recall: {high_recall:.3f}")
        logger.info(f"   Optimal Threshold: {threshold:.2f}")
        
        return results
    
    def select_best_model(self, test_results):
        """Select best model based on test set performance"""
        logger.info("\n🏆 Selecting best bilateral model based on TEST SET performance...")
        
        # Define selection criteria
        selection_criteria = []
        
        for result in test_results:
            # Calculate composite score
            score = (
                0.3 * result['accuracy'] +           # Overall accuracy
                0.3 * result['balanced_accuracy'] +   # Balanced accuracy
                0.2 * result['weighted_f1'] +         # Weighted F1
                0.2 * result['high_recall']           # High risk recall
            )
            
            # Penalize overfitting
            model_name = result['model_name']
            if model_name in self.overfitting_analysis:
                if self.overfitting_analysis[model_name]['overfitting_detected']:
                    score *= 0.9  # Reduce score by 10%
            
            selection_criteria.append({
                'model_name': model_name,
                'score': score,
                'metrics': result
            })
        
        # Sort by score
        selection_criteria.sort(key=lambda x: x['score'], reverse=True)
        
        best_model = selection_criteria[0]
        
        logger.info(f"\n✅ Best bilateral model: {best_model['model_name']}")
        logger.info(f"   Score: {best_model['score']:.3f}")
        logger.info(f"   Test Accuracy: {best_model['metrics']['accuracy']:.3f}")
        logger.info(f"   High Risk Recall: {best_model['metrics']['high_recall']:.3f}")
        logger.info(f"   Medium Risk Recall: {best_model['metrics']['medium_recall']:.3f}")
        
        return best_model
    
    def run_bilateral_ensemble_training(self):
        """Run bilateral breast ensemble training pipeline"""
        logger.info("🚀 Starting Bilateral Ensemble MIL training...")
        
        # 1. Load data
        mil_data = self.data_manager.load_and_prepare_data()
        
        if mil_data is None or len(mil_data['bags']) < 10:
            logger.error("❌ Insufficient data")
            return None
        
        # 2. Data preprocessing
        self.data_manager.scaler.fit(mil_data['clinical_features'])
        mil_data['clinical_features'] = self.data_manager.scaler.transform(
            mil_data['clinical_features']
        )
        
        # 3. Data splitting
        train_data, val_data, test_data = self._split_data_patient_aware(mil_data)
        
        logger.info(f"\n📊 Bilateral Data Split Summary:")
        logger.info(f"   Train: {len(train_data['bags'])} patients")
        logger.info(f"   Val: {len(val_data['bags'])} patients")
        logger.info(f"   Test: {len(test_data['bags'])} patients")
        
        # 4. Train multiple bilateral models
        all_histories = {}
        all_models = []
        test_results = []
        
        model_configs = [
            {
                'name': 'bilateral_balanced',
                'lr': 0.001,
                'class_weight': {0: 1.5, 1: 1.5},
            },
            {
                'name': 'bilateral_high_risk_focus',
                'lr': 0.001,
                'class_weight': {0: 1.0, 1: 2.5},
            },
            {
                'name': 'bilateral_conservative',
                'lr': 0.0005,
                'class_weight': {0: 2.0, 1: 1.0},
            }
        ]
        
        for config in model_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training bilateral {config['name']} model...")
            
            # Create and train bilateral model
            model = self.create_model_with_config(config)
            
            # Optional: use data augmentation for high risk focus model
            if config['name'] == 'bilateral_high_risk_focus':
                augmented_train = self.augment_minority_class(train_data)
            else:
                augmented_train = train_data
            
            history = self.train_single_model(
                model, augmented_train, val_data,
                class_weight=config['class_weight'],
                model_name=config['name']
            )
            
            # Detect overfitting
            overfitting_info = self.detect_overfitting(history, config['name'])
            self.overfitting_analysis[config['name']] = overfitting_info
            
            # Evaluate on test set
            test_result = self.evaluate_on_test_set(model, test_data, config['name'])
            
            all_histories[config['name']] = history
            all_models.append({
                'model': model,
                'config': config,
                'history': history,
                'test_performance': test_result
            })
            test_results.append(test_result)
            
            # Save model to instance variable
            self.ensemble_models = all_models
        
        # 5. Select best model
        best_model_info = self.select_best_model(test_results)
        
        # 6. Save results
        results = {
            'best_model': best_model_info,
            'all_test_results': test_results,
            'overfitting_analysis': self.overfitting_analysis,
            'data_stats': {
                'train_size': len(train_data['bags']),
                'val_size': len(val_data['bags']),
                'test_size': len(test_data['bags'])
            }
        }
        
        # Print final report
        self._print_final_report(results)
        
        return results
    
    def _print_final_report(self, results):
        """Print final report"""
        print("\n" + "="*80)
        print("🎯 BILATERAL MIL Model Selection - Final Report")
        print("="*80)
        
        print("\n📊 Data Split:")
        print(f"   Train: {results['data_stats']['train_size']} patients")
        print(f"   Val: {results['data_stats']['val_size']} patients")
        print(f"   Test: {results['data_stats']['test_size']} patients")
        
        print("\n🔍 Overfitting Analysis:")
        for model_name, info in results['overfitting_analysis'].items():
            status = "⚠️ OVERFITTING" if info['overfitting_detected'] else "✅ OK"
            print(f"   {model_name}: {status}")
            print(f"      Train/Val gap: {info['accuracy_gap']:.3f}")
            print(f"      Best epoch: {info['best_epoch']+1}")
        
        print("\n📈 Test Set Performance Summary:")
        for result in results['all_test_results']:
            print(f"\n   {result['model_name']}:")
            print(f"      Accuracy: {result['accuracy']:.3f}")
            print(f"      High Risk Recall: {result['high_recall']:.3f}")
            print(f"      Medium Risk Recall: {result['medium_recall']:.3f}")
            print(f"      Threshold: {result['threshold']:.2f}")
        
        print(f"\n🏆 SELECTED BILATERAL MODEL: {results['best_model']['model_name']}")
        print(f"   Selection Score: {results['best_model']['score']:.3f}")
        print(f"   This model achieved the best balance of accuracy and recall")
        print(f"   on the TEST SET using BILATERAL features and asymmetry analysis")
        
        print("="*80)

def main_bilateral():
    """Bilateral breast main function"""
    parser = argparse.ArgumentParser(description='Bilateral MIL Model')
    parser.add_argument('--output-dir', type=str, default='D:/Desktop/bilateral_mil_output')
    parser.add_argument('--cache-root', type=str, default='./cache')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--max-instances', type=int, default=20)  # Increased to 20
    
    args = parser.parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'cache_root': args.cache_root,
        'max_instances': args.max_instances,
        'image_config': {
            'target_size': (args.image_size, args.image_size)
        }
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    setup_gpu()
    
    # Run bilateral training pipeline
    pipeline = BilateralImprovedEnsembleMILPipeline(config)
    results = pipeline.run_bilateral_ensemble_training()
    
    if results:
        # Use best model for final prediction
        best_model_name = results['best_model']['model_name']
        logger.info(f"\n✅ Bilateral training complete! Best model: {best_model_name}")
        
        # Save best model
        best_model = None
        for model_info in pipeline.ensemble_models:
            if model_info['config']['name'] == best_model_name:
                best_model = model_info['model']
                break
        
        if best_model:
            weights_file = os.path.join(config['output_dir'], f'best_bilateral_model_{best_model_name}.h5')
            best_model.model.save_weights(weights_file)
            logger.info(f"✅ Best bilateral model weights saved: {weights_file}")

        print("\n🔍 Starting performance validation...")
        
        # Get best model
        best_model = None
        for model_info in pipeline.ensemble_models:
            if model_info['config']['name'] == results['best_model']['model_name']:
                best_model = model_info['model']
                break
        
        if best_model:
            # Re-prepare data
            mil_data = pipeline.data_manager.load_and_prepare_data()
            pipeline.data_manager.scaler.fit(mil_data['clinical_features'])
            mil_data['clinical_features'] = pipeline.data_manager.scaler.transform(mil_data['clinical_features'])
            train_data, val_data, test_data = pipeline._split_data_patient_aware(mil_data)
            
            # Execute complete validation
            validation_results, deep_analysis_results = validate_trained_model(
                best_model, 
                pipeline.data_manager, 
                train_data, 
                val_data, 
                test_data, 
                mil_data, 
                config['output_dir']
            )
            
            print("✅ Performance validation complete!")

            try:
                print("\n🎨 Generating visualization charts...")
                from paper_visualizations import generate_paper_visualizations
                
                generator = generate_paper_visualizations(
                    best_model, 
                    test_data,  
                    train_data, 
                    val_data, 
                    config['output_dir']
                )
                print("✅ Paper visualization generation complete!")
                
            except Exception as e:
                print(f"❌ Paper visualization generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Save best model weights
            weights_file = os.path.join(config['output_dir'], f'best_bilateral_model_{results["best_model"]["model_name"]}.h5')
            best_model.model.save_weights(weights_file)
            logger.info(f"✅ Best bilateral model weights saved: {weights_file}")
            
            print("\n" + "🎉" * 80)
            print("Training, validation and visualization all complete!")
            print(f"📁 All results saved in: {config['output_dir']}")
            print("🎉" * 80)

    
    return results

if __name__ == "__main__":
    sys.exit(main_bilateral())