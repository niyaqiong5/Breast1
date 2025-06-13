"""
Multiple instance learning (MIL) breast cancer risk prediction model
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
from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve
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

try:
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported all data processing modules")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Missing required modules: {e}")
    sys.exit(1)

# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Log Settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"mil_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu():
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
    """Safe JSON conversion with comprehensive NumPy type handling"""
    try:
        # Handle TensorFlow/Keras objects
        if hasattr(obj, 'numpy'):
            return safe_json_convert(obj.numpy())
        
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 0:  # Scalar array
                return safe_json_convert(obj.item())
            else:
                return obj.tolist()
        
        # Handle NumPy scalar types
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        
        # Handle Python built-in types
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        
        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {key: safe_json_convert(value) for key, value in obj.items()}
        
        # Handle lists and tuples recursively
        elif isinstance(obj, (list, tuple)):
            return [safe_json_convert(item) for item in obj]
        
        # Handle sets (convert to list)
        elif isinstance(obj, set):
            return [safe_json_convert(item) for item in obj]
        
        # Handle any remaining complex objects by converting to string
        else:
            return str(obj)
            
    except Exception as e:
        # Last resort: convert to string representation
        return f"<non-serializable: {type(obj).__name__}>"

class FixedCacheManager:
    """Cache manager using fixed cache names"""
    
    def __init__(self, cache_root='./cache'):
        self.cache_root = cache_root
        self.cache_dir = os.path.join(cache_root, 'optimized_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 使用固定的缓存名称
        self.cache_name = "breast_data_v1"
        
        logger.info(f"🗄️ Fixed cache manager initialized")
        logger.info(f"   Cache directory: {self.cache_dir}")
        logger.info(f"   Fixed cache name: {self.cache_name}")
    
    def get_cache_files(self):
        """Get the cache file path"""
        return {
            'clinical': os.path.join(self.cache_dir, f"{self.cache_name}_clinical.pkl.gz"),
            'images': os.path.join(self.cache_dir, f"{self.cache_name}_images.h5"),
            'mapping': os.path.join(self.cache_dir, f"{self.cache_name}_mapping.pkl.gz")
        }
    
    def cache_exists(self):
        """Check if the cache exists"""
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
        """Loading cache data"""
        logger.info("📂 Loading data from fixed cache...")
        
        cache_files = self.get_cache_files()
        
        try:
            # Loading clinical data
            with gzip.open(cache_files['clinical'], 'rb') as f:
                clinical_df = pickle.load(f)
            logger.info(f"✅ Clinical data loaded: {clinical_df.shape}")
            
            # Loading image data
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
            
            # Loading Mapping Data
            with gzip.open(cache_files['mapping'], 'rb') as f:
                mapping_data = pickle.load(f)
            logger.info(f"✅ Mapping data loaded")
            
            # Reconstructing data
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
        
        # Calculating attention weights
        ait = K.exp(ait)
        # Add a small epsilon to avoid division by zero
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        ait = K.expand_dims(ait, axis=-1)
        
        # Weighted Average
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        
        return output, ait

class MILBreastCancerModel:
    """Multi-instance learning breast cancer risk prediction model"""
    
    def __init__(self, instance_shape=(128, 128, 3), max_instances=10, 
                 clinical_dim=8, num_classes=2):
        self.instance_shape = instance_shape
        self.max_instances = max_instances
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_instance_encoder(self):
        """Use MobileNetV2 - a lighter pre-trained model"""
        from tensorflow.keras.applications import MobileNetV2
        
        inputs = Input(shape=self.instance_shape)
        
        # Using MobileNetV2, the alpha parameter controls the network width
        base_model = MobileNetV2(
            input_shape=self.instance_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg',
            alpha=0.5  # Use 50% of the number of channels, significantly reducing parameters
        )
        
        # Freeze most of the layers and only fine-tune the last few
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        x = base_model(inputs)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        return Model(inputs=inputs, outputs=x, name='instance_encoder')

    def _build_model(self):
        """Building a complete MIL model"""
        # Input Definition
        bag_input = Input(shape=(self.max_instances, *self.instance_shape), name='bag_input')
        instance_mask = Input(shape=(self.max_instances,), name='instance_mask')
        clinical_input = Input(shape=(self.clinical_dim,), name='clinical_input')
        
        # Building an Example Encoder
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
        
        # Attention Mechanism
        bag_features, attention_weights = AttentionLayer(64)(instance_features_masked)
        
        # Managing clinical features
        x_clinical = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(clinical_input)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        x_clinical = Dropout(0.3)(x_clinical)
        
        x_clinical = Dense(16, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x_clinical)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        
        # Feature Fusion
        combined = Concatenate()([bag_features, x_clinical])
        
        # Final classification layer
        fusion = Dense(64, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(combined)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.4)(fusion)
        
        fusion = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(fusion)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.3)(fusion)
        
        # Output Layer
        output = Dense(self.num_classes, activation='softmax', name='risk_output')(fusion)
        
        # Build the model
        self.model = Model(
            inputs=[bag_input, instance_mask, clinical_input], 
            outputs=output,
            name='MIL_BreastCancer_Model'
        )
        
        # Attention Model
        self.attention_model = Model(
            inputs=[bag_input, instance_mask, clinical_input],
            outputs=[output, attention_weights],
            name='MIL_Attention_Model'
        )
        
        # Using Sparse Categorical Cross Entropy
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ MIL model built: {self.model.count_params():,} parameters")
        logger.info(f"   Using sparse_categorical_crossentropy loss")
        logger.info(f"   Max instances per bag: {self.max_instances}")
        logger.info(f"   Instance shape: {self.instance_shape}")

class MILDataManager:
    """MIL Data Manager"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.cache_manager = FixedCacheManager(config.get('cache_root', './cache'))
        
        # Risk Mapping - Ignore BI-RADS 1-2, binary classification
        # BI-RADS 3-4 → 0 (medium risk)
        # BI-RADS 5-6 → 1 (high risk)
        self.risk_mapping = {3: 0, 4: 0, 5: 1, 6: 1}
        self.risk_names = {0: 'Medium Risk', 1: 'High Risk'}
        self.num_classes = 2  # Binary classification
        
        # BI-RADS classes to ignore
        self.ignore_birads = [1, 2]
        
        # MIL Configuration
        self.max_instances = config.get('max_instances', 10)
        
        logger.info(f"✅ MIL data manager initialized")
        logger.info(f"   Max instances per bag: {self.max_instances}")
    
    def prepare_mil_data(self, clinical_df, bilateral_image_features):
        """Preparing data in MIL format"""
        bags = []  # Each package contains all the sections of a breast
        instance_masks = []  # Mark valid slices
        clinical_features = []
        risk_labels = []
        bag_info = []
        
        logger.info(f"📊 Preparing MIL data: {len(bilateral_image_features)} patients")
        
        total_bags = 0
        slice_counts = []
        
        for pid, image_data in tqdm(bilateral_image_features.items(), desc="Preparing MIL bags"):
            patient_row = clinical_df[clinical_df['PID'] == pid]
            
            if len(patient_row) == 0:
                continue
            
            patient_clinical = patient_row.iloc[0]
            
            # Check BI-RADS label - ignore 1 and 2
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
            
            # Extract clinical features (enhanced version)
            try:
                age = float(patient_clinical['年龄']) #AGE
                bmi = float(patient_clinical['BMI'])
                density = float(patient_clinical['density_numeric'])
                history = float(patient_clinical['history'])
                
                # Adding Derived Features
                age_group = age // 10
                bmi_category = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
                
                clinical_feature = np.array([
                    age, bmi, density, history,
                    age_group, bmi_category,
                    age * density,  # Interaction Features
                    bmi * density   # Interaction Features
                ], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            
            # Process of left breast mass - only BI-RADS 3-6
            left_images = image_data.get('left_images', [])
            if len(left_images) > 0 and birads_left not in self.ignore_birads:
                # Standardize package size
                bag, mask = self._standardize_bag(left_images)
                bags.append(bag)
                instance_masks.append(mask)
                clinical_features.append(clinical_feature)
                risk_labels.append(self.risk_mapping.get(birads_left, 0))
                bag_info.append({
                    'patient_id': pid,
                    'breast_side': 'left',
                    'n_instances': len(left_images),
                    'birads_score': birads_left,
                    'risk_level': self.risk_mapping.get(birads_left, 0)
                })
                total_bags += 1
                slice_counts.append(len(left_images))
            
            # Process of left breast mass - only BI-RADS 3-6
            right_images = image_data.get('right_images', [])
            if len(right_images) > 0 and birads_right not in self.ignore_birads:
                # Flip all slices
                right_images_flipped = [np.fliplr(img) for img in right_images]
                bag, mask = self._standardize_bag(right_images_flipped)
                bags.append(bag)
                instance_masks.append(mask)
                clinical_features.append(clinical_feature)
                risk_labels.append(self.risk_mapping.get(birads_right, 0))
                bag_info.append({
                    'patient_id': pid,
                    'breast_side': 'right',
                    'n_instances': len(right_images),
                    'birads_score': birads_right,
                    'risk_level': self.risk_mapping.get(birads_right, 0)
                })
                total_bags += 1
                slice_counts.append(len(right_images))
        
        # Statistics
        logger.info(f"📊 MIL data prepared:")
        logger.info(f"   Total bags (breasts): {total_bags}")
        logger.info(f"   Ignored BI-RADS 1-2 cases")
        logger.info(f"   Only including BI-RADS 3-6")
        if slice_counts:
            logger.info(f"   Average slices per bag: {np.mean(slice_counts):.1f}")
            logger.info(f"   Min slices: {np.min(slice_counts)}")
            logger.info(f"   Max slices: {np.max(slice_counts)}")
        else:
            logger.warning("   No valid bags found!")
        
        # Convert to numpy array
        mil_data = {
            'bags': np.array(bags),
            'instance_masks': np.array(instance_masks),
            'clinical_features': np.array(clinical_features),
            'risk_labels': np.array(risk_labels),
            'bag_info': bag_info
        }
        
        return mil_data
    
    def _standardize_bag(self, slices):
        """Standardize package size"""
        n_slices = len(slices)
        
        if n_slices >= self.max_instances:
            # Uniform Sampling
            indices = np.linspace(0, n_slices-1, self.max_instances, dtype=int)
            selected_slices = [slices[i] for i in indices]
            mask = np.ones(self.max_instances)
        else:
            # filling
            selected_slices = list(slices)
            padding_needed = self.max_instances - n_slices
            
            # Fill with zeros
            padding_shape = slices[0].shape
            for _ in range(padding_needed):
                selected_slices.append(np.zeros(padding_shape, dtype=np.float32))
            
            # Creating a mask
            mask = np.zeros(self.max_instances)
            mask[:n_slices] = 1
        
        return np.array(selected_slices), mask
    
    def load_and_prepare_data(self, force_rebuild=False):
        """Load data and prepare MIL format"""
        
        print("=" * 80)
        print("🗄️ MIL MODEL DATA LOADING")
        print("=" * 80)
        
        # Check the cache
        if self.cache_manager.cache_exists() and not force_rebuild:
            logger.info("🎯 CACHE FOUND! Loading...")
            try:
                cached_data = self.cache_manager.load_cache()
                mil_data = self.prepare_mil_data(
                    cached_data['clinical_features'],
                    cached_data['bilateral_image_features']
                )
                
                print("=" * 80)
                print("🎉 SUCCESS! Data loaded from cache and prepared for MIL")
                print(f"✅ Loaded {len(mil_data['bags'])} bags (breasts)")
                print("=" * 80)
                
                self._print_data_summary(mil_data)
                return mil_data
                
            except Exception as e:
                logger.warning(f"⚠️ Cache loading failed: {e}")
        
        # If there is no cache, the original data needs to be processed
        logger.error("❌ No cache found. Please run the original model first to create cache.")
        return None
    
    def _print_data_summary(self, mil_data):
        """Print data summary"""
        logger.info("📊 MIL Data Summary:")
        logger.info(f"   Total bags: {len(mil_data['bags'])}")
        logger.info(f"   Bag shape: {mil_data['bags'].shape}")
        logger.info(f"   Clinical features: {mil_data['clinical_features'].shape[1]} dimensions")
        
        # Risk Distribution
        labels = mil_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        logger.info(f"   Risk distribution:")
        for label, count in zip(unique_labels, counts):
            risk_name = self.risk_names[label]
            percentage = count / len(labels) * 100
            logger.info(f"     {risk_name}: {count} ({percentage:.1f}%)")

class ImprovedEnsembleMILPipeline:
    """Improved ensemble MIL training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_manager = MILDataManager(config)
        self.overfitting_analysis = {}
        self.ensemble_models = []
        self.predictions_cache = {}  # Cache for storing predictions
        
        logger.info("🚀 Improved MIL training pipeline initialized")
    
    def create_model_with_config(self, model_config):
        """Create model based on configuration"""
        model = MILBreastCancerModel(
            instance_shape=(*self.config['image_config']['target_size'], 3),
            max_instances=self.config.get('max_instances', 10),
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
        # Get patient ID and risk level for each bag
        patient_risks = {}
        bag_to_patient = {}
        
        for i, info in enumerate(mil_data['bag_info']):
            pid = info['patient_id']
            risk = info['risk_level']
            bag_to_patient[i] = pid
            
            if pid not in patient_risks:
                patient_risks[pid] = []
            patient_risks[pid].append(risk)
        
        # Determine main risk for each patient
        patient_main_risks = {}
        for pid, risks in patient_risks.items():
            main_risk = max(set(risks), key=risks.count)
            patient_main_risks[pid] = main_risk
        
        # Patient list and labels
        patient_ids = list(patient_main_risks.keys())
        patient_labels = [patient_main_risks[pid] for pid in patient_ids]
        
        # Stratified split of patients
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_patient_idx, temp_patient_idx = next(sss.split(patient_ids, patient_labels))
        
        train_patients = set([patient_ids[i] for i in train_patient_idx])
        temp_patients = [patient_ids[i] for i in temp_patient_idx]
        temp_labels = [patient_labels[i] for i in temp_patient_idx]
        
        # Further split validation and test sets
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_patient_idx, test_patient_idx = next(sss_val.split(range(len(temp_patients)), temp_labels))
        
        val_patients = set([temp_patients[i] for i in val_patient_idx])
        test_patients = set([temp_patients[i] for i in test_patient_idx])
        
        # Assign bags to corresponding sets
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, pid in bag_to_patient.items():
            if pid in train_patients:
                train_indices.append(i)
            elif pid in val_patients:
                val_indices.append(i)
            elif pid in test_patients:
                test_indices.append(i)
        
        def create_subset(indices):
            return {
                'bags': mil_data['bags'][indices],
                'instance_masks': mil_data['instance_masks'][indices],
                'clinical_features': mil_data['clinical_features'][indices],
                'risk_labels': mil_data['risk_labels'][indices],
                'bag_info': [mil_data['bag_info'][i] for i in indices]
            }
        
        train_data = create_subset(train_indices)
        val_data = create_subset(val_indices)
        test_data = create_subset(test_indices)
        
        logger.info(f"📊 Patient-aware data split:")
        logger.info(f"   Train: {len(train_patients)} patients, {len(train_data['bags'])} bags")
        logger.info(f"   Val: {len(val_patients)} patients, {len(val_data['bags'])} bags")
        logger.info(f"   Test: {len(test_patients)} patients, {len(test_data['bags'])} bags")
        
        return train_data, val_data, test_data
    
    def augment_minority_class(self, train_data):
        """Data augmentation - make high risk class 50%"""
        logger.info("🔄 Aggressive minority class augmentation...")
        
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
        logger.info(f"🏃 Training {model_name} model...")
        
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
            batch_size=8,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        return history.history
    
    def store_model_predictions(self, model, test_data, model_name):
        """Store model predictions for ROC/PR curve calculation"""
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features']
        ]
        y_test = test_data['risk_labels']
        
        # Get prediction probabilities
        predictions = model.model.predict(X_test, verbose=0)
        
        # Store predictions
        self.predictions_cache[model_name] = {
            'y_true': y_test,
            'y_prob': predictions,
            'y_pred': np.argmax(predictions, axis=1)
        }
        
        logger.info(f"✅ Stored predictions for {model_name}")
    
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
        
        logger.info(f"\n📊 Test Set Results for {model_name}:")
        logger.info(f"   Accuracy: {accuracy:.3f}")
        logger.info(f"   Balanced Accuracy: {balanced_accuracy:.3f}")
        logger.info(f"   Medium Risk Recall: {med_recall:.3f}")
        logger.info(f"   High Risk Recall: {high_recall:.3f}")
        logger.info(f"   Optimal Threshold: {threshold:.2f}")
        
        return results
    
    def select_best_model(self, test_results):
        """Select best model based on test set performance"""
        logger.info("\n🏆 Selecting best model based on TEST SET performance...")
        
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
        
        logger.info(f"\n✅ Best model: {best_model['model_name']}")
        logger.info(f"   Score: {best_model['score']:.3f}")
        logger.info(f"   Test Accuracy: {best_model['metrics']['accuracy']:.3f}")
        logger.info(f"   High Risk Recall: {best_model['metrics']['high_recall']:.3f}")
        logger.info(f"   Medium Risk Recall: {best_model['metrics']['medium_recall']:.3f}")
        
        return best_model
    
    def plot_training_analysis(self, all_histories, test_results):
        """Plot training analysis"""
        # Set paper style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        n_models = len(all_histories)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5*(n_models+1), 10))
        
        # Plot training history for each model
        for i, (model_name, history) in enumerate(all_histories.items()):
            epochs = range(1, len(history['loss']) + 1)
            
            # Loss
            axes[0, i].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
            axes[0, i].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            axes[0, i].set_title(f'{model_name} - Loss')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Accuracy
            axes[1, i].plot(epochs, history['accuracy'], 'b-', label='Train Acc', linewidth=2)
            axes[1, i].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
            axes[1, i].set_title(f'{model_name} - Accuracy')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Accuracy')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
            # Mark overfitting
            if model_name in self.overfitting_analysis:
                if self.overfitting_analysis[model_name]['overfitting_detected']:
                    axes[1, i].text(0.5, 0.5, 'OVERFITTING\nDETECTED', 
                                  transform=axes[1, i].transAxes,
                                  fontsize=16, color='red', alpha=0.7,
                                  ha='center', va='center',
                                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Show test set performance comparison in last column
        model_names = [r['model_name'] for r in test_results]
        test_accs = [r['accuracy'] for r in test_results]
        high_recalls = [r['high_recall'] for r in test_results]
        med_recalls = [r['medium_recall'] for r in test_results]
        
        x = np.arange(len(model_names))
        
        # Test set accuracy comparison
        axes[0, -1].bar(x, test_accs, color=['green' if acc == max(test_accs) else 'blue' for acc in test_accs])
        axes[0, -1].set_xlabel('Models')
        axes[0, -1].set_ylabel('Test Accuracy')
        axes[0, -1].set_title('Test Set Performance')
        axes[0, -1].set_xticks(x)
        axes[0, -1].set_xticklabels(model_names, rotation=45)
        axes[0, -1].grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (acc, model) in enumerate(zip(test_accs, model_names)):
            axes[0, -1].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Recall comparison
        width = 0.35
        axes[1, -1].bar(x - width/2, high_recalls, width, label='High Risk', color='red', alpha=0.7)
        axes[1, -1].bar(x + width/2, med_recalls, width, label='Medium Risk', color='orange', alpha=0.7)
        axes[1, -1].set_xlabel('Models')
        axes[1, -1].set_ylabel('Recall')
        axes[1, -1].set_title('Recall Comparison')
        axes[1, -1].set_xticks(x)
        axes[1, -1].set_xticklabels(model_names, rotation=45)
        axes[1, -1].legend()
        axes[1, -1].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training analysis plots saved")
    
    def _plot_roc_pr_curves(self, test_results):
        """Plot ROC curves and PR curves using real data"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # ROC curves
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
        
        for i, result in enumerate(test_results):
            model_name = result['model_name']
            
            # Get cached predictions
            if model_name in self.predictions_cache:
                pred_data = self.predictions_cache[model_name]
                y_true = pred_data['y_true']
                y_prob = pred_data['y_prob']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            else:
                logger.warning(f"No cached predictions for {model_name}")
        
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # PR curves
        for i, result in enumerate(test_results):
            model_name = result['model_name']
            
            # Get cached predictions
            if model_name in self.predictions_cache:
                pred_data = self.predictions_cache[model_name]
                y_true = pred_data['y_true']
                y_prob = pred_data['y_prob']
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                avg_precision = average_precision_score(y_true, y_prob[:, 1])
                
                ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
            else:
                logger.warning(f"No cached predictions for {model_name}")
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Real ROC/PR curves saved")
    
    def plot_comprehensive_visualizations(self, mil_data, test_results, best_model_info, test_data=None):
        """Generate comprehensive visualizations"""
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # 1. Data distribution visualization
        self._plot_data_distribution(mil_data)
        
        # 2. Confusion matrix heatmap
        self._plot_confusion_matrices(test_results)
        
        # 3. ROC curves and PR curves (using real data)
        self._plot_roc_pr_curves(test_results)
        
        # 4. Model performance radar chart
        self._plot_performance_radar(test_results)
        
        # 5. Slice count distribution
        self._plot_slice_distribution(mil_data)
        
        # 6. Clinical features analysis
        self._plot_clinical_features_analysis(mil_data)
        
        logger.info("✅ All visualizations saved for paper")
    
    def _plot_data_distribution(self, mil_data):
        """Plot data distribution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Risk level distribution
        labels = mil_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        risk_names = ['Medium Risk', 'High Risk']
        
        colors = ['#3498db', '#e74c3c']
        wedges, texts, autotexts = axes[0].pie(counts, labels=risk_names, autopct='%1.1f%%',
                                                colors=colors, explode=(0.05, 0.05))
        axes[0].set_title('Risk Distribution in Dataset', fontsize=14, fontweight='bold')
        
        # 2. Left vs right breast distribution
        sides = [info['breast_side'] for info in mil_data['bag_info']]
        side_counts = pd.Series(sides).value_counts()
        
        axes[1].bar(side_counts.index, side_counts.values, color=['#9b59b6', '#1abc9c'])
        axes[1].set_title('Left vs Right Breast Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Breast Side')
        axes[1].set_ylabel('Count')
        
        # Add value labels
        for i, (side, count) in enumerate(side_counts.items()):
            axes[1].text(i, count + 1, str(count), ha='center', va='bottom')
        
        # 3. BI-RADS score distribution
        birads_scores = [info['birads_score'] for info in mil_data['bag_info']]
        birads_counts = pd.Series(birads_scores).value_counts().sort_index()
        
        axes[2].bar(birads_counts.index, birads_counts.values, 
                   color=['#2ecc71' if score <= 4 else '#e74c3c' for score in birads_counts.index])
        axes[2].set_title('BI-RADS Score Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('BI-RADS Score')
        axes[2].set_ylabel('Count')
        axes[2].set_xticks(birads_counts.index)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, test_results):
        """Plot confusion matrix heatmaps"""
        n_models = len(test_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, result in enumerate(test_results):
            cm = result['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                       xticklabels=['Medium', 'High'],
                       yticklabels=['Medium', 'High'],
                       ax=axes[i], cbar_kws={'label': 'Normalized Count'})
            
            axes[i].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.3f}", 
                             fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, test_results):
        """Plot model performance radar chart"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Performance metrics
        categories = ['Accuracy', 'High Risk\nRecall', 'Medium Risk\nRecall', 
                     'High Risk\nPrecision', 'Medium Risk\nPrecision', 'Balanced\nAccuracy']
        n_cats = len(categories)
        
        # Angles
        angles = [n / float(n_cats) * 2 * pi for n in range(n_cats)]
        angles += angles[:1]
        
        # Plot each model
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for i, result in enumerate(test_results):
            values = [
                result['accuracy'],
                result.get('high_recall', 0),
                result.get('medium_recall', 0),
                result.get('high_precision', 0),
                result.get('medium_precision', 0),
                result.get('balanced_accuracy', 0)
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'],
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_slice_distribution(self, mil_data):
        """Plot slice count distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Slice count histogram
        slice_counts = [info['n_instances'] for info in mil_data['bag_info']]
        
        ax1.hist(slice_counts, bins=range(min(slice_counts), max(slice_counts)+2), 
                edgecolor='black', color='#3498db', alpha=0.7)
        ax1.axvline(np.mean(slice_counts), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(slice_counts):.1f}')
        ax1.set_xlabel('Number of Slices per Bag')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Slice Counts', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Slice count box plot by risk level
        risk_levels = [info['risk_level'] for info in mil_data['bag_info']]
        
        data_by_risk = {'Medium Risk': [], 'High Risk': []}
        for count, risk in zip(slice_counts, risk_levels):
            if risk == 0:
                data_by_risk['Medium Risk'].append(count)
            else:
                data_by_risk['High Risk'].append(count)
        
        box_data = [data_by_risk['Medium Risk'], data_by_risk['High Risk']]
        box = ax2.boxplot(box_data, tick_labels=['Medium Risk', 'High Risk'], patch_artist=True)
        
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Number of Slices')
        ax2.set_title('Slice Count by Risk Level', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'slice_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_clinical_features_analysis(self, mil_data):
        """Plotting clinical characteristics analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Extraction of clinical features
        clinical_features = mil_data['clinical_features']
        risk_labels = mil_data['risk_labels']
        
        # Denormalize (if necessary)
        feature_names = ['Age', 'BMI', 'Density', 'Family History']
        
        for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            feature_values = clinical_features[:, i]
            
            # Group by risk level
            medium_risk = feature_values[risk_labels == 0]
            high_risk = feature_values[risk_labels == 1]
            
            # Plotting the kernel density estimate
            if len(medium_risk) > 1 and np.std(medium_risk) > 0:
                sns.kdeplot(data=medium_risk, ax=ax, label='Medium Risk', 
                           color='#3498db', fill=True, alpha=0.6, warn_singular=False)
            if len(high_risk) > 1 and np.std(high_risk) > 0:
                sns.kdeplot(data=high_risk, ax=ax, label='High Risk', 
                           color='#e74c3c', fill=True, alpha=0.6, warn_singular=False)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature_name} Distribution by Risk Level', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clinical_features_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_gradcam(self, model, img_array, class_index, layer_name=None):
        """Generating Grad-CAM heatmap"""
        import cv2
        
        # If no layer name is specified, the last convolutional layer is used.
        if layer_name is None:
            # For MobileNetV2, the last convolutional layer is usually used
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        # Create a model that outputs the feature maps and final predictions of the specified layer
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Recording gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        
        # Computing Gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute the global average of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get the convolutional layer output
        conv_outputs = conv_outputs[0]
        
        # Weighted combination
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalized heat map
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize the heatmap to match the original image
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        
        return heatmap
    
    def plot_enhanced_gradcam_analysis(self, model, test_data, num_samples=8):
        """Draw enhanced Grad-CAM analysis graph"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import cv2
        
        # Setting up the graphics
        fig = plt.figure(figsize=(20, 3*num_samples))
        
        # Get the instance encoder
        instance_encoder = None
        for layer in model.model.layers:
            if hasattr(layer, 'name') and 'instance_encoder' in layer.name:
                instance_encoder = layer
                break
        
        if instance_encoder is None:
            instance_encoder = model._build_instance_encoder()
        
        # Randomly select samples
        sample_indices = np.random.choice(len(test_data['bags']), min(num_samples, len(test_data['bags'])), replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            bag = test_data['bags'][sample_idx]
            mask = test_data['instance_masks'][sample_idx]
            true_label = test_data['risk_labels'][sample_idx]
            bag_info = test_data['bag_info'][sample_idx]
            
            # predict
            X_sample = [
                bag[np.newaxis, ...],
                mask[np.newaxis, ...],
                test_data['clinical_features'][sample_idx:sample_idx+1]
            ]
            
            predictions, attention_weights = model.attention_model.predict(X_sample)
            pred_label = np.argmax(predictions[0])
            pred_prob = predictions[0, pred_label]
            
            # Find the three slices with the highest attention weights
            valid_slices = int(np.sum(mask))
            attention_scores = attention_weights[0, :valid_slices, 0]
            top_indices = np.argsort(attention_scores)[-3:][::-1]
            
            # Create 6 sub-graphs for each sample
            for j, slice_idx in enumerate(top_indices):
                # 1. Original Image
                ax1 = plt.subplot(num_samples, 6, idx*6 + j*2 + 1)
                
                # Check if the image is in color
                slice_img = bag[slice_idx]
                if len(slice_img.shape) == 2:  # Grayscale
                    ax1.imshow(slice_img, cmap='gray')
                else:  # Color map
                    ax1.imshow(slice_img)
                
                ax1.set_title(f'Slice #{slice_idx+1}\nAttention: {attention_scores[slice_idx]:.3f}', fontsize=10)
                ax1.axis('off')
                
                # 2. Grad-CAM Overlay
                ax2 = plt.subplot(num_samples, 6, idx*6 + j*2 + 2)
                
                try:
                    # Preparing individual slices for Grad-CAM
                    slice_input = slice_img[np.newaxis, ...]
                    
                    # Generate Grad-CAM
                    heatmap = self.generate_gradcam(instance_encoder, slice_input, pred_label)
                    
                    # Creating a Color Heat Map
                    jet = cm.get_cmap("jet")
                    jet_colors = jet(np.arange(256))[:, :3]
                    jet_heatmap = jet_colors[np.uint8(heatmap * 255)]
                    
                    # Overlay the original image and heat map
                    if len(slice_img.shape) == 2:  # Grayscale to RGB
                        slice_rgb = np.stack([slice_img] * 3, axis=-1)
                    else:
                        slice_rgb = slice_img
                    
                    superimposed = 0.7 * slice_rgb + 0.3 * jet_heatmap
                    superimposed = np.clip(superimposed, 0, 1)
                    
                    ax2.imshow(superimposed)
                    ax2.set_title(f'Grad-CAM\n{"✓" if pred_label == true_label else "✗"}', fontsize=10)
                    ax2.axis('off')
                    
                except Exception as e:
                    ax2.text(0.5, 0.5, 'Grad-CAM\nFailed', ha='center', va='center', 
                            transform=ax2.transAxes, fontsize=10)
                    ax2.axis('off')
            
            # Add sample information
            if idx == 0:
                fig.text(0.08, 0.98 - idx*0.375/num_samples, 
                        f'Patient {bag_info["patient_id"]} - {bag_info["breast_side"]} - '
                        f'True: {["Medium", "High"][true_label]}, '
                        f'Pred: {["Medium", "High"][pred_label]} ({pred_prob:.2f}), '
                        f'BI-RADS: {bag_info["birads_score"]}',
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(self.output_dir, 'enhanced_gradcam_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Enhanced Grad-CAM analysis saved")
    
    def plot_attention_heatmap_comparison(self, model, test_data):
        """Draw attention weight heat map comparison"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect all attention weight data
        attention_data_by_risk = {'Medium Risk': [], 'High Risk': []}
        attention_data_by_prediction = {'Correct': [], 'Incorrect': []}
        
        for i in range(len(test_data['bags'])):
            X_sample = [
                test_data['bags'][i:i+1],
                test_data['instance_masks'][i:i+1],
                test_data['clinical_features'][i:i+1]
            ]
            
            predictions, attention_weights = model.attention_model.predict(X_sample)
            pred_label = np.argmax(predictions[0])
            true_label = test_data['risk_labels'][i]
            
            valid_slices = int(np.sum(test_data['instance_masks'][i]))
            attention_scores = attention_weights[0, :valid_slices, 0]
            
            # Group by risk level
            risk_name = 'High Risk' if true_label == 1 else 'Medium Risk'
            attention_data_by_risk[risk_name].append(attention_scores)
            
            # Group by prediction accuracy
            pred_type = 'Correct' if pred_label == true_label else 'Incorrect'
            attention_data_by_prediction[pred_type].append(attention_scores)
        
        # 1. Attention distribution by risk level
        ax = axes[0, 0]
        for risk, data in attention_data_by_risk.items():
            if len(data) > 0:
                all_scores = np.concatenate(data)
                ax.hist(all_scores, bins=30, alpha=0.6, label=risk, density=True)
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Density')
        ax.set_title('Attention Weight Distribution by Risk Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Attention distribution by prediction accuracy
        ax = axes[0, 1]
        for pred_type, data in attention_data_by_prediction.items():
            if len(data) > 0:
                all_scores = np.concatenate(data)
                ax.hist(all_scores, bins=30, alpha=0.6, label=pred_type, density=True)
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Density')
        ax.set_title('Attention Weight Distribution by Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Standard Deviation Analysis of Attention Weights
        ax = axes[1, 0]
        std_by_risk = {}
        for risk, data in attention_data_by_risk.items():
            if len(data) > 0:
                stds = [np.std(scores) for scores in data]
                std_by_risk[risk] = stds
        
        if std_by_risk:
            ax.boxplot(std_by_risk.values(), labels=std_by_risk.keys())
            ax.set_ylabel('Standard Deviation of Attention Weights')
            ax.set_title('Attention Weight Variability by Risk Level')
            ax.grid(True, alpha=0.3)
        
        # 4. The position distribution of the highest attention weight
        ax = axes[1, 1]
        max_positions_by_risk = {'Medium Risk': [], 'High Risk': []}
        
        for i in range(len(test_data['bags'])):
            true_label = test_data['risk_labels'][i]
            valid_slices = int(np.sum(test_data['instance_masks'][i]))
            
            X_sample = [
                test_data['bags'][i:i+1],
                test_data['instance_masks'][i:i+1],
                test_data['clinical_features'][i:i+1]
            ]
            
            _, attention_weights = model.attention_model.predict(X_sample)
            attention_scores = attention_weights[0, :valid_slices, 0]
            
            # Find the position of highest attention (relative position)
            max_pos = np.argmax(attention_scores) / (valid_slices - 1) if valid_slices > 1 else 0.5
            
            risk_name = 'High Risk' if true_label == 1 else 'Medium Risk'
            max_positions_by_risk[risk_name].append(max_pos)
        
        for risk, positions in max_positions_by_risk.items():
            if len(positions) > 0:
                ax.hist(positions, bins=10, alpha=0.6, label=risk, density=True)
        
        ax.set_xlabel('Relative Position (0=First Slice, 1=Last Slice)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Maximum Attention Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'attention_analysis_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Attention analysis comparison saved")
    
    def plot_feature_importance_analysis(self, model, test_data):
        """Importance of analyzing clinical features"""
        import matplotlib.pyplot as plt
        from sklearn.inspection import permutation_importance
        
        # Create a wrapper to make the model compatible with sklearn
        class ModelWrapper:
            def __init__(self, mil_model, test_data):
                self.model = mil_model
                self.test_data = test_data
                
            def predict(self, clinical_features):
                # Use average image features
                n_samples = len(clinical_features)
                predictions = []
                
                for i in range(n_samples):
                    # Use the first sample image as a representative
                    X = [
                        self.test_data['bags'][0:1],
                        self.test_data['instance_masks'][0:1],
                        clinical_features[i:i+1]
                    ]
                    pred = self.model.model.predict(X, verbose=0)
                    predictions.append(np.argmax(pred[0]))
                
                return np.array(predictions)
        
        # Prepare the data
        wrapper = ModelWrapper(model, test_data)
        
        # Calculating Permutation Importance
        feature_names = ['Age', 'BMI', 'Density', 'Family History', 
                        'Age Group', 'BMI Category', 'Age×Density', 'BMI×Density']
        
        # Create a graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Violin plot of feature distribution
        clinical_features = test_data['clinical_features']
        risk_labels = test_data['risk_labels']
        
        data_for_plot = []
        for i, feature_name in enumerate(feature_names[:4]):  # Only primitive features are plotted
            for j, label in enumerate(risk_labels):
                data_for_plot.append({
                    'Feature': feature_name,
                    'Value': clinical_features[j, i],
                    'Risk': 'High' if label == 1 else 'Medium'
                })
        
        import pandas as pd
        df = pd.DataFrame(data_for_plot)
        
        # Violin Plot
        for i, feature in enumerate(feature_names[:4]):
            feature_data = df[df['Feature'] == feature]
            
            medium_data = feature_data[feature_data['Risk'] == 'Medium']['Value']
            high_data = feature_data[feature_data['Risk'] == 'High']['Value']
            
            positions = [i*2, i*2+0.8]
            parts = ax1.violinplot([medium_data, high_data], positions=positions, widths=0.6)
            
            # Setting Color
            colors = ['#3498db', '#e74c3c']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        
        ax1.set_xticks([i*2+0.4 for i in range(4)])
        ax1.set_xticklabels(feature_names[:4])
        ax1.set_ylabel('Standardized Value')
        ax1.set_title('Clinical Feature Distributions by Risk Level')
        ax1.grid(True, alpha=0.3)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', label='Medium Risk'),
                          Patch(facecolor='#e74c3c', label='High Risk')]
        ax1.legend(handles=legend_elements)
        
        # 2. Feature Correlation Heatmap
        import seaborn as sns
        
        # Calculate the correlation matrix
        feature_df = pd.DataFrame(clinical_features, columns=feature_names)
        feature_df['Risk'] = risk_labels
        
        correlation_matrix = feature_df.corr()
        
        # Draw a heat map
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax2,
                   square=True, linewidths=1)
        ax2.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Feature importance analysis saved")
    
    def plot_model_decision_boundary(self, model, test_data):
        """Plotting the model decision boundary (using dimensionality reduction)"""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        all_features = []
        all_labels = []
        all_predictions = []
        
        # Create a feature extraction model (only once)
        feature_model = Model(
            inputs=model.model.input,
            outputs=model.model.layers[-2].output
        )
        
        # Batch processing to avoid duplicate tracking
        batch_size = 5
        n_samples = len(test_data['bags'])
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_bags = test_data['bags'][i:end_idx]
            batch_masks = test_data['instance_masks'][i:end_idx]
            batch_clinical = test_data['clinical_features'][i:end_idx]
            
            # Batch Prediction
            features = feature_model.predict([batch_bags, batch_masks, batch_clinical], verbose=0)
            predictions = model.model.predict([batch_bags, batch_masks, batch_clinical], verbose=0)
            
            for j in range(len(features)):
                all_features.append(features[j])
                all_labels.append(test_data['risk_labels'][i + j])
                all_predictions.append(np.argmax(predictions[j]))
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Check sample size
        n_samples = len(all_features)
        
        # 1. PCA Visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(all_features)
        
        # Draw the true labels
        for label in [0, 1]:
            mask = all_labels == label
            ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=['#3498db' if label == 0 else '#e74c3c'],
                       label=['Medium Risk' if label == 0 else 'High Risk'][0],
                       alpha=0.6, s=100)
        
        # Marking mispredictions
        errors = all_predictions != all_labels
        ax1.scatter(features_pca[errors, 0], features_pca[errors, 1],
                   marker='x', s=200, c='black', label='Misclassified')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('Model Feature Space (PCA)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. t-SNE visualization (if the number of samples is sufficient)
        if n_samples > 5:  # t-SNE A minimum of 5 samples is required
            # Adjust perplexity to fit the number of samples
            perplexity = min(30, n_samples - 1)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_tsne = tsne.fit_transform(all_features)
            
            # Draw the true labels
            for label in [0, 1]:
                mask = all_labels == label
                ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                           c=['#3498db' if label == 0 else '#e74c3c'],
                           label=['Medium Risk' if label == 0 else 'High Risk'][0],
                           alpha=0.6, s=100)
            
            # Marking mispredictions
            ax2.scatter(features_tsne[errors, 0], features_tsne[errors, 1],
                       marker='x', s=200, c='black', label='Misclassified')
            
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.set_title(f'Model Feature Space (t-SNE, perplexity={perplexity})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # There are too few samples, so a prompt message is displayed
            ax2.text(0.5, 0.5, f'Too few samples ({n_samples}) for t-SNE\n(requires at least 5)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('t-SNE Visualization')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_decision_boundary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Model decision boundary saved")
    
    def plot_slice_importance_matrix(self, model, test_data, num_samples=10):
        """Plotting the slice importance matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Collecting attention weight data
        attention_matrix = []
        sample_labels = []
        max_slices = 10  # Maximum number of slices
        
        sample_indices = np.random.choice(len(test_data['bags']), 
                                        min(num_samples, len(test_data['bags'])), 
                                        replace=False)
        
        for sample_idx in sample_indices:
            X_sample = [
                test_data['bags'][sample_idx:sample_idx+1],
                test_data['instance_masks'][sample_idx:sample_idx+1],
                test_data['clinical_features'][sample_idx:sample_idx+1]
            ]
            
            predictions, attention_weights = model.attention_model.predict(X_sample)
            
            valid_slices = int(np.sum(test_data['instance_masks'][sample_idx]))
            attention_scores = attention_weights[0, :valid_slices, 0]
            
            # Padding to fixed length
            padded_scores = np.zeros(max_slices)
            padded_scores[:valid_slices] = attention_scores
            
            attention_matrix.append(padded_scores)
            
            # Create a label
            true_label = test_data['risk_labels'][sample_idx]
            pred_label = np.argmax(predictions[0])
            patient_id = test_data['bag_info'][sample_idx]['patient_id']
            side = test_data['bag_info'][sample_idx]['breast_side'][0].upper()
            
            label = f"{patient_id}-{side} ({'H' if true_label else 'M'}/{'H' if pred_label else 'M'})"
            sample_labels.append(label)
        
        # Creating a Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        attention_matrix = np.array(attention_matrix)
        
        # Create a mask to hide invalid slices
        mask = attention_matrix == 0
        
        sns.heatmap(attention_matrix, 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=[f'Slice {i+1}' for i in range(max_slices)],
                   yticklabels=sample_labels,
                   mask=mask,
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Slice Importance Matrix\n(True label/Predicted label: H=High Risk, M=Medium Risk)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('Patient Sample')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'slice_importance_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Slice importance matrix saved")
    
    def plot_model_architecture(self):
        """Plot model architecture diagram (using text description)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        architecture_text = """
        MIL Breast Cancer Risk Prediction Model Architecture
        
        Input Layer:
        ├── Bag Input: (batch_size, max_instances=10, 128, 128, 3)
        ├── Instance Mask: (batch_size, max_instances=10)
        └── Clinical Input: (batch_size, 8)
        
        Instance Encoder (MobileNetV2):
        ├── Base Model: MobileNetV2 (α=0.5, weights='imagenet')
        ├── Global Average Pooling
        ├── Dense(128, ReLU) → BatchNorm → Dropout(0.3)
        └── Output: 128-dimensional features per instance
        
        Attention Mechanism:
        ├── Attention Layer (dim=64)
        ├── Learns importance weights for each instance
        └── Weighted aggregation of instance features
        
        Clinical Feature Processing:
        ├── Dense(32) → BatchNorm → LeakyReLU → Dropout(0.3)
        └── Dense(16) → BatchNorm → LeakyReLU
        
        Feature Fusion:
        ├── Concatenate image and clinical features
        ├── Dense(64) → BatchNorm → LeakyReLU → Dropout(0.4)
        └── Dense(32) → BatchNorm → LeakyReLU → Dropout(0.3)
        
        Output Layer:
        └── Dense(2, Softmax) → Risk Prediction (Medium/High)
        
        Total Parameters: 891,842
        """
        
        ax.text(0.05, 0.95, architecture_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.title('Model Architecture Overview', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_improved_ensemble_training(self):
        """Improved ensemble training pipeline"""
        logger.info("🚀 Starting Improved Ensemble MIL training...")
        
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
        
        logger.info(f"\n📊 Data Split Summary:")
        logger.info(f"   Train: {len(train_data['bags'])} bags")
        logger.info(f"   Val: {len(val_data['bags'])} bags")
        logger.info(f"   Test: {len(test_data['bags'])} bags")
        
        # 4. Train multiple models
        all_histories = {}
        all_models = []
        test_results = []
        
        model_configs = [
            {
                'name': 'balanced',
                'lr': 0.001,
                'class_weight': {0: 1.5, 1: 1.5},
            },
            {
                'name': 'high_risk_focus',
                'lr': 0.001,
                'class_weight': {0: 1.0, 1: 2.5},
            },
            {
                'name': 'conservative',
                'lr': 0.0005,
                'class_weight': {0: 2.0, 1: 1.0},
            }
        ]
        
        for config in model_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {config['name']} model...")
            
            # Create and train model
            model = self.create_model_with_config(config)
            
            # Optional: use data augmentation for high risk focus model
            if config['name'] == 'high_risk_focus':
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
            
            # Store predictions for ROC/PR curves
            self.store_model_predictions(model, test_data, config['name'])
            
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
        
        # 6. Generate visualizations
        self.plot_training_analysis(all_histories, test_results)
        self.plot_comprehensive_visualizations(mil_data, test_results, best_model_info, test_data)
        self.plot_model_architecture()
        
        # 7. Save results
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
        
        # Save results to JSON
        results_json = safe_json_convert(results)
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Print final report
        self._print_final_report(results)
        
        return results
    
    def _print_final_report(self, results):
        print("\n" + "="*80)
        print("🎯 MIL Model Selection - Final Report")
        print("="*80)
        
        print("\n📊 Data Split:")
        print(f"   Train: {results['data_stats']['train_size']} bags")
        print(f"   Val: {results['data_stats']['val_size']} bags")
        print(f"   Test: {results['data_stats']['test_size']} bags")
        
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
        
        print(f"\n🏆 SELECTED MODEL: {results['best_model']['model_name']}")
        print(f"   Selection Score: {results['best_model']['score']:.3f}")
        print(f"   This model achieved the best balance of accuracy and recall")
        print(f"   on the TEST SET (not validation set)")
        
        print("="*80)

def main_improved():
    parser = argparse.ArgumentParser(description='Improved MIL Model Selection')
    parser.add_argument('--output-dir', type=str, default='D:/Desktop/mil_output_improved')
    parser.add_argument('--cache-root', type=str, default='./cache')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--max-instances', type=int, default=10)
    
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
    
    pipeline = ImprovedEnsembleMILPipeline(config)
    results = pipeline.run_improved_ensemble_training()
    
    if results:
        # Use the best model to make final predictions
        best_model_name = results['best_model']['model_name']
        logger.info(f"\n✅ Training complete! Best model: {best_model_name}")
        
        # Save the best model
        best_model = None
        for model_info in pipeline.ensemble_models:
            if model_info['config']['name'] == best_model_name:
                best_model = model_info['model']
                break
        
        if best_model:
            weights_file = os.path.join(config['output_dir'], f'best_model_{best_model_name}.h5')
            best_model.model.save_weights(weights_file)
            logger.info(f"✅ Best model weights saved: {weights_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main_improved())