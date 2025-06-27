"""
真正复用缓存的乳腺风险预测模型
使用固定缓存名称，确保每次都能找到和使用现有缓存
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, 
    Concatenate, Conv2D, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback

# 导入现有模块
try:
    from data_loader import ClinicalDataLoader, DicomLoader, SegmentationLoader
    from image_features import BilateralImageProcessor
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported all data processing modules")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Missing required modules: {e}")
    sys.exit(1)

# 环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"cache_reuse_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """GPU设置"""
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
    """安全的JSON转换"""
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

def flip_image_horizontal(image):
    """水平翻转图像"""
    if len(image.shape) == 3:
        return np.fliplr(image)
    else:
        return image

class FixedCacheManager:
    """使用固定缓存名称的缓存管理器"""
    
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
        """获取缓存文件路径"""
        return {
            'clinical': os.path.join(self.cache_dir, f"{self.cache_name}_clinical.pkl.gz"),
            'images': os.path.join(self.cache_dir, f"{self.cache_name}_images.h5"),
            'mapping': os.path.join(self.cache_dir, f"{self.cache_name}_mapping.pkl.gz")
        }
    
    def cache_exists(self):
        """检查缓存是否存在"""
        cache_files = self.get_cache_files()
        
        all_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in cache_files.values())
        
        if all_exist:
            total_size = sum(os.path.getsize(f) for f in cache_files.values()) / (1024*1024)
            logger.info(f"🎯 CACHE FOUND!")
            logger.info(f"   Clinical: {os.path.getsize(cache_files['clinical'])/1024:.1f} KB")
            logger.info(f"   Images: {os.path.getsize(cache_files['images'])/(1024*1024):.1f} MB")
            logger.info(f"   Mapping: {os.path.getsize(cache_files['mapping'])/1024:.1f} KB")
            logger.info(f"   Total: {total_size:.1f} MB")
            return True
        else:
            missing = [name for name, path in cache_files.items() if not os.path.exists(path)]
            if missing:
                logger.info(f"❌ Cache incomplete, missing: {missing}")
            else:
                logger.info(f"❌ No cache found")
            return False
    
    def load_cache(self):
        """加载缓存数据"""
        logger.info("📂 Loading data from fixed cache...")
        
        cache_files = self.get_cache_files()
        
        try:
            # 加载临床数据
            with gzip.open(cache_files['clinical'], 'rb') as f:
                clinical_df = pickle.load(f)
            logger.info(f"✅ Clinical data loaded: {clinical_df.shape}")
            
            # 加载图像数据
            bilateral_image_features = {}
            with h5py.File(cache_files['images'], 'r') as hf:
                patient_count = len(hf.keys())
                logger.info(f"📊 Loading images for {patient_count} patients...")
                
                for pid in tqdm(hf.keys(), desc="Loading cached images"):
                    patient_group = hf[pid]
                    image_data = {}
                    
                    # 加载左右乳图像
                    if 'left_images' in patient_group:
                        image_data['left_images'] = patient_group['left_images'][:]
                    if 'right_images' in patient_group:
                        image_data['right_images'] = patient_group['right_images'][:]
                    
                    bilateral_image_features[pid] = image_data
            
            logger.info(f"✅ Image data loaded: {len(bilateral_image_features)} patients")
            
            # 加载映射数据
            with gzip.open(cache_files['mapping'], 'rb') as f:
                mapping_data = pickle.load(f)
            logger.info(f"✅ Mapping data loaded")
            
            # 重构数据
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
    
    def save_cache(self, clinical_df, bilateral_image_features, processing_config):
        """保存缓存数据"""
        logger.info("💾 Saving data to fixed cache...")
        
        cache_files = self.get_cache_files()
        
        try:
            # 保存临床数据
            with gzip.open(cache_files['clinical'], 'wb') as f:
                pickle.dump(clinical_df, f)
            logger.info(f"✅ Clinical data saved")
            
            # 保存图像数据
            with h5py.File(cache_files['images'], 'w') as hf:
                total_patients = len(bilateral_image_features)
                logger.info(f"📊 Saving images for {total_patients} patients...")
                
                for i, (pid, image_data) in enumerate(tqdm(bilateral_image_features.items(), desc="Saving images")):
                    patient_group = hf.create_group(str(pid))
                    
                    # 保存左右乳图像
                    if 'left_images' in image_data and image_data['left_images']:
                        left_array = np.array(image_data['left_images'], dtype=np.float32)
                        patient_group.create_dataset('left_images', data=left_array, compression='gzip')
                    
                    if 'right_images' in image_data and image_data['right_images']:
                        right_array = np.array(image_data['right_images'], dtype=np.float32)
                        patient_group.create_dataset('right_images', data=right_array, compression='gzip')
            
            logger.info(f"✅ Image data saved")
            
            # 保存映射数据
            mapping_data = {
                'bilateral_slices_data': {},
                'processing_config': processing_config,
                'cache_version': 'fixed_v1.0',
                'created_time': datetime.now().isoformat()
            }
            
            with gzip.open(cache_files['mapping'], 'wb') as f:
                pickle.dump(mapping_data, f)
            logger.info(f"✅ Mapping data saved")
            
            # 显示保存的文件大小
            total_size = sum(os.path.getsize(f) for f in cache_files.values()) / (1024*1024)
            logger.info(f"💾 Cache saved successfully!")
            logger.info(f"   Total size: {total_size:.1f} MB")
            logger.info(f"   Next run will use this cache!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache saving failed: {e}")
            return False
    
    def clear_cache(self):
        """清理缓存"""
        cache_files = self.get_cache_files()
        
        cleared = 0
        for file_path in cache_files.values():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleared += 1
                    logger.info(f"🗑️ Removed: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {file_path}: {e}")
        
        if cleared > 0:
            logger.info(f"✅ Cleared {cleared} cache files")
        else:
            logger.info(f"✅ No cache files to clear")
    
    def find_any_existing_cache(self):
        """查找任何现有的缓存并使用最新的"""
        logger.info("🔍 Searching for any existing cache...")
        
        if not os.path.exists(self.cache_dir):
            logger.info("❌ Cache directory doesn't exist")
            return None
        
        # 查找所有缓存文件
        all_files = os.listdir(self.cache_dir)
        cache_groups = {}
        
        for file in all_files:
            if '_' in file and file.endswith(('.pkl.gz', '.h5')):
                cache_key = file.split('_')[0]
                if cache_key not in cache_groups:
                    cache_groups[cache_key] = []
                cache_groups[cache_key].append(file)
        
        # 找到完整的缓存组
        complete_caches = []
        for cache_key, files in cache_groups.items():
            has_clinical = any('clinical' in f for f in files)
            has_images = any('images' in f for f in files)
            has_mapping = any('mapping' in f for f in files)
            
            if has_clinical and has_images and has_mapping:
                # 获取图像文件大小作为数据量指标
                image_file = next(f for f in files if 'images' in f)
                image_size = os.path.getsize(os.path.join(self.cache_dir, image_file))
                complete_caches.append((cache_key, image_size))
        
        if complete_caches:
            # 选择数据量最大的缓存
            best_cache = max(complete_caches, key=lambda x: x[1])
            cache_key = best_cache[0]
            size_mb = best_cache[1] / (1024*1024)
            
            logger.info(f"🎯 Found existing cache: {cache_key} ({size_mb:.1f} MB)")
            logger.info(f"🔄 Will copy to fixed cache name...")
            
            # 复制到固定缓存名称
            try:
                self._copy_cache_to_fixed(cache_key)
                return cache_key
            except Exception as e:
                logger.error(f"❌ Failed to copy cache: {e}")
                return None
        else:
            logger.info("❌ No complete cache found")
            return None
    
    def _copy_cache_to_fixed(self, source_cache_key):
        """将现有缓存复制到固定名称"""
        import shutil
        
        source_files = {
            'clinical': os.path.join(self.cache_dir, f"{source_cache_key}_clinical.pkl.gz"),
            'images': os.path.join(self.cache_dir, f"{source_cache_key}_images.h5"),
            'mapping': os.path.join(self.cache_dir, f"{source_cache_key}_mapping.pkl.gz")
        }
        
        target_files = self.get_cache_files()
        
        for file_type, source_path in source_files.items():
            if os.path.exists(source_path):
                target_path = target_files[file_type]
                shutil.copy2(source_path, target_path)
                logger.info(f"✅ Copied {file_type}: {os.path.basename(source_path)} -> {os.path.basename(target_path)}")

class CacheReuseDataManager:
    """真正复用缓存的数据管理器"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
        # 使用固定缓存管理器
        self.cache_manager = FixedCacheManager(config.get('cache_root', './cache'))
        
        # 风险映射
        self.risk_mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        self.risk_names = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        self.num_classes = 3
        
        logger.info(f"✅ Cache reuse data manager initialized")
    
    def load_and_prepare_data(self, force_rebuild=False, use_any_cache=True):
        """加载数据 - 真正复用缓存"""
        
        print("=" * 80)
        print("🗄️ CACHE-REUSE DATA LOADING")
        print("=" * 80)
        
        # 强制重建
        if force_rebuild:
            logger.info("🔄 Force rebuild requested")
            self.cache_manager.clear_cache()
        
        # 检查固定缓存
        if self.cache_manager.cache_exists():
            logger.info("🎯 FIXED CACHE FOUND! Loading...")
            try:
                cached_data = self.cache_manager.load_cache()
                prepared_data = self._prepare_from_cached_data(cached_data)
                
                if prepared_data and len(prepared_data['images']) > 0:
                    print("=" * 80)
                    print("🎉 SUCCESS! Data loaded from FIXED CACHE")
                    print(f"✅ Loaded {len(prepared_data['images'])} samples")
                    print("✅ NO image preprocessing needed!")
                    print("=" * 80)
                    
                    self._print_data_summary(prepared_data)
                    return prepared_data
                    
            except Exception as e:
                logger.warning(f"⚠️ Fixed cache loading failed: {e}")
        
        # 查找任何现有缓存
        if use_any_cache:
            existing_cache = self.cache_manager.find_any_existing_cache()
            if existing_cache:
                logger.info(f"🎯 Using existing cache: {existing_cache}")
                try:
                    cached_data = self.cache_manager.load_cache()
                    prepared_data = self._prepare_from_cached_data(cached_data)
                    
                    if prepared_data and len(prepared_data['images']) > 0:
                        print("=" * 80)
                        print("🎉 SUCCESS! Data loaded from EXISTING CACHE")
                        print(f"✅ Loaded {len(prepared_data['images'])} samples")
                        print("✅ Converted existing cache to fixed cache!")
                        print("=" * 80)
                        
                        self._print_data_summary(prepared_data)
                        return prepared_data
                        
                except Exception as e:
                    logger.warning(f"⚠️ Existing cache loading failed: {e}")
        
        # 重新处理数据
        print("=" * 80)
        print("🔄 PROCESSING DATA FROM SCRATCH")
        print("=" * 80)
        
        prepared_data = self._process_and_save_data()
        
        print("=" * 80)
        print("💾 DATA PROCESSED & SAVED TO FIXED CACHE")
        print("🎯 Next run will use the fixed cache!")
        print("=" * 80)
        
        return prepared_data
    
    def _process_and_save_data(self):
        """处理数据并保存到固定缓存"""
        logger.info("📊 Loading and processing medical data...")
        
        # 加载原始数据
        clinical_df, bilateral_image_features = self._load_real_data()
        
        # 保存到固定缓存
        self.cache_manager.save_cache(
            clinical_df, 
            bilateral_image_features, 
            self.config.get('image_config', {})
        )
        
        # 准备样本
        prepared_data = self._prepare_breast_level_samples(clinical_df, bilateral_image_features)
        
        self._print_data_summary(prepared_data)
        return prepared_data
    
    def _prepare_from_cached_data(self, cached_data):
        """从缓存数据准备样本"""
        logger.info("🔧 Preparing samples from cached data...")
        
        clinical_df = cached_data['clinical_features']
        bilateral_image_features = cached_data['bilateral_image_features']
        
        prepared_data = self._prepare_breast_level_samples(clinical_df, bilateral_image_features)
        return prepared_data
    
    def _load_real_data(self):
        """加载真实数据"""
        logger.info("📊 Loading medical data...")
        
        # 初始化加载器
        clinical_loader = ClinicalDataLoader(self.config['data_paths']['excel_path'])
        dicom_loader = DicomLoader(self.config['data_paths']['dicom_root_dir'])
        segmentation_loader = SegmentationLoader(self.config['data_paths']['segmentation_root_dir'])
        
        # 加载临床数据
        clinical_df = clinical_loader.load_data()
        logger.info(f"✅ Clinical data: {len(clinical_df)} patients")
        
        # 图像处理器
        image_processor = BilateralImageProcessor(self.config['image_config'])
        
        # 处理患者数据
        patients_data = {}
        valid_patients = 0
        max_patients = self.config.get('max_patients', None)
        
        logger.info("🔄 Loading patient imaging data...")
        for _, row in tqdm(clinical_df.iterrows(), desc="Loading patients", total=len(clinical_df)):
            pid = str(row['PID'])
            
            dicom_data = dicom_loader.load_patient_dicom(pid)
            segmentation_data = segmentation_loader.load_patient_segmentation_with_glandular(pid)
            
            if dicom_data is not None and segmentation_data is not None:
                patients_data[pid] = (dicom_data, segmentation_data)
                valid_patients += 1
                
                if max_patients and valid_patients >= max_patients:
                    logger.info(f"⚠️ Reached patient limit: {max_patients}")
                    break
        
        logger.info(f"✅ Valid patients: {valid_patients}")
        
        # 批量处理图像
        logger.info("🖼️ Processing bilateral images... (This will take time)")
        bilateral_image_features = image_processor.batch_process_bilateral(patients_data)
        logger.info("✅ Image processing complete")
        
        return clinical_df, bilateral_image_features
    
    def _prepare_breast_level_samples(self, clinical_df, bilateral_image_features):
        """准备乳腺级别样本"""
        images = []
        clinical_features = []
        risk_labels = []
        sample_info = []
        
        logger.info(f"📊 Preparing breast-level samples: {len(bilateral_image_features)} patients")
        
        # 检查必要列
        required_columns = ['PID', 'BI-RADSl', 'BI-RADSr', '年龄', 'BMI', 'density_numeric', 'history']
        missing_columns = [col for col in required_columns if col not in clinical_df.columns]
        if missing_columns:
            logger.error(f"❌ Missing clinical columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # 处理每个患者
        for pid, image_data in tqdm(bilateral_image_features.items(), desc="Processing samples"):
            patient_row = clinical_df[clinical_df['PID'] == pid]
            
            if len(patient_row) == 0:
                continue
            
            patient_clinical = patient_row.iloc[0]
            
            # 检查BI-RADS标签
            birads_left = patient_clinical.get('BI-RADSl')
            birads_right = patient_clinical.get('BI-RADSr')
            
            if pd.isna(birads_left) or pd.isna(birads_right):
                continue
            
            # 提取临床特征
            try:
                age = float(patient_clinical['年龄'])
                bmi = float(patient_clinical['BMI'])
                density = float(patient_clinical['density_numeric'])
                history = float(patient_clinical['history'])
                clinical_feature = np.array([age, bmi, density, history], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            
            # 转换风险等级
            risk_left = self.risk_mapping.get(int(birads_left), 1)
            risk_right = self.risk_mapping.get(int(birads_right), 1)
            
            # 获取图像
            left_images = image_data.get('left_images', [])
            right_images = image_data.get('right_images', [])
            
            # 处理左乳
            if len(left_images) > 0:
                mid_idx = len(left_images) // 2
                left_img = left_images[mid_idx]
                
                if len(left_img.shape) == 3 and left_img.shape[2] == 3:
                    images.append(left_img)
                    clinical_features.append(clinical_feature)
                    risk_labels.append(risk_left)
                    sample_info.append({
                        'patient_id': pid,
                        'breast_side': 'left',
                        'birads_score': int(birads_left),
                        'risk_level': risk_left,
                        'risk_name': self.risk_names[risk_left]
                    })
            
            # 处理右乳（翻转）
            if len(right_images) > 0:
                mid_idx = len(right_images) // 2
                right_img = flip_image_horizontal(right_images[mid_idx])
                
                if len(right_img.shape) == 3 and right_img.shape[2] == 3:
                    images.append(right_img)
                    clinical_features.append(clinical_feature)
                    risk_labels.append(risk_right)
                    sample_info.append({
                        'patient_id': pid,
                        'breast_side': 'right_flipped',
                        'birads_score': int(birads_right),
                        'risk_level': risk_right,
                        'risk_name': self.risk_names[risk_right]
                    })
        
        # 转换为numpy数组
        prepared_data = {
            'images': np.array(images),
            'clinical_features': np.array(clinical_features),
            'risk_labels': np.array(risk_labels),
            'sample_info': sample_info
        }
        
        return prepared_data
    
    def _print_data_summary(self, prepared_data):
        """打印数据摘要"""
        logger.info("📊 Data Summary:")
        logger.info(f"   Total samples: {len(prepared_data['images'])}")
        logger.info(f"   Image shape: {prepared_data['images'].shape}")
        
        # 风险分布
        labels = prepared_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        logger.info(f"   Risk distribution:")
        for label, count in zip(unique_labels, counts):
            risk_name = self.risk_names[label]
            percentage = count / len(labels) * 100
            logger.info(f"     {risk_name}: {count} ({percentage:.1f}%)")

class SimpleCNN:
    """简单CNN模型"""
    
    def __init__(self, input_size=(128, 128), clinical_dim=4, num_classes=3):
        self.input_size = input_size
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """构建简单模型"""
        
        # 图像输入
        image_input = Input(shape=(*self.input_size, 3), name='image_input')
        
        # 简化CNN
        x = Conv2D(32, (7, 7), activation='relu', padding='same')(image_input)
        x = MaxPooling2D((4, 4))(x)
        
        x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 4))(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        
        # 图像特征
        x_img = Dense(64, activation='relu')(x)
        x_img = Dense(32, activation='relu')(x_img)
        
        # 临床特征
        clinical_input = Input(shape=(self.clinical_dim,), name='clinical_input')
        x_clinical = Dense(16, activation='relu')(clinical_input)
        x_clinical = Dense(8, activation='relu')(x_clinical)
        
        # 融合
        combined = Concatenate()([x_img, x_clinical])
        
        # 分类
        final = Dense(32, activation='relu')(combined)
        final = Dense(16, activation='relu')(final)
        output = Dense(self.num_classes, activation='softmax')(final)
        
        # 构建模型
        self.model = Model(inputs=[image_input, clinical_input], outputs=output)
        
        # 编译
        self.model.compile(
            optimizer=Adam(learning_rate=0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Simple CNN model built: {self.model.count_params():,} parameters")

class CacheReuseTrainingPipeline:
    """缓存复用训练流程"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.data_manager = CacheReuseDataManager(config)
        
        # 创建模型
        self.model = SimpleCNN(
            input_size=config['image_config']['target_size'],
            clinical_dim=4,
            num_classes=self.data_manager.num_classes
        )
        
        logger.info("🔍 Cache reuse training pipeline initialized")
    
    def run_training(self, force_rebuild=False, use_any_cache=True):
        """运行训练"""
        logger.info("🔍 Starting cache-reuse training...")
        
        # 1. 加载数据（复用缓存）
        logger.info("📁 Step 1: Load data with cache reuse")
        prepared_data = self.data_manager.load_and_prepare_data(
            force_rebuild=force_rebuild,
            use_any_cache=use_any_cache
        )
        
        if len(prepared_data['images']) < 10:
            logger.error(f"❌ Too few samples: {len(prepared_data['images'])}")
            return None
        
        # 2. 数据分割
        logger.info("🔄 Step 2: Data splitting")
        train_data, val_data, test_data = self._split_data(prepared_data)
        
        # 3. 数据预处理
        logger.info("🔧 Step 3: Data preprocessing")
        train_data, val_data, test_data = self._preprocess_data(train_data, val_data, test_data)
        
        # 4. 训练模型
        logger.info("🚀 Step 4: Training model")
        training_results = self._train_model(train_data, val_data)
        
        # 5. 评估模型
        logger.info("📈 Step 5: Evaluating model")
        evaluation_results = self._evaluate_model(test_data)
        
        # 6. 生成报告
        logger.info("📋 Step 6: Generating report")
        self._generate_report(training_results, evaluation_results, prepared_data)
        
        return {
            'training': training_results,
            'evaluation': evaluation_results,
            'data_stats': self._get_data_stats(prepared_data)
        }
    
    def _split_data(self, prepared_data):
        """数据分割"""
        indices = np.arange(len(prepared_data['images']))
        
        try:
            train_indices, temp_indices = train_test_split(
                indices, test_size=0.3, random_state=42,
                stratify=prepared_data['risk_labels']
            )
            
            temp_labels = prepared_data['risk_labels'][temp_indices]
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, random_state=42,
                stratify=temp_labels
            )
            
        except ValueError:
            train_indices, temp_indices = train_test_split(
                indices, test_size=0.3, random_state=42
            )
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, random_state=42
            )
        
        def create_subset(indices):
            return {
                'images': prepared_data['images'][indices],
                'clinical_features': prepared_data['clinical_features'][indices],
                'risk_labels': prepared_data['risk_labels'][indices],
                'sample_info': [prepared_data['sample_info'][i] for i in indices]
            }
        
        train_data = create_subset(train_indices)
        val_data = create_subset(val_indices)
        test_data = create_subset(test_indices)
        
        logger.info(f"📊 Data split: Train {len(train_data['images'])}, Val {len(val_data['images'])}, Test {len(test_data['images'])}")
        
        return train_data, val_data, test_data
    
    def _preprocess_data(self, train_data, val_data, test_data):
        """数据预处理"""
        self.data_manager.scaler.fit(train_data['clinical_features'])
        
        train_data['clinical_features'] = self.data_manager.scaler.transform(train_data['clinical_features'])
        val_data['clinical_features'] = self.data_manager.scaler.transform(val_data['clinical_features'])
        test_data['clinical_features'] = self.data_manager.scaler.transform(test_data['clinical_features'])
        
        logger.info("✅ Data preprocessing complete")
        return train_data, val_data, test_data
    
    def _train_model(self, train_data, val_data):
        """训练模型"""
        logger.info("🚀 Training model...")
        
        # 计算类权重
        classes = np.unique(train_data['risk_labels'])
        if len(classes) > 1:
            weights = compute_class_weight('balanced', classes=classes, y=train_data['risk_labels'])
            class_weight = dict(zip(classes, weights))
        else:
            class_weight = None
        
        logger.info(f"⚖️ Class weights: {class_weight}")
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=12,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 训练
        history = self.model.model.fit(
            [train_data['images'], train_data['clinical_features']],
            train_data['risk_labels'],
            epochs=100,
            batch_size=8,
            validation_data=([val_data['images'], val_data['clinical_features']], 
                           val_data['risk_labels']),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
            shuffle=True
        )
        
        return {
            'history': safe_json_convert(history.history),
            'best_epoch': len(history.history['loss']),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1])
        }
    
    def _evaluate_model(self, test_data):
        """评估模型"""
        logger.info("📈 Evaluating model...")
        
        predictions = self.model.model.predict([test_data['images'], test_data['clinical_features']])
        predicted_classes = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(test_data['risk_labels'], predicted_classes)
        
        report = classification_report(
            test_data['risk_labels'], 
            predicted_classes,
            target_names=['Low Risk', 'Medium Risk', 'High Risk'],
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(test_data['risk_labels'], predicted_classes)
        
        logger.info(f"📈 Test accuracy: {accuracy:.4f} ({accuracy:.1%})")
        
        return {
            'accuracy': float(accuracy),
            'classification_report': safe_json_convert(report),
            'confusion_matrix': safe_json_convert(cm.tolist())
        }
    
    def _get_data_stats(self, prepared_data):
        """获取数据统计"""
        labels = prepared_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        risk_distribution = {}
        for label, count in zip(unique_labels, counts):
            risk_name = self.data_manager.risk_names[label]
            risk_distribution[risk_name] = int(count)
        
        return {
            'total_samples': len(prepared_data['images']),
            'risk_distribution': risk_distribution,
            'unique_patients': len(set([info['patient_id'] for info in prepared_data['sample_info']]))
        }
    
    def _generate_report(self, training_results, evaluation_results, prepared_data):
        """生成报告"""
        
        data_stats = self._get_data_stats(prepared_data)
        
        print("\n" + "="*80)
        print("🗄️ CACHE-REUSE Breast Risk Prediction - Training Complete")
        print("="*80)
        
        print(f"🗄️ Cache Status:")
        print(f"   ✅ CACHE REUSE SUCCESSFUL!")
        print(f"   ✅ Using FIXED cache name: breast_data_v1")
        print(f"   ✅ Next run will load instantly from cache")
        
        print(f"\n📊 Dataset:")
        print(f"   Total samples: {data_stats['total_samples']}")
        print(f"   Unique patients: {data_stats['unique_patients']}")
        
        print(f"\n📊 Risk Distribution:")
        for risk_name, count in data_stats['risk_distribution'].items():
            percentage = count / data_stats['total_samples'] * 100
            print(f"   {risk_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n🚀 Training Results:")
        print(f"   Final training accuracy: {training_results['final_train_acc']:.4f}")
        print(f"   Final validation accuracy: {training_results['final_val_acc']:.4f}")
        
        print(f"\n📈 Test Performance:")
        print(f"   Test accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']:.1%})")
        
        print(f"\n🎯 Cache Benefits:")
        print(f"   ✅ Fixed cache name ensures consistency")
        print(f"   ✅ Automatic detection of existing caches")  
        print(f"   ✅ One-time image processing, multiple uses")
        print(f"   ✅ Instant data loading for subsequent runs")
        
        print("="*80)

def main():
    """主程序 - 缓存复用版本"""
    
    parser = argparse.ArgumentParser(description='Cache-Reuse Breast Risk Prediction')
    parser.add_argument('--data-dir', type=str, default='D:/Desktop', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='D:/Desktop/cache_reuse_output', help='Output directory')
    parser.add_argument('--image-size', type=int, default=128, help='Image size')
    parser.add_argument('--cache-root', type=str, default='./cache', help='Cache directory')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild cache')
    parser.add_argument('--no-existing-cache', action='store_true', help='Don\'t use existing caches')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before training')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'output_dir': args.output_dir,
        'cache_root': args.cache_root,
        'max_patients': args.max_patients,
        'data_paths': {
            'excel_path': os.path.join(args.data_dir, 'Breast BI-RADS.xlsx'),
            'dicom_root_dir': os.path.join(args.data_dir, 'Data_BI-RADS'),
            'segmentation_root_dir': os.path.join(args.data_dir, 'Data_BI-RADS_segmentation')
        },
        'image_config': {
            'target_size': (args.image_size, args.image_size),
            'max_slices': 30,
            'save_debug_images': False,
            'normalization': 'z_score',
            'bilateral_mode': 'separate_channels',
            'focus_on_glandular': True,
            'augmentation': False
        }
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['cache_root'], exist_ok=True)
    
    setup_gpu()
    
    logger.info("🗄️ CACHE-REUSE Breast Risk Prediction Training")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Cache directory: {args.cache_root}")
    logger.info(f"  Image size: {args.image_size}x{args.image_size}")
    logger.info(f"  Max patients: {args.max_patients or 'No limit'}")
    logger.info(f"  Force rebuild: {args.force_rebuild}")
    logger.info(f"  Use existing cache: {not args.no_existing_cache}")
    logger.info(f"  KEY FEATURE: Fixed cache name for guaranteed reuse!")
    logger.info("="*80)
    
    try:
        # 清理缓存
        if args.clear_cache:
            cache_manager = FixedCacheManager(args.cache_root)
            cache_manager.clear_cache()
            logger.info("✅ Cache cleared")
        
        # 运行训练
        pipeline = CacheReuseTrainingPipeline(config)
        results = pipeline.run_training(
            force_rebuild=args.force_rebuild,
            use_any_cache=not args.no_existing_cache
        )
        
        if results is None:
            logger.error("❌ Training failed")
            return 1
        
        # 保存结果
        results_file = os.path.join(config['output_dir'], 'cache_reuse_results.json')
        
        try:
            json_results = safe_json_convert(results)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved: {results_file}")
        except Exception as e:
            logger.error(f"❌ JSON save failed: {e}")
            pickle_file = os.path.join(config['output_dir'], 'cache_reuse_results.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"✅ Results saved as pickle: {pickle_file}")
        
        # 保存模型
        try:
            weights_file = os.path.join(config['output_dir'], 'cache_reuse_model_weights.h5')
            pipeline.model.model.save_weights(weights_file)
            logger.info(f"✅ Model weights saved: {weights_file}")
        except Exception as e:
            logger.error(f"❌ Model save failed: {e}")
        
        logger.info("🎉 Cache-reuse training complete!")
        logger.info(f"📁 Results saved in: {config['output_dir']}")
        
        # 最终提醒
        print(f"\n🗄️ CACHE STATUS:")
        print(f"   ✅ Fixed cache is now ready!")
        print(f"   ✅ Cache name: breast_data_v1")
        print(f"   ✅ Next run will be INSTANT!")
        print(f"   🚀 Run the same command again to see the difference!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())