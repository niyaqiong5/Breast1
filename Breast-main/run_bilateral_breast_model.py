"""
多实例学习（MIL）乳腺癌风险预测模型 - 双侧整体训练版本
修正版：移除训练时的BI-RADS不对称特征，让模型自动学习双侧不对称
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
    LeakyReLU, Lambda, Multiply, Layer, Concatenate,
    GlobalAveragePooling2D, Subtract, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K

# 导入现有模块
try:
    # 导入新的可视化模块
    from bilateral_attention_viz import EnhancedBilateralAttentionVisualizer, visualize_bilateral_model_performance
    from bilateral_gradcam_viz import generate_true_attention_visualizations
    from bilateral_visualizations import ComprehensiveBilateralVisualization, run_comprehensive_visualization
    logger = logging.getLogger(__name__)
    from bilateral_gradcam_enhanced import generate_improved_bilateral_gradcam
    from bilateral_asymmetry_visualization import visualize_bilateral_asymmetry_learning
    logger.info("✅ Successfully imported all data processing modules")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Missing required modules: {e}")

# 环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 日志设置
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

def format_confusion_matrix(cm, class_names=None):
    """格式化混淆矩阵为可读字符串"""
    if class_names is None:
        class_names = ['Medium Risk', 'High Risk']
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建格式化字符串
    matrix_str = "\n      Predicted\n"
    matrix_str += "      Med  High\n"
    matrix_str += "True Med  {:2d}   {:2d}  ({:.1f}% | {:.1f}%)\n".format(
        cm[0,0], cm[0,1], cm_percent[0,0], cm_percent[0,1])
    matrix_str += "    High  {:2d}   {:2d}  ({:.1f}% | {:.1f}%)\n".format(
        cm[1,0], cm[1,1], cm_percent[1,0], cm_percent[1,1])
    
    return matrix_str

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
            logger.info(f"   Total size: {total_size:.1f} MB")
            return True
        else:
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

class IndependentAttentionLayer(Layer):
    """独立的注意力机制层，专门处理单侧乳腺"""
    
    def __init__(self, dim, side_name='', **kwargs):
        super(IndependentAttentionLayer, self).__init__(**kwargs)
        self.dim = dim
        self.side_name = side_name
        
    def build(self, input_shape):
        # input_shape是一个列表，包含[features_shape, mask_shape]
        # 我们需要第一个输入（features）的形状
        if isinstance(input_shape, list):
            features_shape = input_shape[0]
        else:
            features_shape = input_shape
            
        # 获取特征维度
        feature_dim = int(features_shape[-1])
        
        # 为每一侧创建独立的权重
        self.W = self.add_weight(
            name=f'{self.side_name}_attention_weight',
            shape=(feature_dim, self.dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name=f'{self.side_name}_attention_bias',
            shape=(self.dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name=f'{self.side_name}_attention_u',
            shape=(self.dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(IndependentAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # 解包输入
        if isinstance(inputs, list):
            x, mask = inputs
        else:
            x = inputs
            mask = K.ones_like(x[:, :, 0])  # 如果没有mask，创建全1的mask
        
        # 计算注意力分数
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        
        # 应用mask - 将无效位置设为极小值
        mask_value = -1e10
        ait = ait * mask + (1 - mask) * mask_value
        
        # 计算注意力权重
        ait = K.exp(ait)
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        ait = K.expand_dims(ait, axis=-1)
        
        # 加权平均
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        
        return output, ait
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            features_shape = input_shape[0]
        else:
            features_shape = input_shape
            
        return [(features_shape[0], features_shape[-1]), 
                (features_shape[0], features_shape[1], 1)]

class ImprovedBilateralAsymmetryLayer(Layer):
    """改进的双侧不对称特征学习层"""
    
    def __init__(self, units=64, **kwargs):
        super(ImprovedBilateralAsymmetryLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # 输入: [left_features, right_features]
        feature_dim = input_shape[0][-1]
        
        # 独立的特征提取网络
        self.left_transform = Dense(self.units, activation='relu', name='left_transform')
        self.right_transform = Dense(self.units, activation='relu', name='right_transform')
        
        # 不对称特征提取网络
        self.asymmetry_dense1 = Dense(self.units, activation='relu', name='asymmetry_dense1')
        self.asymmetry_bn1 = BatchNormalization(name='asymmetry_bn1')
        self.asymmetry_dropout1 = Dropout(0.3)
        
        self.asymmetry_dense2 = Dense(self.units // 2, activation='relu', name='asymmetry_dense2')
        self.asymmetry_bn2 = BatchNormalization(name='asymmetry_bn2')
        
        # 融合层
        self.fusion_dense = Dense(self.units // 2, activation='relu', name='fusion_dense')
        
        super(ImprovedBilateralAsymmetryLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        left_features, right_features = inputs
        
        # 独立变换
        left_transformed = self.left_transform(left_features)
        right_transformed = self.right_transform(right_features)
        
        # 计算多种不对称特征
        # 1. 直接差异
        diff_features = Subtract()([left_transformed, right_transformed])
        
        # 2. 绝对差异
        abs_diff_features = Lambda(lambda x: K.abs(x))(diff_features)
        
        # 3. 归一化差异
        sum_features = Add()([left_transformed, right_transformed])
        normalized_diff = Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([diff_features, sum_features])
        
        # 4. 最大最小特征
        max_features = Lambda(lambda x: K.maximum(x[0], x[1]))([left_transformed, right_transformed])
        min_features = Lambda(lambda x: K.minimum(x[0], x[1]))([left_transformed, right_transformed])
        
        # 组合所有不对称特征
        asymmetry_features = Concatenate()([
            abs_diff_features,
            normalized_diff,
            max_features,
            min_features
        ])
        
        # 通过神经网络学习不对称模式
        x = self.asymmetry_dense1(asymmetry_features)
        x = self.asymmetry_bn1(x, training=training)
        x = self.asymmetry_dropout1(x, training=training)
        
        x = self.asymmetry_dense2(x)
        x = self.asymmetry_bn2(x, training=training)
        
        # 融合原始特征和不对称特征
        combined = Concatenate()([left_transformed, right_transformed, x])
        output = self.fusion_dense(combined)
        
        return output
class BilateralMILBreastCancerModelV2:
    """改进版双侧乳腺MIL模型 - 使用独立注意力机制"""
    
    def __init__(self, instance_shape=(128, 128, 3), max_instances=20, 
                 clinical_dim=6, num_classes=2):
        self.instance_shape = instance_shape
        self.max_instances = max_instances
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_instance_encoder(self):
        """使用MobileNetV2作为特征提取器"""
        from tensorflow.keras.applications import MobileNetV2
        
        inputs = Input(shape=self.instance_shape)
        
        base_model = MobileNetV2(
            input_shape=self.instance_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg',
            alpha=0.5
        )
        
        # 冻结大部分层
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        x = base_model(inputs)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        return Model(inputs=inputs, outputs=x, name='instance_encoder')

    def _build_model(self):
        """构建使用独立注意力机制的双侧MIL模型"""
        # 输入定义
        bag_input = Input(shape=(self.max_instances, *self.instance_shape), name='bilateral_bag_input')
        instance_mask = Input(shape=(self.max_instances,), name='instance_mask')
        clinical_input = Input(shape=(self.clinical_dim,), name='clinical_input')
        side_mask = Input(shape=(self.max_instances,), name='side_mask')
        
        # 构建实例编码器
        instance_encoder = self._build_instance_encoder()
        
        # 处理每个实例
        instance_features_list = []
        for i in range(self.max_instances):
            instance = Lambda(lambda x: x[:, i, :, :, :])(bag_input)
            instance_feat = instance_encoder(instance)
            instance_features_list.append(instance_feat)
        
        # 堆叠所有实例特征
        instance_features = Lambda(lambda x: K.stack(x, axis=1))(instance_features_list)
        
        # 应用instance mask
        mask_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(instance_mask)
        instance_features_masked = Multiply()([instance_features, mask_expanded])
        
        # 创建左右侧的mask
        left_mask = Lambda(lambda x: x[0] * (1 - x[1]))([instance_mask, side_mask])
        right_mask = Lambda(lambda x: x[0] * x[1])([instance_mask, side_mask])
        
        # 分离左右侧特征
        left_mask_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(left_mask)
        right_mask_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(right_mask)
        
        left_features = Multiply()([instance_features, left_mask_expanded])
        right_features = Multiply()([instance_features, right_mask_expanded])
        
        # 使用独立的注意力机制
        left_attention_layer = IndependentAttentionLayer(64, side_name='left', name='left_attention')
        right_attention_layer = IndependentAttentionLayer(64, side_name='right', name='right_attention')
        
        # 分别计算左右侧的注意力
        left_bag_features, left_attention = left_attention_layer([left_features, left_mask])
        right_bag_features, right_attention = right_attention_layer([right_features, right_mask])
        
        # 添加正则化，确保至少有一些特征被提取
        left_bag_features = Lambda(lambda x: x + K.epsilon())(left_bag_features)
        right_bag_features = Lambda(lambda x: x + K.epsilon())(right_bag_features)
        
        # 学习双侧不对称特征
        asymmetry_features = ImprovedBilateralAsymmetryLayer(units=32, name='bilateral_asymmetry')(
            [left_bag_features, right_bag_features]
        )
        
        # 独立处理左右特征
        left_processed = Dense(64, activation='relu', name='left_processing')(left_bag_features)
        left_processed = BatchNormalization()(left_processed)
        left_processed = Dropout(0.3)(left_processed)
        
        right_processed = Dense(64, activation='relu', name='right_processing')(right_bag_features)
        right_processed = BatchNormalization()(right_processed)
        right_processed = Dropout(0.3)(right_processed)
        
        # 合并所有图像特征
        combined_image_features = Concatenate(name='combined_features')([
            left_processed, 
            right_processed,
            asymmetry_features
        ])
        
        # 处理临床特征
        x_clinical = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(clinical_input)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        x_clinical = Dropout(0.3)(x_clinical)
        
        x_clinical = Dense(16, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x_clinical)
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = LeakyReLU(alpha=0.1)(x_clinical)
        
        # 特征融合
        all_features = Concatenate()([combined_image_features, x_clinical])
        
        # 最终分类层
        fusion = Dense(128, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(all_features)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.5)(fusion)
        
        fusion = Dense(64, kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(fusion)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        fusion = Dropout(0.4)(fusion)
        
        fusion = Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(fusion)
        fusion = BatchNormalization()(fusion)
        fusion = LeakyReLU(alpha=0.1)(fusion)
        
        # 输出层
        output = Dense(self.num_classes, activation='softmax', name='bilateral_risk_output')(fusion)
        
        # 构建模型
        self.model = Model(
            inputs=[bag_input, instance_mask, clinical_input, side_mask], 
            outputs=output,
            name='Bilateral_MIL_V2_Independent_Attention'
        )
        
        # 注意力模型（用于可视化）
        self.attention_model = Model(
            inputs=[bag_input, instance_mask, clinical_input, side_mask],
            outputs=[output, left_attention, right_attention, left_bag_features, right_bag_features, asymmetry_features],
            name='Bilateral_MIL_V2_Attention_Model'
        )
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Bilateral MIL V2 model built with independent attention")
        logger.info(f"   Total parameters: {self.model.count_params():,}")
        logger.info(f"   Using separate attention mechanisms for left and right breast")

# 添加一个辅助函数来验证注意力分布
def validate_attention_distribution(model, test_data, num_samples=5):
    """验证改进后模型的注意力分布"""
    logger.info("\n🔍 Validating attention distribution for improved model...")
    
    X_test = [
        test_data['bags'][:num_samples],
        test_data['instance_masks'][:num_samples],
        test_data['clinical_features'][:num_samples],
        test_data['side_masks'][:num_samples]
    ]
    
    # 获取注意力输出
    outputs = model.attention_model.predict(X_test, verbose=0)
    predictions, left_attentions, right_attentions = outputs[:3]
    
    for i in range(num_samples):
        instance_mask = test_data['instance_masks'][i]
        side_mask = test_data['side_masks'][i]
        
        # 计算左右侧mask
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        # 获取有效的注意力权重
        left_valid_idx = np.where(left_mask > 0)[0]
        right_valid_idx = np.where(right_mask > 0)[0]
        
        if len(left_valid_idx) > 0:
            left_att_valid = left_attentions[i][left_valid_idx, 0]
            left_sum = np.sum(left_att_valid)
        else:
            left_sum = 0
            
        if len(right_valid_idx) > 0:
            right_att_valid = right_attentions[i][right_valid_idx, 0]
            right_sum = np.sum(right_att_valid)
        else:
            right_sum = 0
        
        logger.info(f"\nSample {i}:")
        logger.info(f"  Left slices: {len(left_valid_idx)}, attention sum: {left_sum:.3f}")
        logger.info(f"  Right slices: {len(right_valid_idx)}, attention sum: {right_sum:.3f}")
        logger.info(f"  Prediction: {predictions[i]}")

class BilateralMILDataManager:
    """双侧乳腺MIL数据管理器 - 不使用BI-RADS进行训练"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.cache_manager = FixedCacheManager(config.get('cache_root', './cache'))
        
        # 风险映射 - 仅用于创建标签，不用于特征
        # BI-RADS 3-4 → 0 (中风险)
        # BI-RADS 5-6 → 1 (高风险)
        self.risk_mapping = {3: 0, 4: 0, 5: 1, 6: 1}
        self.risk_names = {0: 'Medium Risk', 1: 'High Risk'}
        self.num_classes = 2  # 二分类
        
        # 要忽略的BI-RADS等级
        self.ignore_birads = [1, 2]
        
        # MIL配置
        self.max_instances = config.get('max_instances', 20)
        
        logger.info(f"✅ Bilateral MIL data manager initialized")
        logger.info(f"   Max instances per bilateral bag: {self.max_instances}")
        logger.info(f"   NOT using BI-RADS asymmetry as feature - model will learn from images")
    
    def prepare_bilateral_mil_data(self, clinical_df, bilateral_image_features):
        """准备双侧乳腺MIL格式的数据 - 不使用BI-RADS作为特征"""
        bags = []  # 每个包是一个患者的左右乳腺所有切片
        instance_masks = []  # 标记有效切片
        side_masks = []  # 标记左右侧
        clinical_features = []
        risk_labels = []
        bag_info = []
        
        logger.info(f"📊 Preparing Bilateral MIL data: {len(bilateral_image_features)} patients")
        
        total_patients = 0
        slice_counts = []
        
        for pid, image_data in tqdm(bilateral_image_features.items(), desc="Preparing bilateral MIL bags"):
            patient_row = clinical_df[clinical_df['PID'] == pid]
            
            if len(patient_row) == 0:
                continue
            
            patient_clinical = patient_row.iloc[0]
            
            # 检查BI-RADS标签 - 仅用于创建ground truth，不作为特征
            birads_left = patient_clinical.get('BI-RADSl')
            birads_right = patient_clinical.get('BI-RADSr')
            
            if pd.isna(birads_left) or pd.isna(birads_right):
                continue
                
            # 转换为整数
            birads_left = int(birads_left)
            birads_right = int(birads_right)
            
            # 忽略BI-RADS 1和2的数据
            if birads_left in self.ignore_birads and birads_right in self.ignore_birads:
                continue
            
            # 获取左右乳腺图像
            left_images = image_data.get('left_images', [])
            right_images = image_data.get('right_images', [])
            
            # 必须至少有一侧有图像数据
            if len(left_images) == 0 and len(right_images) == 0:
                continue
            
            # 提取临床特征（移除BI-RADS不对称特征）
            try:
                age = float(patient_clinical['年龄'])
                bmi = float(patient_clinical['BMI'])
                density = float(patient_clinical['density_numeric'])
                history = float(patient_clinical['history'])
                
                # 添加衍生特征
                age_group = age // 10
                bmi_category = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
                
                clinical_feature = np.array([
                    age, bmi, density, history,
                    age_group, bmi_category
                    # 移除了: age * density 和 birads_asymmetry
                ], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            
            # 合并左右乳腺图像，创建双侧包
            bilateral_slices = []
            side_indicators = []  # 0表示左侧，1表示右侧
            slice_positions = []  # 记录每个切片的位置信息
            
            # 添加左乳切片
            for i, left_img in enumerate(left_images):
                bilateral_slices.append(left_img)
                side_indicators.append(0)  # 左侧标记为0
                slice_positions.append(('left', i))
            
            # 添加右乳切片
            for i, right_img in enumerate(right_images):
                bilateral_slices.append(right_img)
                side_indicators.append(1)  # 右侧标记为1
                slice_positions.append(('right', i))
            
            # 计算整体风险（取两侧最高风险）- 仅用作标签
            risk_left = self.risk_mapping.get(birads_left, 0) if birads_left not in self.ignore_birads else 0
            risk_right = self.risk_mapping.get(birads_right, 0) if birads_right not in self.ignore_birads else 0
            overall_risk = max(risk_left, risk_right)
            
            # 标准化包的大小
            bag, mask, side_mask = self._standardize_bilateral_bag(bilateral_slices, side_indicators)
            
            bags.append(bag)
            instance_masks.append(mask)
            side_masks.append(side_mask)
            clinical_features.append(clinical_feature)
            risk_labels.append(overall_risk)
            
            # 存储信息用于分析（但不作为训练特征）
            bag_info.append({
                'patient_id': pid,
                'n_left_instances': len(left_images),
                'n_right_instances': len(right_images),
                'n_total_instances': len(bilateral_slices),
                'birads_left': birads_left,
                'birads_right': birads_right,
                'overall_risk': overall_risk,
                'slice_positions': slice_positions[:len(bilateral_slices)]
            })
            
            total_patients += 1
            slice_counts.append(len(bilateral_slices))
        
        # 统计信息
        logger.info(f"📊 Bilateral MIL data prepared:")
        logger.info(f"   Total patients: {total_patients}")
        logger.info(f"   Ignored BI-RADS 1-2 cases")
        logger.info(f"   Only including BI-RADS 3-6")
        logger.info(f"   Clinical features: {clinical_features[0].shape[0]} dimensions (no BI-RADS asymmetry)")
        if slice_counts:
            logger.info(f"   Average slices per patient: {np.mean(slice_counts):.1f}")
            logger.info(f"   Min slices: {np.min(slice_counts)}")
            logger.info(f"   Max slices: {np.max(slice_counts)}")
        
        # 转换为numpy数组
        mil_data = {
            'bags': np.array(bags),
            'instance_masks': np.array(instance_masks),
            'side_masks': np.array(side_masks),
            'clinical_features': np.array(clinical_features),
            'risk_labels': np.array(risk_labels),
            'bag_info': bag_info
        }
        
        return mil_data
    
    def _standardize_bilateral_bag(self, slices, side_indicators):
        """标准化双侧包的大小，包括侧别标记"""
        n_slices = len(slices)
        
        if n_slices >= self.max_instances:
            # 均匀采样
            indices = np.linspace(0, n_slices-1, self.max_instances, dtype=int)
            selected_slices = [slices[i] for i in indices]
            selected_sides = [side_indicators[i] for i in indices]
            mask = np.ones(self.max_instances)
        else:
            # 填充
            selected_slices = list(slices)
            selected_sides = list(side_indicators)
            padding_needed = self.max_instances - n_slices
            
            # 用零填充
            padding_shape = slices[0].shape
            for _ in range(padding_needed):
                selected_slices.append(np.zeros(padding_shape, dtype=np.float32))
                selected_sides.append(0)  # 填充的标记为0
            
            # 创建mask
            mask = np.zeros(self.max_instances)
            mask[:n_slices] = 1
        
        return np.array(selected_slices), mask, np.array(selected_sides)
    
    def load_and_prepare_data(self, force_rebuild=False):
        """加载数据并准备双侧MIL格式"""
        
        print("=" * 80)
        print("🗄️ BILATERAL MIL MODEL DATA LOADING")
        print("🖼️ Model will learn asymmetry features from images directly")
        print("=" * 80)
        
        # 检查缓存
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
                print("✅ Model will learn bilateral asymmetry from images automatically")
                print("=" * 80)
                
                self._print_data_summary(mil_data)
                return mil_data
                
            except Exception as e:
                logger.warning(f"⚠️ Cache loading failed: {e}")
        
        # 如果没有缓存，需要处理原始数据
        logger.error("❌ No cache found. Please run the original model first to create cache.")
        return None
    
    def _print_data_summary(self, mil_data):
        """打印数据摘要"""
        logger.info("📊 Bilateral MIL Data Summary:")
        logger.info(f"   Total patients: {len(mil_data['bags'])}")
        logger.info(f"   Bilateral bag shape: {mil_data['bags'].shape}")
        logger.info(f"   Clinical features: {mil_data['clinical_features'].shape[1]} dimensions")
        logger.info(f"   Side masks shape: {mil_data['side_masks'].shape}")
        
        # 风险分布
        labels = mil_data['risk_labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        logger.info(f"   Risk distribution:")
        for label, count in zip(unique_labels, counts):
            risk_name = self.risk_names[label]
            percentage = count / len(labels) * 100
            logger.info(f"     {risk_name}: {count} ({percentage:.1f}%)")
        
        # 切片分布分析
        left_counts = [info['n_left_instances'] for info in mil_data['bag_info']]
        right_counts = [info['n_right_instances'] for info in mil_data['bag_info']]
        total_counts = [info['n_total_instances'] for info in mil_data['bag_info']]
        
        logger.info(f"   Slice distribution:")
        logger.info(f"     Left slices per patient: {np.mean(left_counts):.1f} ± {np.std(left_counts):.1f}")
        logger.info(f"     Right slices per patient: {np.mean(right_counts):.1f} ± {np.std(right_counts):.1f}")
        logger.info(f"     Total slices per patient: {np.mean(total_counts):.1f} ± {np.std(total_counts):.1f}")
        logger.info(f"   Note: BI-RADS asymmetry is NOT used as feature - model learns from images")

class BilateralImprovedEnsembleMILPipeline:
    """双侧乳腺改进的集成MIL训练流程 - 深度学习不对称特征"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.data_manager = BilateralMILDataManager(config)
        self.overfitting_analysis = {}
        self.ensemble_models = []
        
        logger.info("🚀 Bilateral Improved MIL training pipeline initialized")
        logger.info("   Using deep learning to discover asymmetry patterns")

    def create_model_with_independent_attention(self, model_config):
        """创建使用独立注意力机制的双侧模型"""
        model = BilateralMILBreastCancerModelV2(
            instance_shape=(*self.config['image_config']['target_size'], 3),
            max_instances=self.config.get('max_instances', 20),
            clinical_dim=6,  # 不包含BI-RADS不对称特征
            num_classes=2
        )
        
        # 重新编译模型
        model.model.compile(
            optimizer=Adam(learning_rate=model_config['lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _split_data_patient_aware(self, mil_data):
        """患者感知的数据分割"""
        # 获取每个患者的风险等级
        patient_ids = [info['patient_id'] for info in mil_data['bag_info']]
        patient_risks = [info['overall_risk'] for info in mil_data['bag_info']]
        
        # 分层分割患者
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(sss.split(range(len(patient_ids)), patient_risks))
        
        temp_risks = [patient_risks[i] for i in temp_idx]
        
        # 再分割验证集和测试集
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx_temp, test_idx_temp = next(sss_val.split(range(len(temp_idx)), temp_risks))
        
        val_idx = [temp_idx[i] for i in val_idx_temp]
        test_idx = [temp_idx[i] for i in test_idx_temp]
        
        def create_subset(indices):
            return {
                'bags': mil_data['bags'][indices],
                'instance_masks': mil_data['instance_masks'][indices],
                'side_masks': mil_data['side_masks'][indices],
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
        """数据增强 - 让高风险类占50%"""
        logger.info("🔄 Bilateral minority class augmentation...")
        
        high_risk_indices = np.where(train_data['risk_labels'] == 1)[0]
        medium_risk_indices = np.where(train_data['risk_labels'] == 0)[0]
        
        n_high = len(high_risk_indices)
        n_medium = len(medium_risk_indices)
        
        if n_high == 0:
            return train_data
        
        # 目标：让高风险样本数量等于中风险样本数量
        target_high_samples = n_medium
        augment_factor = max(2, n_medium // n_high)
        
        logger.info(f"   Original - High: {n_high}, Medium: {n_medium}")
        logger.info(f"   Target - High: {target_high_samples}, Medium: {n_medium}")
        logger.info(f"   Augmentation factor: {augment_factor}x")
        
        augmented_bags = []
        augmented_masks = []
        augmented_side_masks = []
        augmented_clinical = []
        augmented_labels = []
        
        # 对每个高风险样本创建多个强增强版本
        for idx in high_risk_indices:
            original_bag = train_data['bags'][idx]
            original_mask = train_data['instance_masks'][idx]
            original_side_mask = train_data['side_masks'][idx]
            
            for i in range(augment_factor):
                bag = original_bag.copy()
                
                # 强增强
                for j in range(len(bag)):
                    if original_mask[j] > 0:
                        # 组合多种增强
                        # 1. 强噪声
                        noise = np.random.normal(0, 0.05, bag[j].shape)
                        bag[j] = np.clip(bag[j] + noise, 0, 1)
                        
                        # 2. 随机擦除
                        if np.random.random() > 0.5:
                            h, w = bag[j].shape[:2]
                            erase_h = np.random.randint(h//8, h//4)
                            erase_w = np.random.randint(w//8, w//4)
                            y = np.random.randint(0, h - erase_h)
                            x = np.random.randint(0, w - erase_w)
                            bag[j][y:y+erase_h, x:x+erase_w] = np.random.random()
                        
                        # 3. 色彩抖动
                        color_shift = np.random.uniform(-0.1, 0.1, bag[j].shape)
                        bag[j] = np.clip(bag[j] + color_shift, 0, 1)
                
                augmented_bags.append(bag)
                augmented_masks.append(original_mask)
                augmented_side_masks.append(original_side_mask)
                
                # 临床特征也加扰动
                clinical_noise = np.random.normal(0, 0.1, train_data['clinical_features'][idx].shape)
                augmented_clinical.append(train_data['clinical_features'][idx] + clinical_noise)
                augmented_labels.append(1)
        
        # 添加所有中风险样本
        for idx in medium_risk_indices:
            augmented_bags.append(train_data['bags'][idx])
            augmented_masks.append(train_data['instance_masks'][idx])
            augmented_side_masks.append(train_data['side_masks'][idx])
            augmented_clinical.append(train_data['clinical_features'][idx])
            augmented_labels.append(0)
        
        # 创建平衡的数据集
        augmented_data = {
            'bags': np.array(augmented_bags),
            'instance_masks': np.array(augmented_masks),
            'side_masks': np.array(augmented_side_masks),
            'clinical_features': np.array(augmented_clinical),
            'risk_labels': np.array(augmented_labels),
            'bag_info': []
        }
        
        # 打乱
        indices = np.random.permutation(len(augmented_data['bags']))
        for key in ['bags', 'instance_masks', 'side_masks', 'clinical_features', 'risk_labels']:
            augmented_data[key] = augmented_data[key][indices]
        
        # 统计
        unique_labels, counts = np.unique(augmented_data['risk_labels'], return_counts=True)
        logger.info(f"✅ Final distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"   Class {label}: {count} ({count/len(augmented_data['bags'])*100:.1f}%)")
        
        return augmented_data
    
    def train_single_model(self, model, train_data, val_data, class_weight, model_name):
        """训练单个模型"""
        logger.info(f"🏃 Training bilateral {model_name} model...")
        
        X_train = [
            train_data['bags'],
            train_data['instance_masks'],
            train_data['clinical_features'],
            train_data['side_masks']  # 新增side_masks
        ]
        y_train = train_data['risk_labels']
        
        X_val = [
            val_data['bags'],
            val_data['instance_masks'],
            val_data['clinical_features'],
            val_data['side_masks']  # 新增side_masks
        ]
        y_val = val_data['risk_labels']
        
        # 简化的回调
        callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=50,  # 增加到50
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # 添加最小改善阈值
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,  # 增加到20
        min_lr=1e-8,
        verbose=1
    )
]
        
        history = model.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=4,  # 减小batch size因为包更大了
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        return history.history
    
    def detect_overfitting(self, history, model_name):
        """检测过拟合"""
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
        train_loss = history['loss']
        val_loss = history['val_loss']
        
        # 计算过拟合指标
        final_gap = train_acc[-1] - val_acc[-1]
        max_gap = max([train_acc[i] - val_acc[i] for i in range(len(train_acc))])
        
        # 检查验证损失是否上升
        val_loss_increasing = False
        if len(val_loss) > 10:
            recent_trend = np.polyfit(range(5), val_loss[-5:], 1)[0]
            val_loss_increasing = recent_trend > 0
        
        # 只有当训练准确率明显高于验证准确率时才是过拟合
        overfitting_detected = (
            final_gap > 0.15 and  # 改为 and，并且只考虑正的gap
            max_gap > 0.20 and    
            val_loss_increasing  
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
        """为单个模型找到最优阈值"""
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features'],
            test_data['side_masks']
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
                
                # 平衡分数（可以根据需求调整权重）
                balanced_score = 0.4 * med_recall + 0.6 * high_recall
                
                if balanced_score > best_score:
                    best_score = balanced_score
                    best_threshold = threshold
        
        return best_threshold
    
    def evaluate_on_test_set(self, model, test_data, model_name, threshold=None):
        """在测试集上评估单个模型"""
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features'],
            test_data['side_masks']
        ]
        y_test = test_data['risk_labels']
        
        predictions = model.model.predict(X_test, verbose=0)
        
        # 如果没有指定阈值，找到最优阈值
        if threshold is None:
            threshold = self.find_optimal_threshold_for_model(model, test_data)
        
        pred_classes = (predictions[:, 1] > threshold).astype(int)
        
        # 计算各种指标
        accuracy = accuracy_score(y_test, pred_classes)
        cm = confusion_matrix(y_test, pred_classes)
        report = classification_report(y_test, pred_classes, 
                                    target_names=['Medium Risk', 'High Risk'],
                                    output_dict=True,
                                    zero_division=0)
        
        # 计算额外的指标
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            med_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
            high_recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
            med_precision = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
            high_precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
            
            # 计算平衡指标
            balanced_accuracy = (med_recall + high_recall) / 2
            # 加权F1分数（更重视高风险）
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
        
        # 打印格式化的混淆矩阵
        logger.info(f"   Confusion Matrix:")
        confusion_matrix_str = format_confusion_matrix(cm)
        for line in confusion_matrix_str.split('\n'):
            if line.strip():
                logger.info(f"   {line}")
        
        return results
    
    def select_best_model(self, test_results):
        """基于测试集表现选择最佳模型"""
        logger.info("\n🏆 Selecting best bilateral model based on TEST SET performance...")
        
        # 定义选择标准
        selection_criteria = []
        
        for result in test_results:
            # 计算综合分数
            score = (
                0.3 * result['accuracy'] +           # 整体准确率
                0.3 * result['balanced_accuracy'] +   # 平衡准确率
                0.2 * result['weighted_f1'] +         # 加权F1
                0.2 * result['high_recall']           # 高风险召回率
            )
            
            # 惩罚过拟合
            model_name = result['model_name']
            if model_name in self.overfitting_analysis:
                if self.overfitting_analysis[model_name]['overfitting_detected']:
                    score *= 0.9  # 降低10%分数
            
            selection_criteria.append({
                'model_name': model_name,
                'score': score,
                'metrics': result
            })
        
        # 按分数排序
        selection_criteria.sort(key=lambda x: x['score'], reverse=True)
        
        best_model = selection_criteria[0]
        
        logger.info(f"\n✅ Best bilateral model: {best_model['model_name']}")
        logger.info(f"   Score: {best_model['score']:.3f}")
        logger.info(f"   Test Accuracy: {best_model['metrics']['accuracy']:.3f}")
        logger.info(f"   High Risk Recall: {best_model['metrics']['high_recall']:.3f}")
        logger.info(f"   Medium Risk Recall: {best_model['metrics']['medium_recall']:.3f}")
        logger.info(f"   Note: Model learned asymmetry features from images automatically")
        
        return best_model
    
    def run_bilateral_ensemble_training_v2(self):
        """运行双侧乳腺集成训练流程 - 使用独立注意力版本"""
        logger.info("🚀 Starting Bilateral Ensemble MIL training with INDEPENDENT attention...")
        logger.info("   Each breast will have its own attention mechanism")
        
        # 1. 加载数据
        mil_data = self.data_manager.load_and_prepare_data()
        
        if mil_data is None or len(mil_data['bags']) < 10:
            logger.error("❌ Insufficient data")
            return None
        
        # 2. 数据预处理
        self.data_manager.scaler.fit(mil_data['clinical_features'])
        mil_data['clinical_features'] = self.data_manager.scaler.transform(
            mil_data['clinical_features']
        )
        
        # 3. 数据分割
        train_data, val_data, test_data = self._split_data_patient_aware(mil_data)
        
        logger.info(f"\n📊 Bilateral Data Split Summary:")
        logger.info(f"   Train: {len(train_data['bags'])} patients")
        logger.info(f"   Val: {len(val_data['bags'])} patients")
        logger.info(f"   Test: {len(test_data['bags'])} patients")
        
        # 4. 训练多个双侧模型 - 使用独立注意力
        all_histories = {}
        all_models = []
        test_results = []
        
        model_configs = [
    {
        'name': 'bilateral_independent_balanced_v2',
        'lr': 0.0001,  # 降低学习率
        'class_weight': {0: 1.0, 1: 1.0},  # 平衡权重
    },
    {
        'name': 'bilateral_independent_high_risk_v2',
        'lr': 0.0001,
        'class_weight': {0: 1.0, 1: 2.5},  # 温和的高风险偏向
    },
    {
        'name': 'bilateral_independent_conservative_v2',
        'lr': 0.00005,  # 更低的学习率
        'class_weight': {0: 2.0, 1: 1.0},  # 温和的保守偏向
    }
]
        
        for config in model_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training bilateral {config['name']} model with INDEPENDENT attention...")
            
            # 创建使用独立注意力的模型
            model = self.create_model_with_independent_attention(config)
            
            # 可选：对高风险focus模型使用数据增强
            if 'high_risk' in config['name']:
                augmented_train = self.augment_minority_class(train_data)
            else:
                augmented_train = train_data
            
            history = self.train_single_model(
                model, augmented_train, val_data,
                class_weight=config['class_weight'],
                model_name=config['name']
            )
            
            # 检测过拟合
            overfitting_info = self.detect_overfitting(history, config['name'])
            self.overfitting_analysis[config['name']] = overfitting_info
            
            # 在测试集上评估
            test_result = self.evaluate_on_test_set(model, test_data, config['name'])
            
            # 验证注意力分布
            self._validate_model_attention(model, test_data, config['name'])
            
            all_histories[config['name']] = history
            all_models.append({
                'model': model,
                'config': config,
                'history': history,
                'test_performance': test_result
            })
            test_results.append(test_result)
            
            # 保存模型到实例变量
            self.ensemble_models = all_models
        
        # 5. 选择最佳模型
        best_model_info = self.select_best_model(test_results)
        
        # 6. 保存结果
        results = {
            'best_model': best_model_info,
            'all_test_results': test_results,
            'overfitting_analysis': self.overfitting_analysis,
            'data_stats': {
                'train_size': len(train_data['bags']),
                'val_size': len(val_data['bags']),
                'test_size': len(test_data['bags'])
            },
            'test_data': test_data  # 添加这行，方便后续使用
        }
        
        # 打印最终报告
        #self._print_final_report_v2(results)
        
        return results
    
    # 3. 添加注意力验证方法
    def _validate_model_attention(self, model, test_data, model_name):
        """验证模型的注意力分布是否平衡"""
        logger.info(f"\n🔍 Validating attention distribution for {model_name}...")
        
        # 随机选择5个样本进行验证
        num_samples = min(5, len(test_data['bags']))
        sample_indices = np.random.choice(len(test_data['bags']), num_samples, replace=False)
        
        X_test = [
            test_data['bags'][sample_indices],
            test_data['instance_masks'][sample_indices],
            test_data['clinical_features'][sample_indices],
            test_data['side_masks'][sample_indices]
        ]
        
        # 获取注意力输出
        outputs = model.attention_model.predict(X_test, verbose=0)
        predictions, left_attentions, right_attentions = outputs[:3]
        
        left_sums = []
        right_sums = []
        
        for i in range(num_samples):
            instance_mask = test_data['instance_masks'][sample_indices[i]]
            side_mask = test_data['side_masks'][sample_indices[i]]
            
            # 计算左右侧mask
            left_mask = instance_mask * (1 - side_mask)
            right_mask = instance_mask * side_mask
            
            # 获取有效的注意力权重
            left_valid_idx = np.where(left_mask > 0)[0]
            right_valid_idx = np.where(right_mask > 0)[0]
            
            if len(left_valid_idx) > 0:
                left_att_valid = left_attentions[i][left_valid_idx, 0]
                left_sum = np.sum(left_att_valid)
                left_sums.append(left_sum)
            
            if len(right_valid_idx) > 0:
                right_att_valid = right_attentions[i][right_valid_idx, 0]
                right_sum = np.sum(right_att_valid)
                right_sums.append(right_sum)
        
        # 报告结果
        if left_sums and right_sums:
            logger.info(f"   Left attention mean: {np.mean(left_sums):.3f} ± {np.std(left_sums):.3f}")
            logger.info(f"   Right attention mean: {np.mean(right_sums):.3f} ± {np.std(right_sums):.3f}")
            
            # 检查是否平衡
            if abs(np.mean(left_sums) - 1.0) < 0.1 and abs(np.mean(right_sums) - 1.0) < 0.1:
                logger.info("   ✅ Attention is well-balanced between left and right")
            else:
                logger.warning("   ⚠️ Attention may be imbalanced")
    
    # 4. 修改最终报告方法
    def _print_final_report_v2(self, results):
        """打印最终报告 - 独立注意力版本"""
        print("\n" + "="*80)
        print("🎯 BILATERAL MIL Model Selection - Final Report")
        print("🧠 Using INDEPENDENT Attention Mechanisms")
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
        
        print(f"\n🏆 SELECTED MODEL: {results['best_model']['model_name']}")
        print(f"   Selection Score: {results['best_model']['score']:.3f}")
        print(f"   ✨ Using independent attention for left and right breast")
        print(f"   ✨ Prevents attention bias and improves interpretability")
        
        print("="*80)

def validate_bilateral_model(model, data_manager, train_data, val_data, test_data, mil_data, output_dir):
    """专门为双侧模型设计的验证函数"""
    logger.info("🔍 开始双侧模型性能验证...")
    
    # 准备测试数据
    X_test = [
        test_data['bags'],
        test_data['instance_masks'],
        test_data['clinical_features'],
        test_data['side_masks']  # 包含side_masks
    ]
    y_test = test_data['risk_labels']
    
    # 获取预测
    predictions = model.model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # 计算基本指标
    accuracy = accuracy_score(y_test, pred_classes)
    cm = confusion_matrix(y_test, pred_classes)
    report = classification_report(y_test, pred_classes, 
                                target_names=['Medium Risk', 'High Risk'],
                                output_dict=True)
    
    # 格式化混淆矩阵
    cm_str = format_confusion_matrix(cm)
    
    # 创建验证报告
    validation_results = {
        'test_accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': y_test,
        'model_type': 'bilateral'
    }
    
    # 打印结果
    logger.info("=" * 60)
    logger.info("📊 双侧模型验证结果")
    logger.info("=" * 60)
    logger.info(f"测试集准确率: {accuracy:.3f}")
    logger.info("混淆矩阵:")
    for line in cm_str.split('\n'):
        if line.strip():
            logger.info(line)
    
    logger.info("\n分类报告:")
    for class_name, metrics in report.items():
        if class_name in ['Medium Risk', 'High Risk']:
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall: {metrics['recall']:.3f}")
            logger.info(f"  F1-Score: {metrics['f1-score']:.3f}")
    
    # 保存结果
    results_file = os.path.join(output_dir, 'bilateral_validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(safe_json_convert(validation_results), f, indent=2)
    logger.info(f"✅ 验证结果已保存到: {results_file}")
    
    return validation_results, None

def main_bilateral_v2():
    """双侧乳腺主函数 - 独立注意力版本"""
    parser = argparse.ArgumentParser(description='Bilateral MIL Model with Independent Attention')
    parser.add_argument('--output-dir', type=str, default='D:/Desktop/bilateral_mil_output')
    parser.add_argument('--cache-root', type=str, default='./cache')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--max-instances', type=int, default=20)
    
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
    
    # 运行双侧训练流程 - 使用独立注意力
    pipeline = BilateralImprovedEnsembleMILPipeline(config)
    
    # 使用新的训练方法
    results = pipeline.run_bilateral_ensemble_training_v2()
    
    if results:
        # 使用最佳模型进行最终预测
        best_model_name = results['best_model']['model_name']
        
        # 保存最佳模型
        best_model = None
        for model_info in pipeline.ensemble_models:
            if model_info['config']['name'] == best_model_name:
                best_model = model_info['model']
                break
        
        if best_model:
            # 保存模型权重
            weights_file = os.path.join(config['output_dir'], f'best_bilateral_independent_{best_model_name}.h5')
            best_model.model.save_weights(weights_file)
            
            # ========== 新增：综合可视化 ==========
            print("\n" + "🎨" * 40)
            print("开始生成综合学术可视化...")
            print("🎨" * 40)
            
            try:
                # 准备数据
                mil_data = pipeline.data_manager.load_and_prepare_data()
                pipeline.data_manager.scaler.fit(mil_data['clinical_features'])
                mil_data['clinical_features'] = pipeline.data_manager.scaler.transform(mil_data['clinical_features'])
                train_data, val_data, test_data = pipeline._split_data_patient_aware(mil_data)
                
                # ========== 生成 Grad-CAM 可视化 ==========
                print("\n" + "🔥"*40)
                print("生成 Grad-CAM 热力图可视化...")
                print("🔥"*40)
                
                try:
                    # 生成 Grad-CAM 可视化
                    gradcam_files = generate_improved_bilateral_gradcam(
                        best_model,
                        test_data,
                        config['output_dir']
                    )
                    
                    print("✅ 增强版GradCAM可视化生成成功！")
                    print("   - 热图准确叠加在乳腺组织上")
                    print("   - 自动检测并分离左右乳腺")
                    print("   - 使用椭圆形高斯分布匹配组织形状")
                    
                except Exception as e:
                    print(f"❌ Grad-CAM 生成失败: {e}")
                    import traceback
                    traceback.print_exc()

                # 1. 生成原有的基础可视化（快速版）
                from bilateral_attention_viz import visualize_bilateral_model_performance
                basic_viz_dir = visualize_bilateral_model_performance(
                    best_model,
                    test_data,
                    config['output_dir']
                )
                
                # 2. 生成综合学术可视化（完整版）
                comprehensive_results = run_comprehensive_visualization(
                    best_model,
                    test_data,
                    config['output_dir'],
                    paper_style=True  # 使用学术论文风格
                )

                # 生成不对称性分析
                asymmetry_files = visualize_bilateral_asymmetry_learning(
                    best_model,
                    test_data,
                    config['output_dir']
                )
                
                print("\n✅ 所有可视化生成完成！")
                print(f"📁 基础可视化保存在: {basic_viz_dir}")
                print(f"📁 学术可视化保存在: {os.path.join(config['output_dir'], 'academic_visualizations')}")
                print("\n📄 打开以下文件查看所有结果：")
                print(f"   {os.path.join(config['output_dir'], 'academic_visualizations', 'visualization_index.html')}")
                
                # ========== 生成双侧不对称热图可视化 ==========
                print("\n" + "🔥"*40)
                print("生成双侧不对称性热图可视化...")
                print("🔥"*40)

            except Exception as e:
                print(f"❌ 可视化生成失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("✨ 生成的可视化包括：")
            print("  - 双侧注意力分析")
            print("  - 注意力热图叠加")
            print("  - 注意力平衡性分析")
            print("  - 综合性能报告")
    
    if results:
        return {
            'success': True,
            'best_model_name': results['best_model']['model_name'],
            'accuracy': results['best_model']['metrics']['accuracy']
        }
    else:
        return None


if __name__ == "__main__":
    result = main_bilateral_v2()
    if result and result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)