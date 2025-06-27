"""
数据处理脚本 - 适配训练文件
集成现有的数据处理逻辑，生成训练所需的数据格式
保持与原代码相同的数据预处理逻辑
"""

import os
import sys
import argparse
import logging
import json
import pickle
import gzip
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy.ndimage import binary_erosion, binary_dilation, center_of_mass, label, distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns

# 设置环境变量
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入现有模块
try:
    from data_loader import ClinicalDataLoader, DicomLoader, SegmentationLoader
    from image_features import BilateralImageProcessor
    logger.info("✅ 成功导入现有数据处理模块")
except ImportError as e:
    logger.error(f"❌ 缺少必需的模块: {e}")
    logger.error("请确保 data_loader.py 和 image_features.py 文件在同一目录")
    sys.exit(1)

def safe_json_convert(obj):
    """安全的JSON转换函数，处理numpy和pandas类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: safe_json_convert(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    else:
        return obj

class CacheManager:
    """缓存管理器 - 使用固定缓存名称"""
    
    def __init__(self, cache_root='./cache'):
        self.cache_root = cache_root
        self.cache_dir = os.path.join(cache_root, 'optimized_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 使用固定的缓存名称
        self.cache_name = "breast_data_v1"
        
        logger.info(f"🗄️ 缓存管理器初始化")
        logger.info(f"   缓存目录: {self.cache_dir}")
        logger.info(f"   缓存名称: {self.cache_name}")
    
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
            logger.info(f"🎯 发现缓存!")
            logger.info(f"   总大小: {total_size:.1f} MB")
            return True
        else:
            return False
    
    def save_cache(self, clinical_df, bilateral_image_features, processing_config):
        """保存缓存数据"""
        logger.info("💾 保存数据到缓存...")
        
        cache_files = self.get_cache_files()
        
        try:
            # 保存临床数据
            with gzip.open(cache_files['clinical'], 'wb') as f:
                pickle.dump(clinical_df, f)
            logger.info(f"✅ 临床数据已保存: {clinical_df.shape}")
            
            # 保存图像数据
            with h5py.File(cache_files['images'], 'w') as hf:
                patient_count = len(bilateral_image_features)
                logger.info(f"📊 保存 {patient_count} 个患者的图像数据...")
                
                for pid, data in tqdm(bilateral_image_features.items(), desc="保存图像缓存"):
                    patient_group = hf.create_group(pid)
                    
                    # 保存左右乳房图像
                    if 'left_images' in data:
                        patient_group.create_dataset('left_images', data=data['left_images'])
                    if 'right_images' in data:
                        patient_group.create_dataset('right_images', data=data['right_images'])
            
            logger.info(f"✅ 图像数据已保存: {len(bilateral_image_features)} 个患者")
            
            # 保存映射数据
            mapping_data = {
                'processing_config': processing_config
            }
            
            with gzip.open(cache_files['mapping'], 'wb') as f:
                pickle.dump(mapping_data, f)
            logger.info(f"✅ 映射数据已保存")
            
            total_size = sum(os.path.getsize(f) for f in cache_files.values()) / (1024*1024)
            logger.info(f"🎉 缓存保存完成! 总大小: {total_size:.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ 缓存保存失败: {e}")
            # 清理可能的不完整文件
            for file_path in cache_files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            raise
    
    def load_cache(self):
        """加载缓存数据"""
        logger.info("📂 从缓存加载数据...")
        
        cache_files = self.get_cache_files()
        
        try:
            # 加载临床数据
            with gzip.open(cache_files['clinical'], 'rb') as f:
                clinical_df = pickle.load(f)
            logger.info(f"✅ 临床数据已加载: {clinical_df.shape}")
            
            # 加载图像数据
            bilateral_image_features = {}
            with h5py.File(cache_files['images'], 'r') as hf:
                patient_count = len(hf.keys())
                logger.info(f"📊 加载 {patient_count} 个患者的图像数据...")
                
                for pid in tqdm(hf.keys(), desc="加载图像缓存"):
                    patient_group = hf[pid]
                    image_data = {}
                    
                    # 加载左右乳房图像
                    if 'left_images' in patient_group:
                        image_data['left_images'] = patient_group['left_images'][:]
                    if 'right_images' in patient_group:
                        image_data['right_images'] = patient_group['right_images'][:]
                    
                    bilateral_image_features[pid] = image_data
            
            logger.info(f"✅ 图像数据已加载: {len(bilateral_image_features)} 个患者")
            
            # 加载映射数据
            with gzip.open(cache_files['mapping'], 'rb') as f:
                mapping_data = pickle.load(f)
            logger.info(f"✅ 映射数据已加载")
            
            # 重构数据
            cached_data = {
                'clinical_features': clinical_df,
                'bilateral_image_features': bilateral_image_features,
                'processing_config': mapping_data.get('processing_config', {})
            }
            
            return cached_data
            
        except Exception as e:
            logger.error(f"❌ 缓存加载失败: {e}")
            raise

class DataProcessor:
    """数据处理器 - 保持原有逻辑"""
    
    def __init__(self, config):
        self.config = config
        self.cache_manager = CacheManager(config.get('cache_root', './cache'))
        
        # 数据路径
        self.data_paths = config['data_paths']
        
        # 图像处理配置
        self.image_config = config['image_config']
        
        # 输出目录
        self.output_dir = config.get('output_dir', './processed_data')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("🚀 数据处理器初始化完成")
    
    def load_and_process_data(self, force_rebuild=False):
        """加载和处理所有数据"""
        
        print("=" * 80)
        print("🗄️ 数据加载和处理")
        print("=" * 80)
        
        # 检查缓存
        if self.cache_manager.cache_exists() and not force_rebuild:
            logger.info("🎯 发现缓存! 直接加载...")
            try:
                cached_data = self.cache_manager.load_cache()
                
                print("=" * 80)
                print("🎉 成功! 从缓存加载数据")
                print(f"✅ 加载 {len(cached_data['bilateral_image_features'])} 个患者的数据")
                print("=" * 80)
                
                self._print_data_summary(cached_data)
                return cached_data
                
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")
                logger.info("🔄 将重新处理数据...")
        
        # 重新处理数据
        logger.info("🔄 开始完整的数据处理流程...")
        
        # 1. 加载临床数据
        logger.info("📊 加载临床数据...")
        clinical_loader = ClinicalDataLoader(self.data_paths['excel_path'])
        clinical_df = clinical_loader.load_data()
        
        # 检查必需的列
        required_columns = ['PID', 'BI-RADSl', 'BI-RADSr', '年龄', 'BMI', 'density', 'history']
        missing_columns = [col for col in required_columns if col not in clinical_df.columns]
        if missing_columns:
            raise ValueError(f"临床数据缺少必需的列: {missing_columns}")
        
        logger.info(f"临床数据加载完成: {len(clinical_df)} 条记录")
        
        # 2. 检查和清理BI-RADS数据
        clinical_df = self._check_and_clean_birads_data(clinical_df)
        
        # 3. 加载和处理图像数据
        logger.info("🖼️ 加载和处理图像数据...")
        bilateral_image_features = self._load_and_process_images(clinical_df)
        
        # 4. 保存到缓存
        self.cache_manager.save_cache(clinical_df, bilateral_image_features, self.image_config)
        
        # 5. 生成数据质量报告
        self._generate_data_quality_report(clinical_df, bilateral_image_features)
        
        cached_data = {
            'clinical_features': clinical_df,
            'bilateral_image_features': bilateral_image_features,
            'processing_config': self.image_config
        }
        
        print("=" * 80)
        print("🎉 数据处理完成并保存到缓存")
        print(f"✅ 处理 {len(bilateral_image_features)} 个患者的数据")
        print("=" * 80)
        
        self._print_data_summary(cached_data)
        
        return cached_data
    
    def _check_and_clean_birads_data(self, clinical_df):
        """检查和清理BI-RADS数据 - 保持原有逻辑"""
        
        logger.info("🔍 检查BI-RADS数据质量...")
        
        original_count = len(clinical_df)
        
        # 检查BI-RADS值的有效性
        valid_birads = [1, 2, 3, 4, 5, 6]
        
        # 过滤无效的BI-RADS值
        valid_mask = (
            clinical_df['BI-RADSl'].isin(valid_birads) & 
            clinical_df['BI-RADSr'].isin(valid_birads)
        )
        
        clinical_df = clinical_df[valid_mask].copy()
        
        filtered_count = len(clinical_df)
        removed_count = original_count - filtered_count
        
        logger.info(f"BI-RADS数据检查完成:")
        logger.info(f"  原始记录: {original_count}")
        logger.info(f"  有效记录: {filtered_count}")
        logger.info(f"  移除记录: {removed_count}")
        
        # 统计BI-RADS分布
        logger.info("BI-RADS分布统计:")
        logger.info(f"  左乳房: {clinical_df['BI-RADSl'].value_counts().sort_index().to_dict()}")
        logger.info(f"  右乳房: {clinical_df['BI-RADSr'].value_counts().sort_index().to_dict()}")
        
        return clinical_df
    
    def _load_and_process_images(self, clinical_df):
        """加载和处理图像数据 - 保持原有逻辑"""
        
        # 创建图像处理器 - 保持原有配置
        image_config = self.image_config.copy()
        image_config['bilateral_mode'] = 'separate_channels'  # 确保分离模式
        
        bilateral_processor = BilateralImageProcessor(image_config)
        
        # 加载DICOM和分割数据
        dicom_loader = DicomLoader(self.data_paths['dicom_root_dir'])
        segmentation_loader = SegmentationLoader(self.data_paths['segmentation_root_dir'])
        
        # 收集患者数据
        patient_data = {}
        successful_patients = 0
        
        unique_pids = clinical_df['PID'].unique()
        logger.info(f"开始处理 {len(unique_pids)} 个患者的图像数据...")
        
        for pid in tqdm(unique_pids, desc="加载患者图像数据"):
            try:
                dicom_data = dicom_loader.load_patient_dicom(pid)
                segmentation_data = segmentation_loader.load_patient_segmentation(pid)
                
                if dicom_data is not None and segmentation_data is not None:
                    patient_data[pid] = (dicom_data, segmentation_data)
                    successful_patients += 1
                    
                    # 内存管理
                    if successful_patients % 10 == 0:
                        import gc
                        gc.collect()
                        
            except Exception as e:
                logger.warning(f"患者 {pid} 图像数据加载失败: {e}")
                continue
        
        logger.info(f"成功加载 {successful_patients} 个患者的图像数据")
        
        if successful_patients == 0:
            raise ValueError("没有成功加载任何患者的图像数据")
        
        # 批量处理图像 - 保持原有逻辑
        logger.info("🔄 批量处理双侧图像...")
        processed_bilateral_data = bilateral_processor.batch_process_bilateral(patient_data)
        
        # 提取2D切片 - 保持原有逻辑
        bilateral_slices_data = bilateral_processor.extract_bilateral_2d_slices(processed_bilateral_data)
        
        # 构建图像特征字典 - 保持原有格式
        bilateral_image_features = {}
        for i, pid in enumerate(bilateral_slices_data['slice_to_patient']):
            if pid not in bilateral_image_features:
                bilateral_image_features[pid] = {
                    'bilateral_images': [],
                    'left_images': [],
                    'right_images': []
                }
            bilateral_image_features[pid]['bilateral_images'].append(bilateral_slices_data['bilateral_slices'][i])
            bilateral_image_features[pid]['left_images'].append(bilateral_slices_data['left_slices'][i])
            bilateral_image_features[pid]['right_images'].append(bilateral_slices_data['right_slices'][i])
        
        logger.info(f"图像处理完成，共处理 {len(bilateral_image_features)} 个患者")
        
        return bilateral_image_features
    
    def _generate_data_quality_report(self, clinical_df, bilateral_image_features):
        """生成数据质量报告"""
        
        logger.info("📋 生成数据质量报告...")
        
        report = {
            'clinical_data': {
                'total_patients': int(len(clinical_df)),
                'age_range': [float(clinical_df['年龄'].min()), float(clinical_df['年龄'].max())],
                'age_mean': float(clinical_df['年龄'].mean()),
                'bmi_range': [float(clinical_df['BMI'].min()), float(clinical_df['BMI'].max())],
                'bmi_mean': float(clinical_df['BMI'].mean()),
                'density_distribution': {k: int(v) for k, v in clinical_df['density'].value_counts().to_dict().items()},
                'history_distribution': {k: int(v) for k, v in clinical_df['history'].value_counts().to_dict().items()},
                'birads_left_distribution': {int(k): int(v) for k, v in clinical_df['BI-RADSl'].value_counts().sort_index().to_dict().items()},
                'birads_right_distribution': {int(k): int(v) for k, v in clinical_df['BI-RADSr'].value_counts().sort_index().to_dict().items()}
            },
            'image_data': {
                'patients_with_images': int(len(bilateral_image_features)),
                'total_slices': int(sum(len(data['bilateral_images']) for data in bilateral_image_features.values()))
            }
        }
        
        # 匹配统计
        clinical_pids = set(clinical_df['PID'].astype(str))
        image_pids = set(bilateral_image_features.keys())
        common_pids = clinical_pids.intersection(image_pids)
        
        report['data_matching'] = {
            'clinical_only': int(len(clinical_pids - image_pids)),
            'image_only': int(len(image_pids - clinical_pids)),
            'both_available': int(len(common_pids)),
            'coverage_rate': float(len(common_pids) / len(clinical_pids) if clinical_pids else 0)
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'data_quality_report.json')
        
        # 使用safe_json_convert确保所有数据都能序列化
        safe_report = safe_json_convert(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(safe_report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"数据质量报告已保存: {report_path}")
        logger.info(f"数据覆盖率: {report['data_matching']['coverage_rate']:.2%}")
    
    def _print_data_summary(self, data):
        """打印数据摘要"""
        clinical_df = data['clinical_features']
        bilateral_image_features = data['bilateral_image_features']
        
        logger.info("📊 数据摘要:")
        logger.info(f"   临床数据: {len(clinical_df)} 个患者")
        logger.info(f"   图像数据: {len(bilateral_image_features)} 个患者")
        
        # BI-RADS分布
        logger.info("   BI-RADS分布:")
        for side, col in [('左乳房', 'BI-RADSl'), ('右乳房', 'BI-RADSr')]:
            dist = clinical_df[col].value_counts().sort_index()
            logger.info(f"     {side}: {dist.to_dict()}")
        
        # 图像切片统计
        total_slices = sum(len(data['bilateral_images']) for data in bilateral_image_features.values())
        avg_slices = total_slices / len(bilateral_image_features) if bilateral_image_features else 0
        logger.info(f"   总切片数: {total_slices}")
        logger.info(f"   平均切片数: {avg_slices:.1f}")

def create_training_data_format(processed_data, output_dir):
    """创建训练数据格式"""
    
    logger.info("📝 创建训练数据格式...")
    
    clinical_df = processed_data['clinical_features']
    bilateral_image_features = processed_data['bilateral_image_features']
    
    # 匹配临床和图像数据
    clinical_pids = set(clinical_df['PID'].astype(str))
    image_pids = set(bilateral_image_features.keys())
    common_pids = list(clinical_pids.intersection(image_pids))
    
    logger.info(f"共同患者数: {len(common_pids)}")
    
    # 创建训练数据格式的说明文档
    readme_content = f"""
# 训练数据格式说明

## 数据结构
- 临床数据: {len(clinical_df)} 个患者
- 图像数据: {len(bilateral_image_features)} 个患者  
- 匹配患者: {len(common_pids)} 个患者

## 数据访问方式

### 1. 临床数据 (clinical_features)
```python
clinical_df = processed_data['clinical_features']
# 包含列: PID, BI-RADSl, BI-RADSr, 年龄, BMI, density, history 等
```

### 2. 图像数据 (bilateral_image_features)
```python
bilateral_image_features = processed_data['bilateral_image_features']
# 字典格式: {{patient_id: {{'left_images': [...], 'right_images': [...], 'bilateral_images': [...]}}}}
```

### 3. 使用示例
```python
# 遍历所有患者
for pid in common_pids:
    # 获取临床数据
    patient_clinical = clinical_df[clinical_df['PID'].astype(str) == pid].iloc[0]
    
    # 获取图像数据
    patient_images = bilateral_image_features[pid]
    left_images = patient_images['left_images']      # 左乳房图像列表
    right_images = patient_images['right_images']    # 右乳房图像列表
    bilateral_images = patient_images['bilateral_images']  # 双侧图像列表
    
    # 获取标签
    birads_left = patient_clinical['BI-RADSl']
    birads_right = patient_clinical['BI-RADSr']
```

## 数据预处理特点
- 图像已经标准化到指定尺寸
- 已经进行了双侧分离
- 保持了原有的预处理逻辑
- 临床特征已经清理和验证

## 风险分层
- BI-RADS 1-3: 中低风险
- BI-RADS 4-6: 高风险

生成时间: {pd.Timestamp.now()}
"""
    
    # 保存说明文档
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # 创建数据访问示例
    example_code = f"""
# 数据访问示例代码
import pickle
import gzip

# 加载处理后的数据
def load_processed_data():
    # 这里应该用您的实际数据加载方法
    # 例如使用缓存管理器
    cache_manager = CacheManager('./cache')
    processed_data = cache_manager.load_cache()
    return processed_data

# 使用示例
processed_data = load_processed_data()
clinical_df = processed_data['clinical_features']
bilateral_image_features = processed_data['bilateral_image_features']

# 匹配患者
clinical_pids = set(clinical_df['PID'].astype(str))
image_pids = set(bilateral_image_features.keys())
common_pids = list(clinical_pids.intersection(image_pids))

print(f"可用于训练的患者数: {{len(common_pids)}}")

# 遍历患者进行训练数据准备
for pid in common_pids[:5]:  # 示例：前5个患者
    # 临床数据
    patient_row = clinical_df[clinical_df['PID'].astype(str) == pid].iloc[0]
    
    # 图像数据
    image_data = bilateral_image_features[pid]
    
    print(f"患者 {{pid}}:")
    print(f"  BI-RADS左: {{patient_row['BI-RADSl']}}")
    print(f"  BI-RADS右: {{patient_row['BI-RADSr']}}")
    print(f"  左乳房图像数: {{len(image_data['left_images'])}}")
    print(f"  右乳房图像数: {{len(image_data['right_images'])}}")
    
    # 这里可以添加您的训练数据准备逻辑
"""
    
    example_path = os.path.join(output_dir, 'data_access_example.py')
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    logger.info(f"✅ 训练数据格式文档已创建:")
    logger.info(f"   说明文档: {readme_path}")
    logger.info(f"   示例代码: {example_path}")

def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(description='数据处理脚本 - 适配训练文件')
    parser.add_argument('--data_dir', type=str, default='D:/Desktop/', help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='输出目录')
    parser.add_argument('--cache_root', type=str, default='./cache', help='缓存根目录')
    parser.add_argument('--image_size', type=int, default=128, help='图像尺寸')
    parser.add_argument('--max_slices', type=int, default=20, help='最大切片数')
    parser.add_argument('--force_rebuild', action='store_true', help='强制重新处理')
    
    args = parser.parse_args()
    
    # 配置设置
    config = {
        'output_dir': args.output_dir,
        'cache_root': args.cache_root,
        'data_paths': {
            'excel_path': os.path.join(args.data_dir, 'Breast BI-RADS.xlsx'),
            'dicom_root_dir': os.path.join(args.data_dir, 'Data_BI-RADS'),
            'segmentation_root_dir': os.path.join(args.data_dir, 'Data_BI-RADS_segmentation')
        },
        'image_config': {
            'target_size': (args.image_size, args.image_size),
            'max_slices': args.max_slices,
            'save_debug_images': False,
            'normalization': 'z_score',
            'bilateral_mode': 'separate_channels',
            'output_dir': os.path.join(args.output_dir, 'processed_images')
        }
    }
    
    logger.info("🚀 开始数据处理")
    logger.info("="*80)
    logger.info(f"配置信息:")
    logger.info(f"  数据目录: {args.data_dir}")
    logger.info(f"  输出目录: {config['output_dir']}")
    logger.info(f"  缓存目录: {config['cache_root']}")
    logger.info(f"  图像尺寸: {config['image_config']['target_size']}")
    logger.info(f"  最大切片数: {config['image_config']['max_slices']}")
    logger.info("="*80)
    
    try:
        # 创建数据处理器
        processor = DataProcessor(config)
        
        # 加载和处理数据
        processed_data = processor.load_and_process_data(force_rebuild=args.force_rebuild)
        
        # 创建训练数据格式文档
        create_training_data_format(processed_data, args.output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("🎯 数据处理完成!")
        logger.info("="*80)
        logger.info("📁 输出文件:")
        logger.info(f"   缓存目录: {config['cache_root']}/optimized_cache/")
        logger.info(f"   质量报告: {args.output_dir}/data_quality_report.json")
        logger.info(f"   使用说明: {args.output_dir}/README.md")
        logger.info(f"   示例代码: {args.output_dir}/data_access_example.py")
        logger.info("\n🎉 数据已准备好用于训练!")
        logger.info("请在训练脚本中使用 CacheManager 加载处理后的数据")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)