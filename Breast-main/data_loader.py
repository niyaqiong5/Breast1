"""
Breast cancer risk stratification model - data loading module
Responsible for loading and preliminary processing of various data sources
"""

import os
import pandas as pd
import numpy as np
import pydicom
import logging

logger = logging.getLogger(__name__)

class ClinicalDataLoader:
    """临床数据加载器"""
    
    def __init__(self, excel_path):
        """
        Initialize clinical data loader

        Parameters:
        excel_path: Path to the Excel file containing clinical data
        """
        self.excel_path = excel_path
        self.clinical_df = None
    
    def load_data(self):
        """加载Excel中的临床数据"""
        logger.info("Loading clinical data...")
        try:
            self.clinical_df = pd.read_excel(self.excel_path)

            self.clinical_df = self.clinical_df.sort_values('PID').reset_index(drop=True)
            logger.info(f"数据按PID排序后前5行: {self.clinical_df['PID'].head().tolist()}")
            
            # 检查数据列
            required_columns = ['PID', 'density', 'label', 'history', '年龄', 'BMI']
            for col in required_columns:
                if col not in self.clinical_df.columns:
                    raise ValueError(f"Required columns are missing in Excel file: {col}")
            
            # 数据预处理
            # 将risk转换为数值: 低=0, 中=1, 高=2
            #risk_mapping = {'低': 0, '中': 1, '高': 2}
            #self.clinical_df['risk_numeric'] = self.clinical_df['risk'].map(risk_mapping)
            
            # 将density转换为数值: A=0, B=1, C=2, D=3
            density_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            self.clinical_df['density_numeric'] = self.clinical_df['density'].map(density_mapping)
            
            # 确保PID列是字符串类型
            self.clinical_df['PID'] = self.clinical_df['PID'].astype(str)
            
            logger.info(f"临床数据加载成功，共 {len(self.clinical_df)} 条记录")
            return self.clinical_df
            
        except Exception as e:
            logger.error(f"Failed to load clinical data: {str(e)}")
            raise
    
    def get_patient_info(self, pid):
        """Obtain clinical information for a specific patient"""
        if self.clinical_df is None:
            self.load_data()
        
        # 确保pid是字符串
        pid_str = str(pid)
        
        if pid_str in self.clinical_df['PID'].values:
            return self.clinical_df[self.clinical_df['PID'] == pid_str].iloc[0].to_dict()
        else:
            logger.warning(f"未找到PID为 {pid_str} 的患者记录")
            return None
    
    def get_data_summary(self):
        """Get data summary statistics"""
        if self.clinical_df is None:
            self.load_data()
        
        summary = {
            "Total number of samples": len(self.clinical_df),
            "Number of positive samples (cancer patients)": self.clinical_df['label'].sum(),
            "Number of negative samples (healthy)": (self.clinical_df['label'] == 0).sum(),
            "Risk Distribution": self.clinical_df['risk'].value_counts().to_dict(),
            "Breast density distribution": self.clinical_df['density'].value_counts().to_dict(),
            "Age Range": [self.clinical_df['年龄'].min(), self.clinical_df['年龄'].max()],
            "BMI Range": [self.clinical_df['BMI'].min(), self.clinical_df['BMI'].max()]
        }
        
        return summary


class DicomLoader:
    """DICOM数据加载器"""
    
    def __init__(self, dicom_root_dir):
        """
        Initialize DICOM data loader

        Parameters:
        dicom_root_dir: DICOM data root directory
        """
        self.dicom_root_dir = dicom_root_dir
        # 存储文件夹格式信息
        self._folder_format = self._detect_folder_format()
    
    def _detect_folder_format(self):
        """Detect folder naming format"""
        if not os.path.exists(self.dicom_root_dir):
            logger.warning(f"DICOM根目录 {self.dicom_root_dir} 不存在")
            return None
            
        try:
            # 获取目录列表
            dirs = [d for d in os.listdir(self.dicom_root_dir) 
                    if os.path.isdir(os.path.join(self.dicom_root_dir, d))]
            
            if not dirs:
                logger.warning(f"DICOM根目录 {self.dicom_root_dir} 中没有子目录")
                return None
                
            # 检查是否有以0开头的目录
            zero_prefixed = any(d.startswith('0') for d in dirs)
            
            # 确定填充长度
            lengths = [len(d) for d in dirs]
            common_length = max(set(lengths), key=lengths.count) if lengths else 6
            
            return {
                'zero_prefixed': zero_prefixed,
                'common_length': common_length
            }
        except Exception as e:
            logger.error(f"检测文件夹格式时出错: {str(e)}")
            return None
    
    def _format_pid(self, pid):
        """Format the PID according to the detected folder format"""
        pid_str = str(pid)
        
        if self._folder_format and self._folder_format['zero_prefixed']:
            # 如果目录使用前导0，则添加前导0到PID
            return pid_str.zfill(self._folder_format['common_length'])
        else:
            return pid_str
    
    def _find_patient_directory(self, pid):
        """查找匹配患者ID的目录"""
        # 尝试直接使用格式化的PID
        formatted_pid = self._format_pid(pid)
        direct_path = os.path.join(self.dicom_root_dir, formatted_pid)
        
        if os.path.exists(direct_path):
            return direct_path
        
        # 如果直接路径不存在，尝试查找匹配的目录
        pid_str = str(pid).lstrip('0')  # 移除前导0
        
        try:
            for dir_name in os.listdir(self.dicom_root_dir):
                dir_path = os.path.join(self.dicom_root_dir, dir_name)
                if os.path.isdir(dir_path) and dir_name.lstrip('0') == pid_str:
                    return dir_path
        except Exception as e:
            logger.error(f"查找患者目录时出错: {str(e)}")
        
        return None
    
    def explore_structure(self, num_examples=3):
        """探索DICOM数据结构"""
        logger.info("探索DICOM数据结构...")
        
        # 获取所有study目录
        study_dirs = [d for d in os.listdir(self.dicom_root_dir) 
                      if os.path.isdir(os.path.join(self.dicom_root_dir, d))]
        
        logger.info(f"共找到 {len(study_dirs)} 个study目录")
        
        structure_info = {}
        
        # 探索前num_examples个study的结构
        for i, study_dir in enumerate(study_dirs[:num_examples]):
            study_path = os.path.join(self.dicom_root_dir, study_dir)
            logger.info(f"探索Study {i+1}: {study_dir}")
            
            study_info = {}
            
            # 获取该study下的所有series目录
            series_dirs = [d for d in os.listdir(study_path) 
                          if os.path.isdir(os.path.join(study_path, d))]
            
            study_info['series_count'] = len(series_dirs)
            
            # 探索第一个series的结构
            if series_dirs:
                series_path = os.path.join(study_path, series_dirs[0])
                instance_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
                
                study_info['first_series'] = {
                    'series_id': series_dirs[0],
                    'instance_count': len(instance_files)
                }
                
                # 加载第一个DICOM文件并显示一些基本信息
                if instance_files:
                    dicom_path = os.path.join(series_path, instance_files[0])
                    try:
                        dicom_data = pydicom.dcmread(dicom_path)
                        study_info['dicom_sample'] = {
                            'rows': dicom_data.Rows,
                            'columns': dicom_data.Columns,
                            'pixel_spacing': getattr(dicom_data, 'PixelSpacing', None),
                            'slice_thickness': getattr(dicom_data, 'SliceThickness', None)
                        }
                    except Exception as e:
                        logger.error(f"读取DICOM文件失败: {e}")
                        study_info['dicom_sample'] = {'error': str(e)}
            
            structure_info[study_dir] = study_info
        
        return structure_info
    
    def load_patient_dicom(self, pid):
        """
        Load DICOM data for a specific patient

        Parameters:
        pid: Patient ID

        Returns:
        Dictionary containing DICOM instance and metadata
        """
        study_path = self._find_patient_directory(pid)
        
        if study_path is None:
            logger.warning(f"未找到PID为 {pid} 的DICOM数据目录")
            return None
        
        # 获取该study下的所有series目录
        series_dirs = [d for d in os.listdir(study_path) 
                      if os.path.isdir(os.path.join(study_path, d))]
        
        if not series_dirs:
            logger.warning(f"PID为 {pid} 的DICOM数据目录中没有series")
            return None
        
        # 处理第一个series (可以根据需要扩展为处理所有series)
        series_path = os.path.join(study_path, series_dirs[0])
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        if not dicom_files:
            logger.warning(f"PID为 {pid} 的series中没有DICOM文件")
            return None
        
        # 加载所有DICOM文件
        dicom_data = []
        for dicom_file in dicom_files:
            file_path = os.path.join(series_path, dicom_file)
            try:
                dcm = pydicom.dcmread(file_path)
                dicom_data.append({
                    'instance': dcm,
                    'file_path': file_path,
                    'pixel_array': dcm.pixel_array
                })
            except Exception as e:
                logger.error(f"读取DICOM文件 {file_path} 失败: {e}")
        
        # 按照实例号排序
        dicom_data.sort(key=lambda x: getattr(x['instance'], 'InstanceNumber', 0))
        
        return {
            'pid': pid,
            'series_id': series_dirs[0],
            'dicom_count': len(dicom_data),
            'dicom_data': dicom_data
        }


class SegmentationLoader:
    """分割数据加载器"""
    
    def __init__(self, segmentation_root_dir):
        """
        Initialize segmentation data loader

        Parameters:
        segmentation_root_dir: segmentation data root directory
        """
        self.segmentation_root_dir = segmentation_root_dir
        # 存储文件夹格式信息
        self._folder_format = self._detect_folder_format()
    
    def _detect_folder_format(self):
        """检测文件夹命名格式"""
        if not os.path.exists(self.segmentation_root_dir):
            logger.warning(f"分割数据根目录 {self.segmentation_root_dir} 不存在")
            return None
            
        try:
            # 获取目录列表
            dirs = [d for d in os.listdir(self.segmentation_root_dir) 
                    if os.path.isdir(os.path.join(self.segmentation_root_dir, d))]
            
            if not dirs:
                logger.warning(f"分割数据根目录 {self.segmentation_root_dir} 中没有子目录")
                return None
                
            # 检查是否有以0开头的目录
            zero_prefixed = any(d.startswith('0') for d in dirs)
            
            # 确定填充长度
            lengths = [len(d) for d in dirs]
            common_length = max(set(lengths), key=lengths.count) if lengths else 6
            
            return {
                'zero_prefixed': zero_prefixed,
                'common_length': common_length
            }
        except Exception as e:
            logger.error(f"检测文件夹格式时出错: {str(e)}")
            return None
    
    def _format_pid(self, pid):
        """根据检测到的文件夹格式格式化PID"""
        pid_str = str(pid)
        
        if self._folder_format and self._folder_format['zero_prefixed']:
            # 如果目录使用前导0，则添加前导0到PID
            return pid_str.zfill(self._folder_format['common_length'])
        else:
            return pid_str
    
    def _find_patient_directory(self, pid):
        """查找匹配患者ID的目录"""
        # 尝试直接使用格式化的PID
        formatted_pid = self._format_pid(pid)
        direct_path = os.path.join(self.segmentation_root_dir, formatted_pid)
        
        if os.path.exists(direct_path):
            return direct_path
        
        # 如果直接路径不存在，尝试查找匹配的目录
        pid_str = str(pid).lstrip('0')  # 移除前导0
        
        try:
            for dir_name in os.listdir(self.segmentation_root_dir):
                dir_path = os.path.join(self.segmentation_root_dir, dir_name)
                if os.path.isdir(dir_path) and dir_name.lstrip('0') == pid_str:
                    return dir_path
        except Exception as e:
            logger.error(f"查找患者目录时出错: {str(e)}")
        
        return None
    
    def explore_structure(self, num_examples=3):
        """探索分割数据结构"""
        logger.info("探索分割数据结构...")
        
        # 获取所有study目录
        study_dirs = [d for d in os.listdir(self.segmentation_root_dir) 
                      if os.path.isdir(os.path.join(self.segmentation_root_dir, d))]
        
        logger.info(f"共找到 {len(study_dirs)} 个分割study目录")
        
        structure_info = {}
        
        # 探索前num_examples个study的结构
        for i, study_dir in enumerate(study_dirs[:num_examples]):
            study_path = os.path.join(self.segmentation_root_dir, study_dir)
            logger.info(f"探索Study {i+1}: {study_dir}")
            
            study_info = {}
            
            # 获取该study下的所有series目录
            series_dirs = [d for d in os.listdir(study_path) 
                          if os.path.isdir(os.path.join(study_path, d))]
            
            study_info['series_count'] = len(series_dirs)
            
            # 探索第一个series的结构
            if series_dirs:
                series_path = os.path.join(study_path, series_dirs[0])
                npy_files = [f for f in os.listdir(series_path) if f.endswith('.npy')]
                
                study_info['first_series'] = {
                    'series_id': series_dirs[0],
                    'npy_files': npy_files
                }
                
                # 加载分割mask并显示基本信息
                if 'breast_mask.npy' in npy_files:
                    mask_path = os.path.join(series_path, 'breast_mask.npy')
                    try:
                        mask_data = np.load(mask_path)
                        study_info['mask_sample'] = {
                            'shape': mask_data.shape,
                            'nonzero_ratio': float(np.mean(mask_data > 0))
                        }
                    except Exception as e:
                        logger.error(f"读取分割mask失败: {e}")
                        study_info['mask_sample'] = {'error': str(e)}
            
            structure_info[study_dir] = study_info
        
        return structure_info
    
    def load_patient_segmentation(self, pid):
        """
        Load segmentation data for a specific patient

        Parameters:
        pid: patient ID

        Returns:
        Dictionary containing segmentation masks
        """
        study_path = self._find_patient_directory(pid)
        
        if study_path is None:
            logger.warning(f"未找到PID为 {pid} 的分割数据目录")
            return None
        
        # 获取该study下的所有series目录
        series_dirs = [d for d in os.listdir(study_path) 
                      if os.path.isdir(os.path.join(study_path, d))]
        
        if not series_dirs:
            logger.warning(f"PID为 {pid} 的分割数据目录中没有series")
            return None
        
        # 处理第一个series (可以根据需要扩展为处理所有series)
        series_path = os.path.join(study_path, series_dirs[0])
        
        # 检查是否存在breast_mask.npy文件
        mask_path = os.path.join(series_path, 'breast_mask.npy')
        
        # 检查是否存在glandular_tissue_mask.npy文件
        glandular_path = os.path.join(series_path, 'glandular_tissue_mask.npy')
        has_glandular = os.path.exists(glandular_path)
        
        if not os.path.exists(mask_path):
            logger.warning(f"PID为 {pid} 的分割数据中没有breast_mask.npy文件")
            return None
        
        # 加载分割mask
        try:
            breast_mask = np.load(mask_path)
            result = {
                'pid': pid,
                'series_id': series_dirs[0],
                'mask_path': mask_path,
                'breast_mask': breast_mask,
                'shape': breast_mask.shape,
                'nonzero_ratio': float(np.mean(breast_mask > 0))
            }
            
            # 如果存在腺体组织掩码，也加载它
            if has_glandular:
                try:
                    glandular_mask = np.load(glandular_path)
                    result['glandular_tissue_mask'] = glandular_mask
                    result['glandular_path'] = glandular_path
                    result['glandular_nonzero_ratio'] = float(np.mean(glandular_mask > 0))
                except Exception as e:
                    logger.error(f"读取glandular_tissue_mask {glandular_path} 失败: {e}")
            
            return result
        except Exception as e:
            logger.error(f"读取分割mask {mask_path} 失败: {e}")
            return None
        
    """
    Improved implementation of glandular tissue extraction function
    """

    def _approximate_glandular_tissue(self, breast_mask):
        """
        Approximately extract glandular tissue regions from breast mask
        This is an improved method that uses morphological operations and intensity analysis to extract glandular tissue

        Parameters:
        breast_mask: breast region mask

        Returns:
        Approximate glandular tissue mask
        """
        from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt
        
        # 复制乳腺掩码
        glandular_mask = np.zeros_like(breast_mask)
        
        # 对每个切片进行处理
        for i in range(breast_mask.shape[0]):
            # 获取当前切片
            mask_slice = breast_mask[i]
            
            # 跳过全零切片
            if np.sum(mask_slice) == 0:
                continue
            
            # 计算距离变换 - 到乳腺边缘的距离
            distance_map = distance_transform_edt(mask_slice)
            
            # 归一化距离图
            max_dist = np.max(distance_map)
            if max_dist > 0:
                distance_map = distance_map / max_dist
            
            # 提取中心区域(距离边缘较远的区域)
            # 这里假设腺体组织主要位于乳腺的中心区域
            threshold = 0.4  # 调整阈值以控制提取区域大小
            glandular_region = distance_map > threshold
            
            # 应用高斯平滑使边缘更自然
            glandular_region = gaussian_filter(glandular_region.astype(float), sigma=1.0) > 0.5
            
            # 确保腺体区域被包含在乳腺区域内
            glandular_region = glandular_region & mask_slice
            
            # 保存到结果中
            glandular_mask[i] = glandular_region
        
        return glandular_mask

    def load_patient_segmentation_with_glandular(self, pid):
        """
        加载特定患者的分割数据，并确保有腺体组织掩码
        
        参数:
        pid: 患者ID
        
        返回:
        包含分割mask和腺体组织mask的字典
        """
        # 首先尝试加载常规分割数据
        segmentation_data = self.load_patient_segmentation(pid)
        
        if segmentation_data is None:
            return None
        
        # 检查是否已经有腺体组织掩码
        if 'glandular_tissue_mask' not in segmentation_data:
            # 如果没有，使用近似方法生成
            logger.info(f"PID={pid}没有腺体组织掩码，正在生成近似掩码...")
            breast_mask = segmentation_data['breast_mask']
            glandular_mask = self._approximate_glandular_tissue(breast_mask)
            
            # 添加到结果中
            segmentation_data['glandular_tissue_mask'] = glandular_mask
            segmentation_data['glandular_nonzero_ratio'] = float(np.mean(glandular_mask > 0))
            segmentation_data['glandular_is_approximated'] = True
        
        return segmentation_data