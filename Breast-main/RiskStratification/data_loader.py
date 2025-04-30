"""
Breast cancer risk stratification model - data loading module
"""

import os
import pandas as pd
import numpy as np
import pydicom
import logging

logger = logging.getLogger(__name__)

class ClinicalDataLoader:
    
    def __init__(self, excel_path):
        """
        Initialize clinical data loader

        Parameters:
        excel_path: Path to the Excel file containing clinical data
        """
        self.excel_path = excel_path
        self.clinical_df = None
    
    def load_data(self):
        """Load clinical data from Excel"""
        logger.info("Loading clinical data...")
        self.clinical_df = pd.read_excel(self.excel_path)
        required_columns = ['PID', 'density', 'label', 'risk', 'history', '年龄', 'BMI', 'biopsy']
        for col in required_columns:
            if col not in self.clinical_df.columns:
                raise ValueError(f"Required columns are missing in Excel file: {col}")
    
        # Convert risk to a numerical value: low=0, medium=1, high=2
        risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
        self.clinical_df['risk_numeric'] = self.clinical_df['risk'].map(risk_mapping)
            
        # Convert density to numeric value: A=0, B=1, C=2, D=3
        density_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.clinical_df['density_numeric'] = self.clinical_df['density'].map(density_mapping)
            
        # Make sure the PID column is of type string
        self.clinical_df['PID'] = self.clinical_df['PID'].astype(str)
        return self.clinical_df
    
    def get_patient_info(self, pid):
        """Obtain clinical information for a specific patient"""
        if self.clinical_df is None:
            self.load_data()

        pid_str = str(pid)
        
        if pid_str in self.clinical_df['PID'].values:
            return self.clinical_df[self.clinical_df['PID'] == pid_str].iloc[0].to_dict()
        else:
            logger.warning(f"No patient record found for PID {pid_str}")
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
            "Age Range": [self.clinical_df['age'].min(), self.clinical_df['age'].max()],
            "BMI Range": [self.clinical_df['BMI'].min(), self.clinical_df['BMI'].max()]
        }
        
        return summary


class DicomLoader   
    def __init__(self, dicom_root_dir):
        """
        Initialize DICOM data loader
        """
        self.dicom_root_dir = dicom_root_dir
        self._folder_format = self._detect_folder_format()
            
        try:
            dirs = [d for d in os.listdir(self.dicom_root_dir) 
                    if os.path.isdir(os.path.join(self.dicom_root_dir, d))]
            
            if not dirs:
                logger.warning(f"There are no subdirectories in the DICOM root directory {self.dicom_root_dir}")
                return None
                
            # Check if there is a directory starting with 0
            zero_prefixed = any(d.startswith('0') for d in dirs)
            
            # Determine the fill length
            lengths = [len(d) for d in dirs]
            common_length = max(set(lengths), key=lengths.count) if lengths else 6
            
            return {
                'zero_prefixed': zero_prefixed,
                'common_length': common_length
            }
        except Exception as e:
            logger.error(f"Error detecting folder format: {str(e)}")
            return None
    
    def _format_pid(self, pid):
        """Format the PID according to the detected folder format"""
        pid_str = str(pid)
        
        if self._folder_format and self._folder_format['zero_prefixed']:
            # If the directory uses leading 0, add leading 0 to the PID
            return pid_str.zfill(self._folder_format['common_length'])
        else:
            return pid_str
    
    def _find_patient_directory(self, pid):
        formatted_pid = self._format_pid(pid)
        direct_path = os.path.join(self.dicom_root_dir, formatted_pid)
        
        if os.path.exists(direct_path):
            return direct_path
        
        # If the direct path does not exist, try to find a matching directory
        pid_str = str(pid).lstrip('0')  # Remove leading zeros

        for dir_name in os.listdir(self.dicom_root_dir):
            dir_path = os.path.join(self.dicom_root_dir, dir_name)
            if os.path.isdir(dir_path) and dir_name.lstrip('0') == pid_str:
                return dir_path
                
        return None
    
    def explore_structure(self, num_examples=3):

        study_dirs = [d for d in os.listdir(self.dicom_root_dir) 
                      if os.path.isdir(os.path.join(self.dicom_root_dir, d))]

        for i, study_dir in enumerate(study_dirs[:num_examples]):
            study_path = os.path.join(self.dicom_root_dir, study_dir)

            series_dirs = [d for d in os.listdir(study_path) 
                          if os.path.isdir(os.path.join(study_path, d))]
            
            study_info['series_count'] = len(series_dirs)
  
            if series_dirs:
                series_path = os.path.join(study_path, series_dirs[0])
                instance_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
                
                study_info['first_series'] = {
                    'series_id': series_dirs[0],
                    'instance_count': len(instance_files)
                }

                if instance_files:
                    dicom_path = os.path.join(series_path, instance_files[0])
              
                    dicom_data = pydicom.dcmread(dicom_path)
                    study_info['dicom_sample'] = {
                            'rows': dicom_data.Rows,
                            'columns': dicom_data.Columns,
                            'pixel_spacing': getattr(dicom_data, 'PixelSpacing', None),
                            'slice_thickness': getattr(dicom_data, 'SliceThickness', None)
                        }
        
            structure_info[study_dir] = study_info
        
        return structure_info
    
    def load_patient_dicom(self, pid):
        """
        Load DICOM data for a specific patient
        """
        study_path = self._find_patient_directory(pid)

        # Get all series directories under the study
        series_dirs = [d for d in os.listdir(study_path) 
                      if os.path.isdir(os.path.join(study_path, d))]
       
        series_path = os.path.join(study_path, series_dirs[0])
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        dicom_data = []
        for dicom_file in dicom_files:
            file_path = os.path.join(series_path, dicom_file)
            dcm = pydicom.dcmread(file_path)
            dicom_data.append({
                    'instance': dcm,
                    'file_path': file_path,
                    'pixel_array': dcm.pixel_array
                })
        
        dicom_data.sort(key=lambda x: getattr(x['instance'], 'InstanceNumber', 0))
        
        return {
            'pid': pid,
            'series_id': series_dirs[0],
            'dicom_count': len(dicom_data),
            'dicom_data': dicom_data
        }


class SegmentationLoader:
    
    def __init__(self, segmentation_root_dir):
        """
        Initialize segmentation data loader
        """
        self.segmentation_root_dir = segmentation_root_dir
        # Storage folder format information
        self._folder_format = self._detect_folder_format()
    
    def _detect_folder_format(self):
        """Detect folder naming format"""
            
        try:
            dirs = [d for d in os.listdir(self.segmentation_root_dir) 
                    if os.path.isdir(os.path.join(self.segmentation_root_dir, d))]

            # Check if there is a directory starting with 0
            zero_prefixed = any(d.startswith('0') for d in dirs)
            
            # Determine the fill length
            lengths = [len(d) for d in dirs]
            common_length = max(set(lengths), key=lengths.count) if lengths else 6
            
            return {
                'zero_prefixed': zero_prefixed,
                'common_length': common_length
            }
        except Exception as e:
            logger.error(f"Error detecting folder format: {str(e)}")
            return None
    
    def _format_pid(self, pid):
        """Format the PID according to the detected folder format"""
        pid_str = str(pid)
        
        if self._folder_format and self._folder_format['zero_prefixed']:
            # If the directory uses leading 0, add leading 0 to the PID
            return pid_str.zfill(self._folder_format['common_length'])
        else:
            return pid_str
    
    def _find_patient_directory(self, pid):
        """Find a directory matching the patient ID"""
        formatted_pid = self._format_pid(pid)
        direct_path = os.path.join(self.segmentation_root_dir, formatted_pid)
        
        # If the direct path does not exist, try to find a matching directory
        pid_str = str(pid).lstrip('0')

        for dir_name in os.listdir(self.segmentation_root_dir):
            dir_path = os.path.join(self.segmentation_root_dir, dir_name)
            if os.path.isdir(dir_path) and dir_name.lstrip('0') == pid_str:
                return dir_path
        
        return None
    
    def explore_structure(self, num_examples=3):

        study_dirs = [d for d in os.listdir(self.segmentation_root_dir) 
                      if os.path.isdir(os.path.join(self.segmentation_root_dir, d))]
        
        
        # Explore the structure of the first num_examples studies
        for i, study_dir in enumerate(study_dirs[:num_examples]):
            study_path = os.path.join(self.segmentation_root_dir, study_dir)

            series_dirs = [d for d in os.listdir(study_path) 
                          if os.path.isdir(os.path.join(study_path, d))]
            
            study_info['series_count'] = len(series_dirs)

            if series_dirs:
                series_path = os.path.join(study_path, series_dirs[0])
                npy_files = [f for f in os.listdir(series_path) if f.endswith('.npy')]
                
                study_info['first_series'] = {
                    'series_id': series_dirs[0],
                    'npy_files': npy_files
                }

                if 'breast_mask.npy' in npy_files:
                    mask_path = os.path.join(series_path, 'breast_mask.npy')
                    mask_data = np.load(mask_path)
                    study_info['mask_sample'] = {
                            'shape': mask_data.shape,
                            'nonzero_ratio': float(np.mean(mask_data > 0))
                        }
            
            structure_info[study_dir] = study_info
        
        return structure_info
    
    def load_patient_segmentation(self, pid):
        """
        Load segmentation data for a specific patient
        """
        study_path = self._find_patient_directory(pid)

        series_dirs = [d for d in os.listdir(study_path) 
                      if os.path.isdir(os.path.join(study_path, d))]

        series_path = os.path.join(study_path, series_dirs[0])

        mask_path = os.path.join(series_path, 'breast_mask.npy')

        glandular_path = os.path.join(series_path, 'glandular_tissue_mask.npy')
        has_glandular = os.path.exists(glandular_path)

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
        
    """
    Improved implementation of glandular tissue extraction function
    """

    def _approximate_glandular_tissue(self, breast_mask):
        """
        Approximately extract glandular tissue regions from breast mask
        This is an improved method that uses morphological operations and intensity analysis to extract glandular tissue
        """
        from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt

        glandular_mask = np.zeros_like(breast_mask)
        
        # Process each slice
        for i in range(breast_mask.shape[0]):
            mask_slice = breast_mask[i]

            if np.sum(mask_slice) == 0:
                continue
            
            # Calculate distance transform - distance to breast edge
            distance_map = distance_transform_edt(mask_slice)
            
            # Normalized distance map
            max_dist = np.max(distance_map)
            if max_dist > 0:
                distance_map = distance_map / max_dist
            
            # Extract the central area (the area farthest from the edge)
            threshold = 0.4  # Adjust the threshold to control the size of the extracted area
            glandular_region = distance_map > threshold
            
            # Apply Gaussian smoothing to make the edges more natural
            glandular_region = gaussian_filter(glandular_region.astype(float), sigma=1.0) > 0.5
            
            # Make sure the glandular area is contained within the mammary gland area
            glandular_region = glandular_region & mask_slice
            
            # save
            glandular_mask[i] = glandular_region
        
        return glandular_mask

    def load_patient_segmentation_with_glandular(self, pid):
        
        segmentation_data = self.load_patient_segmentation(pid)

        return segmentation_data
