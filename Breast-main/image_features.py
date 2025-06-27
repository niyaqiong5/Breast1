"""
Breast Cancer Risk Stratification Model - Bilateral Image Processing Module
Responsible for processing both left and right breast simultaneously
"""

import numpy as np
import os
import logging
import cv2
import pydicom
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy.ndimage import binary_erosion, binary_dilation, center_of_mass, label
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BilateralImageProcessor:
    """双侧乳腺同时处理的图像处理器"""
    
    def __init__(self, config):
        """
        Initialize bilateral image processor

        Parameters:
        config: Configuration dictionary containing processing parameters
        """
        self.output_dir = config.get('output_dir', 'output/images')
        self.target_size = config.get('target_size', (224, 224))
        self.max_slices_per_patient = config.get('max_slices', None)
        self.save_debug_images = config.get('save_debug_images', False)
        self.normalization_method = config.get('normalization', 'minmax')
        self.augmentation = config.get('augmentation', False)
        self.focus_on_glandular = config.get('focus_on_glandular', True)
        
        # 双侧处理特定配置
        self.bilateral_mode = config.get('bilateral_mode', 'separate_channels')  # 'separate_channels' or 'side_by_side'
        
        # Create output directories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        if self.save_debug_images:
            self.debug_dir = os.path.join(self.output_dir, 'debug')
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
    
    def separate_left_right_breast(self, breast_mask):
        """
        Separate left and right breast from the combined breast mask
        
        Parameters:
        breast_mask: Combined breast mask (3D array)
        
        Returns:
        Tuple of (left_mask, right_mask)
        """
        left_mask = np.zeros_like(breast_mask)
        right_mask = np.zeros_like(breast_mask)
        
        for slice_idx in range(breast_mask.shape[0]):
            mask_slice = breast_mask[slice_idx]
            
            if np.sum(mask_slice) == 0:
                continue
            
            # Find connected components
            labeled_mask, num_labels = label(mask_slice)
            
            if num_labels == 0:
                continue
            elif num_labels == 1:
                # Only one component, determine left/right based on center of mass
                center_y, center_x = center_of_mass(mask_slice)
                image_center_x = mask_slice.shape[1] // 2
                
                if center_x < image_center_x:
                    # Right side in image (patient's right breast)
                    right_mask[slice_idx] = mask_slice
                else:
                    # Left side in image (patient's left breast)
                    left_mask[slice_idx] = mask_slice
            else:
                # Multiple components, separate by position
                for component_label in range(1, num_labels + 1):
                    component_mask = (labeled_mask == component_label)
                    center_y, center_x = center_of_mass(component_mask)
                    image_center_x = mask_slice.shape[1] // 2
                    
                    if center_x < image_center_x:
                        # Right side in image (patient's right breast)
                        right_mask[slice_idx] += component_mask
                    else:
                        # Left side in image (patient's left breast)
                        left_mask[slice_idx] += component_mask
        
        # Ensure masks are binary
        left_mask = (left_mask > 0).astype(np.float32)
        right_mask = (right_mask > 0).astype(np.float32)
        
        return left_mask, right_mask
    
    def process_patient_data_bilateral(self, dicom_data, segmentation_data, pid):
        """
        Process DICOM and segmentation data for both breasts simultaneously

        Parameters:
        dicom_data: dictionary containing DICOM instances
        segmentation_data: dictionary containing segmentation masks
        pid: patient ID

        Returns:
        Dictionary containing processed bilateral data
        """
        if dicom_data is None or segmentation_data is None:
            logger.warning(f"PID={pid}的DICOM数据或分割数据为空")
            return None
        
        # Get segmentation masks and DICOM data
        breast_mask = segmentation_data.get('breast_mask')
        glandular_mask = segmentation_data.get('glandular_tissue_mask')
        dicom_instances = dicom_data.get('dicom_data', [])
        
        if breast_mask is None or not dicom_instances:
            logger.warning(f"PID={pid}的分割掩码或DICOM实例为空")
            return None
        
        # Separate left and right breast masks
        left_breast_mask, right_breast_mask = self.separate_left_right_breast(breast_mask)
        
        # Handle glandular tissue separation if available
        left_glandular_mask = None
        right_glandular_mask = None
        if glandular_mask is not None:
            left_glandular_mask, right_glandular_mask = self.separate_left_right_breast(glandular_mask)
        elif self.focus_on_glandular:
            # Generate approximate glandular tissue for each side
            left_glandular_mask = self._approximate_glandular_tissue(left_breast_mask)
            right_glandular_mask = self._approximate_glandular_tissue(right_breast_mask)
        
        processed_left_images = []
        processed_right_images = []
        processed_bilateral_images = []
        processed_masks = []
        
        # Select number of slices to process
        total_slices = min(len(dicom_instances), breast_mask.shape[0])
        
        if self.max_slices_per_patient is None or self.max_slices_per_patient >= total_slices:
            slice_indices = range(total_slices)
        else:
            slice_indices = np.linspace(0, total_slices-1, 
                                    self.max_slices_per_patient, 
                                    dtype=int)
        
        logger.info(f"处理PID={pid}的{len(slice_indices)}个切片 (双侧)")
        
        for idx in slice_indices:
            try:
                # Get DICOM pixel data
                pixel_array = dicom_instances[idx]['pixel_array']
                
                if idx >= breast_mask.shape[0]:
                    logger.warning(f"PID={pid}的切片索引{idx}超出范围")
                    continue
                
                # Get masks for both sides
                left_mask_slice = left_breast_mask[idx].astype(np.float32)
                right_mask_slice = right_breast_mask[idx].astype(np.float32)
                
                # Handle glandular tissue masks
                if self.focus_on_glandular:
                    if left_glandular_mask is not None and idx < left_glandular_mask.shape[0]:
                        left_active_mask = left_glandular_mask[idx].astype(np.float32)
                    else:
                        left_active_mask = left_mask_slice
                    
                    if right_glandular_mask is not None and idx < right_glandular_mask.shape[0]:
                        right_active_mask = right_glandular_mask[idx].astype(np.float32)
                    else:
                        right_active_mask = right_mask_slice
                else:
                    left_active_mask = left_mask_slice
                    right_active_mask = right_mask_slice
                
                # Resize masks if necessary
                if left_active_mask.shape != pixel_array.shape:
                    left_active_mask = resize(left_active_mask, pixel_array.shape, 
                                            preserve_range=True, anti_aliasing=True)
                    left_active_mask = (left_active_mask > 0.5).astype(np.float32)
                
                if right_active_mask.shape != pixel_array.shape:
                    right_active_mask = resize(right_active_mask, pixel_array.shape, 
                                             preserve_range=True, anti_aliasing=True)
                    right_active_mask = (right_active_mask > 0.5).astype(np.float32)
                
                # Apply masks to extract breast regions
                left_masked_image = pixel_array * left_active_mask
                right_masked_image = pixel_array * right_active_mask
                
                # Preprocess each side separately
                left_processed = self._preprocess_image(left_masked_image, pid, idx, 'left')
                right_processed = self._preprocess_image(right_masked_image, pid, idx, 'right')
                
                if left_processed is not None and right_processed is not None:
                    # Create bilateral representation
                    if self.bilateral_mode == 'separate_channels':
                        # Stack left and right as separate channels (6 channels total)
                        bilateral_image = self._create_bilateral_channels(left_processed, right_processed)
                    elif self.bilateral_mode == 'side_by_side':
                        # Place left and right side by side
                        bilateral_image = self._create_side_by_side(left_processed, right_processed)
                    else:
                        # Default: use separate channels
                        bilateral_image = self._create_bilateral_channels(left_processed, right_processed)
                    
                    processed_left_images.append(left_processed)
                    processed_right_images.append(right_processed)
                    processed_bilateral_images.append(bilateral_image)
                    processed_masks.append({
                        'left': left_active_mask,
                        'right': right_active_mask,
                        'combined': left_active_mask + right_active_mask
                    })
                    
                    # Save debug images
                    if self.save_debug_images:
                        self._save_bilateral_debug_images(
                            pixel_array, left_active_mask, right_active_mask,
                            left_masked_image, right_masked_image,
                            left_processed, right_processed, bilateral_image,
                            pid, idx
                        )
                
            except Exception as e:
                logger.error(f"处理PID={pid}的切片{idx}时出错: {str(e)}")
        
        if not processed_bilateral_images:
            logger.warning(f"PID={pid}没有成功处理的双侧切片")
            return None
        
        # Convert to arrays
        result = {
            'bilateral_images': np.array(processed_bilateral_images),
            'left_images': np.array(processed_left_images),
            'right_images': np.array(processed_right_images),
            'masks': processed_masks,
            'num_slices': len(processed_bilateral_images)
        }
        
        logger.info(f"成功处理PID={pid}的{len(processed_bilateral_images)}个双侧切片")
        
        return result
    
    def _create_bilateral_channels(self, left_image, right_image):
        """
        Create 6-channel bilateral image (3 channels for each side)
        
        Parameters:
        left_image: Processed left breast image (H, W, 3)
        right_image: Processed right breast image (H, W, 3)
        
        Returns:
        6-channel bilateral image (H, W, 6)
        """
        bilateral_image = np.concatenate([left_image, right_image], axis=2)
        return bilateral_image
    
    def _create_side_by_side(self, left_image, right_image):
        """
        Create side-by-side bilateral image
        
        Parameters:
        left_image: Processed left breast image (H, W, 3)
        right_image: Processed right breast image (H, W, 3)
        
        Returns:
        Side-by-side bilateral image (H, 2*W, 3)
        """
        bilateral_image = np.concatenate([left_image, right_image], axis=1)
        return bilateral_image
    
    def _approximate_glandular_tissue(self, breast_mask):
        """
        Approximately extract glandular tissue regions from breast mask
        """
        glandular_mask = breast_mask.copy()
        
        for i in range(breast_mask.shape[0]):
            mask_slice = breast_mask[i]
            
            if np.sum(mask_slice) == 0:
                continue
            
            # Use morphological operations
            eroded_mask = binary_erosion(mask_slice, iterations=5)
            smooth_mask = binary_dilation(eroded_mask, iterations=2)
            glandular_mask[i] = smooth_mask
        
        glandular_mask = glandular_mask * breast_mask
        return glandular_mask
    
    def _preprocess_image(self, image, pid, slice_idx, side):
        """Enhanced image preprocessing for bilateral analysis"""
        try:
            img = image.copy()
            
            if np.sum(img > 0) == 0:
                logger.warning(f"PID={pid}的{side}侧切片{slice_idx}全为零，跳过")
                return None
            
            nonzero_mask = img > 0
            
            # CLAHE enhancement
            if np.sum(nonzero_mask) > 0:
                img_8bit = np.zeros_like(img, dtype=np.uint8)
                if np.max(img[nonzero_mask]) > 0:
                    img_8bit[nonzero_mask] = (255 * (img[nonzero_mask] - np.min(img[nonzero_mask])) / 
                                            (np.max(img[nonzero_mask]) - np.min(img[nonzero_mask]))).astype(np.uint8)
                
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                img_enhanced = clahe.apply(img_8bit)
                img = img_enhanced.astype(np.float32) / 255.0
            
            # Resize
            resized_img = resize(img, self.target_size, preserve_range=True, anti_aliasing=True)
            resized_img = resized_img.astype(np.float32)
            resized_img = np.clip(resized_img, 0, 1)
            
            # Convert to 3 channels
            if len(resized_img.shape) == 2:
                resized_img = np.stack([resized_img] * 3, axis=-1)
            
            # Normalize each channel
            for c in range(3):
                channel_data = resized_img[:,:,c][resized_img[:,:,c] > 0]
                if len(channel_data) > 0:
                    channel_mean = np.mean(channel_data)
                    channel_std = np.std(channel_data)
                    if channel_std > 0:
                        mask = resized_img[:,:,c] > 0
                        resized_img[:,:,c][mask] = (resized_img[:,:,c][mask] - channel_mean) / channel_std
                        resized_img[:,:,c] = np.clip(resized_img[:,:,c], -3, 3)
            
            return resized_img
            
        except Exception as e:
            logger.error(f"预处理PID={pid}的{side}侧切片{slice_idx}时出错: {str(e)}")
            return None
    
    def _save_bilateral_debug_images(self, original, left_mask, right_mask, 
                                   left_masked, right_masked, left_processed, 
                                   right_processed, bilateral, pid, slice_idx):
        """
        Save debug images for bilateral processing
        """
        try:
            plt.figure(figsize=(24, 8))
            
            # Original image
            plt.subplot(2, 4, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            # Left mask
            plt.subplot(2, 4, 2)
            plt.imshow(left_mask, cmap='gray')
            plt.title('Left Mask')
            plt.axis('off')
            
            # Right mask
            plt.subplot(2, 4, 3)
            plt.imshow(right_mask, cmap='gray')
            plt.title('Right Mask')
            plt.axis('off')
            
            # Combined masks
            plt.subplot(2, 4, 4)
            combined_mask = left_mask + right_mask
            plt.imshow(combined_mask, cmap='gray')
            plt.title('Combined Masks')
            plt.axis('off')
            
            # Left processed
            plt.subplot(2, 4, 5)
            if len(left_processed.shape) == 3 and left_processed.shape[2] == 3:
                plt.imshow(left_processed)
            else:
                plt.imshow(left_processed, cmap='gray')
            plt.title('Left Processed')
            plt.axis('off')
            
            # Right processed
            plt.subplot(2, 4, 6)
            if len(right_processed.shape) == 3 and right_processed.shape[2] == 3:
                plt.imshow(right_processed)
            else:
                plt.imshow(right_processed, cmap='gray')
            plt.title('Right Processed')
            plt.axis('off')
            
            # Bilateral result
            plt.subplot(2, 4, 7)
            if self.bilateral_mode == 'side_by_side':
                plt.imshow(bilateral)
            else:
                # For 6-channel, show first 3 channels
                plt.imshow(bilateral[:,:,:3])
            plt.title('Bilateral Result')
            plt.axis('off')
            
            # Add side indicators
            plt.subplot(2, 4, 8)
            plt.imshow(original, cmap='gray')
            plt.axvline(x=original.shape[1]//2, color='red', linestyle='--', linewidth=2)
            plt.text(original.shape[1]//4, 20, 'R', color='red', fontsize=16, fontweight='bold')
            plt.text(3*original.shape[1]//4, 20, 'L', color='red', fontsize=16, fontweight='bold')
            plt.title('L/R Reference')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, f'pid_{pid}_slice_{slice_idx}_bilateral_debug.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"保存双侧调试图像时出错: {str(e)}")
    
    def batch_process_bilateral(self, patients_data):
        """
        Batch process multiple patients' bilateral data

        Parameters:
        patients_data: patient data dictionary

        Returns:
        Processed bilateral data dictionary
        """
        results = {}
        
        logger.info(f"开始批量处理{len(patients_data)}个患者的双侧数据")
        
        for pid, (dicom_data, segmentation_data) in tqdm(patients_data.items(), desc="处理患者双侧数据"):
            bilateral_data = self.process_patient_data_bilateral(dicom_data, segmentation_data, pid)
            
            if bilateral_data is not None:
                results[pid] = bilateral_data
        
        logger.info(f"成功处理{len(results)}个患者的双侧数据")
        return results
    
    def extract_bilateral_2d_slices(self, processed_bilateral_data):
        """
        Extract 2D slices from bilateral processed data
        
        Parameters:
        processed_bilateral_data: Processed bilateral data dictionary
        
        Returns:
        Arrays of bilateral slices and corresponding patient IDs
        """
        all_bilateral_slices = []
        all_left_slices = []
        all_right_slices = []
        slice_to_patient = []
        
        for pid, data in processed_bilateral_data.items():
            bilateral_images = data['bilateral_images']
            left_images = data['left_images']
            right_images = data['right_images']
            
            for i in range(len(bilateral_images)):
                all_bilateral_slices.append(bilateral_images[i])
                all_left_slices.append(left_images[i])
                all_right_slices.append(right_images[i])
                slice_to_patient.append(pid)
        
        result = {
            'bilateral_slices': np.array(all_bilateral_slices),
            'left_slices': np.array(all_left_slices),
            'right_slices': np.array(all_right_slices),
            'slice_to_patient': slice_to_patient
        }
        
        logger.info(f"从{len(processed_bilateral_data)}个患者的数据中提取了{len(all_bilateral_slices)}个双侧2D切片")
        
        return result