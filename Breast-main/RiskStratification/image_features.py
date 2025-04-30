"""
Responsible for extracting and preprocessing image data from DICOM images
"""

import numpy as np
import os
import logging
import cv2
import pydicom
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImprovedImageProcessor:
    
    def __init__(self, config):
        self.output_dir = config.get('output_dir', 'output/images')
        self.target_size = config.get('target_size', (224, 224))
        # Set max_slices_per_patient to None or a very large number to handle all slices
        self.max_slices_per_patient = config.get('max_slices', None)
        self.save_debug_images = config.get('save_debug_images', False)
        self.normalization_method = config.get('normalization', 'minmax')  # 'minmax', 'zscore', 'hist_eq'
        self.augmentation = config.get('augmentation', False)

        self.focus_on_glandular = config.get('focus_on_glandular', True)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        if self.save_debug_images:
            self.debug_dir = os.path.join(self.output_dir, 'debug')
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
    
    def process_patient_data(self, dicom_data, segmentation_data, pid):
        """
        Process DICOM and segmentation data of a single patient
        """
        breast_mask = segmentation_data.get('breast_mask')
 
        glandular_mask = segmentation_data.get('glandular_tissue_mask')
        
        dicom_instances = dicom_data.get('dicom_data', [])
        
        processed_images = []
        processed_masks = []
        
        # Select the number of slices to process
        total_slices = min(len(dicom_instances), breast_mask.shape[0])
        
        # If max_slices_per_patient is None or greater than total_slices, all slices are processed.
        # Otherwise, select the specified number of slices uniformly.
        if self.max_slices_per_patient is None or self.max_slices_per_patient >= total_slices:
            slice_indices = range(total_slices)
        else:
            slice_indices = np.linspace(0, total_slices-1, 
                                    self.max_slices_per_patient, 
                                    dtype=int)
        
        logger.info(f"处理PID={pid}的{len(slice_indices)}个切片")
        
        for idx in slice_indices:
            try:
                # Get DICOM pixel data
                pixel_array = dicom_instances[idx]['pixel_array']
                
                # Make sure the mask and image dimensions match
                if idx < breast_mask.shape[0]:
                    mask_slice = breast_mask[idx].astype(np.float32)
                    
                    # If you need to focus on glandular_tissue and have a corresponding mask
                    if self.focus_on_glandular and glandular_mask is not None and idx < glandular_mask.shape[0]:
                        glandular_slice = glandular_mask[idx].astype(np.float32)
                        # Make sure the glandular mask size matches the image
                        if glandular_slice.shape != pixel_array.shape:
                            glandular_slice = resize(glandular_slice, pixel_array.shape, 
                                                  preserve_range=True, anti_aliasing=True)
                            glandular_slice = (glandular_slice > 0.5).astype(np.float32)
                        
                        # Using glandular_tissue mask
                        active_mask = glandular_slice
                    else:
                        active_mask = mask_slice
                    
                    # If the dimensions don't match, resize the mask
                    if active_mask.shape != pixel_array.shape:
                        active_mask = resize(active_mask, pixel_array.shape, 
                                          preserve_range=True, anti_aliasing=True)
                        active_mask = (active_mask > 0.5).astype(np.float32)
                else:
                    logger.warning(f"Mask slice index {idx} out of range for PID={pid}")
                    continue
                
                # Apply mask for segmentation
                masked_image = pixel_array * active_mask
                
                # Preprocessing images
                processed_img = self._preprocess_image(masked_image, pid, idx)
                
                if processed_img is not None:
                    processed_images.append(processed_img)
                    processed_masks.append(active_mask)
 
                    if self.save_debug_images:
                        self._save_debug_images(pixel_array, active_mask, masked_image, 
                                              processed_img, pid, idx)
                
            except Exception as e:
                logger.error(f"Error processing slice {idx} for PID={pid}: {str(e)}")
        
        if not processed_images:
            logger.warning(f"PID={pid} No slices were processed successfully")
            return None, None
        
        # Merge the processed images into one array
        processed_images_array = np.array(processed_images)
        processed_masks_array = np.array(processed_masks)
        
        logger.info(f"Successfully processed {len(processed_images)} slices of PID={pid}")
        
        return processed_images_array, processed_masks_array
    
    def _approximate_glandular_tissue(self, breast_mask):
        """
        Approximately extract the glandular tissue region from the breast mask
        """
        # Copy the breast mask
        glandular_mask = breast_mask.copy()

        for i in range(breast_mask.shape[0]):
            mask_slice = breast_mask[i]

            if np.sum(mask_slice) == 0:
                continue
            
            # Use erosion to shrink mask edges
            eroded_mask = binary_erosion(mask_slice, iterations=5)
            
            # Then use the dilation operation to smooth the edges.
            smooth_mask = binary_dilation(eroded_mask, iterations=2)
            
            # Update Mask
            glandular_mask[i] = smooth_mask
        
        # Ensure that the glandular tissue area is no larger than the original breast area
        glandular_mask = glandular_mask * breast_mask
        
        return glandular_mask
    
    def _preprocess_image(self, image, pid, slice_idx):
        try:
            img = image.copy()
            
            if np.sum(img > 0) == 0:
                return None

            nonzero_mask = img > 0
            
            # Apply CLAHE 
            if np.sum(nonzero_mask) > 0:
                # Scale the image to the range 0-255 (uint8 type)
                img_8bit = np.zeros_like(img, dtype=np.uint8)
                if np.max(img[nonzero_mask]) > 0:
                    img_8bit[nonzero_mask] = (255 * (img[nonzero_mask] - np.min(img[nonzero_mask])) / 
                                            (np.max(img[nonzero_mask]) - np.min(img[nonzero_mask]))).astype(np.uint8)
        
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                if len(img.shape) == 2:  
                    img_enhanced = clahe.apply(img_8bit)
                else:
                    img_enhanced = np.zeros_like(img_8bit)
                    for i in range(img.shape[2]):
                        img_enhanced[:,:,i] = clahe.apply(img_8bit[:,:,i])
                
                # Normalize back to the 0-1 range
                img = img_enhanced.astype(np.float32) / 255.0
            
            # resize
            from skimage.transform import resize
            resized_img = resize(img, self.target_size, 
                            preserve_range=True, anti_aliasing=True)
            
            # Make sure the image is a floating point number and is in the range 0-1
            resized_img = resized_img.astype(np.float32)
            resized_img = np.clip(resized_img, 0, 1)
            
            # For deep learning models, three-channel images
            if len(resized_img.shape) == 2:
                resized_img = np.stack([resized_img] * 3, axis=-1)
            
            # Normalized to ImageNet mean and standard deviation
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # Normalize each channel
            for c in range(3):
                resized_img[:,:,c] = (resized_img[:,:,c] - mean[c]) / std[c]
                
            return resized_img
            
        except Exception as e:
            logger.error(f"Error preprocessing slice {slice_idx} for PID={pid}: {str(e)}")
            return None
   
    def batch_process(self, patients_data):
        """
        Batch process multiple patients' data
        """
        results = {}
        
        for pid, (dicom_data, segmentation_data) in tqdm(patients_data.items()"):
            images, masks = self.process_patient_data(dicom_data, segmentation_data, pid)
            
            if images is not None and masks is not None:
                results[pid] = (images, masks)

        return results
    
    def augment_data(self, images, masks):
        augmented_images = []
        augmented_masks = []
        
        # Copy the original data
        augmented_images.extend(images)
        augmented_masks.extend(masks)
        
        for i in range(len(images)):
            img = images[i]
            mask = masks[i]
            
            # 1. Flip Horizontal
            flipped_img = np.fliplr(img)
            flipped_mask = np.fliplr(mask)
            augmented_images.append(flipped_img)
            augmented_masks.append(flipped_mask)
            
            # 2.Random rotation (±15 degrees)
            angle = np.random.uniform(-15, 15)
            from scipy.ndimage import rotate
            rotated_img = rotate(img, angle, reshape=False, mode='nearest')
            rotated_mask = rotate(mask, angle, reshape=False, mode='nearest') > 0.5
            augmented_images.append(rotated_img)
            augmented_masks.append(rotated_mask)
            
            # 3.Brightness and contrast changes
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            adjusted_img = np.clip(img * brightness, 0, 1)  # 亮度调整
            adjusted_img = np.clip((adjusted_img - 0.5) * contrast + 0.5, 0, 1)  # 对比度调整
            augmented_images.append(adjusted_img)
            augmented_masks.append(mask)
            
            # 4.Slight zoom
            scale = np.random.uniform(0.9, 1.1)
            from skimage.transform import rescale
            h, w = img.shape[:2]
            scaled_img = rescale(img, scale, multichannel=True, mode='reflect', anti_aliasing=True)
            scaled_mask = rescale(mask, scale, multichannel=False, mode='reflect', anti_aliasing=True) > 0.5
            
            # Crop or pad to original size
            if scale > 1:  # Zoomed in, need to crop the center area
                h_diff = int((scaled_img.shape[0] - h) / 2)
                w_diff = int((scaled_img.shape[1] - w) / 2)
                scaled_img = scaled_img[h_diff:h_diff+h, w_diff:w_diff+w]
                scaled_mask = scaled_mask[h_diff:h_diff+h, w_diff:w_diff+w]
            else:  # Shrink, need to be filled
                h_diff = int((h - scaled_img.shape[0]) / 2)
                w_diff = int((w - scaled_img.shape[1]) / 2)
                padded_img = np.zeros_like(img)
                padded_mask = np.zeros_like(mask)
                padded_img[h_diff:h_diff+scaled_img.shape[0], w_diff:w_diff+scaled_img.shape[1]] = scaled_img
                padded_mask[h_diff:h_diff+scaled_mask.shape[0], w_diff:w_diff+scaled_mask.shape[1]] = scaled_mask
                scaled_img = padded_img
                scaled_mask = padded_mask
            
            augmented_images.append(scaled_img)
            augmented_masks.append(scaled_mask)
        
        return np.array(augmented_images), np.array(augmented_masks)

    
    def save_processed_data(self, processed_data, prefix='processed'):

        save_dir = os.path.join(self.output_dir, 'processed_arrays')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save metadata
        metadata = {
            'num_patients': len(processed_data),
            'patient_ids': list(processed_data.keys()),
            'target_size': self.target_size,
            'normalization': self.normalization_method
        }
        
        # Save each patient's data
        for pid, (images, masks) in processed_data.items():
            np.save(os.path.join(save_dir, f'{prefix}_images_pid_{pid}.npy'), images)
            np.save(os.path.join(save_dir, f'{prefix}_masks_pid_{pid}.npy'), masks)
    
        np.save(os.path.join(save_dir, f'{prefix}_metadata.npy'), metadata)

        return save_dir
    
    @classmethod
    def load_processed_data(cls, data_dir, patient_ids=None, prefix='processed'):

        result = {}
        
        # Loading metadata
        metadata_path = os.path.join(data_dir, f'{prefix}_metadata.npy')
        os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        available_pids = metadata['patient_ids']

        pids_to_load = patient_ids if patient_ids is not None else available_pids
        
        # Load each patient's data
        for pid in pids_to_load:
            if pid in available_pids:
                images_path = os.path.join(data_dir, f'{prefix}_images_pid_{pid}.npy')
                masks_path = os.path.join(data_dir, f'{prefix}_masks_pid_{pid}.npy')
                
                if os.path.exists(images_path) and os.path.exists(masks_path):
                    images = np.load(images_path)
                    masks = np.load(masks_path)
                    result[pid] = (images, masks)
                else:
                    logger.warning(f"The data file for patient {pid} does not exist")
            else:
                logger.warning(f"Patient {pid} is not in the list of available IDs")

        return result, metadata
    
    def extract_2d_slices(self, processed_data):
        """
        Extract 2D slices from 3D data for 2D CNN model
        """
        all_slices = []
        all_masks = []
        slice_to_patient = []
        
        for pid, (images, masks) in processed_data.items():
            for i in range(len(images)):
                all_slices.append(images[i])
                all_masks.append(masks[i])
                slice_to_patient.append(pid)

        all_slices = np.array(all_slices)
        all_masks = np.array(all_masks)
        
        logger.info(f"{{len(all_slices)} 2D slices extracted from the data of len(processed_data)} patients")
        
        return all_slices, all_masks, slice_to_patient
