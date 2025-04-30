import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import cv2
from skimage import measure, morphology, feature
from scipy import ndimage
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# Define attention block for the model
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate attention weights
        attention = self.sigmoid(self.conv(x))
        # Apply attention
        return x * attention


# Define an improved 3D model architecture with attention mechanism
class BreastDensity3DNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BreastDensity3DNet, self).__init__()
        
        # Input: [batch, channels=20, height=256, width=256]
        # 20 channels: 10 for breast mask slices, 10 for glandular mask slices
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention1 = AttentionBlock(32)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention2 = AttentionBlock(64)
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention3 = AttentionBlock(128)
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers with improved dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Apply convolutional blocks with attention
        x = self.conv_block1(x)
        x = self.attention1(x)
        
        x = self.conv_block2(x)
        x = self.attention2(x)
        
        x = self.conv_block3(x)
        x = self.attention3(x)
        
        x = self.conv_block4(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


class BreastDensity3DDataset(Dataset):
    def __init__(self, data_root, segmentation_root, patient_data, is_training=True):
        self.data_root = Path(data_root)
        self.segmentation_root = Path(segmentation_root)
        self.patient_data = patient_data
        self.is_training = is_training
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        
        for idx, row in self.patient_data.iterrows():
            patient_id = row['PID']
            label = row['density_encoded']
            
            # Find corresponding segmentation directory
            patient_seg_dirs = list(self.segmentation_root.glob(f"*{patient_id}*"))
            
            if not patient_seg_dirs:
                continue
            
            # Use the first matching directory
            patient_seg_dir = patient_seg_dirs[0]
            
            # Find the series directory
            series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]
            
            if not series_dirs:
                continue
            
            # Use the first series directory
            series_dir = series_dirs[0]
            
            # Check for breast_mask.npy and glandular_tissue_mask.npy files
            breast_mask_path = series_dir / "breast_mask.npy"
            glandular_mask_path = series_dir / "glandular_tissue_mask.npy"
            
            if not breast_mask_path.exists() or not glandular_mask_path.exists():
                continue
            
            # Store patient information and mask paths
            samples.append({
                'patient_id': patient_id,
                'label': label,
                'breast_mask_path': str(breast_mask_path),
                'glandular_mask_path': str(glandular_mask_path)
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def create_balanced_sampler(self, dataset_indices, all_labels):
        """
        Create a weighted sampler that assigns higher weights to rare classes (A, B, and D)
        """
        # Calculate the number of samples for each category
        class_counts = np.bincount(all_labels)
        print(f"Category distribution: {class_counts}")
        
        # Assign weights to each sample (inversely proportional to class frequency, with extra weights for classes A, B and D)
        weights = np.ones_like(all_labels, dtype=np.float32)
        
        for idx, label in enumerate(all_labels):
            # Base weight: the inverse of the category frequency
            base_weight = 1.0 / class_counts[label]
            if label == 0 or label == 3:
                weights[idx] = base_weight * 1.0
            else:
                weights[idx] = base_weight
        
        # Creating a Weighted Sampler
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset_indices),
            replacement=True
        )
        
        return sampler
    
    
    def apply_augmentations(self, breast_slices, glandular_slices, label=None, is_synthetic=False):
        """
       Adjusting the augmentation strategy based on whether it is a synthetic sample
        """
        if not self.is_training:
            return breast_slices, glandular_slices
        
        if is_synthetic:
            # For synthetic samples, mild augmentation is used
            # Mainly geometric transformation, maintaining basic features
            
            # Flip Horizontal
            if np.random.rand() > 0.5:
                breast_slices = np.flip(breast_slices, axis=2).copy()
                glandular_slices = np.flip(glandular_slices, axis=2).copy()
            
            # Random rotation (smaller angles)
            if np.random.rand() > 0.7:
                angle = np.random.uniform(-10, 10) 
                for i in range(breast_slices.shape[0]):
                    breast_slices[i] = ndimage.rotate(
                        breast_slices[i], angle=angle, reshape=False, mode='nearest'
                    )
                    glandular_slices[i] = ndimage.rotate(
                        glandular_slices[i], angle=angle, reshape=False, mode='nearest'
                    )
            
            # For synthetic samples, no more additional noise or complex transformations are added
            # Because the synthesis process has introduced enough variability
        else:
            # For the original sample, use more dynamic and standard enhancements
            
            # Flip Horizontal
            if np.random.rand() > 0.5:
                breast_slices = np.flip(breast_slices, axis=2).copy()
                glandular_slices = np.flip(glandular_slices, axis=2).copy()

             # Additional enhancements for rare class
            if label in [0, 1, 3]:
                # Increase vertical flip probability to improve generalization ability
                if np.random.rand() > 0.7:
                    breast_slices = np.flip(breast_slices, axis=1).copy()
                    glandular_slices = np.flip(glandular_slices, axis=1).copy()
                    
                # Add a larger angle of rotation
                if np.random.rand() > 0.6:  # Increase the probability
                    angle = np.random.uniform(-20, 20)  # Increase the rotation angle range
                    for i in range(breast_slices.shape[0]):
                        breast_slices[i] = ndimage.rotate(
                            breast_slices[i], angle=angle, reshape=False, mode='nearest'
                        )
                        glandular_slices[i] = ndimage.rotate(
                            glandular_slices[i], angle=angle, reshape=False, mode='nearest'
                        )
            
            # Random rotation
            if np.random.rand() > 0.7:
                angle = np.random.uniform(-15, 15)
                for i in range(breast_slices.shape[0]):
                    breast_slices[i] = ndimage.rotate(
                        breast_slices[i], angle=angle, reshape=False, mode='nearest'
                    )
                    glandular_slices[i] = ndimage.rotate(
                        glandular_slices[i], angle=angle, reshape=False, mode='nearest'
                    )
            
            # Brightness Adjustment
            if np.random.rand() > 0.7:
                scale = np.random.uniform(0.85, 1.15)
                breast_slices = np.clip(breast_slices * scale, 0, 1)
                glandular_slices = np.clip(glandular_slices * scale, 0, 1)
            
            # Adding Noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 0.05, breast_slices.shape)
                breast_slices = np.clip(breast_slices + noise, 0, 1)
                noise = np.random.normal(0, 0.05, glandular_slices.shape)
                glandular_slices = np.clip(glandular_slices + noise, 0, 1)
            
            # Random elastic deformation (mild, for pristine samples)
            if np.random.rand() > 0.8:
                for i in range(breast_slices.shape[0]):
                    if np.sum(breast_slices[i]) > 0:
                        shape = breast_slices[i].shape
                        dx = ndimage.gaussian_filter(np.random.randn(*shape) * 2, sigma=4)
                        dy = ndimage.gaussian_filter(np.random.randn(*shape) * 2, sigma=4)
                        
                        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
                        
                        # Apply deformation
                        if np.sum(breast_slices[i]) > 0:
                            breast_slices[i] = ndimage.map_coordinates(
                                breast_slices[i], indices, order=1
                            ).reshape(shape)
                        
                        if np.sum(glandular_slices[i]) > 0:
                            glandular_slices[i] = ndimage.map_coordinates(
                                glandular_slices[i], indices, order=1
                            ).reshape(shape)

            glandular_slices = np.minimum(glandular_slices, breast_slices)
        
        return breast_slices, glandular_slices
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset and enhance the processing of the synthetic sample
        """
        sample = self.samples[idx]

        # Gets the flag of whether the sample is a synthetic sample
        is_synthetic = sample.get('is_synthetic', False)
        
        breast_mask = np.load(sample['breast_mask_path'])
        glandular_mask = np.load(sample['glandular_mask_path'])
        label = sample['label']

        breast_slice_indices = np.where(np.sum(breast_mask, axis=(1, 2)) > 0)[0]
        
        if len(breast_slice_indices) == 0:
            # If there is no breast tissue, create an empty array
            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
        else:
            # Select the central area of ​​the breast volume
            mid_idx = len(breast_slice_indices) // 2
            start_idx = max(0, mid_idx - 5)
            end_idx = min(breast_mask.shape[0], start_idx + 10)

            selected_indices = breast_slice_indices[start_idx:end_idx]

            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)

            for i, slice_idx in enumerate(selected_indices):
                if i >= 10: 
                    break
 
                breast_slice = breast_mask[slice_idx]
                glandular_slice = glandular_mask[slice_idx]
                
                # resize
                breast_slice_resized = cv2.resize(breast_slice.astype(np.float32), (256, 256))
                glandular_slice_resized = cv2.resize(glandular_slice.astype(np.float32), (256, 256))
                
                processed_breast_mask[i] = breast_slice_resized
                processed_glandular_mask[i] = glandular_slice_resized
        
        # Applying Data Augmentation - Passing the Synthetic Samples Flag
        processed_breast_mask, processed_glandular_mask = self.apply_augmentations(
            processed_breast_mask, processed_glandular_mask, label, is_synthetic
        )
        
        # Create combined input: 20 channels
        input_tensor = np.concatenate([processed_breast_mask, processed_glandular_mask], axis=0)

        input_tensor = torch.from_numpy(input_tensor).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        is_synthetic_tensor = torch.tensor(is_synthetic, dtype=torch.bool)

        return input_tensor, label_tensor, is_synthetic_tensor

class BreastDensityPipeline:
    def __init__(self, data_root, segmentation_root, excel_path, output_dir):
        self.data_root = Path(data_root)
        self.segmentation_root = Path(segmentation_root)
        self.excel_path = excel_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patient_data = None
        self.features = None
        self.labels = None
        self.model = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_patient_data(self):
        df = pd.read_excel(self.excel_path)

        df = df[['PID', 'density']]
        
        # Keep only patients with density labels
        labeled_patients = df.dropna(subset=['density']).copy()

        le = LabelEncoder()
        labeled_patients['density_encoded'] = le.fit_transform(labeled_patients['density'])

        with open(self.output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)

        self.patient_data = labeled_patients
        return labeled_patients
    
    def synthesize_minority_samples(self, minority_samples, n_synthetic=5):
        """
        Create synthetic samples for rare classes (A, B, and D)
        """
        print(f"Generate {n_synthetic} times more synthetic samples for {len(minority_samples)} rare class samples")
        synthetic_samples = []
        
        for sample in tqdm(minority_samples):
            patient_id = sample['patient_id']
            label = sample['label']
            breast_mask_path = sample['breast_mask_path']
            glandular_mask_path = sample['glandular_mask_path']
   
            actual_n_synthetic = n_synthetic
            if label == 0:  # class A are extremely rare, more synthetic samples are needed
                actual_n_synthetic = n_synthetic * 2
            elif label == 1:  # Class B also adds some
                actual_n_synthetic = int(n_synthetic * 1.5)
            elif label == 3: # Also for D
                actual_n_synthetic = int(n_synthetic * 1.5)
         
            breast_mask = np.load(breast_mask_path)
            glandular_mask = np.load(glandular_mask_path)
            
            for i in range(actual_n_synthetic):
                syn_breast = breast_mask.copy()
                syn_glandular = glandular_mask.copy()
                
                # Apply different transformations based on the classes
                if label == 0: 
                    # Randomly adjust the amount of glandular tissue (decrease or slight increase)
                    factor = np.random.uniform(0.7, 1.1)
                    syn_glandular = np.clip(syn_glandular * factor, 0, 1)
                    
                    # Randomly move gland positions
                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            shift_y = np.random.randint(-5, 6)
                            shift_x = np.random.randint(-5, 6)
                            syn_glandular[slice_idx] = ndimage.shift(
                                syn_glandular[slice_idx], 
                                (shift_y, shift_x), 
                                mode='constant', 
                                cval=0
                            )
                  
                            mask = syn_glandular[slice_idx] > 0
                            if np.sum(mask) > 0:
                                noise = np.random.normal(0, 0.1, syn_glandular[slice_idx].shape)
                                syn_glandular[slice_idx][mask] += noise[mask]
                                syn_glandular[slice_idx] = np.clip(syn_glandular[slice_idx], 0, 1)

                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            binary = syn_glandular[slice_idx] > 0.3
                            if np.sum(binary) > 0:
                                # Apply the open operation to remove small areas
                                opened = morphology.opening(binary, morphology.disk(1))
                                # Set the non-zero fields to their original values
                                syn_glandular[slice_idx] = syn_glandular[slice_idx] * opened
                
                elif label == 1:  # B
                    # Randomly adjust the amount of glandular tissue (slightly increase or decrease)
                    factor = np.random.uniform(0.8, 1.2)
                    syn_glandular = np.clip(syn_glandular * factor, 0, 1)
                    
                    # Randomize the gland positions to make them more dispersed
                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            shape = syn_glandular[slice_idx].shape
                            # Create a random displacement field - Class B requires more dispersed glands
                            dx = ndimage.gaussian_filter(np.random.randn(*shape) * 2, sigma=3)
                            dy = ndimage.gaussian_filter(np.random.randn(*shape) * 2, sigma=3)

                            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

                            syn_glandular[slice_idx] = ndimage.map_coordinates(
                                syn_glandular[slice_idx], 
                                indices, 
                                order=1
                            ).reshape(shape)
                    
                    # Randomly add or remove small areas
                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            if np.random.rand() > 0.5: 
                                breast_mask_slice = syn_breast[slice_idx] > 0
                                if np.sum(breast_mask_slice) > 0:
                                    points = np.where(breast_mask_slice)
                                    num_points = min(len(points[0]), 5)  
                                    if num_points > 0:
                                        random_indices = np.random.choice(len(points[0]), num_points, replace=False)
                                        for idx in random_indices:
                                            y, x = points[0][idx], points[1][idx]
                         
                                            radius = np.random.randint(2, 5)
                                            y_min, y_max = max(0, y-radius), min(breast_mask_slice.shape[0], y+radius+1)
                                            x_min, x_max = max(0, x-radius), min(breast_mask_slice.shape[1], x+radius+1)
                        
                                            y_grid, x_grid = np.ogrid[y_min-y:y_max-y, x_min-x:x_max-x]
                                            mask_region = x_grid**2 + y_grid**2 <= radius**2
                   
                                            region_shape = (y_max - y_min, x_max - x_min)
                                            if mask_region.shape != region_shape:
                                                mask_region = mask_region[:region_shape[0], :region_shape[1]]
                           
                                            syn_glandular[slice_idx][y_min:y_max, x_min:x_max][mask_region] = np.random.uniform(0.3, 0.6)
                    
                elif label == 3:  # D
                    # Random increase in glandular tissue
                    factor = np.random.uniform(1.0, 1.3)
                    syn_glandular = np.clip(syn_glandular * factor, 0, 1)
                    
                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            binary = syn_glandular[slice_idx] > 0.3
                            if np.sum(binary) > 0:
                                # Apply closure operations to connect regions
                                closed = morphology.closing(binary, morphology.disk(np.random.randint(1, 3)))
                                # Set the new zone value to a slightly lower value
                                new_regions = closed & (~binary)
                                syn_glandular[slice_idx][new_regions] = np.random.uniform(0.4, 0.7)

                                gradient_mask = morphology.dilation(closed, morphology.disk(2)) & (~closed)
                                syn_glandular[slice_idx][gradient_mask] = np.random.uniform(0.2, 0.4)

                    for slice_idx in range(syn_glandular.shape[0]):
                        if np.sum(syn_glandular[slice_idx]) > 0:
                            shape = syn_glandular[slice_idx].shape
                            dx = ndimage.gaussian_filter(np.random.randn(*shape) * 3, sigma=5)
                            dy = ndimage.gaussian_filter(np.random.randn(*shape) * 3, sigma=5)
           
                            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
                   
                            syn_glandular[slice_idx] = ndimage.map_coordinates(
                                syn_glandular[slice_idx], 
                                indices, 
                                order=1
                            ).reshape(shape)
                
                # Make sure the synthetic glands remain within the breast mask
                syn_glandular = np.minimum(syn_glandular, syn_breast)

                syn_patient_id = f"{patient_id}_syn{i+1}"
                syn_breast_path = str(self.output_dir / f"synthetic/{syn_patient_id}_breast.npy")
                syn_glandular_path = str(self.output_dir / f"synthetic/{syn_patient_id}_glandular.npy")
     
                os.makedirs(os.path.dirname(syn_breast_path), exist_ok=True)

                np.save(syn_breast_path, syn_breast)
                np.save(syn_glandular_path, syn_glandular)

                synthetic_samples.append({
                    'patient_id': syn_patient_id,
                    'label': label,
                    'breast_mask_path': syn_breast_path,
                    'glandular_mask_path': syn_glandular_path,
                    'is_synthetic': True 
                })

        return synthetic_samples
    
    def extract_features_from_segmentation(self):
        features_list = []
        patient_ids = []
        valid_patients = []

        for idx, row in tqdm(self.patient_data.iterrows(), total=len(self.patient_data)):
            patient_id = row['PID']

            patient_seg_dirs = list(self.segmentation_root.glob(f"*{patient_id}*"))
            patient_seg_dir = patient_seg_dirs[0]

            series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]
            series_dir = series_dirs[0]

            breast_mask_path = series_dir / "breast_mask.npy"
            glandular_mask_path = series_dir / "glandular_tissue_mask.npy"

            breast_mask = np.load(breast_mask_path)
            glandular_mask = np.load(glandular_mask_path)
            
            # Extract features
            patient_features = self._compute_patient_features(breast_mask, glandular_mask)

            features_list.append(patient_features)
            patient_ids.append(patient_id)
            valid_patients.append(row)

        # Create a feature DataFrame
        features_df = pd.DataFrame(features_list)
        features_df['PID'] = patient_ids

        valid_patients_df = pd.DataFrame(valid_patients)

        merged_df = pd.merge(features_df, valid_patients_df, on='PID')

        self.features = merged_df.drop(['PID', 'density', 'density_encoded'], axis=1, errors='ignore')
        self.labels = merged_df['density_encoded'].values

        merged_df.to_csv(self.output_dir / 'extracted_features.csv', index=False)
        
        return merged_df
    
    def _compute_patient_features(self, breast_mask, glandular_mask):
        # Calculate features for a single patient
        breast_volume = np.sum(breast_mask)
        glandular_volume = np.sum(glandular_mask)

        density_ratio = glandular_volume / breast_volume if breast_volume > 0 else 0

        features = {
            'breast_volume': breast_volume,
            'glandular_volume': glandular_volume,
            'density_ratio': density_ratio,
        }

        slice_features = []
        for slice_idx in range(breast_mask.shape[0]):
            breast_slice = breast_mask[slice_idx]
            glandular_slice = glandular_mask[slice_idx]

            slice_breast_area = np.sum(breast_slice)
            if slice_breast_area == 0:
                continue
            
            slice_glandular_area = np.sum(glandular_slice)
            slice_density_ratio = slice_glandular_area / slice_breast_area
            
            # Calculate the morphological characteristics of glandular tissue
            if np.any(glandular_slice):
                labeled_glandular, num_regions = measure.label(glandular_slice, return_num=True)

                if num_regions > 0:
                    regions = measure.regionprops(labeled_glandular)

                    region_areas = [region.area for region in regions]

                    slice_features.append({
                        'slice_idx': slice_idx,
                        'slice_breast_area': slice_breast_area,
                        'slice_glandular_area': slice_glandular_area,
                        'slice_density_ratio': slice_density_ratio,
                        'num_glandular_regions': num_regions,
                        'max_glandular_region_area': max(region_areas) if region_areas else 0,
                        'mean_glandular_region_area': np.mean(region_areas) if region_areas else 0
                    })
        
        # If there are slice features, calculate their aggregate statistics
        if slice_features:
            slice_df = pd.DataFrame(slice_features)

            for col in ['slice_breast_area', 'slice_glandular_area', 'slice_density_ratio', 
                       'num_glandular_regions', 'max_glandular_region_area', 'mean_glandular_region_area']:
                features[f'{col}_mean'] = slice_df[col].mean()
                features[f'{col}_std'] = slice_df[col].std()
                features[f'{col}_max'] = slice_df[col].max()

            max_density_slice = slice_df.loc[slice_df['slice_density_ratio'].idxmax()]
            features['max_density_slice_idx'] = max_density_slice['slice_idx']
            features['max_density_slice_breast_area'] = max_density_slice['slice_breast_area']
            features['max_density_slice_glandular_area'] = max_density_slice['slice_glandular_area']
            
            # Count the number of sections with glandular tissue
            features['num_slices_with_glandular'] = len(slice_df)
            features['pct_slices_with_glandular'] = len(slice_df) / breast_mask.shape[0]
        
        # Fill missing features with 0
        for key in features.keys():
            if features[key] is None or np.isnan(features[key]):
                features[key] = 0
        
        return features
    
    def calculate_class_weights(self, all_labels):
        """
        Calculate weights for each class based on their frequency
        """
        # Convert to numpy array if not already
        labels_array = np.array(all_labels)
        
        # Count occurrences of each class
        class_counts = np.bincount(labels_array)
        
        # Calculate weights (inverse of frequency)
        n_samples = len(labels_array)
        n_classes = len(class_counts)
        
        # Handle possible missing classes in the dataset
        if len(class_counts) < 4:  # Assuming 4 classes (A, B, C, D)
            expanded_counts = np.zeros(4)
            expanded_counts[:len(class_counts)] = class_counts
            class_counts = expanded_counts
            n_classes = 4
        
        # Calculate weights - inverse of frequency
        weights = n_samples / (n_classes * class_counts)
        
        # If a class is missing, set a high weight
        weights[class_counts == 0] = max(weights) * 2
        
        return torch.FloatTensor(weights).to(self.device)
    
    def train_with_cross_validation(self, epochs=50, batch_size=4, learning_rate=0.0001, 
                       use_class_weights=True, n_folds=5, 
                       early_stopping_patience=10, use_mixed_precision=True,
                       generate_synthetic=True, rare_classes=[0, 1, 3]):
        """
        Use cross-validation to train multiple models and select the best model
        """        
        train_dataset = BreastDensity3DDataset(
            self.data_root,
            self.segmentation_root,
            self.patient_data,
            is_training=True
        )
        
        print(f"找到 {len(train_dataset)} 个有效患者数据")

        # Verify dataset integrity
        valid_samples = []
        problematic_samples = []

        for idx, sample in enumerate(tqdm(train_dataset.samples, desc="Check the samples")):
            if os.path.exists(sample['breast_mask_path']) and os.path.exists(sample['glandular_mask_path']):
                breast_mask = np.load(sample['breast_mask_path'])
                glandular_mask = np.load(sample['glandular_mask_path'])
                    
                if (breast_mask.ndim >= 2 and glandular_mask.ndim >= 2 and 
                    np.sum(breast_mask) > 0):
                    valid_samples.append(sample)
                else:
                    problematic_samples.append((idx, "维度不正确或为空", sample['patient_id']))
            else:
                problematic_samples.append((idx, "文件不存在", sample['patient_id']))

        # Update dataset sample list
        original_count = len(train_dataset.samples)
        train_dataset.samples = valid_samples

        all_labels = []
        all_is_synthetic = []
        for sample in train_dataset.samples:
            all_labels.append(sample['label'])
            all_is_synthetic.append(sample.get('is_synthetic', False))
        
        all_labels = np.array(all_labels)
        all_is_synthetic = np.array(all_is_synthetic)
        
        # Only use real samples for cross validation splits
        real_indices = np.where(~all_is_synthetic)[0].tolist()
        real_labels = all_labels[real_indices]
        
        # Additional handling of rare classes - ensure that every fold has rare classes
        rare_indices_by_class = {}
        for cls in rare_classes:
            rare_indices_by_class[cls] = [i for i, (idx, label) in enumerate(zip(real_indices, real_labels)) if label == cls]
        
        # Use Stratified K-fold Cross Validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
  
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(real_indices, real_labels)):

            all_labels = []
            all_is_synthetic = []
            for sample in train_dataset.samples:
                all_labels.append(sample['label'])
                all_is_synthetic.append(sample.get('is_synthetic', False))
            
            all_labels = np.array(all_labels)
            all_is_synthetic = np.array(all_is_synthetic)

            real_indices = np.where(~all_is_synthetic)[0].tolist()
            real_labels = all_labels[real_indices]

            fold_train_indices = [real_indices[i] for i in train_idx]
            fold_val_indices = [real_indices[i] for i in val_idx]
            
            # Check if each rare class is included in the validation set
            val_labels = [all_labels[i] for i in fold_val_indices]
            val_classes_present = set(val_labels)
            
            # Make sure each rare class has at least one example in the validation set
            for cls in rare_classes:
                if cls not in val_classes_present and rare_indices_by_class.get(cls):
                    # If there is no rare category in the validation set, but there is in the dataset
                    if rare_indices_by_class[cls]:
                        total_samples_in_class = len(rare_indices_by_class[cls])
                        
                        # If there is only one example of this class, create a copy for validation
                        if total_samples_in_class == 1:
                            sample_to_use = real_indices[rare_indices_by_class[cls][0]]
                            # Not removed from the training set
                            fold_val_indices.append(sample_to_use)  # Add to validation set
                        else:
                            # Move a sample from the training set to the validation set
                            sample_to_move = real_indices[rare_indices_by_class[cls][0]]
                            if sample_to_move in fold_train_indices:
                                fold_train_indices.remove(sample_to_move)
                                fold_val_indices.append(sample_to_move) 
            
            # If using synthetic samples, add all synthetic samples to the training set
            if generate_synthetic:
                synthetic_indices = np.where(all_is_synthetic)[0].tolist()
                if synthetic_indices:
                    fold_train_indices.extend(synthetic_indices)
                    print(f"Add {len(synthetic_indices)} synthetic examples to the training set")
                else:
                    minority_samples = []
                    for i in fold_train_indices:
                        if all_labels[i] in rare_classes:
                            minority_samples.append(train_dataset.samples[i])
                    
                    if minority_samples:
                        print(f"Find {len(minority_samples)} rare class samples for synthesis")
                        synthetic_samples = self.synthesize_minority_samples(minority_samples)
     
                        original_len = len(train_dataset.samples)
                        train_dataset.samples.extend(synthetic_samples)

                        new_indices = list(range(original_len, len(train_dataset.samples)))
                        fold_train_indices.extend(new_indices)
                        print(f"Add {len(new_indices)} newly generated synthetic samples to the training set")

                        for sample in synthetic_samples:
                            all_labels = np.append(all_labels, sample['label'])
                            all_is_synthetic = np.append(all_is_synthetic, True)

            fold_train_set = Subset(train_dataset, fold_train_indices)
            fold_val_set = Subset(train_dataset, fold_val_indices)

            fold_train_labels = [all_labels[i] for i in fold_train_indices]
            fold_val_labels = [all_labels[i] for i in fold_val_indices]
            
            print("Training set category distribution:")
            train_class_counts = np.bincount(fold_train_labels, minlength=4)
            for cls, count in enumerate(train_class_counts):
                if count > 0:
                    print(f"  Class {cls}: {count} ({count/len(fold_train_labels)*100:.1f}%)")
            
            print("Validation set category distribution:")
            val_class_counts = np.bincount(fold_val_labels, minlength=4)
            for cls, count in enumerate(val_class_counts):
                if count > 0:
                    print(f"  Class {cls}: {count} ({count/len(fold_val_labels)*100:.1f}%)")

            fold_train_loader = DataLoader(
                fold_train_set, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            fold_val_loader = DataLoader(
                fold_val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            model = BreastDensity3DNet().to(self.device)

            if use_class_weights:
                smooth_weights = np.ones(4, dtype=np.float32)
                smooth_weights[0] = 1.5  # A类
                smooth_weights[1] = 1.0  # B类
                smooth_weights[2] = 0.7  # C类
                smooth_weights[3] = 1.2  # D类
                
                class_weights = torch.FloatTensor(smooth_weights).to(self.device)
                print(f"Use class weights: {class_weights}")
                criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            else:
                criterion = FocalLoss(gamma=2.0)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.5, weight_decay=2e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=1e-6
            )

            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_model_state = None
            no_improve_epochs = 0
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
 
            scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
            
            # train
            print(f"Start training for {epochs} epochs")
            start_time = time.time()
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                rare_class_correct = {cls: 0 for cls in rare_classes}
                rare_class_total = {cls: 0 for cls in rare_classes}
                
                for batch in tqdm(fold_train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                    inputs, labels, is_synthetic = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    
                    if use_mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    for cls in rare_classes:
                        cls_mask = (labels == cls)
                        if torch.sum(cls_mask) > 0:
                            rare_class_total[cls] += torch.sum(cls_mask).item()
                            rare_class_correct[cls] += torch.sum(predicted[cls_mask] == labels[cls_mask]).item()
                
                # Calculate the average loss and accuracy
                train_loss = train_loss / train_total if train_total > 0 else 0
                train_acc = train_correct / train_total if train_total > 0 else 0
                
                # validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in tqdm(fold_val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        inputs, labels, _ = batch
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
   
                val_loss = val_loss / val_total if val_total > 0 else float('inf')
                val_acc = val_correct / val_total if val_total > 0 else 0

                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                print(f"  Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                for cls in rare_classes:
                    if rare_class_total[cls] > 0:
                        cls_acc = rare_class_correct[cls] / rare_class_total[cls]
                        print(f"    Class {cls}: Acc: {cls_acc:.4f}, Samples: {rare_class_total[cls]}")
                
                # Check if it is the best model
                is_best = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                elif val_loss == best_val_loss and val_acc > best_val_acc:
                    is_best = True
                
                if is_best:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    no_improve_epochs = 0

                    fold_model_path = self.output_dir / f'fold_{fold+1}_best_model.pth'
                    torch.save({
                        'model_state_dict': best_model_state,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'epoch': epoch
                    }, fold_model_path)
                    
                    print(f"Find the new best model. Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= early_stopping_patience:
                        break
            
            # Load the best model for evaluation
            model.load_state_dict(best_model_state)

            model.eval()
            val_correct = 0
            val_total = 0
            fold_all_preds = []
            fold_all_labels = []
            fold_class_correct = {cls: 0 for cls in range(4)}
            fold_class_total = {cls: 0 for cls in range(4)}
            
            with torch.no_grad():
                for batch in fold_val_loader:
                    inputs, labels, _ = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Accuracy by category
                    for cls in range(4):
                        idx = (labels == cls)
                        if idx.sum() > 0:
                            fold_class_total[cls] += idx.sum().item()
                            fold_class_correct[cls] += (predicted[idx] == labels[idx]).sum().item()
                    
                    fold_all_preds.extend(predicted.cpu().numpy())
                    fold_all_labels.extend(labels.cpu().numpy())
            
            # Calculate the final index
            fold_acc = val_correct / val_total if val_total > 0 else 0

            conf_matrix = confusion_matrix(fold_all_labels, fold_all_preds)
            class_acc = {}
            for cls in range(4):
                if fold_class_total[cls] > 0:
                    class_acc[cls] = fold_class_correct[cls] / fold_class_total[cls]
                else:
                    class_acc[cls] = 0.0

            fold_result = {
                'fold': fold + 1,
                'val_acc': fold_acc,
                'val_loss': best_val_loss,
                'class_acc': class_acc,
                'conf_matrix': conf_matrix,
                'model_state': best_model_state,
            }
            
            fold_results.append(fold_result)
            print(f"Fold {fold+1} 完成，验证准确率: {fold_acc:.4f}")
            print("类别准确率:")
            for cls, acc in class_acc.items():
                if fold_class_total[cls] > 0:
                    print(f"  Class {cls}: {acc:.4f} ({fold_class_total[cls]} samples)")
        
        # Analyze all fold results
        fold_accs = [res['val_acc'] for res in fold_results]
        avg_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        print(f"Average validation accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print("Verification accuracy of each fold:")
        for i, acc in enumerate(fold_accs):
            print(f"  Fold {i+1}: {acc:.4f}")

        best_fold_idx = np.argmax(fold_accs)
        best_fold = fold_results[best_fold_idx]
        
        # Save the best model
        torch.save({
            'model_state_dict': best_fold['model_state'],
            'val_acc': best_fold['val_acc'],
            'val_loss': best_fold['val_loss'],
            'class_acc': best_fold['class_acc'],
            'cv_results': {
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'fold_accs': fold_accs
            }
        }, self.output_dir / 'best_3d_dl_model.pth')
        
        # Save all cross validation results
        with open(self.output_dir / 'cv_results.pkl', 'wb') as f:
            pickle.dump({
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'fold_results': fold_results
            }, f)
        
        return {
            'model': best_fold['model_state'],
            'val_acc': best_fold['val_acc'],
            'class_acc': best_fold['class_acc'],
            'cv_results': {
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'fold_accs': fold_accs
            }
        }

    def ensure_balanced_validation_split(self, all_labels, indices, min_samples_per_class=2, target_ratio=0.2, force_rare_classes=True, rare_classes=[0, 1, 3]):
        """
        Make sure the validation set contains a certain percentage of samples from each existing class, with special emphasis on rare classes.
        """
        all_labels_array = np.array(all_labels)
        unique_classes = np.unique(all_labels_array)
        train_indices = []
        val_indices = []
        
        # Count the number of samples for each category
        class_counts = {}
        for cls in unique_classes:
            class_counts[cls] = np.sum(all_labels_array == cls)
        
        # Calculate the number of samples that should be assigned to the validation set for each category
        #large classes do not completely occupy the validation set
        total_samples = len(all_labels)
        
        # First, make sure each class has at least min_samples_per_class samples
        allocated_samples = 0
        class_val_sizes = {}
        
        # Treat rare classes specially - make sure there is at least one sample
        for cls in rare_classes:
            if cls in class_counts and class_counts[cls] > 0:
                # Make sure the validation set has at least one example but no more than half of the total number of examples in the class
                class_val_sizes[cls] = min(max(1, int(class_counts[cls] * target_ratio)), class_counts[cls] // 2)
                if class_val_sizes[cls] == 0 and class_counts[cls] > 1:
                    class_val_sizes[cls] = 1
                allocated_samples += class_val_sizes[cls]
        
        # Processing other calasses
        for cls in unique_classes:
            if cls not in class_val_sizes:  # Skip processed rare categories
                class_val_sizes[cls] = min(max(min_samples_per_class, int(class_counts[cls] * target_ratio)), class_counts[cls] // 2)
                allocated_samples += class_val_sizes[cls]
        
        # Special handling for very small classes
        for cls, count in class_counts.items():
            if count <= 2 and cls not in rare_classes:
                if count == 1:
                    class_val_sizes[cls] = 0  # If there is only 1 sample, put it into the training set
                elif count == 2:
                    class_val_sizes[cls] = 1  # If there are 2 samples, 1 is a validation set
        
        # Enforce rare class
        if force_rare_classes:
            for cls in rare_classes:
                if cls in class_counts and class_counts[cls] > 0:
                    # Ensure that there is at least one sample, even if the total number is small
                    if class_val_sizes[cls] == 0:
                        class_val_sizes[cls] = 1
        
        # Performing the actual segmentation
        for cls in unique_classes:
            class_indices = np.where(all_labels_array == cls)[0]
            class_indices = [indices[i] for i in class_indices] 
            
            if len(class_indices) == 0:
                continue
            
            # Get the number of samples that this category should be assigned to the validation set
            val_size = class_val_sizes[cls]
            
            # Handling various situations
            if val_size == 0:
                train_indices.extend(class_indices)
            elif val_size >= len(class_indices):
                if len(class_indices) == 1:
                    # If there is only one sample and it needs to be put into the validation set, put it into both the training set and the validation set
                    val_indices.append(class_indices[0])
                    train_indices.append(class_indices[0])
                else:
                    # Keep at least one in the training set
                    np.random.seed(42 + int(cls))
                    np.random.shuffle(class_indices)
                    val_indices.extend(class_indices[:-1])
                    train_indices.append(class_indices[-1])
            else:
                np.random.shuffle(class_indices)
                val_indices.extend(class_indices[:val_size])
                train_indices.extend(class_indices[val_size:])
        
        total_indices = len(train_indices) + len(val_indices)
        
        return train_indices, val_indices

    def train_with_model_selection(self, model_names=None, epochs=50, batch_size=4, 
                                learning_rate=0.0001, use_class_weights=True,
                                early_stopping_patience=10, use_mixed_precision=True,
                                use_balanced_sampling=True, generate_synthetic=True, 
                                rare_classes=[0, 1, 3], validation_split=0.2):
        """
        Using model selection to train an optimal breast density classification model
        """
        from model_selection import ModelSelector, model_registry, FocalLoss

        train_dataset = BreastDensity3DDataset(
            self.data_root,
            self.segmentation_root,
            self.patient_data,
            is_training=True
        )
        
        # Verify dataset integrity
        valid_samples = []
        problematic_samples = []

        for idx, sample in enumerate(tqdm(train_dataset.samples, desc="检查样本")):
            try:
                if os.path.exists(sample['breast_mask_path']) and os.path.exists(sample['glandular_mask_path']):
                    breast_mask = np.load(sample['breast_mask_path'])
                    glandular_mask = np.load(sample['glandular_mask_path'])
                    
                    if (breast_mask.ndim >= 2 and glandular_mask.ndim >= 2 and 
                        np.sum(breast_mask) > 0):
                        valid_samples.append(sample)
                    else:
                        problematic_samples.append((idx, "维度不正确或为空", sample['patient_id']))
                else:
                    problematic_samples.append((idx, "文件不存在", sample['patient_id']))
            except Exception as e:
                problematic_samples.append((idx, f"加载错误: {e}", sample['patient_id']))

        # Update dataset sample list
        original_count = len(train_dataset.samples)
        train_dataset.samples = valid_samples
        
        # Automatically adjust batch size if there are too few samples
        if len(valid_samples) < batch_size * 2:
            batch_size = max(1, len(valid_samples) // 4)

        # Create a copy of the validation set, also using the valid sample list
        val_dataset = BreastDensity3DDataset(
            self.data_root,
            self.segmentation_root,
            self.patient_data,
            is_training=False
        )
        val_dataset.samples = valid_samples.copy()

        all_labels = []
        for sample in train_dataset.samples:
            all_labels.append(sample['label'])
        
        all_labels = np.array(all_labels)
        
        # Check if you need to generate synthetic samples for rare classes
        if generate_synthetic:

            minority_samples = []
            for i, label in enumerate(all_labels):
                if label in rare_classes and i < len(train_dataset.samples):
                    minority_samples.append(train_dataset.samples[i])
            
            if minority_samples:
                synthetic_samples = self.synthesize_minority_samples(minority_samples)

                if synthetic_samples:
                    train_dataset.samples.extend(synthetic_samples)
        
        indices = list(range(len(train_dataset)))
               
        updated_all_labels = []
        is_synthetic_sample = []
        for sample in train_dataset.samples:
            updated_all_labels.append(sample['label'])
            is_synthetic_sample.append(sample.get('is_synthetic', False))
            
        # First put all synthetic samples into the training set
        synthetic_indices = [i for i, is_syn in enumerate(is_synthetic_sample) if is_syn]
        real_indices = [i for i, is_syn in enumerate(is_synthetic_sample) if not is_syn]
            
        # Perform stratified sampling on real samples
        real_labels = [updated_all_labels[i] for i in real_indices]
            
        # Perform balanced partitioning on real samples
        real_train_indices, val_indices = self.ensure_balanced_validation_split(
                real_labels, 
                real_indices, 
                min_samples_per_class=2,
                target_ratio=validation_split,
                force_rare_classes=True,
                rare_classes=rare_classes
            )

        train_indices = synthetic_indices + real_train_indices

        train_set = torch.utils.data.Subset(train_dataset, train_indices)
        val_set = torch.utils.data.Subset(val_dataset, val_indices)

        val_batch_size = min(batch_size, len(val_set))

        if use_balanced_sampling:
            train_labels = [updated_all_labels[i] for i in train_indices]

            weights = np.ones_like(train_labels, dtype=np.float32)
            class_sample_counts = np.bincount(train_labels)
            
            for idx, label in enumerate(train_labels):
                if class_sample_counts[label] > 0:
                    base_weight = 1.0 / np.sqrt(class_sample_counts[label])
                    
                    # Distinguishing between original and synthetic samples
                    sample_idx = train_indices[idx]
                    if sample_idx < len(train_dataset.samples):
                        sample = train_dataset.samples[sample_idx]
                        is_synthetic = sample.get('is_synthetic', False)
                        
                        if is_synthetic:
                            weights[idx] = base_weight * 0.8
                        else:
                            weights[idx] = base_weight
                else:
                    weights[idx] = 1.0

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_indices),
                replacement=True
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=sampler,
                num_workers=2, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True
            )

        val_loader = torch.utils.data.DataLoader(
            val_set, 
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        if use_class_weights:
            # Use flat class weights
            smooth_weights = np.ones(4, dtype=np.float32)
            smooth_weights[0] = 2.0  # A类
            smooth_weights[1] = 1.5  # B类
            smooth_weights[2] = 0.7  # C类
            smooth_weights[3] = 1.5  # D类
            
            class_weights = torch.FloatTensor(smooth_weights).to(self.device)
            print(f"使用平缓的类别权重: {class_weights}")
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            criterion = FocalLoss(gamma=2.0)
        model_selector = ModelSelector(self.device)

        model_names is None:
        # Use all registered models
        models = model_selector.add_all_models(num_classes=4)
        
        # Run model selection
        optimizer_params_list = [
            {'lr': learning_rate * 0.75, 'weight_decay': 2e-5},  # 轻量模型
            {'lr': learning_rate * 0.5, 'weight_decay': 2e-5},   # 默认模型
            {'lr': learning_rate * 0.25, 'weight_decay': 3e-5},  # 深层模型
            {'lr': learning_rate * 0.25, 'weight_decay': 3e-5}   # ResNet模型
        ]
        
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        scheduler_params = {
            'T_0': 15,
            'T_mult': 2,
            'eta_min': 1e-6
        }
        
        best_model_name, results = model_selector.run_model_selection(
            train_loader,
            val_loader,
            criterion,
            optimizer_class=torch.optim.Adam,
            optimizer_params_list=optimizer_params_list,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            use_mixed_precision=use_mixed_precision,
            rare_classes=rare_classes
        )
        
        # Plotting the comparison results
        output_dir = self.output_dir / 'model_selection_results'
        model_selector.plot_comparison_results(output_dir, rare_classes)
        
        # save best model
        best_model_state = results[best_model_name]['model_state']
        best_model = model_selector.models[best_model_name]
        
        torch.save({
            'model_name': best_model_name,
            'model_state_dict': best_model_state,
            'accuracy': results[best_model_name]['accuracy'],
            'val_loss': results[best_model_name]['best_val_loss'],
            'class_weights': class_weights if use_class_weights else None,
            'rare_class_metrics': results[best_model_name]['rare_class_metrics'],
            'results_summary': {name: {'accuracy': res['accuracy'], 'val_loss': res['best_val_loss']} 
                            for name, res in results.items()}
        }, self.output_dir / 'best_selected_model.pth')

        return best_model_name, best_model, results[best_model_name]


    def predict_with_selected_model(self, patient_id, model_name=None):
        from model_selection import model_registry
        
        # Find the corresponding segmentation result directory
        patient_seg_dirs = list(self.segmentation_root.glob(f"*{patient_id}*"))

        model_path = self.output_dir / 'best_selected_model.pth'
      
        checkpoint = torch.load(model_path, map_location=self.device)

        if model_name is None:
            if 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
            else:
                 model_name = "BreastDensity3DNet"
                    
        print(f"Use model architecture: {model_name}")
 
        model_class = model_registry.get_model(model_name)
        model = model_class(num_classes=4).to(self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        patient_seg_dir = patient_seg_dirs[0]

        series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]
        series_dir = series_dirs[0]

        breast_mask_path = series_dir / "breast_mask.npy"
        glandular_mask_path = series_dir / "glandular_tissue_mask.npy"

        breast_mask = np.load(breast_mask_path)
        glandular_mask = np.load(glandular_mask_path)

        breast_slice_indices = np.where(np.sum(breast_mask, axis=(1, 2)) > 0)[0]
        
        if len(breast_slice_indices) == 0:
            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
        else:
            mid_idx = len(breast_slice_indices) // 2
            start_idx = max(0, mid_idx - 5)
            end_idx = min(breast_mask.shape[0], start_idx + 10)
            selected_indices = breast_slice_indices[start_idx:end_idx]
     
            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
    
            for i, slice_idx in enumerate(selected_indices):
                if i >= 10:
                    break
       
                breast_slice = breast_mask[slice_idx]
                glandular_slice = glandular_mask[slice_idx]
                
                # resize
                breast_slice_resized = cv2.resize(breast_slice.astype(np.float32), (256, 256))
                glandular_slice_resized = cv2.resize(glandular_slice.astype(np.float32), (256, 256))
                
                processed_breast_mask[i] = breast_slice_resized
                processed_glandular_mask[i] = glandular_slice_resized

        input_tensor = np.concatenate([processed_breast_mask, processed_glandular_mask], axis=0)

        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        with open(self.output_dir / 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        density_class = le.inverse_transform([predicted.item()])[0]
        
        # Calculate class probabilities
        probs = probabilities.cpu().numpy()[0]
        prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
        
        # Extract other information relevant to the prediction, such as volume scale
        stats = {
            'breast_volume': float(np.sum(breast_mask)),
            'glandular_volume': float(np.sum(glandular_mask)),
            'density_ratio': float(np.sum(glandular_mask) / np.sum(breast_mask)) if np.sum(breast_mask) > 0 else 0
        }
        
        return {
            'patient_id': patient_id,
            'model_name': model_name,
            'predicted_class': density_class,
            'class_probabilities': prob_dict,
            'stats': stats
        }
    
    def _plot_training_curves(self, history):
        plt.figure(figsize=(15, 10))
        
        # Plotting the loss curve
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Draw the accuracy curve
        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png')
        plt.close()
    
    def _evaluate_model(self, model, data_loader):
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels, _  in tqdm(data_loader, desc="Evaluating the Model"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Try loading the label encoder to get the category names
        try:
            with open(self.output_dir / 'label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            class_names = le.classes_
        except:
            # If unable to load, use default category name
            class_names = ['A', 'B', 'C', 'D']

        print(f"Accuracy: {accuracy:.4f}")
        
        # Plotting the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
        
        # 返回评估结果
        eval_results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        return eval_results
    
    def predict_with_3d_model(self, patient_id):
 
        model_path = self.output_dir / 'best_3d_dl_model.pth'
        
        try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Creating a Model Instance
                model = BreastDensity3DNet().to(self.device)
                
                # Choose the correct state dictionary based on the structure of the checkpoint
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    # If the checkpoint is directly the state dictionary
                    model.load_state_dict(checkpoint)

                model.eval()
        except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        # Find the corresponding segmentation result directory
        patient_seg_dirs = list(self.segmentation_root.glob(f"*{patient_id}*"))
        patient_seg_dir = patient_seg_dirs[0]

        series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]

        series_dir = series_dirs[0]

        breast_mask_path = series_dir / "breast_mask.npy"
        glandular_mask_path = series_dir / "glandular_tissue_mask.npy"

        breast_mask = np.load(breast_mask_path)
        glandular_mask = np.load(glandular_mask_path)

        breast_slice_indices = np.where(np.sum(breast_mask, axis=(1, 2)) > 0)[0]
        
        if len(breast_slice_indices) == 0:
            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
        else:
            mid_idx = len(breast_slice_indices) // 2
            start_idx = max(0, mid_idx - 5)
            end_idx = min(breast_mask.shape[0], start_idx + 10)

            selected_indices = breast_slice_indices[start_idx:end_idx]

            processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
            processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
  
            for i, slice_idx in enumerate(selected_indices):
                if i >= 10: 
                    break

                breast_slice = breast_mask[slice_idx]
                glandular_slice = glandular_mask[slice_idx]

                breast_slice_resized = cv2.resize(breast_slice.astype(np.float32), (256, 256))
                glandular_slice_resized = cv2.resize(glandular_slice.astype(np.float32), (256, 256))
                
                processed_breast_mask[i] = breast_slice_resized
                processed_glandular_mask[i] = glandular_slice_resized

        input_tensor = np.concatenate([processed_breast_mask, processed_glandular_mask], axis=0)

        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        with open(self.output_dir / 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        
        # Convert predicted numbers to density categories
        density_class = le.inverse_transform([predicted.item()])[0]
        
        # Calculate class probabilities
        probs = probabilities.cpu().numpy()[0]
        prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
        
        # Extract other information relevant to the prediction, such as volume scale
        stats = {
            'breast_volume': float(np.sum(breast_mask)),
            'glandular_volume': float(np.sum(glandular_mask)),
            'density_ratio': float(np.sum(glandular_mask) / np.sum(breast_mask)) if np.sum(breast_mask) > 0 else 0
        }
        
        return {
            'patient_id': patient_id,
            'predicted_class': density_class,
            'class_probabilities': prob_dict,
            'stats': stats
        }
    
    def predict_all_unlabeled_with_3d(self):
        """
        Predicting breast density of all unlabeled patients using a 3D deep learning model
        """
        df = pd.read_excel(self.excel_path)
        
        # Find all patients without a density label
        unlabeled_patients = df[df['density'].isna()].copy()
        
        results = []
        
        # Iterate over each unlabeled patient
        for idx, row in tqdm(unlabeled_patients.iterrows(), total=len(unlabeled_patients)):
            patient_id = row['PID']

            prediction = self.predict_with_3d_model(patient_id)
            
            if prediction:
                results.append(prediction)
        
        print(f"Successfully predicted breast density for {len(results)}/{len(unlabeled_patients)} patients")

        if results:
            results_df = pd.DataFrame([
                {
                    'PID': r['patient_id'],
                    'predicted_density': r['predicted_class'],
                    'probability_A': r['class_probabilities'].get('A', 0),
                    'probability_B': r['class_probabilities'].get('B', 0),
                    'probability_C': r['class_probabilities'].get('C', 0),
                    'probability_D': r['class_probabilities'].get('D', 0),
                    'breast_volume': r['stats']['breast_volume'],
                    'glandular_volume': r['stats']['glandular_volume'],
                    'density_ratio': r['stats']['density_ratio']
                }
                for r in results
            ])
            
            # Save prediction results
            results_df.to_csv(self.output_dir / '3d_dl_unlabeled_predictions.csv', index=False)
            
            # Visualizing the Prediction Distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x='predicted_density', data=results_df, palette='viridis')
            plt.title('Predicted distribution of breast density in unlabeled patients')
            plt.xlabel('Predicted breast density class')
            plt.ylabel('Number of patients')
            plt.tight_layout()
            plt.savefig(self.output_dir / '3d_dl_prediction_distribution.png')
            plt.close()
            
            # Visualize the relationship between density ratio and predicted density level
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='predicted_density', y='density_ratio', data=results_df, palette='viridis')
            plt.title('Relationship between predicted density class and density ratio')
            plt.xlabel('Predicted density class')
            plt.ylabel('Density ratio (glandular volume/breast volume)')
            plt.tight_layout()
            plt.savefig(self.output_dir / '3d_dl_density_ratio_by_predicted_class.png')
            plt.close()
            
            return results_df
        
        return None
    
    def visual_feature_analysis(self, rare_classes=[0,1, 3]):
        """
        Visualize the relationship between features and breast density, with a special focus on rare categories
        """
        # Loading the Tag Encoder
        with open(self.output_dir / 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        
        # Create a DataFrame containing feature and density labels
        df = self.features.copy()
        df['density_class'] = le.inverse_transform(self.labels)
        df['density_numeric'] = self.labels  # Add value labels
        df['is_rare_class'] = df['density_numeric'].isin(rare_classes)  # Marking rare classes
        
        # Analyze the relationship between density ratio and density level, and highlight rare categories
        plt.figure(figsize=(12, 8))
        if 'density_ratio' in df.columns:
            ax = sns.boxplot(x='density_class', y='density_ratio', data=df, palette='viridis')
            
            # Add a separate scatter plot on the rare category to make it more visible
            rare_df = df[df['is_rare_class']]
            sns.stripplot(x='density_class', y='density_ratio', data=rare_df, 
                        color='red', jitter=True, size=8, marker='X', linewidth=1)
            
            # Add labels for rare classes
            for idx, row in rare_df.iterrows():
                plt.text(
                    list(le.classes_).index(row['density_class']), 
                    row['density_ratio'], 
                    f"{row.get('patient_id', 'ID')}",
                    horizontalalignment='left', 
                    size='small', 
                    color='darkred'
                )
            
            plt.title('Relationship between breast density ratio and density class (rare classes highlighted)')
            plt.xlabel('Density class')
            plt.ylabel('Density ratio (glandular tissue/breast volume)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'density_ratio_by_class_rare_highlighted.png')
            plt.close()
        
        # Create a feature distribution map for each rare class and compare it to other classes
        for rare_class in rare_classes:
            rare_label = le.inverse_transform([rare_class])[0]
            print(f"分析稀有类别 {rare_label} (code: {rare_class}) 的特征分布...")
            
            # Select the most important features for visualization
            important_features = ['breast_volume', 'glandular_volume', 'density_ratio']
            
            for feat in additional_features:
                if feat in df.columns:
                    important_features.append(feat)

            
            fig, axes = plt.subplots(len(important_features), 1, figsize=(10, 4*len(important_features)))
        
        # Create multi-feature scatter plots for samples in rare categories, highlighting their feature space locations
        plt.figure(figsize=(10, 8))
        
        # Select two important features for a 2D scatter plot
        if all(feat in df.columns for feat in ['breast_volume', 'glandular_volume']):
            plt.scatter(
                df[df['is_rare_class'] == False]['breast_volume'],
                df[df['is_rare_class'] == False]['glandular_volume'],
                c=df[df['is_rare_class'] == False]['density_numeric'],
                cmap='viridis',
                alpha=0.6,
                s=50,
                label='Common Classes'
            )
            
            # Use different shapes to mark rare categories
            markers = ['*', 'X', 'P', 'D']
            for i, rare_class in enumerate(rare_classes):
                rare_df = df[df['density_numeric'] == rare_class]
                if not rare_df.empty:
                    plt.scatter(
                        rare_df['breast_volume'],
                        rare_df['glandular_volume'],
                        marker=markers[i % len(markers)],
                        s=200,
                        edgecolor='black',
                        label=f'Class {le.inverse_transform([rare_class])[0]}'
                    )
                    
                    # Add labels
                    for idx, row in rare_df.iterrows():
                        plt.annotate(
                            f"{row.get('patient_id', 'ID')}",
                            (row['breast_volume'], row['glandular_volume']),
                            fontsize=8,
                            color='black',
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
            
            plt.title('Breast Volume vs. Glandular Volume')
            plt.xlabel('Breast Volume')
            plt.ylabel('Glandular Volume')
            plt.colorbar(label='Density Class')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'rare_classes_feature_space.png')
            plt.close()
    
    def visualize_predictions(self, patient_id, rare_classes=[0,1, 3]):
        """
        Visualize predictions for a single patient
        """

        patient_seg_dirs = list(self.segmentation_root.glob(f"*{patient_id}*"))
        patient_seg_dir = patient_seg_dirs[0]

        series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]

        series_dir = series_dirs[0]

        breast_mask_path = series_dir / "breast_mask.npy"
        glandular_mask_path = series_dir / "glandular_tissue_mask.npy"

        breast_mask = np.load(breast_mask_path)
        glandular_mask = np.load(glandular_mask_path)
        
        # Get predictions for all available models
        predictions = {}

        dl_model_path = self.output_dir / 'final_3d_dl_model.pth'


        dl_prediction = self.predict_with_3d_model_enhanced(patient_id)
           
        if dl_prediction:
            predictions['3D DL'] = dl_prediction
        
        # Get the true label
        patient_row = self.patient_data[self.patient_data['PID'] == patient_id]
        true_label = None
        density_encoded = None
        is_rare_class = False

        # 1. Find the slice with breast tissue
        breast_indices = np.where(np.sum(breast_mask, axis=(1, 2)) > 0)[0]

        # 2. Select a representative slice (center slice)
        mid_idx = breast_indices[len(breast_indices) // 2]
        
        # 3. Creating custom window functions for better display of CT images
        def apply_window(image, window_center=50, window_width=400):
            window_min = window_center - window_width // 2
            window_max = window_center + window_width // 2
            
            windowed = np.clip(image, window_min, window_max)
            windowed = (windowed - window_min) / (window_max - window_min)
            
            return windowed
        
        # 4. Creating a Visual Catalog
        viz_dir = self.output_dir / 'patient_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 5. Visualize the center slice
        fig = plt.figure(figsize=(15, 10))

        title = f"Breast segmentation and density prediction for patient {patient_id}"
        if true_label:
            title += f" (True density level: {true_label})"
        fig.suptitle(title, fontsize=16)
        
        # 5.1 Breast Mask
        ax1 = plt.subplot(2, 3, 1)
        breast_slice = breast_mask[mid_idx]
        plt.imshow(breast_slice, cmap='Blues')
        plt.title(f"Breast Mask (slice {mid_idx})")
        plt.axis('off')
        
        # 5.2 Glandular tissue mask
        ax2 = plt.subplot(2, 3, 2)
        glandular_slice = glandular_mask[mid_idx]
        plt.imshow(glandular_slice, cmap='Reds')
        plt.title(f"Glandular tissue mask (slice {mid_idx})")
        plt.axis('off')
        
        # 5.3 Mask Coverage
        ax3 = plt.subplot(2, 3, 3)
        plt.imshow(breast_slice, cmap='Blues', alpha=0.5)
        plt.imshow(glandular_slice, cmap='Reds', alpha=0.5)
        plt.title("Breast and glandular tissue coverage")
        plt.axis('off')
        
        # 5.4 Density Ratio Pie Chart
        ax4 = plt.subplot(2, 3, 4)
        breast_volume = np.sum(breast_mask)
        glandular_volume = np.sum(glandular_mask)
        density_ratio = glandular_volume / breast_volume if breast_volume > 0 else 0
        
        labels = ['Glandular tissue', 'Non Glandular tissue']
        sizes = [glandular_volume, breast_volume - glandular_volume]
        colors = ['#ff9999', '#66b3ff']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        plt.title(f"Breast tissue composition\nDensity ratio: {density_ratio:.4f}")
        
        # 5.5 Bar chart of forecast results
        ax5 = plt.subplot(2, 3, 5)
        
        if predictions:
                isinstance(predictions.get('3D DL', {}), dict) and 'predicted_class' in predictions['3D DL']:
                prediction_info = predictions['3D DL']
                predicted_class = prediction_info.get('predicted_class')
                class_probs = prediction_info.get('class_probabilities', {})
  
                class_names = sorted(class_probs.keys())
                class_values = [class_probs[c] for c in class_names]
  
                colors = []
                for cls in class_names:
                    if cls == predicted_class:
                        colors.append('#ff9900')  
                    elif cls == true_label:
                        colors.append('#33cc33')  
                    else:
                        colors.append('#cccccc')  
    
                bars = plt.bar(class_names, class_values, color=colors)
                
                # Add value labels
                for bar, val in zip(bars, class_values):
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.02,
                        f'{val:.2f}',
                        ha='center', va='bottom',
                        fontsize=9
                    )
                
                # Add a legend
                legend_elements = [
                    plt.Rectangle((0,0),1,1, color='#ff9900', label='Prediction Class'),
                    plt.Rectangle((0,0),1,1, color='#33cc33', label='True class') if true_label else None,
                    plt.Rectangle((0,0),1,1, color='#cccccc', label='other classes')
                ]
                legend_elements = [e for e in legend_elements if e]
                plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
                plt.title(f"Density category probability distribution\n(prediction: {predicted_class})")
                plt.ylim(0, 1.1)
                plt.ylabel('Probability')
    
        else:
            plt.text(0.5, 0.5, "No prediction results", ha='center', va='center', fontsize=12)
            plt.axis('off')

        output_path = viz_dir / f"patient_{patient_id}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        
        # Creating a multi-slice visualization
        # Find all the slides with breast tissue
        valid_slices = [i for i in breast_indices if i < breast_mask.shape[0]]
        
        if len(valid_slices) >= 3:
            # Select some representative slices (beginning, middle, end)
            if len(valid_slices) <= 6:
                selected_slices = valid_slices
            else:
                # Uniform selection of slices
                step = len(valid_slices) // 6
                selected_slices = valid_slices[::step][:6]
 
            fig, axes = plt.subplots(2, len(selected_slices), figsize=(15, 6))

            title = f"Multi-slice breast analysis for patient {patient_id}"
            if true_label:
                title += f" (Density level: {true_label})"
            if is_rare_class:
                title += f" - rare classes"
                   
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

            multi_slice_path = viz_dir / f"patient_{patient_id}_multi_slice.png"
            plt.savefig(multi_slice_path, dpi=150, bbox_inches='tight')
            plt.close()

        
        # Returns visualization result information
        result = {
            'patient_id': patient_id,
            'true_label': true_label,
            'density_encoded': density_encoded,
            'is_rare_class': is_rare_class,
            'predictions': predictions,
            'breast_volume': breast_volume,
            'glandular_volume': glandular_volume,
            'density_ratio': density_ratio,
            'visualization_paths': {
                'main': str(output_path)
            }
        }
        
        return result
