#版本2
import os
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology, segmentation, filters
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import binary_erosion, binary_dilation, disk

class BreastSegmentation:
    def __init__(self, dicom_folder):
        self.dicom_folder = dicom_folder
        self.ct_image = None
        self.image_spacing = None
        self.skin_mask = None
        self.thoracic_cavity_mask = None
        self.fat_mask = None
        self.combined_tissues_mask = None
        self.glandular_tissue_mask = None
        self.anterior_muscles_mask = None
        self.posterior_muscles_mask = None
        self.breast_mask = None

    def smooth_profile(self, profile, window_size=5):
        """Smooth the Profile curve to reduce noise effects"""
        return np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        
    def load_dicom_series(self, study_id, series_id):
        """
        Load a DICOM series from the dataset
        """
        series_folder = os.path.join(self.dicom_folder, study_id, series_id)
        if not os.path.exists(series_folder):
            raise ValueError(f"Series folder {series_folder} does not exist")
        
        # Get list of DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_folder):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_folder}")
        
        print(f"Loading {len(dicom_files)} DICOM files...")
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        
        # Convert to numpy array and reorient if needed
        self.ct_image = sitk.GetArrayFromImage(image)
        self.image_spacing = image.GetSpacing()

            # Check if it is a color image (has 3 channels)
        if len(self.ct_image.shape) == 4 and self.ct_image.shape[3] == 3:
            # Convert to grayscale (take average or use specific channel)
            self.ct_image = np.mean(self.ct_image, axis=3).astype(self.ct_image.dtype)
        
    def apply_hounsfield_window(self, image, window_center, window_width):
        """
        Apply a Hounsfield window to the CT image
        """
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        
        windowed_image = np.clip(image, window_min, window_max)
        windowed_image = (windowed_image - window_min) / (window_max - window_min)
        
        return windowed_image
    
    def segment_skin(self):
        """
        Segment skin from the CT image (Step 1)
        Based on the method described in the paper reference [7]
        """
        print("Segmenting skin...")
        
        # Apply soft tissue window
        windowed_image = self.apply_hounsfield_window(self.ct_image, window_center=50, window_width=400)
        
        # Create a binary mask of everything outside air
        # Threshold is typically -500 to -700 HU for air/tissue boundary
        air_threshold = -600
        tissue_mask = self.ct_image > air_threshold
        
        # Find the skin boundary using morphological operations
        self.skin_mask = np.zeros_like(tissue_mask)
        
        for i in range(tissue_mask.shape[0]):
            # Process each slice
            slice_mask = tissue_mask[i].copy()

             # Check dimension of slice_mask before applying morphological operations
           # print(f"Processing slice {i}, shape: {slice_mask.shape}")

            # Ensure we're using a 2D structuring element for 2D data
            struct_elem = morphology.disk(3)

            # Erode to remove skin
            eroded = morphology.binary_erosion(slice_mask, struct_elem)
            
            # Skin is the difference between the original and eroded mask
            skin_boundary = slice_mask & ~eroded
            
            # Dilate the boundary to get complete skin
            skin = morphology.binary_dilation(skin_boundary, morphology.disk(1))
            
            # Fill holes in the body to get ROI
            filled = ndimage.binary_fill_holes(slice_mask)
            
            # Store the result
            self.skin_mask[i] = filled
        
        return self.skin_mask
    
    def segment_thoracic_cavity(self):
        """
        Approximate the thoracic cavity using a convex hull of sternum and ribs
        """
        print("Segmenting thoracic cavity...")
        
        # For this implementation, use a simplified approach since i don't have
        # rib and sternum segmentation readily available
        
        self.thoracic_cavity_mask = np.zeros_like(self.skin_mask)
        
        # In each slice, use the fact that lungs are dark (low HU values)
        for i in range(self.ct_image.shape[0]):
            # Create binary mask for lungs (air regions inside the body)
            lung_threshold = -400  # Typical HU for lung tissue
            slice_mask = (self.ct_image[i] < lung_threshold) & self.skin_mask[i]
            
            # Label connected components to find lungs
            labeled_mask, num_labels = measure.label(slice_mask, return_num=True)
            
            if num_labels > 0:
                # Get region properties
                regions = measure.regionprops(labeled_mask)
                
                # Sort regions by area (descending)
                regions = sorted(regions, key=lambda x: x.area, reverse=True)
                
                # Take the two largest regions (left and right lungs)
                lungs_mask = np.zeros_like(slice_mask)
                for j in range(min(2, len(regions))):
                    if j < len(regions):  # Check if regions has at least j+1 elements
                        lungs_mask[labeled_mask == regions[j].label] = 1
                
                # Create a convex hull to approximate thoracic cavity
                if np.any(lungs_mask):
                    # Get the convex hull of the lungs
                    hull_mask = morphology.convex_hull_image(lungs_mask)
                    
                    # Dilate to include surrounding tissues
                    hull_mask = morphology.binary_dilation(hull_mask, morphology.disk(10))
                    
                    # Ensure we don't extend beyond the skin
                    hull_mask = hull_mask & self.skin_mask[i]
                    
                    self.thoracic_cavity_mask[i] = hull_mask
        
        return self.thoracic_cavity_mask
    
    def visualize_multiple_slices(self, interval=5, num_slices=3):
        """
        Visualize the segmentation results across multiple slices
        """
        if self.breast_mask is None:
            print("Please run segment_breast() first")
            return
        
        # Find slices with breast tissue
        tissue_per_slice = np.sum(self.breast_mask, axis=(1, 2))
        valid_slices = np.where(tissue_per_slice > 0)[0]
        
        if len(valid_slices) == 0:
            print("No breast tissue found in segmentation")
            return
        
        # Select center slice and slices at regular intervals
        center_idx = valid_slices[len(valid_slices) // 2]
        
        # Generate slice indices at regular intervals
        start_idx = max(0, center_idx - (num_slices//2) * interval)
        slice_indices = [start_idx + i * interval for i in range(num_slices)]
        slice_indices = [idx for idx in slice_indices if idx < self.ct_image.shape[0]]
        
        # Create a figure with rows for each slice and columns for each segmentation
        fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 5 * len(slice_indices)))
        
        # If only one slice, make sure axes is 2D
        if len(slice_indices) == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i, slice_idx in enumerate(slice_indices):
            # Window the image
            windowed_image = self.apply_hounsfield_window(self.ct_image[slice_idx], 50, 400)
            
            # Original image with glandular tissue overlay
            axes[i, 0].imshow(windowed_image, cmap='gray')
            axes[i, 0].imshow(self.glandular_tissue_mask[slice_idx], alpha=0.5, cmap='Reds')
            axes[i, 0].set_title(f'Slice {slice_idx}: Glandular Tissue')
            axes[i, 0].axis('off')
            
            # Original image with fat tissue overlay
            axes[i, 1].imshow(windowed_image, cmap='gray')
            axes[i, 1].imshow(self.fat_mask[slice_idx], alpha=0.5, cmap='YlOrBr')
            axes[i, 1].set_title(f'Slice {slice_idx}: Fat Tissue')
            axes[i, 1].axis('off')
            
            # Original image with breast segmentation overlay
            axes[i, 2].imshow(windowed_image, cmap='gray')
            axes[i, 2].imshow(self.breast_mask[slice_idx], alpha=0.5, cmap='Greens')
            axes[i, 2].set_title(f'Slice {slice_idx}: Breast Segmentation')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def segment_fat_tissues(self):
        """
        Segment fat tissues based on HU values (Step 2.2)
        """
        print("Segmenting fat tissues...")
        
        # Fat tissue typically has HU values between -100 and -50
        #use a slightly wider range to account for variations in LDCT
        fat_lower = -150
        fat_upper = -20 
        
        self.fat_mask = (self.ct_image >= fat_lower) & (self.ct_image <= fat_upper) & self.skin_mask
        
        # Apply morphological operations to clean up the fat mask
        for i in range(self.fat_mask.shape[0]):
            # First perform an opening operation to remove small noise points, then perform a closing operation to fill small holes
            self.fat_mask[i] = morphology.binary_opening(self.fat_mask[i], morphology.disk(1))
            self.fat_mask[i] = morphology.binary_closing(self.fat_mask[i], morphology.disk(3))
        
        return self.fat_mask
    
    def segment_combined_tissues(self):
        """
        Segment combined tissues (MG) - muscles and mammary glands (Step 2.2)
        """
        print("Segmenting combined tissues (MG)...")
        
        # Update region of interest to exclude thoracic cavity
        ROI = self.skin_mask & ~self.thoracic_cavity_mask
        
        # Adjusted HU ranges for glandular and muscle tissue to better capture all relevant tissues
        # Glandular and muscle tissues typically have HU values between 20 and 100
        glandular_lower = 0 
        glandular_upper = 120 
        dense_tissue_mask = (self.ct_image >= glandular_lower) & (self.ct_image <= glandular_upper)
        
        # Changing the way composite tissue is defined to ensure all non-fat tissue is captured
        # Combined tissues are non-fat tissues in ROI
        self.combined_tissues_mask = ROI & (~self.fat_mask) & (self.ct_image > -20)
        
        # Adding an additional glandular tissue enhancement step
        # Pay special attention to dense tissue
        dense_mask = (self.ct_image >= 20) & (self.ct_image <= 100) & ROI
        self.combined_tissues_mask = self.combined_tissues_mask | dense_mask
        
        # Apply morphological operations to clean up the mask
        for i in range(self.combined_tissues_mask.shape[0]):
            # Adjust the order of morphological operations and the size of the structuring element
            # First close to fill small holes
            self.combined_tissues_mask[i] = morphology.binary_closing(self.combined_tissues_mask[i], morphology.disk(3))
            # Then open to remove small isolated regions
            self.combined_tissues_mask[i] = morphology.binary_opening(self.combined_tissues_mask[i], morphology.disk(1))
        
        return self.combined_tissues_mask
    
    def separate_posterior_muscles(self):
        """
        Separate the mammary area from the posterior muscles
        """
        print("Separation of the posterior muscles...")
        
        self.posterior_muscles_mask = np.zeros_like(self.skin_mask)
        updated_roi = np.zeros_like(self.skin_mask)
        
        # Process each axial slice
        for i in range(self.ct_image.shape[0]):
            # Skip slices without grouped tissue
            if not np.any(self.combined_tissues_mask[i]):
                continue
            
            # Identify lung regions as anatomical references
            lung_mask = self.get_hounsfield_range_mask(self.ct_image[i], -1000, -650)
            lung_mask = morphology.remove_small_objects(lung_mask, min_size=50)
            
            # If lungs are detected, use them to help locate the anterior and posterior regions
            if np.any(lung_mask):
                labeled_lungs = measure.label(lung_mask)
                lung_regions = measure.regionprops(labeled_lungs)
                
                if lung_regions:
                    # Calculate the center point of the lung
                    lung_pixels = np.argwhere(lung_mask)
                    if len(lung_pixels) > 0:
                        lung_centroid = np.mean(lung_pixels, axis=0)
                        
                        #Anterior region (in front of the lungs)
                        height, width = self.ct_image[i].shape
                        posterior_mask = np.zeros_like(self.ct_image[i], dtype=bool)
                        
                        # The posterior region is defined as below the center of the lung
                        for y in range(height):
                            if y > lung_centroid[0]:
                                posterior_mask[y, :] = True
                        
                        # Identifying the posterior muscles
                        self.posterior_muscles_mask[i] = self.combined_tissues_mask[i] & posterior_mask
                        
                        # Front area reserved
                        updated_roi[i] = self.skin_mask[i] & ~self.thoracic_cavity_mask[i] & ~self.posterior_muscles_mask[i]
                        continue
            
            # If lungs are not detected, a simple front-to-back segmentation is used
            height = self.ct_image[i].shape[0]
            # Define the back region as the back 1/2 of the image
            for y in range(height):
                if y > height//2:
                    self.posterior_muscles_mask[i, y, :] = self.combined_tissues_mask[i, y, :]
                    
            updated_roi[i] = self.skin_mask[i] & ~self.thoracic_cavity_mask[i] & ~self.posterior_muscles_mask[i]
        
        return updated_roi
    
    def get_hounsfield_range_mask(self, volume, min_hu, max_hu):
        """
        Creates a mask for voxels within a specified Hounsfield unit range
        """
        return (volume >= min_hu) & (volume <= max_hu)

    
    def separate_anterior_muscles(self, updated_roi):
        """
        Separate the mammary glandular tissue from the anterior muscles and restrict identification to the upper front area of ​​the chest
        """
        print("Separate the anterior muscles and identify the glandular tissue...")
        
        # Initialize the anterior muscle mask
        self.anterior_muscles_mask = np.zeros_like(self.skin_mask)
        
        # Acquire regions of interest for combined tissue
        mg_roi = self.combined_tissues_mask & updated_roi
        
        # 1: Initial Location-Based Growth
        # The muscle is located on the surface of the chest cavity and is separated from the external mammary gland by a layer of fat.
        
        # Initialize muscles using voxels from the thorax boundary
        boundary = morphology.binary_dilation(self.thoracic_cavity_mask, morphology.ball(2)) & \
                ~self.thoracic_cavity_mask
        initial_muscles = boundary & mg_roi
        
        # Create a distance map to fat tissue for each slice
        distance_to_fat = np.zeros_like(self.ct_image, dtype=float)
        
        for i in range(self.ct_image.shape[0]):
            if np.any(mg_roi[i]) and np.any(self.fat_mask[i]):
                # Calculate the distance to the nearest fat tissue
                fat_distance_map = ndimage.distance_transform_edt(~self.fat_mask[i])
                # Normalize distances to reduce inter-slice variability
                if np.max(fat_distance_map) > 0:
                    fat_distance_map = fat_distance_map / np.max(fat_distance_map) * 10
                distance_to_fat[i] = fat_distance_map
        
        # Initial muscle growth based on distance to fat
        grown_muscles = initial_muscles.copy()
        
        for i in range(self.ct_image.shape[0]):
            if np.any(grown_muscles[i]):
                # Maximum distance for muscle growth
                max_distance = 8
                
                # Create expansion structures to expand the initial muscle
                dilated_muscles = morphology.binary_dilation(
                    grown_muscles[i], 
                    morphology.disk(max_distance)
                )
                
                # Includes MG voxels near the chest cavity within the expansion area
                growth_area = dilated_muscles & (distance_to_fat[i] <= max_distance) & mg_roi[i]
                
                # Consider the CT value for additional constraints: Muscles are usually in the range of HU values ​​30-80
                muscle_hu_mask = (self.ct_image[i] >= 30) & (self.ct_image[i] <= 80)
                growth_area = growth_area & muscle_hu_mask
                
                grown_muscles[i] |= growth_area
        
        self.anterior_muscles_mask = grown_muscles.copy()
        
        # 2. Further growth based on rules
        
        # Process each slice
        for i in range(self.ct_image.shape[0]):
            # Skip slices without grouped tissue
            if not np.any(mg_roi[i]):
                continue
            
            # Applying connected component labeling to identify candidate muscle components
            labeled_mask, num_labels = measure.label(mg_roi[i] & ~self.anterior_muscles_mask[i], return_num=True)
            
            if num_labels > 0:
                # Get region attributes
                regions = measure.regionprops(labeled_mask)
                
                # Process each component
                for region in regions:
                    y0, x0, y1, x1 = region.bbox
                    
                    # Based on the paper's criteria:
                    # 1. Lateral distance to the nearest existing muscle
                    # 2. Distance to lungs/chest cavity
                    # 3. Connectivity with existing muscles
                    
                    # Check the distance to the existing muscle (lateral distance)
                    dilated_muscles = morphology.binary_dilation(
                        self.anterior_muscles_mask[i], 
                        morphology.disk(5) 
                    )
                    
                    # Region Proposal Mask
                    region_mask = labeled_mask == region.label
                    
                    # Check if the area is a candidate muscle (near existing muscles
                    is_candidate = np.any(dilated_muscles & region_mask)
                    
                    if is_candidate:
                        # Compute connectivity with existing muscles
                        dilated_region = morphology.binary_dilation(region_mask, morphology.disk(3))
                        connectivity = np.sum(dilated_region & self.anterior_muscles_mask[i])
                        
                        # Calculate distance to chest/lungs
                        if np.any(self.thoracic_cavity_mask[i] & region_mask):
                            # If the region overlaps with the chest cavity, the distance is 0
                            distance_to_cavity = 0
                        else:
                            # Calculate the minimum distance to the chest cavity
                            cavity_distance = ndimage.distance_transform_edt(~self.thoracic_cavity_mask[i])
                            distance_to_cavity = np.min(cavity_distance[region_mask]) if np.any(region_mask) else np.inf
                        
                        # Judgment standard parameters
                        high_connectivity_threshold = 30  #  σ3
                        medium_connectivity_threshold = 20  # σ2 
                        low_connectivity_threshold = 10   # σ1
                        
                        close_cavity_threshold = 5    # τ1
                        medium_cavity_threshold = 10  #  τ2
                        far_cavity_threshold = 15     # τ3
                        
                        # Implement the three judgment rules described in the paper
                        is_muscle = False
                        
                        # Rule 1: High connectivity and close to the chest
                        if connectivity > high_connectivity_threshold and distance_to_cavity < close_cavity_threshold:
                            is_muscle = True
                        # Rule 2: Moderate connectivity and moderate proximity to the chest
                        elif connectivity > medium_connectivity_threshold and distance_to_cavity < medium_cavity_threshold:
                            is_muscle = True
                        # Rule 3: Low connectivity but very close to the chest
                        elif connectivity > low_connectivity_threshold and distance_to_cavity < far_cavity_threshold:
                            is_muscle = True
                            
                        # Additional HU value constraints: Muscles are usually in the range of 30-80
                        # set to 20-90 here
                        if is_muscle:
                            region_hu = self.ct_image[i][region_mask]
                            if len(region_hu) > 0:
                                mean_hu = np.mean(region_hu)
                                if mean_hu < 20 or mean_hu > 90:
                                    is_muscle = False
                        
                        if is_muscle:
                            # Include this area in the muscles
                            self.anterior_muscles_mask[i] |= region_mask
        
        # Update ROI to exclude anterior muscles
        final_roi = updated_roi & ~self.anterior_muscles_mask
        
        # Initialize the glandular tissue mask
        self.glandular_tissue_mask = np.zeros_like(self.skin_mask)
        
        # Process each slice, identify glandular tissue and apply vertical constraints
        for i in range(self.ct_image.shape[0]):
            # Skip slices without grouped tissue
            if not np.any(final_roi[i]) or not np.any(self.combined_tissues_mask[i]):
                continue
            
            # Create a tighter vertical constraint to limit the upper front area
            height, width = self.ct_image[i].shape
            
            # 1. Basic upper area constraint 
            basic_upper_constraint = np.zeros_like(self.ct_image[i], dtype=bool)
            basic_upper_constraint[:height//2, :] = True
            
            # 2. Frontal area constraint based on the chest cavity
            anterior_constraint = np.zeros_like(self.ct_image[i], dtype=bool)
            
            if np.any(self.thoracic_cavity_mask[i]):
                # Get the front boundary point of the chest cavity
                thoracic_points = np.argwhere(self.thoracic_cavity_mask[i])
                if len(thoracic_points) > 0:
                    # Calculate the anterior border of the thorax
                    min_y_for_x = {}
                    for y, x in thoracic_points:
                        if x not in min_y_for_x or y < min_y_for_x[x]:
                            min_y_for_x[x] = y

                    buffer = 40  
                    
                    # Only the area in front of the chest is preserved
                    for x in range(width):
                        front_y = min_y_for_x.get(x, height//2)  #If not found, use the default value
                        if front_y < height//2:  # Make sure it is the front area
                            # Fills only a limited area in front of the chest
                            anterior_constraint[:min(front_y+buffer, height//3), x] = True
                        else:
                            # If the thorax border is not in the front, use the default front 1/4 region
                            anterior_constraint[:height//4, x] = True
            
            # 3. Create lateral constraints (excluding the center area)
            lateral_constraint = np.ones_like(self.ct_image[i], dtype=bool)
            
            # Find the midpoint
            if np.any(self.thoracic_cavity_mask[i]):
                col_sums = np.sum(self.thoracic_cavity_mask[i], axis=0)
                col_sums_smooth = self.smooth_profile(col_sums)
                midpoint = np.argmax(col_sums_smooth)
            else:
                midpoint = width // 2
            
            # Left and right side areas (excluding the center)
            center_exclusion = width // 10  # Width of the central exclusion zone
            # Left area
            lateral_constraint[:, :midpoint-center_exclusion] = True
            # Right area
            lateral_constraint[:, midpoint+center_exclusion:] = True
            
            # 4. Combine all constraints to create the final constraint mask
            # Combining a basic top constraint with a chest-based front constraint
            vertical_position_constraint = basic_upper_constraint | anterior_constraint
            
            # Combining positional and lateral constraints
            combined_constraint = vertical_position_constraint & lateral_constraint
            
            # Restraints applied inside the skin and not inside the chest cavity
            final_constraint = combined_constraint & self.skin_mask[i] & ~self.thoracic_cavity_mask[i]
            
            # Glandular tissue is the remaining combined tissue in the ROI, but is now tightly constrained
            initial_glandular = self.combined_tissues_mask[i] & final_roi[i] & final_constraint
            
            # Use more precise HU value ranges to further constrain
            glandular_hu_mask = (self.ct_image[i] >= -45) & (self.ct_image[i] <= 100)
            initial_glandular = initial_glandular & glandular_hu_mask
            
            # Calculate the amount of glandular tissue on the left and right sides (based on the results after constraints)
            left_glandular = np.sum(initial_glandular[:, :midpoint])
            right_glandular = np.sum(initial_glandular[:, midpoint:])
            
            # Add symmetry constraints to ensure that the glandular tissue recognition results of the left and right breasts have a certain degree of symmetry
            imbalance_threshold = 3.0  # 允许3倍不平衡
            
            # Treat the left and right breasts separately, staying within the constraints
            # Left breast
            left_mask = np.zeros_like(initial_glandular)
            left_mask[:, :midpoint] = initial_glandular[:, :midpoint]
            
            # The dilation operation needs to be performed on the entire image, and then the required part is cut out.
            dilated_left = morphology.binary_dilation(left_mask, morphology.disk(3))
            # Now slice and make sure the dimensions match
            potential_left = dilated_left[:, :midpoint] & final_roi[i, :, :midpoint] & final_constraint[:, :midpoint]
            
            # right breast
            right_mask = np.zeros_like(initial_glandular)
            right_mask[:, midpoint:] = initial_glandular[:, midpoint:]
            
            # If there is significantly less glandular tissue on the right side than on the left, try to balance them but still stay within the constraints
            if left_glandular > imbalance_threshold * right_glandular and right_glandular > 0:
                print(f"Enhanced slice {i} of right glandular tissue: left {left_glandular}, right {right_glandular}")
                
                # Use a looser HU value range for the right side, but still within the constraint
                right_dense_tissue = (self.ct_image[i, :, midpoint:] >= -10) & (self.ct_image[i, :, midpoint:] <= 100)
                right_dense_tissue = right_dense_tissue & final_roi[i, :, midpoint:] & final_constraint[:, midpoint:]
    
                # Update the right mask to keep the dimensions consistent
                right_mask[:, midpoint:] |= right_dense_tissue
            
            # Expand the potential gland area on the right side but keep it within the constraints
            dilated_right = morphology.binary_dilation(right_mask, morphology.disk(3))
            potential_right = dilated_right[:, midpoint:] & final_roi[i, :, midpoint:] & final_constraint[:, midpoint:]
            
            # If the left glandular tissue is significantly less than the right, treat the left side similarly.
            if right_glandular > imbalance_threshold * left_glandular and left_glandular > 0:
                print(f"Left glandular tissue of enhanced slice {i}: left {left_glandular}, right {right_glandular}")
                
                # Use a looser HU value range for the left side, but still within the constraint
                left_dense_tissue = (self.ct_image[i, :, :midpoint] >= -10) & (self.ct_image[i, :, :midpoint] <= 100)
                left_dense_tissue = left_dense_tissue & final_roi[i, :, :midpoint] & final_constraint[:, :midpoint]
                
                potential_left |= left_dense_tissue
            
            # Combine the left and right glandular tissues to ensure they are still within the constraints
            combined_glandular = np.zeros_like(initial_glandular)
            # Use the sliced ​​result directly to avoid shape mismatch
            combined_glandular[:, :midpoint] = potential_left
            combined_glandular[:, midpoint:] = potential_right
            
            # Reapply the constraints to ensure that the constraints are not exceeded.
            combined_glandular = combined_glandular & final_constraint
            
            # Additional chest distance constraint applied
            if np.any(self.thoracic_cavity_mask[i]):
                cavity_distance = ndimage.distance_transform_edt(~self.thoracic_cavity_mask[i])
          
                distance_constraint = cavity_distance <= 50
                combined_glandular = combined_glandular & distance_constraint
            
            # Apply morphological operations to clean up the results
            if np.any(combined_glandular):
                # Remove small isolated areas
                combined_glandular = morphology.remove_small_objects(combined_glandular, min_size=20)
                # Fill small holes and smooth edges
                combined_glandular = morphology.binary_closing(combined_glandular, morphology.disk(2))
                combined_glandular = morphology.binary_opening(combined_glandular, morphology.disk(1))
                # Apply the constraints again to ensure that the morphological operation is still within the constraints.
                combined_glandular = combined_glandular & final_constraint
            
            # Save the final constrained glandular tissue result
            self.glandular_tissue_mask[i] = combined_glandular
        
        return self.glandular_tissue_mask
    
    def determine_vertical_extents(self):
        """
        Determine the vertical (upper and lower) extent of the breast, limited to the upper chest area, and connect the left and right breasts
        """
        print("Determine the vertical extent and connect the left and right breasts...")

        # Initialize breast_mask (make sure it is not None)
        self.breast_mask = np.zeros_like(self.skin_mask)
        
        # Use the ribcage as the primary reference point
        thoracic_per_slice = np.sum(self.thoracic_cavity_mask, axis=(1, 2))
        thoracic_slices = np.where(thoracic_per_slice > 0)[0]
        
        if len(thoracic_slices) > 0:
            # Top of the rib cage
            thoracic_top = thoracic_slices[0]
            
            # Limits breast appearance to the upper chest area
            upper_limit = thoracic_top
            lower_limit = thoracic_top + int(0.4 * len(thoracic_slices))
            
            # Creating the final breast mask
            self.breast_mask = np.zeros_like(self.skin_mask)
            
            # Process each slice
            for i in range(upper_limit, lower_limit + 1):
                if i < 0 or i >= self.glandular_tissue_mask.shape[0]:
                    continue
                
                # First, create an anterior chest wall region mask
                # Find the front border of the thorax and expand forward
                if np.any(self.thoracic_cavity_mask[i]):
                    # Get the outer contour points of the chest mask
                    thoracic_points = np.argwhere(self.thoracic_cavity_mask[i])
                    
                    if len(thoracic_points) > 0:
                        # Calculate the anterior border contour of the thorax
                        height, width = self.ct_image[i].shape
                        min_y_for_each_x = {}
                        
                        for y, x in thoracic_points:
                            if x not in min_y_for_each_x or y < min_y_for_each_x[x]:
                                min_y_for_each_x[x] = y
                        
                        # Create anterior chest wall region mask
                        chest_wall_mask = np.zeros_like(self.ct_image[i], dtype=bool)
                        
                        # For each column, label all points from 0 to the front boundary
                        for x in range(width):
                            min_y = min_y_for_each_x.get(x, height//2) 
                            chest_wall_mask[:min_y, x] = True
                        
                        # Expand anteriorly to include all anterior chest wall area
                        chest_wall_mask = morphology.binary_dilation(chest_wall_mask, morphology.disk(5))
                        
                        # Apply a mask to keep only the inner area of ​​the skin
                        chest_wall_mask = chest_wall_mask & self.skin_mask[i]
                  
                        midpoint = width // 2
                        
                        # Extraction of glandular tissue from the left and right breast areas
                        left_glandular = self.glandular_tissue_mask[i].copy()
                        left_glandular[:, midpoint:] = False
                        
                        right_glandular = self.glandular_tissue_mask[i].copy()
                        right_glandular[:, :midpoint] = False
                        
                        # If there is glandular tissue on both sides, connect them
                        if np.any(left_glandular) and np.any(right_glandular):
                            # Create a larger expansion radius to secure the connection
                            dilated_left = morphology.binary_dilation(left_glandular, morphology.disk(20))
                            dilated_right = morphology.binary_dilation(right_glandular, morphology.disk(20))
                     
                            connected_breasts = dilated_left | dilated_right
                            
                            # Limit connections to the anterior chest wall area
                            connected_breasts = connected_breasts & chest_wall_mask
                            
                            # Make sure not to include the thoracic cavity
                            connected_breasts = connected_breasts & ~self.thoracic_cavity_mask[i]
                            
                            # Apply fat tissue constraint - the connected area should be mostly fat tissue
                            fat_in_slice = (self.ct_image[i] >= -150) & (self.ct_image[i] <= -20)
          
                            final_breast = (connected_breasts & (fat_in_slice | self.glandular_tissue_mask[i]))
                            
                            # 应用闭操作填充小洞
                            final_breast = morphology.binary_closing(final_breast, morphology.disk(5))
                            
                            # If the join result is empty, the original glandular tissue is used
                            if not np.any(final_breast):
                                final_breast = self.glandular_tissue_mask[i]
                            
                            self.breast_mask[i] = final_breast
                        else:
                            # If only one side has glandular tissue, use it directly
                            self.breast_mask[i] = self.glandular_tissue_mask[i]
                    else:
                        # If no thoracic contour point is found, use the original glandular tissue
                        self.breast_mask[i] = self.glandular_tissue_mask[i]
                else:
                    # If the slice does not have a thoracic cavity, use the original glandular tissue
                    self.breast_mask[i] = self.glandular_tissue_mask[i]
            
            # Clear all segmentation results below the lower limit
            self.breast_mask[lower_limit+1:] = False
            self.glandular_tissue_mask[lower_limit+1:] = False
            
            return self.breast_mask
        
    def connect_left_right_breasts(self):
        """
        Use the convex hull method to connect the left and right breast regions
        """
        print("Use the convex hull method to connect the left and right breast regions...")

        # Check if breast_mask is initialized
        if self.breast_mask is None:
            self.breast_mask = np.zeros_like(self.skin_mask)
            return
        
        # Process each slice
        for i in range(self.breast_mask.shape[0]):
            # Only sections with breast tissue were processed
            if not np.any(self.breast_mask[i]):
                continue
            
                # Add symmetry checking and adjustment code：
            width = self.breast_mask[i].shape[1]
            midpoint = width // 2
            
            # Calculate the amount of glandular tissue on the left and right sides
            left_glandular = np.sum(self.glandular_tissue_mask[i, :, :midpoint])
            right_glandular = np.sum(self.glandular_tissue_mask[i, :, midpoint:])
            
            # Set a reasonable imbalance threshold, allowing for some natural asymmetry
            imbalance_threshold = 3.0  # 3倍不平衡
            
            # Dealing with left-right imbalance
            if left_glandular > imbalance_threshold * right_glandular and right_glandular > 0:
                print(f"增强切片 {i} 的右侧腺体组织：左侧 {left_glandular}，右侧 {right_glandular}")
                
                # Use a looser HU value range for the right side
                right_dense_tissue = (self.ct_image[i, :, midpoint:] >= -10) & (self.ct_image[i, :, midpoint:] <= 120)
                right_dense_tissue = right_dense_tissue & self.skin_mask[i, :, midpoint:] & ~self.thoracic_cavity_mask[i, :, midpoint:]
                
                # Update right gland mask
                self.glandular_tissue_mask[i, :, midpoint:] |= right_dense_tissue
                
            elif right_glandular > imbalance_threshold * left_glandular and left_glandular > 0:
                print(f"增强切片 {i} 的左侧腺体组织：左侧 {left_glandular}，右侧 {right_glandular}")
                
            # 创建前胸壁区域掩码
            height, width = self.breast_mask[i].shape
            front_chest_mask = np.zeros_like(self.breast_mask[i], dtype=bool)
            
            # 使用胸腔位置确定前胸壁区域
            if np.any(self.thoracic_cavity_mask[i]):
                # 找到胸腔的前边界
                thoracic_points = np.argwhere(self.thoracic_cavity_mask[i])
                if len(thoracic_points) > 0:
                    # 计算胸腔的前边界
                    min_y_for_x = {}
                    for y, x in thoracic_points:
                        if x not in min_y_for_x or y < min_y_for_x[x]:
                            min_y_for_x[x] = y
                    
                    # 设置前胸壁区域 - 在胸腔前方
                    for x in range(width):
                        min_y = min_y_for_x.get(x, height//2)  # 默认值
                        front_chest_mask[:min_y, x] = True
                else:
                    # 默认前1/3为前胸壁区域
                    front_chest_mask[:height//3, :] = True
            else:
                # 默认前1/3为前胸壁区域
                front_chest_mask[:height//3, :] = True
            
            # 将乳腺组织限制在前胸壁区域
            breast_tissue_front = self.breast_mask[i] & front_chest_mask
            
            # 如果前胸壁中有乳腺组织
            if np.any(breast_tissue_front):
                # 使用凸包方法连接左右乳腺
                # 凸包会生成一个包含所有点的最小凸多边形
                try:
                    hull_mask = morphology.convex_hull_image(breast_tissue_front)
                    
                    # 应用限制条件：
                    # 1. 保持在前胸壁内
                    hull_mask = hull_mask & front_chest_mask
                    # 2. 保持在皮肤内
                    hull_mask = hull_mask & self.skin_mask[i]
                    # 3. 排除胸腔
                    hull_mask = hull_mask & ~self.thoracic_cavity_mask[i]
                    
                    # 更新乳腺掩码
                    self.breast_mask[i] = hull_mask
                    
                    # 填充小孔洞
                    self.breast_mask[i] = morphology.remove_small_holes(self.breast_mask[i], area_threshold=200)
                except Exception as e:
                    print(f"处理切片 {i} 时出错: {e}")
                    continue
    
    def segment_breast(self):
        # 使用皮肤分割识别前侧和侧向范围
        self.segment_skin()
        
        #分割胸腔、脂肪和组合组织
        self.segment_thoracic_cavity()
        self.segment_fat_tissues()
        self.segment_combined_tissues()
        
        # 从后侧肌肉分离
        updated_roi = self.separate_posterior_muscles()
        
        # 从前侧肌肉分离
        self.separate_anterior_muscles(updated_roi)
        
        #确定垂直范围
        self.determine_vertical_extents()

        #连接左右乳腺
        self.connect_left_right_breasts()
        
        # 如果仍然没有分割结果，进行最后的修正
        if not np.any(self.breast_mask):
            print("最终检查：未找到乳腺组织，应用默认方法...")
            
            # 使用胸腔作为参考
            thoracic_per_slice = np.sum(self.thoracic_cavity_mask, axis=(1, 2))
            thoracic_slices = np.where(thoracic_per_slice > 0)[0]
            
            if len(thoracic_slices) > 0:
                # 胸腔顶部
                thoracic_top = thoracic_slices[0]
                lower_limit = thoracic_top + int(0.4 * len(thoracic_slices))
                
                # 对每个切片应用简单的几何约束
                for i in range(thoracic_top, lower_limit + 1):
                    if i >= self.ct_image.shape[0]:
                        continue
                    
                    # 找到分界线
                    width = self.ct_image.shape[2]
                    midpoint = width // 2
                    
                    # 前部区域掩码
                    height = self.ct_image[i].shape[0]
                    front_mask = np.zeros_like(self.ct_image[i], dtype=bool)
                    front_mask[:height//2, :] = True
                    
                    # 侧向掩码
                    side_mask = np.zeros_like(self.ct_image[i], dtype=bool)
                    side_mask[:, :midpoint//2] = True  # 左侧
                    side_mask[:, midpoint + midpoint//2:] = True  # 右侧
                    
                    # 组合掩码
                    combined_mask = front_mask & side_mask & self.skin_mask[i] & ~self.thoracic_cavity_mask[i]
                    
                    # 应用HU值范围
                    tissue_mask = ((self.ct_image[i] >= -150) & (self.ct_image[i] <= 100)) & combined_mask
                    
                    # 清理
                    if np.any(tissue_mask):
                        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=20)
                        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(2))
                        self.breast_mask[i] = tissue_mask
        
        return self.breast_mask
    
    def visualize_segmentation(self, slice_idx=None):
        """
        Visualize the segmentation results for a specific slice
        """
        if self.breast_mask is None:
            print("Please run segment_breast() first")
            return
        
        # If slice_idx is not provided, find a slice with breast tissue
        if slice_idx is None:
            tissue_per_slice = np.sum(self.breast_mask, axis=(1, 2))
            valid_slices = np.where(tissue_per_slice > 0)[0]
            
            if len(valid_slices) > 0:
                slice_idx = valid_slices[len(valid_slices) // 2]  # Middle slice with tissue
            else:
                slice_idx = self.ct_image.shape[0] // 2  # Middle slice
        
        # Create visualizations
        windowed_image = self.apply_hounsfield_window(self.ct_image[slice_idx], 50, 400)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(windowed_image, cmap='gray')
        axes[0, 0].set_title('Original CT Slice')
        
        # Skin mask
        axes[0, 1].imshow(windowed_image, cmap='gray')
        axes[0, 1].imshow(self.skin_mask[slice_idx], alpha=0.3, cmap='Blues')
        axes[0, 1].set_title('Skin Segmentation')
        
        # Thoracic cavity mask
        axes[0, 2].imshow(windowed_image, cmap='gray')
        axes[0, 2].imshow(self.thoracic_cavity_mask[slice_idx], alpha=0.3, cmap='Purples')
        axes[0, 2].set_title('Thoracic Cavity')
        
        # Fat tissues
        axes[1, 0].imshow(windowed_image, cmap='gray')
        axes[1, 0].imshow(self.fat_mask[slice_idx], alpha=0.3, cmap='YlOrBr')
        axes[1, 0].set_title('Fat Tissues')
        
        # Glandular tissues
        axes[1, 1].imshow(windowed_image, cmap='gray')
        axes[1, 1].imshow(self.glandular_tissue_mask[slice_idx], alpha=0.3, cmap='Reds')
        axes[1, 1].set_title('Glandular Tissues')
        
        # Final breast segmentation
        axes[1, 2].imshow(windowed_image, cmap='gray')
        axes[1, 2].imshow(self.breast_mask[slice_idx], alpha=0.3, cmap='Greens')
        axes[1, 2].set_title('Breast Segmentation')
        
        for ax in axes.flatten():
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        


def process_dicom_dataset(dataset_path, output_path=None):
    """
    Process all DICOM sequences in the dataset, visualize the slices and save them in a format suitable for deep learning
    """
    dataset_path = Path(dataset_path)

    studies = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not studies:
        print(f"No studies found in {dataset_path}")
        return

    for study in studies:
        study_id = study.name
        print(f"\nProcessing study: {study_id}")

        series_list = [d for d in study.iterdir() if d.is_dir()]
        
        if not series_list:
            print(f"No series found in study {study_id}")
            continue

        for series in series_list:
            series_id = series.name
            print(f"\nProcessing series: {series_id}")

            segmenter = BreastSegmentation(str(dataset_path))
            
            try:
                # 加载 DICOM 序列
                segmenter.load_dicom_series(study_id, series_id)
                
                # 执行分割
                breast_mask = segmenter.segment_breast()

                if output_path:
                    output_dir = Path(output_path) / study_id / series_id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 创建可视化目录
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    
                    # 保存乳腺掩码为 numpy 数组
                    np.save(output_dir / "breast_mask.npy", breast_mask)
                    
                    # 保存腺体组织掩码
                    np.save(output_dir / "glandular_tissue_mask.npy", segmenter.glandular_tissue_mask)
                    
                    # 保存为 NIFTI 格式
                    try:
                        # 创建 SimpleITK 图像
                        mask_sitk = sitk.GetImageFromArray(breast_mask.astype(np.uint8))
                        # 使用原始 DICOM 图像的元数据
                        mask_sitk.SetSpacing(segmenter.image_spacing)
                        # 保存为bii文件
                        sitk.WriteImage(mask_sitk, str(output_dir / "breast_mask.nii.gz"))
                        print(f"  Saved NIFTI mask to {output_dir / 'breast_mask.nii.gz'}")
                    except Exception as nii_error:
                        print(f"  Error saving NIFTI: {nii_error}")
                    
                    # 可视化并保存每个包含乳腺组织的切片
                    tissue_per_slice = np.sum(breast_mask, axis=(1, 2))
                    valid_slices = np.where(tissue_per_slice > 0)[0]
                    
                    print(f"  Saving visualizations for {len(valid_slices)} slices with breast tissue...")
                    
                    # 处理每个包含乳腺组织的切片
                    for slice_idx in valid_slices:
                        # 窗口化图像
                        windowed_image = segmenter.apply_hounsfield_window(segmenter.ct_image[slice_idx], 50, 400)
                        
                        # 创建可视化
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # 原始图像
                        axes[0].imshow(windowed_image, cmap='gray')
                        axes[0].set_title(f'Slice {slice_idx}: Original')
                        axes[0].axis('off')
                        
                        # 乳腺掩码覆盖
                        axes[1].imshow(windowed_image, cmap='gray')
                        axes[1].imshow(breast_mask[slice_idx], alpha=0.5, cmap='Greens')
                        axes[1].set_title(f'Slice {slice_idx}: Breast Segmentation')
                        axes[1].axis('off')
                        
                        # 腺体组织掩码覆盖
                        axes[2].imshow(windowed_image, cmap='gray')
                        axes[2].imshow(segmenter.glandular_tissue_mask[slice_idx], alpha=0.5, cmap='Reds')
                        axes[2].set_title(f'Slice {slice_idx}: Glandular Tissue')
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(viz_dir / f"slice_{slice_idx:04d}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # 同时单独保存每个切片的掩码为 PNG
                        plt.figure(figsize=(5, 5))
                        plt.imshow(breast_mask[slice_idx], cmap='binary')
                        plt.axis('off')
                        plt.savefig(viz_dir / f"mask_{slice_idx:04d}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    
                    
                    # 保存所有切片的组合预览
                    try:
                        # 选择一些代表性切片来显示
                        preview_slices = []
                        if len(valid_slices) <= 9:
                            preview_slices = valid_slices
                        else:
                            # 选择大约 9 个均匀分布的切片
                            step = len(valid_slices) // 9
                            preview_slices = valid_slices[::step][:9]
                        
                        rows = int(np.ceil(len(preview_slices) / 3))
                        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
                        
                        # 确保在只有一行的情况下 axes 仍然是 2D
                        if rows == 1:
                            axes = np.expand_dims(axes, axis=0)
                        
                        for i, slice_idx in enumerate(preview_slices):
                            row = i // 3
                            col = i % 3
                            
                            # 窗口化图像
                            windowed_image = segmenter.apply_hounsfield_window(segmenter.ct_image[slice_idx], 50, 400)
                            
                            # 带乳腺分割覆盖的原始图像
                            axes[row, col].imshow(windowed_image, cmap='gray')
                            axes[row, col].imshow(breast_mask[slice_idx], alpha=0.5, cmap='Greens')
                            axes[row, col].set_title(f'Slice {slice_idx}')
                            axes[row, col].axis('off')
                            
                        # 隐藏未使用的子图
                        for i in range(len(preview_slices), rows * 3):
                            row = i // 3
                            col = i % 3
                            axes[row, col].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(output_dir / "preview_all_slices.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    except Exception as preview_error:
                        print(f"  Error creating preview: {preview_error}")
                    
                    print(f"  Segmentation masks and visualizations saved to {output_dir}")
                
                print(f"Segmentation of series {series_id} completed successfully")
                
            except Exception as e:
                print(f"Error processing series {series_id}: {e}")
                import traceback
                traceback.print_exc()
                continue



if __name__ == "__main__":
    process_dicom_dataset("D:/Data_noncontrast", "D:/Desktop/Data_noncontrast_segmentation")
