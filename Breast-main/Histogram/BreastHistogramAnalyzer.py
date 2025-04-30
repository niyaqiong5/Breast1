import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
import SimpleITK as sitk
import os
from pathlib import Path


class BreastHistogramAnalyzer:
    """
    A class for analyzing local histograms of CT intensity values in segmented breast regions.
    Works with the output of the BreastSegmentation class.
    """
    
    def __init__(self, ct_image=None, breast_mask=None, glandular_mask=None, fat_mask=None):
        """
        Initialize the breast histogram analyzer.
        """
        self.ct_image = ct_image
        self.breast_mask = breast_mask
        self.glandular_mask = glandular_mask
        self.fat_mask = fat_mask
        
        # Default histogram parameters
        self.hu_range = (-200, 200)  # HU value range for histogram
        self.num_bins = 100  # Number of histogram bins
        
    def load_from_segmenter(self, segmenter):
        """
        Load data from a BreastSegmentation instance.
        """
        self.ct_image = segmenter.ct_image
        self.breast_mask = segmenter.breast_mask
        self.glandular_mask = segmenter.glandular_tissue_mask
        self.fat_mask = segmenter.fat_mask
        
    def load_from_files(self, ct_file, breast_mask_file, glandular_mask_file=None, fat_mask_file=None):
        """
        Load data from saved files.
        """
        # Determine file type and load accordingly
        if ct_file.endswith('.npy'):
            self.ct_image = np.load(ct_file)
        elif ct_file.endswith('.dcm'):
            # Load DICOM file using pydicom
            import pydicom
            ds = pydicom.dcmread(ct_file)
            self.ct_image = ds.pixel_array
        else:
            # Try loading as a DICOM series using SimpleITK
            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(os.path.dirname(ct_file))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                self.ct_image = sitk.GetArrayFromImage(image)
            except Exception as e:
                raise ValueError(f"Unsupported CT image format: {e}")
        
        # Load mask files
        self.breast_mask = np.load(breast_mask_file)
        
        if glandular_mask_file and os.path.exists(glandular_mask_file):
            self.glandular_mask = np.load(glandular_mask_file)
        
        if fat_mask_file and os.path.exists(fat_mask_file):
            self.fat_mask = np.load(fat_mask_file)
    
    def compute_global_histogram(self, mask=None, range_hu=None, bins=None):
        """
        Compute a global histogram of CT values within the specified mask.
        """
        if self.ct_image is None or self.breast_mask is None:
            raise ValueError("CT image and breast mask must be loaded first")
        
        # Use default mask if none provided
        if mask is None:
            mask = self.breast_mask
            
        # Use default parameters if not specified
        if range_hu is None:
            range_hu = self.hu_range
        if bins is None:
            bins = self.num_bins
            
        # Extract CT values within the mask
        masked_values = self.ct_image[mask > 0]
        
        if len(masked_values) == 0:
            print("Warning: No voxels found in the specified mask")
            return np.array([]), np.array([])
            
        # Compute histogram
        hist, bin_edges = np.histogram(masked_values, bins=bins, range=range_hu)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return hist, bin_centers
    
    def compute_local_histograms(self, window_size=5, stride=1, range_hu=None, bins=None):
        """
        Compute local histograms of CT values using a sliding window approach.
        """
        if self.ct_image is None or self.breast_mask is None:
            raise ValueError("CT image and breast mask must be loaded first")
        
        # Use default parameters if not specified
        if range_hu is None:
            range_hu = self.hu_range
        if bins is None:
            bins = self.num_bins
            
        # Prepare histogram bin edges
        bin_edges = np.linspace(range_hu[0], range_hu[1], bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize window_size and stride to 3D
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        elif len(window_size) == 2:
            window_size = (1, window_size[0], window_size[1])
            
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        elif len(stride) == 2:
            stride = (1, stride[0], stride[1])
        
        # Get shape of the CT volume
        depth, height, width = self.ct_image.shape
        
        # Calculate the number of windows in each dimension
        num_windows_z = max(1, (depth - window_size[0]) // stride[0] + 1)
        num_windows_y = max(1, (height - window_size[1]) // stride[1] + 1)
        num_windows_x = max(1, (width - window_size[2]) // stride[2] + 1)
        
        # Initialize results containers
        local_histograms = []
        positions = []
        metadata = {
            'window_size': window_size,
            'stride': stride,
            'hu_range': range_hu,
            'num_bins': bins,
            'bin_centers': bin_centers
        }
        
        # Slide window and compute local histograms
        for z in range(0, depth - window_size[0] + 1, stride[0]):
            for y in range(0, height - window_size[1] + 1, stride[1]):
                for x in range(0, width - window_size[2] + 1, stride[2]):
                    # Extract local window
                    window_ct = self.ct_image[
                        z:z+window_size[0],
                        y:y+window_size[1],
                        x:x+window_size[2]
                    ]
                    
                    window_mask = self.breast_mask[
                        z:z+window_size[0],
                        y:y+window_size[1],
                        x:x+window_size[2]
                    ]
                    
                    # Skip windows that don't contain breast tissue
                    if np.sum(window_mask) < 0.1 * window_size[0] * window_size[1] * window_size[2]:
                        continue
                    
                    # Extract CT values within the mask
                    masked_values = window_ct[window_mask > 0]
                    
                    if len(masked_values) > 0:
                        # Compute histogram
                        hist, _ = np.histogram(masked_values, bins=bin_edges)
                        
                        # Store results
                        local_histograms.append(hist)
                        positions.append((z, y, x))
        
        return {
            'histograms': np.array(local_histograms),
            'positions': np.array(positions),
            'metadata': metadata
        }
    
    def compute_adaptive_local_histograms(self, min_tissue_percent=10, max_regions=100, range_hu=None, bins=None):
        """
        Compute local histograms using adaptive regions based on tissue distribution.
        """
        if self.ct_image is None or self.breast_mask is None:
            raise ValueError("CT image and breast mask must be loaded first")
        
        # Use default parameters if not specified
        if range_hu is None:
            range_hu = self.hu_range
        if bins is None:
            bins = self.num_bins
            
        # Prepare histogram bin edges
        bin_edges = np.linspace(range_hu[0], range_hu[1], bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize results containers
        local_histograms = []
        region_centers = []
        region_sizes = []
        
        # Get slices with breast tissue
        tissue_per_slice = np.sum(self.breast_mask, axis=(1, 2))
        valid_slices = np.where(tissue_per_slice > 0)[0]
        
        if len(valid_slices) == 0:
            print("Warning: No slices with breast tissue found")
            return {
                'histograms': np.array([]),
                'positions': np.array([]),
                'region_sizes': np.array([]),
                'metadata': {
                    'hu_range': range_hu,
                    'num_bins': bins,
                    'bin_centers': bin_centers
                }
            }
        
        # Process each valid slice
        regions_added = 0
        
        for slice_idx in valid_slices:
            # Skip if we've reached the maximum number of regions
            if regions_added >= max_regions:
                break
                
            # Get the breast mask for this slice
            slice_mask = self.breast_mask[slice_idx]
            
            if np.sum(slice_mask) == 0:
                continue
                
            # Label connected components
            labeled_mask, num_labels = ndimage.label(slice_mask)
            
            if num_labels == 0:
                continue
                
            # Get region properties
            regions = ndimage.find_objects(labeled_mask)
            
            for i, region in enumerate(regions):
                # Skip if we've reached the maximum number of regions
                if regions_added >= max_regions:
                    break
                    
                if region is None:
                    continue
                    
                # Extract region coordinates
                y_slice, x_slice = region
                
                # Calculate region center
                y_center = (y_slice.start + y_slice.stop) // 2
                x_center = (x_slice.start + x_slice.stop) // 2
                
                # Calculate region size
                region_height = y_slice.stop - y_slice.start
                region_width = x_slice.stop - x_slice.start
                
                # Skip very small regions
                if region_height < 3 or region_width < 3:
                    continue
                
                # Extract region image and mask
                region_image = self.ct_image[slice_idx, y_slice, x_slice]
                region_mask = labeled_mask[y_slice, x_slice] == i + 1
                
                # Calculate percentage of voxels containing breast tissue
                tissue_percent = np.sum(region_mask) / (region_height * region_width) * 100
                
                if tissue_percent < min_tissue_percent:
                    continue
                
                # Extract CT values within the mask
                masked_values = region_image[region_mask]
                
                if len(masked_values) > 0:
                    # Compute histogram
                    hist, _ = np.histogram(masked_values, bins=bin_edges)
                    
                    # Store results
                    local_histograms.append(hist)
                    region_centers.append((slice_idx, y_center, x_center))
                    region_sizes.append((1, region_height, region_width))
                    
                    regions_added += 1
        
        return {
            'histograms': np.array(local_histograms),
            'positions': np.array(region_centers),
            'region_sizes': np.array(region_sizes),
            'metadata': {
                'hu_range': range_hu,
                'num_bins': bins,
                'bin_centers': bin_centers,
                'adaptive': True
            }
        }
    
    def analyze_tissue_composition(self, range_hu=None, bins=None):
        """
        Analyze the tissue composition based on HU values.
        """
        if self.ct_image is None or self.breast_mask is None:
            raise ValueError("CT image and breast mask must be loaded first")
        
        # Use default parameters if not specified
        if range_hu is None:
            range_hu = self.hu_range
        if bins is None:
            bins = self.num_bins
        
        # Define HU ranges for different tissue types
        fat_range = (-150, -20)
        fibroglandular_range = (-20, 100)
        
        # Get the global histogram for the entire breast
        global_hist, bin_centers = self.compute_global_histogram(range_hu=range_hu, bins=bins)
        
        # Calculate percentage of different tissue types
        if len(bin_centers) == 0:
            return {
                'error': 'No voxels found in the breast mask'
            }
        
        # Create masks for different tissue types
        fat_mask = (bin_centers >= fat_range[0]) & (bin_centers <= fat_range[1])
        fibroglandular_mask = (bin_centers >= fibroglandular_range[0]) & (bin_centers <= fibroglandular_range[1])
        
        # Calculate total counts
        total_counts = np.sum(global_hist)
        
        if total_counts == 0:
            return {
                'error': 'No voxels found in the specified HU range'
            }
        
        # Calculate percentages
        fat_percent = np.sum(global_hist[fat_mask]) / total_counts * 100
        fibroglandular_percent = np.sum(global_hist[fibroglandular_mask]) / total_counts * 100
        other_percent = 100 - fat_percent - fibroglandular_percent
        
        # Calculate statistics
        masked_values = self.ct_image[self.breast_mask > 0]
        mean_hu = np.mean(masked_values)
        median_hu = np.median(masked_values)
        std_hu = np.std(masked_values)
        
        # Return results
        return {
            'fat_percent': fat_percent,
            'fibroglandular_percent': fibroglandular_percent,
            'other_percent': other_percent,
            'mean_hu': mean_hu,
            'median_hu': median_hu,
            'std_hu': std_hu,
            'histogram': global_hist,
            'bin_centers': bin_centers
        }
    
    def visualize_histograms(self, histograms_result, max_histograms=9, normalize=True):
        """
        Visualize local histograms from compute_local_histograms or compute_adaptive_local_histograms.
        """
        histograms = histograms_result['histograms']
        positions = histograms_result['positions']
        metadata = histograms_result['metadata']
        
        if len(histograms) == 0:
            print("No histograms to visualize")
            return None
        
        # Select a subset of histograms to visualize
        num_histograms = min(max_histograms, len(histograms))
        indices = np.linspace(0, len(histograms) - 1, num_histograms, dtype=int)
        
        # Create subplots
        rows = int(np.ceil(num_histograms / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        
        # Flatten axes if necessary
        if rows == 1 and num_histograms < 3:
            axes = np.array([axes])
        elif rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()
        
        # Get bin centers
        bin_centers = metadata['bin_centers']
        
        # Plot each histogram
        for i, idx in enumerate(indices):
            if i >= len(axes_flat):
                break
                
            hist = histograms[idx]
            pos = positions[idx]
            
            # Normalize if requested
            if normalize and np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            
            # Plot histogram
            axes_flat[i].bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]), alpha=0.7)
            axes_flat[i].set_title(f"Position: {pos}")
            axes_flat[i].set_xlabel("HU Value")
            axes_flat[i].set_ylabel("Frequency" if not normalize else "Normalized Frequency")
            
        # Hide unused subplots
        for i in range(num_histograms, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_global_histogram(self, composition_result=None, range_hu=None, bins=None):
        """
        Visualize the global histogram of CT values in the breast.
        """
        if composition_result is None:
            composition_result = self.analyze_tissue_composition(range_hu=range_hu, bins=bins)
        
        if 'error' in composition_result:
            print(f"Error: {composition_result['error']}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get histogram data
        hist = composition_result['histogram']
        bin_centers = composition_result['bin_centers']
        
        # Define HU ranges for different tissue types
        fat_range = (-150, -20)
        fibroglandular_range = (-20, 100)
        
        # Create masks for different tissue types
        fat_mask = (bin_centers >= fat_range[0]) & (bin_centers <= fat_range[1])
        fibroglandular_mask = (bin_centers >= fibroglandular_range[0]) & (bin_centers <= fibroglandular_range[1])
        other_mask = ~(fat_mask | fibroglandular_mask)
        
        # Plot histogram with tissue type coloring
        width = bin_centers[1] - bin_centers[0]
        ax.bar(bin_centers[fat_mask], hist[fat_mask], width=width, color='orange', alpha=0.7, label='Fat')
        ax.bar(bin_centers[fibroglandular_mask], hist[fibroglandular_mask], width=width, color='red', alpha=0.7, label='Fibroglandular')
        ax.bar(bin_centers[other_mask], hist[other_mask], width=width, color='gray', alpha=0.7, label='Other')
        
        # Add tissue composition annotation
        ax.text(0.02, 0.95, 
                f"Tissue Composition:\n"
                f"Fat: {composition_result['fat_percent']:.1f}%\n"
                f"Fibroglandular: {composition_result['fibroglandular_percent']:.1f}%\n"
                f"Other: {composition_result['other_percent']:.1f}%\n\n"
                f"Mean HU: {composition_result['mean_hu']:.1f}\n"
                f"Median HU: {composition_result['median_hu']:.1f}\n"
                f"Std Dev: {composition_result['std_hu']:.1f}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top', horizontalalignment='left')
        
        # Add title and labels
        ax.set_title("Global Histogram of CT Values in Breast Tissue")
        ax.set_xlabel("Hounsfield Units (HU)")
        ax.set_ylabel("Frequency")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_histogram_map(self, slice_idx=None, window_size=5, stride=3, range_hu=None, bins=None):
        """
        Visualize a map of local histograms over a CT slice.
        """
        if self.ct_image is None or self.breast_mask is None:
            raise ValueError("CT image and breast mask must be loaded first")
        
        # Find a slice with breast tissue if not specified
        if slice_idx is None:
            tissue_per_slice = np.sum(self.breast_mask, axis=(1, 2))
            valid_slices = np.where(tissue_per_slice > 0)[0]
            
            if len(valid_slices) > 0:
                slice_idx = valid_slices[len(valid_slices) // 2]  # Middle slice with tissue
            else:
                print("No slices with breast tissue found")
                return None
        
        # Use default parameters if not specified
        if range_hu is None:
            range_hu = self.hu_range
        if bins is None:
            bins = self.num_bins
            
        # Window the CT image for display
        ct_slice = self.ct_image[slice_idx]
        windowed_slice = np.clip(ct_slice, -100, 200)
        windowed_slice = (windowed_slice - (-100)) / (200 - (-100))
        
        # Get the breast mask for this slice
        mask_slice = self.breast_mask[slice_idx]
        
        # Calculate local histograms for this slice
        height, width = ct_slice.shape
        histogram_grid = np.zeros((
            (height - window_size) // stride + 1,
            (width - window_size) // stride + 1,
            bins
        ))
        valid_grid = np.zeros(histogram_grid.shape[:2], dtype=bool)
        
        # Prepare histogram bin edges
        bin_edges = np.linspace(range_hu[0], range_hu[1], bins + 1)
        
        # Slide window and compute local histograms
        for i, y in enumerate(range(0, height - window_size + 1, stride)):
            for j, x in enumerate(range(0, width - window_size + 1, stride)):
                # Extract local window
                window_ct = ct_slice[y:y+window_size, x:x+window_size]
                window_mask = mask_slice[y:y+window_size, x:x+window_size]
                
                # Skip windows that don't contain breast tissue
                tissue_percentage = np.sum(window_mask) / (window_size * window_size) * 100
                if tissue_percentage < 30:  # At least 30% of window should be breast tissue
                    continue
                
                # Extract CT values within the mask
                masked_values = window_ct[window_mask > 0]
                
                if len(masked_values) > 0:
                    # Compute histogram
                    hist, _ = np.histogram(masked_values, bins=bin_edges)
                    
                    # Normalize the histogram
                    if np.sum(hist) > 0:
                        hist = hist / np.sum(hist)
                    
                    # Store in grid
                    histogram_grid[i, j] = hist
                    valid_grid[i, j] = True
        
        # Create a visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display the CT slice
        axes[0].imshow(windowed_slice, cmap='gray')
        axes[0].set_title(f"CT Slice {slice_idx}")
        axes[0].axis('off')
        
        # Display the breast mask
        axes[1].imshow(windowed_slice, cmap='gray')
        axes[1].imshow(mask_slice, alpha=0.5, cmap='Greens')
        axes[1].set_title(f"Breast Segmentation")
        axes[1].axis('off')
        
        # Create a statistical map (e.g., mean HU value)
        mean_hu_map = np.zeros(valid_grid.shape)
        mean_hu_map.fill(np.nan)  # Fill with NaN for areas without histograms
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for i in range(histogram_grid.shape[0]):
            for j in range(histogram_grid.shape[1]):
                if valid_grid[i, j]:
                    # Calculate mean HU value from the histogram
                    mean_hu_map[i, j] = np.sum(histogram_grid[i, j] * bin_centers)
        
        # Resize the mean HU map to match the original image size
        from scipy.ndimage import zoom
        scale_y = height / mean_hu_map.shape[0]
        scale_x = width / mean_hu_map.shape[1]
        
        # Create a mask for valid areas (not NaN)
        valid_mask = ~np.isnan(mean_hu_map)
        mean_hu_map_valid = mean_hu_map.copy()
        mean_hu_map_valid[~valid_mask] = 0
        
        # Resize the valid mask and mean HU map
        valid_mask_resized = zoom(valid_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
        mean_hu_map_resized = zoom(mean_hu_map_valid, (scale_y, scale_x), order=1)
        
        # Create a masked array for plotting
        import numpy.ma as ma
        masked_map = ma.array(mean_hu_map_resized, mask=~valid_mask_resized)
        
        # Display the statistical map
        map_img = axes[2].imshow(windowed_slice, cmap='gray')
        map_overlay = axes[2].imshow(masked_map, cmap='jet', alpha=0.7, vmin=-80, vmax=40)
        axes[2].set_title(f"Mean HU Value Map")
        axes[2].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(map_overlay, ax=axes[2], orientation='vertical', shrink=0.8)
        cbar.set_label('Mean HU Value')
        
        plt.tight_layout()
        return fig

def extract_histogram_features(histograms_result):
    """
    Extract statistical features from histograms for machine learning or analysis.
    """
    histograms = histograms_result['histograms']
    bin_centers = histograms_result['metadata']['bin_centers']
    
    if len(histograms) == 0:
        return {'error': 'No histograms available'}
    
    # Normalize histograms
    normalized_histograms = []
    for hist in histograms:
        total = np.sum(hist)
        if total > 0:
            normalized_histograms.append(hist / total)
        else:
            normalized_histograms.append(hist)
    
    normalized_histograms = np.array(normalized_histograms)
    
    # Calculate statistical features
    features = {}
    
    # 1. First-order statistics of HU values
    for i, hist in enumerate(normalized_histograms):
        # Calculate mean, variance, skewness, kurtosis
        mean = np.sum(hist * bin_centers)
        variance = np.sum(hist * (bin_centers - mean)**2)
        std_dev = np.sqrt(variance) if variance > 0 else 0
        
        # Skewness (3rd moment)
        skewness = np.sum(hist * ((bin_centers - mean) / std_dev)**3) if std_dev > 0 else 0
        
        # Kurtosis (4th moment)
        kurtosis = np.sum(hist * ((bin_centers - mean) / std_dev)**4) if std_dev > 0 else 0
        
        # 10th and 90th percentiles
        cumsum = np.cumsum(hist)
        p10_idx = np.argmax(cumsum >= 0.1)
        p90_idx = np.argmax(cumsum >= 0.9)
        
        p10 = bin_centers[p10_idx] if p10_idx < len(bin_centers) else bin_centers[-1]
        p90 = bin_centers[p90_idx] if p90_idx < len(bin_centers) else bin_centers[-1]
        
        # Entropy
        nonzero_hist = hist[hist > 0]
        entropy = -np.sum(nonzero_hist * np.log2(nonzero_hist))
        
        # Store features for this histogram
        features[f'histogram_{i}'] = {
            'mean': mean,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'p10': p10,
            'p90': p90,
            'entropy': entropy
        }
    
    # 2. Global statistics across all histograms
    all_means = [features[f'histogram_{i}']['mean'] for i in range(len(normalized_histograms))]
    all_stds = [features[f'histogram_{i}']['std_dev'] for i in range(len(normalized_histograms))]
    all_entropies = [features[f'histogram_{i}']['entropy'] for i in range(len(normalized_histograms))]
    
    features['global'] = {
        'mean_of_means': np.mean(all_means),
        'std_of_means': np.std(all_means),
        'mean_of_stds': np.mean(all_stds),
        'std_of_stds': np.std(all_stds),
        'mean_entropy': np.mean(all_entropies),
        'num_histograms': len(normalized_histograms)
    }
    
    return features

def process_breast_histograms(segmenter, output_dir=None):
    """
    Process and analyze histograms of CT values in segmented breast regions.
    """
    # Create histogram analyzer
    analyzer = BreastHistogramAnalyzer()
    
    # Load data from segmenter
    analyzer.load_from_segmenter(segmenter)
    
    # Set HU range for histograms
    analyzer.hu_range = (-150, 150)
    analyzer.num_bins = 60
    
    # Analyze tissue composition
    composition = analyzer.analyze_tissue_composition()
    
    if 'error' in composition:
        print(f"Error analyzing tissue composition: {composition['error']}")
        return analyzer
    
    print(f"Breast tissue composition:")
    print(f"  Fat: {composition['fat_percent']:.1f}%")
    print(f"  Fibroglandular: {composition['fibroglandular_percent']:.1f}%")
    print(f"  Mean HU: {composition['mean_hu']:.1f}")
    
    # Compute global histogram
    global_hist_fig = analyzer.visualize_global_histogram(composition)
    
    # Compute local histograms with adaptive regions
    local_histograms = analyzer.compute_adaptive_local_histograms(
        min_tissue_percent=20, 
        max_regions=50
    )
    
    # Visualize local histograms
    local_hist_fig = analyzer.visualize_histograms(local_histograms, max_histograms=9)
    
    # Create histogram maps for representative slices
    tissue_per_slice = np.sum(analyzer.breast_mask, axis=(1, 2))
    valid_slices = np.where(tissue_per_slice > 0)[0]
    
    map_figures = []
    if len(valid_slices) > 0:
        # Choose up to 3 representative slices
        num_slices = min(3, len(valid_slices))
        slice_indices = np.linspace(0, len(valid_slices)-1, num_slices, dtype=int)
        
        for i in slice_indices:
            slice_idx = valid_slices[i]
            map_fig = analyzer.visualize_histogram_map(slice_idx, window_size=7, stride=3)
            if map_fig is not None:
                map_figures.append((slice_idx, map_fig))
    
    # Extract histogram features
    features = extract_histogram_features(local_histograms)
    
    # Save results if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the global histogram visualization
        if global_hist_fig:
            global_hist_fig.savefig(output_dir / "global_histogram.png", dpi=150, bbox_inches='tight')
            
        # Save the local histograms visualization
        if local_hist_fig:
            local_hist_fig.savefig(output_dir / "local_histograms.png", dpi=150, bbox_inches='tight')
            
        # Save the histogram maps
        for slice_idx, fig in map_figures:
            fig.savefig(output_dir / f"histogram_map_slice_{slice_idx}.png", dpi=150, bbox_inches='tight')
            
        # Save the tissue composition data
        import json
        composition_data = {
            'fat_percent': float(composition['fat_percent']),
            'fibroglandular_percent': float(composition['fibroglandular_percent']),
            'other_percent': float(composition['other_percent']),
            'mean_hu': float(composition['mean_hu']),
            'median_hu': float(composition['median_hu']),
            'std_hu': float(composition['std_hu']),
        }
        
        with open(output_dir / "tissue_composition.json", 'w') as f:
            json.dump(composition_data, f, indent=2)
            
        # Save the histogram data
        np.save(output_dir / "global_histogram.npy", {
            'histogram': composition['histogram'],
            'bin_centers': composition['bin_centers']
        })
        
        np.save(output_dir / "local_histograms.npy", local_histograms)
        
        # Save the extracted features
        with open(output_dir / "histogram_features.json", 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_compatible_features = {}
            
            # Handle global features
            json_compatible_features['global'] = {
                k: float(v) for k, v in features['global'].items() 
                if k != 'num_histograms'
            }
            json_compatible_features['global']['num_histograms'] = features['global']['num_histograms']
            
            # Handle individual histogram features (just store the first 10 to keep file manageable)
            for i in range(min(10, len(local_histograms['histograms']))):
                key = f'histogram_{i}'
                if key in features:
                    json_compatible_features[key] = {
                        k: float(v) for k, v in features[key].items()
                    }
            
            json.dump(json_compatible_features, f, indent=2)
        
        print(f"Analysis results saved to {output_dir}")
    
    return analyzer
