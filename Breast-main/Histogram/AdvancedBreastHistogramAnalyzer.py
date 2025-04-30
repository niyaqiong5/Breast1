import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from scipy import ndimage
from skimage import measure, morphology
from matplotlib.colors import LinearSegmentedColormap

class AdvancedBreastHistogramAnalyzer:
    """
    Class for advanced analysis of breast tissue histograms, including:
    - Textural feature extraction
    - Density classification
    - Comparative analysis across different regions
    - Visualization of spatial distribution of HU values
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the advanced histogram analyzer.
        """
        self.analyzer = analyzer
        
    def set_analyzer(self, analyzer):
        """Set the BreastHistogramAnalyzer instance to use for analysis"""
        self.analyzer = analyzer
        
    def compute_tissue_type_histograms(self, hu_ranges=None):
        """
        Compute separate histograms for different tissue types based on HU ranges.
        """
        if self.analyzer is None or self.analyzer.ct_image is None:
            raise ValueError("No analyzer or CT image available")
            
        if hu_ranges is None:
            # Default HU ranges for common tissue types
            hu_ranges = {
                'air': (-1000, -600),
                'fat': (-150, -20),
                'fibroglandular': (-20, 100),
                'skin': (20, 100)
            }
        
        # Prepare histogram parameters
        bins = self.analyzer.num_bins
        total_range = (min([r[0] for r in hu_ranges.values()]), 
                       max([r[1] for r in hu_ranges.values()]))
        bin_edges = np.linspace(total_range[0], total_range[1], bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Dictionary to store results
        tissue_histograms = {
            'bin_centers': bin_centers,
            'histograms': {},
            'masks': {},
            'volume_cm3': {},
            'percentages': {}
        }
        
        # Total breast volume
        breast_volume_voxels = np.sum(self.analyzer.breast_mask)
        
        # Compute histograms for each tissue type
        for tissue_type, (min_hu, max_hu) in hu_ranges.items():
            # Create mask for this tissue type
            tissue_mask = (self.analyzer.ct_image >= min_hu) & (self.analyzer.ct_image <= max_hu)
            # Restrict to breast region
            tissue_mask = tissue_mask & self.analyzer.breast_mask
            
            # Calculate volume
            tissue_volume_voxels = np.sum(tissue_mask)
            
            # Skip if no voxels found
            if tissue_volume_voxels == 0:
                tissue_histograms['histograms'][tissue_type] = np.zeros(bins)
                tissue_histograms['masks'][tissue_type] = tissue_mask
                tissue_histograms['volume_cm3'][tissue_type] = 0
                tissue_histograms['percentages'][tissue_type] = 0
                continue
            
            # Compute histogram for this tissue type
            hist, _ = np.histogram(self.analyzer.ct_image[tissue_mask], bins=bin_edges)
            
            # Calculate volume in cubic centimeters (assuming typical DICOM voxel size)
            # This is approximate; for real analysis, use the actual voxel dimensions from DICOM
            voxel_volume_mm3 = 1.0 * 1.0 * 1.0  # Example: 1mm x 1mm x 1mm voxels
            tissue_volume_cm3 = tissue_volume_voxels * voxel_volume_mm3 / 1000.0
            
            # Calculate percentage of total breast volume
            tissue_percentage = tissue_volume_voxels / breast_volume_voxels * 100 if breast_volume_voxels > 0 else 0
            
            # Store results
            tissue_histograms['histograms'][tissue_type] = hist
            tissue_histograms['masks'][tissue_type] = tissue_mask
            tissue_histograms['volume_cm3'][tissue_type] = tissue_volume_cm3
            tissue_histograms['percentages'][tissue_type] = tissue_percentage
            
        return tissue_histograms
    
    def visualize_tissue_histograms(self, tissue_histograms, normalize=True):
        """
        Visualize histograms for different tissue types.
        """
        # Check if there are any histograms to visualize
        if not tissue_histograms['histograms']:
            print("No tissue histograms to visualize")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get bin centers
        bin_centers = tissue_histograms['bin_centers']
        
        # Color map for different tissue types
        colors = {
            'air': 'black',
            'fat': 'yellow',
            'fibroglandular': 'red',
            'skin': 'brown',
            'calcification': 'white',
            'tumor': 'purple'
        }
        
        # Plot each tissue type histogram
        for tissue_type, hist in tissue_histograms['histograms'].items():
            if np.sum(hist) == 0:
                continue
                
            # Normalize if requested
            if normalize and np.sum(hist) > 0:
                hist = hist / np.sum(hist)
                
            # Plot histogram
            color = colors.get(tissue_type, 'gray')
            ax.plot(bin_centers, hist, label=f"{tissue_type.capitalize()} ({tissue_histograms['percentages'][tissue_type]:.1f}%)", 
                   linewidth=2, color=color)
            
        # Add title and labels
        ax.set_title("Histograms by Tissue Type")
        ax.set_xlabel("Hounsfield Units (HU)")
        ax.set_ylabel("Frequency" if not normalize else "Normalized Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compute_density_map(self, slice_idx=None, window_size=10, stride=5):
        """
        Compute a density map based on the HU values within the breast.
        """
        if self.analyzer is None or self.analyzer.ct_image is None:
            raise ValueError("No analyzer or CT image available")
            
        # Find a slice with breast tissue if not specified
        if slice_idx is None:
            tissue_per_slice = np.sum(self.analyzer.breast_mask, axis=(1, 2))
            valid_slices = np.where(tissue_per_slice > 0)[0]
            
            if len(valid_slices) > 0:
                slice_idx = valid_slices[len(valid_slices) // 2]  # Middle slice with tissue
            else:
                print("No slices with breast tissue found")
                return None
        
        # Get the CT slice and mask
        ct_slice = self.analyzer.ct_image[slice_idx]
        mask_slice = self.analyzer.breast_mask[slice_idx]
        
        # Skip if no breast tissue in this slice
        if not np.any(mask_slice):
            print(f"No breast tissue in slice {slice_idx}")
            return None
            
        # Get shape of the slice
        height, width = ct_slice.shape
        
        # Initialize density map
        density_map = np.zeros((
            (height - window_size) // stride + 1,
            (width - window_size) // stride + 1
        ))
        density_map.fill(np.nan)  # Fill with NaN for areas without breast tissue
        
        # Define HU ranges for density calculation
        fat_range = (-150, -20)
        fibroglandular_range = (-20, 100)
        
        # Slide window and compute local density
        for i, y in enumerate(range(0, height - window_size + 1, stride)):
            for j, x in enumerate(range(0, width - window_size + 1, stride)):
                # Extract local window
                window_ct = ct_slice[y:y+window_size, x:x+window_size]
                window_mask = mask_slice[y:y+window_size, x:x+window_size]
                
                # Skip windows with insufficient breast tissue
                tissue_percentage = np.sum(window_mask) / (window_size * window_size) * 100
                if tissue_percentage < 30:  # At least 30% should be breast tissue
                    continue
                
                # Extract CT values within the mask
                masked_values = window_ct[window_mask > 0]
                
                if len(masked_values) == 0:
                    continue
                    
                # Count voxels in fat and fibroglandular ranges
                fat_voxels = np.sum((masked_values >= fat_range[0]) & (masked_values <= fat_range[1]))
                fibro_voxels = np.sum((masked_values >= fibroglandular_range[0]) & (masked_values <= fibroglandular_range[1]))
                
                total_voxels = fat_voxels + fibro_voxels
                
                # Calculate density as percentage of fibroglandular tissue
                if total_voxels > 0:
                    density = fibro_voxels / total_voxels * 100
                    density_map[i, j] = density
        
        return {
            'density_map': density_map,
            'slice_idx': slice_idx,
            'window_size': window_size,
            'stride': stride,
            'shape': ct_slice.shape
        }
    
    def visualize_density_map(self, density_result):
        """
        Visualize the density map overlaid on the CT slice.
        """
        if density_result is None:
            print("No density map to visualize")
            return None
            
        # Extract data from result
        density_map = density_result['density_map']
        slice_idx = density_result['slice_idx']
        window_size = density_result['window_size']
        stride = density_result['stride']
        original_shape = density_result['shape']
        
        # Get the CT slice and mask
        ct_slice = self.analyzer.ct_image[slice_idx]
        mask_slice = self.analyzer.breast_mask[slice_idx]
        
        # Window the CT image for display
        windowed_slice = np.clip(ct_slice, -100, 200)
        windowed_slice = (windowed_slice - (-100)) / (200 - (-100))
        
        # Create figure
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
        
        # Resize the density map to match the original image size
        from scipy.ndimage import zoom
        
        # Create a mask for valid areas (not NaN)
        valid_mask = ~np.isnan(density_map)
        density_map_valid = density_map.copy()
        density_map_valid[~valid_mask] = 0
        
        # Calculate scale factors
        scale_y = original_shape[0] / density_map.shape[0]
        scale_x = original_shape[1] / density_map.shape[1]
        
        # Resize the valid mask and density map
        valid_mask_resized = zoom(valid_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
        density_map_resized = zoom(density_map_valid, (scale_y, scale_x), order=1)
        
        # Create a masked array for plotting
        import numpy.ma as ma
        masked_map = ma.array(density_map_resized, mask=~valid_mask_resized)
        
        # Create custom colormap: blue (low density) to red (high density)
        cmap = LinearSegmentedColormap.from_list('density_cmap', [
            (0.0, 'blue'),     # Low density (mostly fat)
            (0.3, 'cyan'),
            (0.5, 'yellow'),
            (0.7, 'orange'),
            (1.0, 'red')       # High density (mostly fibroglandular)
        ])
        
        # Display the density map
        axes[2].imshow(windowed_slice, cmap='gray')
        density_img = axes[2].imshow(masked_map, cmap=cmap, alpha=0.7, vmin=0, vmax=100)
        axes[2].set_title(f"Breast Density Map")
        axes[2].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(density_img, ax=axes[2], orientation='vertical', shrink=0.8)
        cbar.set_label('Breast Density (%)')
        
        plt.tight_layout()
        return fig
    
    def classify_breast_density(self, tissue_histograms=None):
        """
        Classify breast density according to BI-RADS categories.
        """
        if tissue_histograms is None:
            tissue_histograms = self.compute_tissue_type_histograms()
        
        # Extract percentages of different tissue types
        fat_percent = tissue_histograms['percentages'].get('fat', 0)
        fibroglandular_percent = tissue_histograms['percentages'].get('fibroglandular', 0)
        
        # Calculate total tissue volume
        total_tissue_volume = sum(tissue_histograms['volume_cm3'].values())
        
        # Skip if no tissue found
        if total_tissue_volume == 0:
            return {
                'error': 'No breast tissue found'
            }
        
        # Calculate density as ratio of fibroglandular tissue to total tissue
        total_analyzed_tissue = fat_percent + fibroglandular_percent
        if total_analyzed_tissue > 0:
            density_percentage = fibroglandular_percent / total_analyzed_tissue * 100
        else:
            density_percentage = 0
        
        # Classify according to BI-RADS categories
        # These thresholds are approximate and for illustration only
        if density_percentage < 25:
            category = 'A'
            description = 'Almost entirely fatty'
        elif density_percentage < 50:
            category = 'B'
            description = 'Scattered areas of fibroglandular density'
        elif density_percentage < 75:
            category = 'C'
            description = 'Heterogeneously dense'
        else:
            category = 'D'
            description = 'Extremely dense'
        
        return {
            'density_percentage': density_percentage,
            'birads_category': category,
            'description': description,
            'fat_volume_cm3': tissue_histograms['volume_cm3'].get('fat', 0),
            'fibroglandular_volume_cm3': tissue_histograms['volume_cm3'].get('fibroglandular', 0),
            'total_volume_cm3': total_tissue_volume
        }
    
    def extract_textural_features(self, window_size=10, num_regions=20):
        """
        Extract textural features from the breast tissue.
        """
        if self.analyzer is None or self.analyzer.ct_image is None:
            raise ValueError("No analyzer or CT image available")
        
        # Find slices with breast tissue
        tissue_per_slice = np.sum(self.analyzer.breast_mask, axis=(1, 2))
        valid_slices = np.where(tissue_per_slice > 0)[0]
        
        if len(valid_slices) == 0:
            return {'error': 'No breast tissue found'}
        
        # Initialize features dictionary
        texture_features = {
            'contrast': [],
            'homogeneity': [],
            'energy': [],
            'correlation': [],
            'mean_hu': [],
            'std_hu': [],
            'positions': []
        }
        
        # Randomly sample regions from the breast
        np.random.seed(42)  # For reproducibility
        
        regions_sampled = 0
        max_attempts = 100
        attempts = 0
        
        while regions_sampled < num_regions and attempts < max_attempts:
            # Randomly select a slice with breast tissue
            slice_idx = np.random.choice(valid_slices)
            
            # Get the breast mask for this slice
            mask_slice = self.analyzer.breast_mask[slice_idx]
            
            # Find coordinates of breast voxels
            breast_coords = np.argwhere(mask_slice)
            
            if len(breast_coords) == 0:
                attempts += 1
                continue
            
            # Randomly select a coordinate within the breast
            coord_idx = np.random.randint(0, len(breast_coords))
            y, x = breast_coords[coord_idx]
            
            # Ensure the window fits within the image bounds
            height, width = mask_slice.shape
            if (y < window_size//2 or x < window_size//2 or 
                y + window_size//2 >= height or x + window_size//2 >= width):
                attempts += 1
                continue
            
            # Extract window
            window_start_y = y - window_size//2
            window_start_x = x - window_size//2
            
            window_ct = self.analyzer.ct_image[slice_idx, 
                                              window_start_y:window_start_y+window_size, 
                                              window_start_x:window_start_x+window_size]
            
            window_mask = mask_slice[window_start_y:window_start_y+window_size, 
                                    window_start_x:window_start_x+window_size]
            
            # Skip if window doesn't contain enough breast tissue
            if np.sum(window_mask) < 0.5 * window_size * window_size:
                attempts += 1
                continue
            
            # Calculate textural features
            # Note: For real texture analysis, consider using more sophisticated methods
            # or libraries like scikit-image's GLCM features
            
            # Basic statistical features
            masked_values = window_ct[window_mask > 0]
            
            if len(masked_values) == 0:
                attempts += 1
                continue
            
            mean_hu = np.mean(masked_values)
            std_hu = np.std(masked_values)
            
            # Create a normalized grayscale image for texture analysis
            grayscale = (window_ct - np.min(window_ct)) / (np.max(window_ct) - np.min(window_ct))
            grayscale = grayscale * window_mask  # Apply mask
            
            # Simple textures based on local differences
            dx = np.diff(grayscale, axis=1)
            dy = np.diff(grayscale, axis=0)
            
            # Contrast (sum of squared differences)
            contrast = np.mean(dx**2) + np.mean(dy**2)
            
            # Homogeneity (inverse of contrast)
            homogeneity = 1.0 / (1.0 + contrast) if contrast > 0 else 1.0
            
            # Energy (sum of squared values)
            energy = np.mean(grayscale**2)
            
            # Correlation (relationship between neighboring pixels)
            # This is a simplified approximation
            correlation = np.mean(grayscale[:-1, :-1] * grayscale[1:, 1:])
            
            # Store features
            texture_features['contrast'].append(contrast)
            texture_features['homogeneity'].append(homogeneity)
            texture_features['energy'].append(energy)
            texture_features['correlation'].append(correlation)
            texture_features['mean_hu'].append(mean_hu)
            texture_features['std_hu'].append(std_hu)
            texture_features['positions'].append((slice_idx, y, x))
            
            regions_sampled += 1
        
        # Convert lists to numpy arrays
        for key in texture_features:
            if key != 'positions':
                texture_features[key] = np.array(texture_features[key])
        
        # Add summary statistics
        texture_features['summary'] = {
            'mean_contrast': np.mean(texture_features['contrast']),
            'mean_homogeneity': np.mean(texture_features['homogeneity']),
            'mean_energy': np.mean(texture_features['energy']),
            'mean_correlation': np.mean(texture_features['correlation']),
            'mean_of_mean_hu': np.mean(texture_features['mean_hu']),
            'mean_of_std_hu': np.mean(texture_features['std_hu']),
            'num_regions_sampled': regions_sampled
        }
        
        return texture_features
    
    def visualize_textural_features(self, texture_features):
        """
        Visualize textural features extracted from the breast tissue.
        """
        if 'error' in texture_features:
            print(f"Error: {texture_features['error']}")
            return None
            
        # Extract data
        contrast = texture_features['contrast']
        homogeneity = texture_features['homogeneity']
        energy = texture_features['energy']
        correlation = texture_features['correlation']
        mean_hu = texture_features['mean_hu']
        std_hu = texture_features['std_hu']
        
        # Check if there are enough samples
        if len(contrast) == 0:
            print("No texture samples to visualize")
            return None
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot of contrast vs. homogeneity
        axes[0, 0].scatter(contrast, homogeneity, alpha=0.7)
        axes[0, 0].set_xlabel('Contrast')
        axes[0, 0].set_ylabel('Homogeneity')
        axes[0, 0].set_title('Contrast vs. Homogeneity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot of energy vs. correlation
        axes[0, 1].scatter(energy, correlation, alpha=0.7)
        axes[0, 1].set_xlabel('Energy')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_title('Energy vs. Correlation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot of mean HU vs. std HU
        scatter = axes[1, 0].scatter(mean_hu, std_hu, c=contrast, alpha=0.7, cmap='viridis')
        axes[1, 0].set_xlabel('Mean HU')
        axes[1, 0].set_ylabel('Standard Deviation of HU')
        axes[1, 0].set_title('Mean HU vs. Std HU (colored by contrast)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Contrast')
        
        # Summary statistics
        summary = texture_features['summary']
        summary_text = "\n".join([
            f"Mean Contrast: {summary['mean_contrast']:.4f}",
            f"Mean Homogeneity: {summary['mean_homogeneity']:.4f}",
            f"Mean Energy: {summary['mean_energy']:.4f}",
            f"Mean Correlation: {summary['mean_correlation']:.4f}",
            f"Mean of Mean HU: {summary['mean_of_mean_hu']:.1f}",
            f"Mean of Std HU: {summary['mean_of_std_hu']:.1f}",
            f"Regions Sampled: {summary['num_regions_sampled']}"
        ])
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def run_complete_analysis(self, output_dir=None):
        """
        Run a complete analysis of breast histograms and save results.
        """
        if self.analyzer is None or self.analyzer.ct_image is None:
            raise ValueError("No analyzer or CT image available")
            
        print("Running complete advanced histogram analysis...")
        
        # Initialize results dictionary
        results = {}
        
        # 1. Analyze tissue types
        print("Analyzing tissue types...")
        tissue_histograms = self.compute_tissue_type_histograms()
        results['tissue_histograms'] = tissue_histograms
        
        # Visualize tissue histograms
        if 'error' not in tissue_histograms:
            tissue_hist_fig = self.visualize_tissue_histograms(tissue_histograms)
            results['tissue_hist_fig'] = tissue_hist_fig
        
        # 2. Classify breast density
        print("Classifying breast density...")
        density_classification = self.classify_breast_density(tissue_histograms)
        results['density_classification'] = density_classification
        
        # 3. Compute density map for a representative slice
        print("Computing density map...")
        density_map = self.compute_density_map()
        results['density_map'] = density_map
        
        # Visualize density map
        if density_map is not None:
            density_map_fig = self.visualize_density_map(density_map)
            results['density_map_fig'] = density_map_fig
        
        # 4. Extract textural features
        print("Extracting textural features...")
        texture_features = self.extract_textural_features()
        results['texture_features'] = texture_features
        
        # Visualize textural features
        if 'error' not in texture_features:
            texture_fig = self.visualize_textural_features(texture_features)
            results['texture_fig'] = texture_fig
        
        # Save results if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tissue histograms
            if 'error' not in tissue_histograms:
                # Save visualization
                if 'tissue_hist_fig' in results:
                    results['tissue_hist_fig'].savefig(output_dir / "tissue_histograms.png", 
                                                     dpi=150, bbox_inches='tight')
                
                # Save data as JSON
                tissue_data = {
                    'percentages': {k: float(v) for k, v in tissue_histograms['percentages'].items()},
                    'volume_cm3': {k: float(v) for k, v in tissue_histograms['volume_cm3'].items()}
                }
                
                with open(output_dir / "tissue_histograms.json", 'w') as f:
                    json.dump(tissue_data, f, indent=2)
            
            # Save density classification
            if 'error' not in density_classification:
                with open(output_dir / "density_classification.json", 'w') as f:
                    # Convert numpy values to Python native types
                    density_data = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in density_classification.items()
                    }
                    json.dump(density_data, f, indent=2)
            
            # Save density map visualization
            if 'density_map_fig' in results:
                results['density_map_fig'].savefig(output_dir / "density_map.png", 
                                                 dpi=150, bbox_inches='tight')
            
            # Save textural features
            if 'error' not in texture_features:
                # Save visualization
                if 'texture_fig' in results:
                    results['texture_fig'].savefig(output_dir / "texture_features.png", 
                                                 dpi=150, bbox_inches='tight')
                
                # Save summary as JSON
                texture_summary = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in texture_features['summary'].items()
                }
                
                with open(output_dir / "texture_features.json", 'w') as f:
                    json.dump(texture_summary, f, indent=2)
            
            print(f"Advanced analysis results saved to {output_dir}")
        
        return results
    
    def create_summary_report(self, results, output_path=None):
        """
        Create a comprehensive summary report of the analysis results.
        """
        # Initialize report
        report = []
        report.append("# Breast Histogram Analysis Summary Report")
        report.append("")
        
        # Add date and time
        from datetime import datetime
        now = datetime.now()
        report.append(f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. Tissue Composition
        report.append("## 1. Tissue Composition")
        
        if 'tissue_histograms' in results and 'error' not in results['tissue_histograms']:
            tissue_histograms = results['tissue_histograms']
            
            # Create a table of tissue percentages
            report.append("| Tissue Type | Volume (cm³) | Percentage |")
            
            for tissue_type, percentage in tissue_histograms['percentages'].items():
                volume = tissue_histograms['volume_cm3'].get(tissue_type, 0)
                report.append(f"| {tissue_type.capitalize()} | {volume:.2f} | {percentage:.1f}% |")
                
            report.append("")
        else:
            report.append("No tissue composition data available.")
            report.append("")
        
        # 2. Breast Density Classification
        report.append("## 2. Breast Density Classification")
        
        if 'density_classification' in results and 'error' not in results['density_classification']:
            density = results['density_classification']
            
            report.append(f"**BI-RADS Category:** {density['birads_category']}")
            report.append(f"**Description:** {density['description']}")
            report.append(f"**Density Percentage:** {density['density_percentage']:.1f}%")
            report.append(f"**Fibroglandular Volume:** {density['fibroglandular_volume_cm3']:.2f} cm³")
            report.append(f"**Fat Volume:** {density['fat_volume_cm3']:.2f} cm³")
            report.append(f"**Total Breast Volume:** {density['total_volume_cm3']:.2f} cm³")
            report.append("")
        else:
            report.append("No breast density classification data available.")
            report.append("")
        
        # 3. Texture Analysis
        report.append("## 3. Texture Analysis")
        
        if 'texture_features' in results and 'error' not in results['texture_features'] and 'summary' in results['texture_features']:
            texture = results['texture_features']['summary']
            
            report.append("| Feature | Value |")
            report.append(f"| Mean Contrast | {texture['mean_contrast']:.4f} |")
            report.append(f"| Mean Homogeneity | {texture['mean_homogeneity']:.4f} |")
            report.append(f"| Mean Energy | {texture['mean_energy']:.4f} |")
            report.append(f"| Mean Correlation | {texture['mean_correlation']:.4f} |")
            report.append(f"| Mean HU Value | {texture['mean_of_mean_hu']:.1f} |")
            report.append(f"| Mean HU Standard Deviation | {texture['mean_of_std_hu']:.1f} |")
            report.append("")
        else:
            report.append("No texture analysis data available.")
            report.append("")
        
        # 4. Analysis Notes and Recommendations
        report.append("## 4. Analysis Notes and Recommendations")
        
        # Add recommendations based on density classification
        if 'density_classification' in results and 'error' not in results['density_classification']:
            density = results['density_classification']
            birads = density['birads_category']
            
            if birads == 'A' or birads == 'B':
                report.append("- **Screening Recommendation:** Standard mammography screening at recommended intervals.")
                report.append("- **Risk Assessment:** Lower breast density is associated with reduced risk of masking bias.")
            elif birads == 'C':
                report.append("- **Screening Recommendation:** Consider supplemental screening methods in addition to mammography.")
                report.append("- **Risk Assessment:** Increased breast density may slightly increase breast cancer risk and reduce sensitivity of mammography.")
            elif birads == 'D':
                report.append("- **Screening Recommendation:** Consider supplemental screening with ultrasound or MRI.")
                report.append("- **Risk Assessment:** High breast density may significantly reduce mammographic sensitivity and is associated with increased breast cancer risk.")
            
            report.append("")
        
        # 5. Visualization References
        report.append("## 5. Visualization References")
        report.append("")
        report.append("The following visualizations were generated during analysis:")
        report.append("")
        report.append("- Tissue Histogram: Distribution of CT values across different tissue types")
        report.append("- Density Map: Spatial distribution of breast density")
        report.append("- Texture Analysis: Quantitative assessment of tissue heterogeneity")
        report.append("")
        
        # Create final report
        report_text = "\n".join(report)
        
        # Save report if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
                
            print(f"Summary report saved to {output_path}")
        
        return report_text
