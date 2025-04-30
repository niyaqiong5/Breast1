import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk

from Histogram.BreastHistogramAnalyzer import BreastHistogramAnalyzer
from Histogram.AdvancedBreastHistogramAnalyzer import AdvancedBreastHistogramAnalyzer
from Histogram.PixelWiseBreastAnalyzer import PixelWiseBreastAnalyzer

def analyze_breast_local_histograms(dicom_folder, segmentation_dir, output_dir, study_id, series_id):
    """
    Analyze using only the best slice
    """
    print(f"Processing patient {study_id}, series {series_id}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load segmentation masks
    seg_path = Path(segmentation_dir) / study_id / series_id
    breast_mask_file = seg_path / "breast_mask.npy"
    glandular_mask_file = seg_path / "glandular_tissue_mask.npy"
    
    if not breast_mask_file.exists():
        print(f"Breast mask not found: {breast_mask_file}")
        return
    
    print(f"Loading breast mask: {breast_mask_file}")
    breast_mask = np.load(breast_mask_file)
    
    glandular_mask = None
    if glandular_mask_file.exists():
        print(f"Loading glandular mask: {glandular_mask_file}")
        glandular_mask = np.load(glandular_mask_file)
    
    # Load DICOM images
    series_folder = Path(dicom_folder) / study_id / series_id
    
    if not series_folder.exists():
        print(f"DICOM folder not found: {series_folder}")
        return
    
    print(f"Loading DICOM data: {series_folder}")
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(str(series_folder))
    reader.SetFileNames(dicom_filenames)
    itk_image = reader.Execute()
    ct_image = sitk.GetArrayFromImage(itk_image)
    
    print(f"CT image shape: {ct_image.shape}")
    print(f"Breast mask shape: {breast_mask.shape}")
    
    # Create analyzers
    analyzer = BreastHistogramAnalyzer()
    analyzer.ct_image = ct_image
    analyzer.breast_mask = breast_mask
    analyzer.glandular_mask = glandular_mask
    
    # Create fat mask
    fat_mask = (ct_image >= -150) & (ct_image <= -20) & breast_mask
    analyzer.fat_mask = fat_mask
    
    # Create advanced analyzer and pixel-wise analyzer
    adv_analyzer = AdvancedBreastHistogramAnalyzer(analyzer)
    pixel_analyzer = PixelWiseBreastAnalyzer(adv_analyzer)
    
    # Find the best slice (with most glandular or breast tissue)
    best_slice = pixel_analyzer.find_best_slice()
    print(f"Best slice index: {best_slice}")
    
    # Get CT values and masks for the best slice
    ct_slice = ct_image[best_slice]
    mask_slice = breast_mask[best_slice]
    
    if glandular_mask is not None:
        glandular_slice = glandular_mask[best_slice]
    else:
        glandular_slice = None
    
    # Calculate global histogram
    composition = analyzer.analyze_tissue_composition()
    
    # Calculate local histograms using the best slice - with different window sizes
    window_sizes = [3, 5, 7, 9]
    local_hist_dir = output_path / "local_histograms"
    local_hist_dir.mkdir(exist_ok=True)
    
    for window_size in window_sizes:
        print(f"Calculating local histograms with window size {window_size}...")
        
        # 1. Calculate local histograms - using the selected window size
        local_histograms = analyzer.compute_adaptive_local_histograms(
            min_tissue_percent=20,
            max_regions=9,  # Select 9 representative regions
            range_hu=(-150, 100),
            bins=50
        )
        
        # Visualize local histograms
        hist_fig = analyzer.visualize_histograms(local_histograms, max_histograms=9, normalize=True)
        hist_fig.suptitle(f'Local CT Value Histograms - Slice {best_slice}, Window Size {window_size}', fontsize=16)
        hist_fig.savefig(local_hist_dir / f"local_histograms_w{window_size}.png", dpi=200, bbox_inches='tight')
        plt.close(hist_fig)
        
        # 2. Visualize histogram map - showing local histogram distribution across the slice
        print(f"Generating histogram map (window size {window_size})...")
        histogram_map_fig = analyzer.visualize_histogram_map(best_slice, window_size=window_size, stride=3)
        if histogram_map_fig:
            histogram_map_fig.savefig(local_hist_dir / f"histogram_map_w{window_size}.png", dpi=200, bbox_inches='tight')
            plt.close(histogram_map_fig)
    
    # Create pixel-wise feature maps
    print("Computing pixel-wise features...")
    feature_maps = pixel_analyzer.compute_pixel_wise_features(best_slice, window_size=7)
    feature_fig = pixel_analyzer.visualize_features()
    feature_fig.savefig(output_path / "pixel_features.png", dpi=200, bbox_inches='tight')
    plt.close(feature_fig)
    
    # Create comprehensive visualization - original CT, segmentation, feature overlay, fat/glandular ratio
    print("Creating comprehensive visualization...")
    
    # Window CT values for display
    min_hu, max_hu = -150, 200
    ct_windowed = np.clip(ct_slice, min_hu, max_hu)
    ct_windowed_norm = (ct_windowed - min_hu) / (max_hu - min_hu)
    
    # Create custom colormap - from blue(fat) to red(glandular)
    colors1 = plt.cm.Blues(np.linspace(0.3, 1, 128))
    colors2 = plt.cm.Reds(np.linspace(0, 0.7, 128))
    fat_gland_colors = np.vstack((colors1, colors2))
    fat_gland_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('fat_gland_cmap', fat_gland_colors)
    
    # Perform pixel-wise analysis to get fat and glandular ratio
    # Define breast tissue HU value ranges
    fat_range = (-150, -20)     # Fat tissue
    gland_range = (-20, 100)    # Glandular tissue
    
    # Create fat proportion map and glandular proportion map
    fat_proportion_map = np.zeros_like(ct_slice, dtype=float)
    gland_proportion_map = np.zeros_like(ct_slice, dtype=float)
    
    # Simple method - directly classify by HU values
    fat_mask_slice = (ct_slice >= fat_range[0]) & (ct_slice <= fat_range[1]) & mask_slice
    gland_mask_slice = (ct_slice >= gland_range[0]) & (ct_slice <= gland_range[1]) & mask_slice
    
    fat_proportion_map[mask_slice] = 100.0 * fat_mask_slice[mask_slice]
    gland_proportion_map[mask_slice] = 100.0 * gland_mask_slice[mask_slice]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original CT image
    axes[0, 0].imshow(ct_windowed_norm, cmap='gray')
    axes[0, 0].set_title('Original CT Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Breast segmentation mask
    axes[0, 1].imshow(ct_windowed_norm, cmap='gray')
    axes[0, 1].imshow(mask_slice, alpha=0.5, cmap='Greens')
    axes[0, 1].set_title('Breast Segmentation', fontsize=14)
    axes[0, 1].axis('off')
    
    # Glandular segmentation mask (if available)
    axes[0, 2].imshow(ct_windowed_norm, cmap='gray')
    if glandular_slice is not None:
        axes[0, 2].imshow(glandular_slice, alpha=0.5, cmap='Reds')
        axes[0, 2].set_title('Glandular Segmentation', fontsize=14)
    else:
        axes[0, 2].imshow(fat_mask_slice, alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('Fat Segmentation (Estimated)', fontsize=14)
    axes[0, 2].axis('off')
    
    # Fat proportion map
    axes[1, 0].imshow(ct_windowed_norm, cmap='gray')
    fat_img = axes[1, 0].imshow(fat_proportion_map, cmap='Blues', alpha=0.7, vmin=0, vmax=100)
    axes[1, 0].set_title('Fat Tissue (HU: -150 to -20)', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(fat_img, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Fat Percentage (%)')
    
    # Glandular proportion map
    axes[1, 1].imshow(ct_windowed_norm, cmap='gray')
    gland_img = axes[1, 1].imshow(gland_proportion_map, cmap='Reds', alpha=0.7, vmin=0, vmax=100)
    axes[1, 1].set_title('Glandular Tissue (HU: -20 to 100)', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(gland_img, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Glandular Percentage (%)')
    
    # Fat-glandular distribution map
    # Calculate fat/glandular ratio (-1=all fat, 1=all glandular)
    tissue_ratio = np.zeros_like(fat_proportion_map)
    tissue_mask = fat_mask_slice | gland_mask_slice
    if np.any(tissue_mask):
        tissue_ratio[fat_mask_slice] = -1  # All fat
        tissue_ratio[gland_mask_slice] = 1  # All glandular
    
    axes[1, 2].imshow(ct_windowed_norm, cmap='gray')
    ratio_img = axes[1, 2].imshow(tissue_ratio, cmap=fat_gland_cmap, vmin=-1, vmax=1, alpha=0.7)
    axes[1, 2].set_title('Tissue Type Distribution', fontsize=14)
    axes[1, 2].axis('off')
    
    cbar = plt.colorbar(ratio_img, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.set_label('Tissue Type')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Fat', 'Mixed', 'Glandular'])
    
    # Add overall density information
    fat_percent = composition['fat_percent']
    fibro_percent = composition['fibroglandular_percent']
    
    density_info = f"Breast Density Analysis:\n"
    density_info += f"Fat Tissue: {fat_percent:.1f}%\n"
    density_info += f"Glandular Tissue: {fibro_percent:.1f}%\n"
    
    # Determine BI-RADS classification based on density
    if fibro_percent < 25:
        density_info += "Estimated BI-RADS: A (Almost entirely fatty)"
    elif fibro_percent < 50:
        density_info += "Estimated BI-RADS: B (Scattered fibroglandular densities)"
    elif fibro_percent < 75:
        density_info += "Estimated BI-RADS: C (Heterogeneously dense)"
    else:
        density_info += "Estimated BI-RADS: D (Extremely dense)"
    
    plt.figtext(0.5, 0.01, density_info, ha="center", fontsize=12, 
              bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.suptitle(f'Breast Tissue Local CT Value Analysis - Slice {best_slice}', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.93, bottom=0.12)
    
    plt.savefig(output_path / f"breast_tissue_analysis_slice_{best_slice}.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Generate global CT value histogram
    hist_fig = analyzer.visualize_global_histogram(composition)
    hist_fig.suptitle(f'Global CT Value Histogram Analysis', fontsize=16)
    hist_fig.savefig(output_path / "global_histogram.png", dpi=200, bbox_inches='tight')
    plt.close(hist_fig)
    
    # Create density analysis report
    generate_density_report(analyzer, best_slice, composition, output_path)
    
    print(f"Analysis complete, results saved to {output_path}")
    return {
        'best_slice': best_slice,
        'composition': composition
    }

def generate_density_report(analyzer, slice_idx, composition, output_path):
    """Generate density analysis report"""
    report = []
    report.append("# Breast Tissue Density and CT Value Analysis Report")
    report.append("")
    
    # Global tissue composition
    report.append("## 1. Tissue Composition Analysis")
    report.append("")
    report.append(f"- **Fat Tissue Percentage**: {composition['fat_percent']:.1f}%")
    report.append(f"- **Glandular Tissue Percentage**: {composition['fibroglandular_percent']:.1f}%")
    report.append(f"- **Other Tissue Percentage**: {composition['other_percent']:.1f}%")
    report.append("")
    report.append(f"- **Mean HU Value**: {composition['mean_hu']:.1f}")
    report.append(f"- **Median HU Value**: {composition['median_hu']:.1f}")
    report.append(f"- **HU Value Standard Deviation**: {composition['std_hu']:.1f}")
    report.append("")
    
    # Density assessment
    report.append("## 2. Breast Density Assessment")
    report.append("")
    
    fibro_percent = composition['fibroglandular_percent']
    if fibro_percent < 25:
        birads_category = "A"
        description = "Almost entirely fatty"
    elif fibro_percent < 50:
        birads_category = "B"
        description = "Scattered fibroglandular densities"
    elif fibro_percent < 75:
        birads_category = "C"
        description = "Heterogeneously dense"
    else:
        birads_category = "D"
        description = "Extremely dense"
    
    report.append(f"- **Glandular Tissue Percentage**: {fibro_percent:.1f}%")
    report.append(f"- **Estimated BI-RADS Classification**: {birads_category}")
    report.append(f"- **Description**: {description}")
    report.append("")
    
    # Local histogram analysis
    report.append("## 3. Local CT Value Distribution Analysis")
    report.append("")
    report.append("The generated local histograms show the HU value distribution in different regions of the breast tissue. These local distributions can be used for:")
    report.append("")
    report.append("- Assessing tissue heterogeneity")
    report.append("- Identifying potential abnormal regions (areas that differ significantly from the global distribution)")
    report.append("- Quantifying fat/glandular ratio differences across regions")
    report.append("")
    
    # Save report
    with open(output_path / "density_report.md", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    dicom_folder = "D:/Dataset_only_breast"  # Modify to your DICOM file path
    segmentation_dir = "D:/Dataset_only_breast_segmentation"  # Modify to your segmentation data path
    output_dir = "D:/Output/Pixelwise_breast_local_histograms"  # Modify to your output path
    
    # Set patient identifiers
    study_id = "106620"
    series_id = "106620"

    analyze_breast_local_histograms(dicom_folder, segmentation_dir, output_dir, study_id, series_id)
