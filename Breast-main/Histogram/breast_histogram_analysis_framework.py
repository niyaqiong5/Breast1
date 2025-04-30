import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

from breast_segmentation import BreastSegmentation
from Histogram.BreastHistogramAnalyzer import BreastHistogramAnalyzer, process_breast_histograms
from Histogram.AdvancedBreastHistogramAnalyzer import AdvancedBreastHistogramAnalyzer
from Histogram.BreastHistogramResearch import BreastHistogramResearch

# 1: Single Patient Analysis
def analyze_single_patient():
    """Example workflow for analyzing a single patient's LDCT breast histograms"""
    
    # Path to your DICOM dataset
    dicom_folder = "D:/Dataset_only_breast"
    output_dir = "D:/Output/single_patient_analysis"
    
    # Patient identifiers
    study_id = "106620"
    series_id = "106620"
    
    print(f"Analyzing patient {study_id}, series {series_id}")
    
    # Step 1: Create segmentation object and load DICOM data
    segmenter = BreastSegmentation(dicom_folder)
    
    # Load DICOM series
    segmenter.load_dicom_series(study_id, series_id)
    
    # Step 2: Perform breast segmentation
    breast_mask = segmenter.segment_breast()
    
    # Visualize and save the segmentation result
    seg_dir = Path(output_dir) / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    seg_fig = segmenter.visualize_segmentation()
    seg_fig.savefig(seg_dir / "breast_segmentation.png", dpi=150, bbox_inches='tight')
    plt.close(seg_fig)
    
    # Also visualize multiple slices
    multi_slice_fig = segmenter.visualize_multiple_slices(interval=10, num_slices=3)
    if multi_slice_fig:
        multi_slice_fig.savefig(seg_dir / "multi_slice_view.png", dpi=150, bbox_inches='tight')
        plt.close(multi_slice_fig)
    
    # Step 3: Analyze basic histograms
    hist_dir = Path(output_dir) / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = process_breast_histograms(segmenter, hist_dir)
    
    # Step 4: Perform advanced histogram analysis
    adv_dir = Path(output_dir) / "advanced_analysis"
    adv_dir.mkdir(parents=True, exist_ok=True)
    
    adv_analyzer = AdvancedBreastHistogramAnalyzer(analyzer)
    adv_results = adv_analyzer.run_complete_analysis(adv_dir)
    
    # Step 5: Generate a comprehensive report
    report_path = Path(output_dir) / "patient_report.md"
    adv_analyzer.create_summary_report(adv_results, report_path)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    return {
        'segmenter': segmenter,
        'analyzer': analyzer,
        'adv_analyzer': adv_analyzer,
        'results': adv_results
    }

# Research Study on Multiple Patients

def run_research_study():
    """Example workflow for conducting a research study on multiple patients"""
    
    # Path to your DICOM dataset and output directory
    dicom_folder = "D:/Dataset_only_breast"
    output_base = "D:/Output/research_study"
    
    # Create the research study manager
    research = BreastHistogramResearch(dicom_folder, output_base)
    
    # Option 1: Discover and process all patients
    # research.process_all_patients()
    
    # Option 2: Process a specific list of patients
    patient_list = [
        ("106620", "106620"),
        ("075811", "075811"),
        ("999422", "999422"),
        ("083039", "083039"),
        ("085211", "085211")
    ]
    
    research.process_all_patients(patient_list)
    
    # Analyze the results
    research.analyze_results()
    
    # Generate a comprehensive research report
    report_path = research.generate_research_report()
    
    print(f"Research study complete! Report saved to {report_path}")

# 3. Analyzing Existing Segmentation Results

def analyze_existing_segmentation():
    """Example workflow for analyzing existing segmentation results"""
    
    # Path to your existing segmentation results
    segmentation_dir = "D:/Dataset_only_breast_segmentation"
    output_dir = "D:/Output/existing_segmentation_analysis"
    
    # Patient identifiers
    study_id = "106620"
    series_id = "106620"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the segmentation masks from files
    seg_path = Path(segmentation_dir) / study_id / series_id
    
    if not seg_path.exists():
        print(f"Segmentation path {seg_path} does not exist")
        return
    
    # Load breast mask and glandular tissue mask
    breast_mask_file = seg_path / "breast_mask.npy"
    glandular_mask_file = seg_path / "glandular_tissue_mask.npy"
    
    if not breast_mask_file.exists() or not glandular_mask_file.exists():
        print("Required segmentation files not found")
        return
    
    breast_mask = np.load(breast_mask_file)
    glandular_mask = np.load(glandular_mask_file)
    
    # Create the histogram analyzer directly
    analyzer = BreastHistogramAnalyzer()
    
    # You need the original CT image too - let's assume we can load it
    # For a real script, you would need to load this from DICOM files
    # Here we use a placeholder approach just for the example
    try:
        # Create a temporary segmenter to load the CT image
        temp_segmenter = BreastSegmentation("D:/Dataset_only_breast")
        temp_segmenter.load_dicom_series(study_id, series_id)
        ct_image = temp_segmenter.ct_image
        
        # Now set up the analyzer with the loaded data
        analyzer.ct_image = ct_image
        analyzer.breast_mask = breast_mask
        analyzer.glandular_mask = glandular_mask
        
        # Optionally create a fat mask if you need it
        fat_mask = (ct_image >= -150) & (ct_image <= -20) & breast_mask
        analyzer.fat_mask = fat_mask
        
        # Now we can analyze the histograms
        hist_dir = Path(output_dir) / "histograms"
        hist_dir.mkdir(exist_ok=True)
        
        # Compute and visualize global histogram
        composition = analyzer.analyze_tissue_composition()
        global_hist_fig = analyzer.visualize_global_histogram(composition)
        global_hist_fig.savefig(hist_dir / "global_histogram.png", dpi=150, bbox_inches='tight')
        plt.close(global_hist_fig)
        
        # Compute and visualize local histograms
        local_histograms = analyzer.compute_adaptive_local_histograms(
            min_tissue_percent=20, 
            max_regions=50
        )
        local_hist_fig = analyzer.visualize_histograms(local_histograms, max_histograms=9)
        local_hist_fig.savefig(hist_dir / "local_histograms.png", dpi=150, bbox_inches='tight')
        plt.close(local_hist_fig)
        
        # 创建一个直方图映射，选择腺体组织最多的切片
        # 计算每个切片中包含腺体组织的体素数量
        glandular_per_slice = np.sum(glandular_mask, axis=(1, 2))
        # 找出所有包含腺体组织的切片的索引
        valid_slices_glandular = np.where(glandular_per_slice > 0)[0]

        if len(valid_slices_glandular) > 0:
            # 找出腺体组织最多的切片
            slice_idx = valid_slices_glandular[np.argmax(glandular_per_slice[valid_slices_glandular])]
            print(f"选择腺体组织最多的切片: {slice_idx}，包含 {glandular_per_slice[slice_idx]} 个腺体体素")
            
            map_fig = analyzer.visualize_histogram_map(slice_idx, window_size=7, stride=3)
            if map_fig:
                map_fig.savefig(hist_dir / f"histogram_map_slice_{slice_idx}_max_glandular.png", dpi=150, bbox_inches='tight')
                plt.close(map_fig)
        else:
            # 如果没有检测到腺体组织，回退到使用乳腺组织总量
            tissue_per_slice = np.sum(breast_mask, axis=(1, 2))
            valid_slices = np.where(tissue_per_slice > 0)[0]
            
            if len(valid_slices) > 0:
                # 选择乳腺组织最多的切片
                slice_idx = valid_slices[np.argmax(tissue_per_slice[valid_slices])]
                print(f"未找到腺体组织，选择乳腺组织最多的切片: {slice_idx}")
                
                map_fig = analyzer.visualize_histogram_map(slice_idx, window_size=7, stride=3)
                if map_fig:
                    map_fig.savefig(hist_dir / f"histogram_map_slice_{slice_idx}_max_breast.png", dpi=150, bbox_inches='tight')
                    plt.close(map_fig)
        
        # Perform advanced analysis
        adv_dir = Path(output_dir) / "advanced_analysis"
        adv_dir.mkdir(exist_ok=True)
        
        adv_analyzer = AdvancedBreastHistogramAnalyzer(analyzer)
        adv_results = adv_analyzer.run_complete_analysis(adv_dir)
        
        print(f"Analysis of existing segmentation complete! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing existing segmentation: {e}")
        import traceback
        traceback.print_exc()

#  Batch Processing with Customized Analysis

def batch_processing_with_custom_analysis():
    """Example of batch processing with customized analysis parameters"""
    
    # Path to your DICOM dataset
    dicom_folder = "D:/Dataset_only_breast"
    output_base = "D:/Output/custom_batch_analysis"
    
    # List of patients to process
    patient_list = [
        ("106620", "106620"),
        ("130512", "130512"),
        ("145926", "145926")
    ]
    
    # Create output directory
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each patient with custom settings
    for study_id, series_id in patient_list:
        print(f"\nProcessing patient {study_id}, series {series_id} with custom settings")
        
        # Create patient-specific output directory
        patient_dir = output_dir / f"{study_id}_{series_id}"
        patient_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize and run breast segmentation
            segmenter = BreastSegmentation(str(dicom_folder))
            segmenter.load_dicom_series(study_id, series_id)
            breast_mask = segmenter.segment_breast()
            
            # Check if segmentation was successful
            if not np.any(breast_mask):
                print(f"No breast tissue was segmented for {study_id}/{series_id}")
                continue
                
            # Save a visualization of the segmentation
            seg_fig = segmenter.visualize_segmentation()
            if seg_fig:
                seg_fig.savefig(patient_dir / "segmentation.png", dpi=150, bbox_inches='tight')
                plt.close(seg_fig)
            
            # Create and configure the histogram analyzer
            analyzer = BreastHistogramAnalyzer()
            analyzer.load_from_segmenter(segmenter)
            
            # Custom settings for histogram analysis
            analyzer.hu_range = (-180, 180)  # Wider HU range
            analyzer.num_bins = 120  # More bins for higher resolution
            
            # Compute histograms with custom parameters
            composition = analyzer.analyze_tissue_composition()
            
            # Save tissue composition data
            with open(patient_dir / "tissue_composition.json", 'w') as f:
                composition_data = {
                    'fat_percent': float(composition['fat_percent']),
                    'fibroglandular_percent': float(composition['fibroglandular_percent']),
                    'other_percent': float(composition['other_percent']),
                    'mean_hu': float(composition['mean_hu']),
                    'median_hu': float(composition['median_hu']),
                    'std_hu': float(composition['std_hu']),
                }
                json.dump(composition_data, f, indent=2)
            
            # Create custom histogram visualizations
            global_hist_fig = analyzer.visualize_global_histogram(composition)
            global_hist_fig.savefig(patient_dir / "global_histogram.png", dpi=150, bbox_inches='tight')
            plt.close(global_hist_fig)
            
            # Compute high-resolution local histograms
            local_histograms = analyzer.compute_adaptive_local_histograms(
                min_tissue_percent=15,  # Lower threshold to include more regions
                max_regions=80,  # More regions for better coverage
                range_hu=(-180, 180),  # Custom HU range
                bins=120  # More bins for higher resolution
            )
            
            # Custom visualization with more histograms
            local_hist_fig = analyzer.visualize_histograms(local_histograms, max_histograms=16, normalize=True)
            local_hist_fig.savefig(patient_dir / "local_histograms.png", dpi=150, bbox_inches='tight')
            plt.close(local_hist_fig)
            
            # Advanced analysis with custom tissue HU ranges
            adv_analyzer = AdvancedBreastHistogramAnalyzer(analyzer)
            
            # Custom HU ranges for different tissue types
            custom_hu_ranges = {
                'air': (-1000, -600),
                'fat': (-180, -30),  # Wider fat range
                'fibroglandular': (-30, 120),  # Adjusted fibroglandular range
                'skin': (20, 100)
            }
            
            # Compute tissue histograms with custom ranges
            tissue_histograms = adv_analyzer.compute_tissue_type_histograms(hu_ranges=custom_hu_ranges)
            
            # Visualize tissue histograms
            tissue_hist_fig = adv_analyzer.visualize_tissue_histograms(tissue_histograms)
            tissue_hist_fig.savefig(patient_dir / "tissue_histograms.png", dpi=150, bbox_inches='tight')
            plt.close(tissue_hist_fig)
            
            # Generate a density map with custom parameters
            density_map = adv_analyzer.compute_density_map(window_size=12, stride=4)
            if density_map:
                density_map_fig = adv_analyzer.visualize_density_map(density_map)
                density_map_fig.savefig(patient_dir / "density_map.png", dpi=150, bbox_inches='tight')
                plt.close(density_map_fig)
            
            # Classify breast density
            density_classification = adv_analyzer.classify_breast_density(tissue_histograms)
            
            # Save density classification
            with open(patient_dir / "density_classification.json", 'w') as f:
                density_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                               for k, v in density_classification.items()}
                json.dump(density_data, f, indent=2)
            
            print(f"Custom analysis for {study_id}/{series_id} completed successfully")
            
        except Exception as e:
            print(f"Error processing {study_id}/{series_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Batch processing with custom analysis complete! Results saved to {output_base}")


def main():
    """Main function to demonstrate usage of the breast histogram analysis framework"""
    
    print("LDCT Breast Histogram Analysis Framework Demo")
    
    while True:
        print("\nChoose an example to run:")
        print("1. Analyze a single patient")
        print("2. Run a research study on multiple patients")
        print("3. Analyze existing segmentation results")
        print("4. Batch processing with customized analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            analyze_single_patient()
        elif choice == '2':
            run_research_study()
        elif choice == '3':
            analyze_existing_segmentation()
        elif choice == '4':
            batch_processing_with_custom_analysis()
        elif choice == '5':
            print("\nExiting the demo. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
