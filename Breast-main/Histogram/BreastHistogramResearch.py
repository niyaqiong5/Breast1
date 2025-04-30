import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import json
from breast_segmentation import BreastSegmentation
from Histogram.BreastHistogramAnalyzer import BreastHistogramAnalyzer, process_breast_histograms
from Histogram.AdvancedBreastHistogramAnalyzer import AdvancedBreastHistogramAnalyzer

# Set the visual style for plots
#plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("viridis")

class BreastHistogramResearch:
    """
  conducting research studies on breast histograms in LDCT images.
    This workflow supports:
    1. Processing multiple patient datasets
    2. Extracting histogram features
    3. Comparing features across patients
    4. Statistical analysis of results
    5. Generating research-grade visualizations
    """
    
    def __init__(self, dicom_folder, output_base):
        """
        Initialize the research workflow.
        """
        self.dicom_folder = Path(dicom_folder)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Create directories for different types of outputs
        self.segmentation_dir = self.output_base / "segmentations"
        self.histogram_dir = self.output_base / "histograms"
        self.advanced_dir = self.output_base / "advanced_analysis"
        self.results_dir = self.output_base / "results"
        
        self.segmentation_dir.mkdir(exist_ok=True)
        self.histogram_dir.mkdir(exist_ok=True)
        self.advanced_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Dictionary to store results
        self.patient_results = {}
        
    def discover_patients(self):
        """
        Discover all patients/studies in the DICOM folder.
        """
        patient_list = []
        
        # List all study directories
        for study_dir in self.dicom_folder.iterdir():
            if study_dir.is_dir():
                study_id = study_dir.name
                
                # List all series directories in this study
                for series_dir in study_dir.iterdir():
                    if series_dir.is_dir():
                        series_id = series_dir.name
                        patient_list.append((study_id, series_id))
        
        print(f"Discovered {len(patient_list)} patient datasets")
        return patient_list
    
    def process_patient(self, study_id, series_id):
        """
        Process a single patient dataset.
        """
        print(f"\nProcessing patient {study_id}, series {series_id}")
        
        # Create patient-specific output directories
        patient_key = f"{study_id}_{series_id}"
        seg_dir = self.segmentation_dir / patient_key
        hist_dir = self.histogram_dir / patient_key
        adv_dir = self.advanced_dir / patient_key
        
        seg_dir.mkdir(exist_ok=True)
        hist_dir.mkdir(exist_ok=True)
        adv_dir.mkdir(exist_ok=True)
        
        # Initialize results dictionary
        result = {
            'study_id': study_id,
            'series_id': series_id,
            'patient_key': patient_key,
            'segmentation_success': False,
            'histogram_success': False,
            'advanced_success': False,
            'errors': []
        }
        
        try:
            # Step 1: Initialize and run breast segmentation
            segmenter = BreastSegmentation(str(self.dicom_folder))
            
            # Load DICOM series
            segmenter.load_dicom_series(study_id, series_id)
            
            # Perform breast segmentation
            breast_mask = segmenter.segment_breast()
            
            # Check if segmentation was successful
            if not np.any(breast_mask):
                result['errors'].append("No breast tissue was segmented")
                return result
            
            # Save a visualization of the segmentation
            seg_fig = segmenter.visualize_segmentation()
            if seg_fig:
                seg_fig.savefig(seg_dir / "segmentation.png", dpi=150, bbox_inches='tight')
                plt.close(seg_fig)
            
            result['segmentation_success'] = True
            
            # Step 2: Run basic histogram analysis
            analyzer = process_breast_histograms(segmenter, hist_dir)
            
            # Check if histogram analysis was successful
            if analyzer is None:
                result['errors'].append("Histogram analysis failed")
                return result
            
            result['histogram_success'] = True
            
            # Step 3: Run advanced histogram analysis
            adv_analyzer = AdvancedBreastHistogramAnalyzer(analyzer)
            adv_results = adv_analyzer.run_complete_analysis(adv_dir)
            
            # Generate a summary report
            report_path = adv_dir / "summary_report.md"
            adv_analyzer.create_summary_report(adv_results, report_path)
            
            # Extract key metrics for the patient results
            self._extract_patient_metrics(result, analyzer, adv_results)
            
            result['advanced_success'] = True
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            result['errors'].append(error_msg)
            traceback.print_exc()
        
        # Store result in patient_results dictionary
        self.patient_results[patient_key] = result
        
        return result
    
    def _extract_patient_metrics(self, result, analyzer, adv_results):
        """
        Extract key metrics from analysis results for the patient.
        """
        # Basic histogram metrics
        try:
            composition_file = self.histogram_dir / result['patient_key'] / "tissue_composition.json"
            if os.path.exists(composition_file):
                with open(composition_file, 'r') as f:
                    composition = json.load(f)
                
                result['fat_percent'] = composition.get('fat_percent', 0)
                result['fibroglandular_percent'] = composition.get('fibroglandular_percent', 0)
                result['mean_hu'] = composition.get('mean_hu', 0)
                result['median_hu'] = composition.get('median_hu', 0)
        except Exception as e:
            result['errors'].append(f"Error extracting basic metrics: {str(e)}")
        
        # Advanced metrics - density classification
        try:
            if 'density_classification' in adv_results and 'error' not in adv_results['density_classification']:
                density = adv_results['density_classification']
                result['density_percentage'] = density.get('density_percentage', 0)
                result['birads_category'] = density.get('birads_category', '')
                result['breast_volume_cm3'] = density.get('total_volume_cm3', 0)
        except Exception as e:
            result['errors'].append(f"Error extracting density metrics: {str(e)}")
        
        # Advanced metrics - texture features
        try:
            if 'texture_features' in adv_results and 'error' not in adv_results['texture_features']:
                if 'summary' in adv_results['texture_features']:
                    texture = adv_results['texture_features']['summary']
                    result['mean_contrast'] = texture.get('mean_contrast', 0)
                    result['mean_homogeneity'] = texture.get('mean_homogeneity', 0)
                    result['mean_energy'] = texture.get('mean_energy', 0)
                    result['mean_correlation'] = texture.get('mean_correlation', 0)
        except Exception as e:
            result['errors'].append(f"Error extracting texture metrics: {str(e)}")
    
    def process_all_patients(self, patient_list=None):
        """
        Process all patients in the list or discover them automatically.
        """
        if patient_list is None:
            patient_list = self.discover_patients()
        
        for study_id, series_id in patient_list:
            self.process_patient(study_id, series_id)
        
        # Generate summary of all patients
        self.generate_summary_table()
        
        return self.patient_results
    
    def generate_summary_table(self):
        """
        Generate a summary table of results for all patients.
        """
        # Create list of dictionaries for each patient
        patient_data = []
        
        for patient_key, result in self.patient_results.items():
            # Skip patients with errors
            if not (result['segmentation_success'] and result['histogram_success']):
                continue
            
            # Create a dictionary of patient metrics
            patient_dict = {
                'PatientID': result['study_id'],
                'SeriesID': result['series_id'],
                'FatPercent': result.get('fat_percent', 0),
                'FibroglandularPercent': result.get('fibroglandular_percent', 0),
                'MeanHU': result.get('mean_hu', 0),
                'MedianHU': result.get('median_hu', 0),
                'DensityPercentage': result.get('density_percentage', 0),
                'BIRADSCategory': result.get('birads_category', ''),
                'BreastVolume': result.get('breast_volume_cm3', 0),
                'MeanContrast': result.get('mean_contrast', 0),
                'MeanHomogeneity': result.get('mean_homogeneity', 0),
                'MeanEnergy': result.get('mean_energy', 0),
                'MeanCorrelation': result.get('mean_correlation', 0)
            }
            
            patient_data.append(patient_dict)
        
        # Create DataFrame
        if not patient_data:
            print("No valid patient data for summary table")
            return None
        
        df = pd.DataFrame(patient_data)
        
        # Save to CSV
        csv_path = self.results_dir / "patient_summary.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Summary table saved to {csv_path}")
        
        return df
    
    def analyze_results(self, df=None):
        """
        Perform statistical analysis on the results.
        """
        if df is None:
            csv_path = self.results_dir / "patient_summary.csv"
            if not os.path.exists(csv_path):
                print("No summary table found")
                return None
            
            df = pd.read_csv(csv_path)
        
        if df.empty:
            print("No data for analysis")
            return None
        
        print(f"Analyzing results for {len(df)} patients")
        
        # Initialize results dictionary
        analysis = {}
        
        # 1. Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate mean, median, min, max, std for numeric columns
        stats = df[numeric_cols].describe()
        analysis['basic_stats'] = stats
        
        # Save to CSV
        stats_path = self.results_dir / "descriptive_stats.csv"
        stats.to_csv(stats_path)
        
        # 2. Analyze breast density distribution
        birads_counts = df['BIRADSCategory'].value_counts().reset_index()
        birads_counts.columns = ['Category', 'Count']
        analysis['birads_distribution'] = birads_counts
        
        # Save to CSV
        birads_path = self.results_dir / "birads_distribution.csv"
        birads_counts.to_csv(birads_path, index=False)
        
        # 3. Correlation analysis
        corr_features = [
            'FatPercent', 'FibroglandularPercent', 'MeanHU', 
            'DensityPercentage', 'BreastVolume', 
            'MeanContrast', 'MeanHomogeneity', 'MeanEnergy', 'MeanCorrelation'
        ]
        
        # Make sure all columns exist
        corr_features = [col for col in corr_features if col in df.columns]
        
        if len(corr_features) > 1:
            corr_matrix = df[corr_features].corr()
            analysis['correlation_matrix'] = corr_matrix
            
            # Save to CSV
            corr_path = self.results_dir / "correlation_matrix.csv"
            corr_matrix.to_csv(corr_path)
        
        # 4. Generate visualizations
        self.generate_analysis_visualizations(df, analysis)
        
        return analysis
    
    def generate_analysis_visualizations(self, df, analysis):
        """
        Generate visualizations for the analysis results.
        """
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Histogram of density percentages
        plt.figure(figsize=(10, 6))
        sns.histplot(df['DensityPercentage'], kde=True)
        plt.title('Distribution of Breast Density Percentages')
        plt.xlabel('Density Percentage (%)')
        plt.ylabel('Count')
        plt.savefig(viz_dir / "density_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. BI-RADS category distribution
        if 'BIRADSCategory' in df.columns and not df['BIRADSCategory'].isna().all():
            plt.figure(figsize=(10, 6))
            birads_order = ['A', 'B', 'C', 'D']
            
            # Filter to only include valid categories
            valid_categories = [cat for cat in birads_order if cat in df['BIRADSCategory'].values]
            if valid_categories:
                sns.countplot(data=df, x='BIRADSCategory', order=valid_categories)
                plt.title('Distribution of BI-RADS Categories')
                plt.xlabel('BI-RADS Category')
                plt.ylabel('Count')
                plt.savefig(viz_dir / "birads_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation heatmap
        if 'correlation_matrix' in analysis:
            plt.figure(figsize=(12, 10))
            sns.heatmap(analysis['correlation_matrix'], annot=True, cmap='coolwarm', 
                       vmin=-1, vmax=1, center=0, fmt='.2f')
            plt.title('Correlation Matrix of Breast Features')
            plt.tight_layout()
            plt.savefig(viz_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Scatter plot of fat vs. fibroglandular percentage
        if 'FatPercent' in df.columns and 'FibroglandularPercent' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='FatPercent', y='FibroglandularPercent', 
                           hue='DensityPercentage' if 'DensityPercentage' in df.columns else None,
                           palette='viridis', size='BreastVolume' if 'BreastVolume' in df.columns else None,
                           sizes=(20, 200), alpha=0.7)
            plt.title('Fat vs. Fibroglandular Tissue Percentage')
            plt.xlabel('Fat Percentage (%)')
            plt.ylabel('Fibroglandular Percentage (%)')
            plt.savefig(viz_dir / "fat_vs_fibroglandular.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Relationship between texture features and density
        texture_features = ['MeanContrast', 'MeanHomogeneity', 'MeanEnergy', 'MeanCorrelation']
        texture_features = [col for col in texture_features if col in df.columns]
        
        if texture_features and 'DensityPercentage' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(texture_features[:4]):  # Limit to 4 features
                if i < len(axes):
                    sns.scatterplot(data=df, x='DensityPercentage', y=feature, ax=axes[i])
                    axes[i].set_title(f'Density vs. {feature}')
                    axes[i].set_xlabel('Density Percentage (%)')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "texture_vs_density.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 6. Box plot of HU values by BI-RADS category
        if 'BIRADSCategory' in df.columns and 'MeanHU' in df.columns:
            valid_categories = df['BIRADSCategory'].dropna().unique()
            if len(valid_categories) > 1:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x='BIRADSCategory', y='MeanHU')
                plt.title('Mean HU Values by BI-RADS Category')
                plt.xlabel('BI-RADS Category')
                plt.ylabel('Mean HU Value')
                plt.savefig(viz_dir / "hu_by_birads.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    def generate_research_report(self):
        """
        Generate a comprehensive research report summarizing the findings.
        """
        # Load results data
        csv_path = self.results_dir / "patient_summary.csv"
        if not os.path.exists(csv_path):
            print("No summary table found, cannot generate report")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Generate analysis if not done already
        analysis = self.analyze_results(df)
        
        if analysis is None:
            print("Analysis failed, cannot generate report")
            return None
        
        # Initialize report
        report = []
        report.append("# Breast Histogram Analysis Research Report")
        report.append("")
        
        # Add date and time
        from datetime import datetime
        now = datetime.now()
        report.append(f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. Executive Summary
        report.append("## 1. Executive Summary")
        report.append("")
        report.append("This report presents the results of a comprehensive analysis of breast tissue characteristics")
        report.append("in low-dose CT (LDCT) images. The analysis includes automated segmentation of breast tissue,")
        report.append("quantification of tissue composition, and assessment of breast density and texture features.")
        report.append("")
        
        # Add summary statistics
        if 'basic_stats' in analysis:
            stats = analysis['basic_stats']
            
            report.append("### Key Findings:")
            report.append("")
            
            # Number of patients
            report.append(f"- **Total Patients Analyzed:** {len(df)}")
            
            # Mean density percentage
            if 'DensityPercentage' in stats:
                mean_density = stats.loc['mean', 'DensityPercentage']
                report.append(f"- **Mean Breast Density:** {mean_density:.1f}%")
            
            # BI-RADS distribution
            if 'birads_distribution' in analysis:
                birads_dist = analysis['birads_distribution']
                report.append("- **BI-RADS Category Distribution:**")
                for _, row in birads_dist.iterrows():
                    report.append(f"  - Category {row['Category']}: {row['Count']} patients ({row['Count']/len(df)*100:.1f}%)")
            
            report.append("")
        
        # 2. Methodology
        report.append("## 2. Methodology")
        report.append("")
        report.append("### 2.1 Data Collection")
        report.append("")
        report.append(f"LDCT images were collected from {len(df)} patients. ")
        report.append("All images were acquired using a standardized imaging protocol.")
        report.append("")
        
        report.append("### 2.2 Image Processing")
        report.append("")
        report.append("The following steps were performed for each patient dataset:")
        report.append("")
        report.append("1. **Breast Segmentation:** Automated segmentation of breast tissue using a multi-stage algorithm")
        report.append("   that identifies skin, thoracic cavity, fat, and glandular tissues.")
        report.append("2. **Histogram Analysis:** Computation of local and global histograms of CT intensity values within")
        report.append("   the segmented breast region.")
        report.append("3. **Tissue Classification:** Classification of breast tissue into fat and fibroglandular components")
        report.append("   based on Hounsfield Unit (HU) ranges.")
        report.append("4. **Density Assessment:** Calculation of breast density percentage and BI-RADS category assignment.")
        report.append("5. **Texture Analysis:** Extraction of textural features including contrast, homogeneity, energy,")
        report.append("   and correlation from local regions within the breast.")
        report.append("")
        
        # 3. Results
        report.append("## 3. Results")
        report.append("")
        
        # 3.1 Tissue Composition
        report.append("### 3.1 Tissue Composition")
        report.append("")
        
        if 'FatPercent' in df.columns and 'FibroglandularPercent' in df.columns:
            mean_fat = df['FatPercent'].mean()
            mean_fibro = df['FibroglandularPercent'].mean()
            std_fat = df['FatPercent'].std()
            std_fibro = df['FibroglandularPercent'].std()
            
            report.append(f"The mean fat percentage was {mean_fat:.1f}% (SD: {std_fat:.1f}%), while the mean")
            report.append(f"fibroglandular tissue percentage was {mean_fibro:.1f}% (SD: {std_fibro:.1f}%).")
            report.append("")
        
        # Add reference to visualization
        report.append("Figure 1 shows the relationship between fat and fibroglandular tissue percentages across all patients.")
        report.append("")
        
        # 3.2 Breast Density
        report.append("### 3.2 Breast Density")
        report.append("")
        
        if 'DensityPercentage' in df.columns:
            mean_density = df['DensityPercentage'].mean()
            median_density = df['DensityPercentage'].median()
            min_density = df['DensityPercentage'].min()
            max_density = df['DensityPercentage'].max()
            
            report.append(f"The mean breast density was {mean_density:.1f}% with a median of {median_density:.1f}%.")
            report.append(f"Density values ranged from {min_density:.1f}% to {max_density:.1f}%.")
            report.append("")
        
        # BI-RADS distribution
        if 'BIRADSCategory' in df.columns and not df['BIRADSCategory'].isna().all():
            # Calculate category distribution
            birads_counts = df['BIRADSCategory'].value_counts()
            birads_percents = (birads_counts / len(df) * 100).round(1)
            
            report.append("The distribution of BI-RADS density categories was as follows:")
            report.append("")
            report.append("| BI-RADS Category | Count | Percentage |")
            report.append("|------------------|-------|------------|")
            
            for category in ['A', 'B', 'C', 'D']:
                if category in birads_counts:
                    count = birads_counts[category]
                    percent = birads_percents[category]
                    report.append(f"| {category} | {count} | {percent}% |")
            
            report.append("")
        
        report.append("Figure 2 shows the distribution of BI-RADS categories across all patients.")
        report.append("")
        
        # 3.3 Texture Analysis
        report.append("### 3.3 Texture Analysis")
        report.append("")
        
        texture_features = ['MeanContrast', 'MeanHomogeneity', 'MeanEnergy', 'MeanCorrelation']
        texture_features_present = [col for col in texture_features if col in df.columns]
        
        if texture_features_present:
            report.append("Texture features were extracted to characterize tissue heterogeneity:")
            report.append("")
            report.append("| Feature | Mean | Standard Deviation | Minimum | Maximum |")
            report.append("|---------|------|-------------------|---------|---------|")
            
            for feature in texture_features_present:
                if feature in df.columns:
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    
                    # Format based on the magnitude of values
                    if abs(mean_val) < 0.01:
                        report.append(f"| {feature} | {mean_val:.6f} | {std_val:.6f} | {min_val:.6f} | {max_val:.6f} |")
                    elif abs(mean_val) < 0.1:
                        report.append(f"| {feature} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} |")
                    else:
                        report.append(f"| {feature} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |")
            
            report.append("")
        
        report.append("Figure 3 illustrates the relationship between breast density and texture features.")
        report.append("")
        
        # 3.4 Correlation Analysis
        report.append("### 3.4 Correlation Analysis")
        report.append("")
        
        if 'correlation_matrix' in analysis:
            # Extract some notable correlations
            corr_matrix = analysis['correlation_matrix']
            
            # Find the strongest positive and negative correlations
            upper_triangle = np.triu(corr_matrix.values, k=1)
            strongest_positive = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
            strongest_negative = np.unravel_index(np.argmin(upper_triangle), upper_triangle.shape)
            
            pos_corr_val = upper_triangle[strongest_positive]
            neg_corr_val = upper_triangle[strongest_negative]
            
            pos_feature1 = corr_matrix.index[strongest_positive[0]]
            pos_feature2 = corr_matrix.columns[strongest_positive[1]]
            
            neg_feature1 = corr_matrix.index[strongest_negative[0]]
            neg_feature2 = corr_matrix.columns[strongest_negative[1]]
            
            report.append("Correlation analysis revealed relationships between various breast tissue features:")
            report.append("")
            
            if abs(pos_corr_val) > 0.3:
                report.append(f"- The strongest positive correlation was between {pos_feature1} and {pos_feature2} (r = {pos_corr_val:.2f}),")
                report.append(f"  suggesting that these features tend to increase together.")
                report.append("")
            
            if abs(neg_corr_val) > 0.3:
                report.append(f"- The strongest negative correlation was between {neg_feature1} and {neg_feature2} (r = {neg_corr_val:.2f}),")
                report.append(f"  indicating an inverse relationship between these features.")
                report.append("")
            
            # Specific focus on density correlations
            if 'DensityPercentage' in corr_matrix.index:
                density_corrs = corr_matrix['DensityPercentage'].drop('DensityPercentage')
                density_corrs = density_corrs.sort_values(ascending=False)
                
                report.append("The features most strongly correlated with breast density were:")
                report.append("")
                
                for feature, corr_val in density_corrs.items()[:3]:
                    if abs(corr_val) > 0.2:
                        direction = "positive" if corr_val > 0 else "negative"
                        report.append(f"- {feature}: {corr_val:.2f} ({direction} correlation)")
                
                report.append("")
        
        report.append("Figure 4 presents a correlation heatmap of all analyzed features.")
        report.append("")
        
        # 4. Discussion
        report.append("## 4. Discussion")
        report.append("")
        report.append("This analysis demonstrates the feasibility of extracting quantitative breast tissue")
        report.append("characteristics from LDCT images. The histogram-based approach allows for automated")
        report.append("assessment of breast density and tissue composition, which may have clinical applications")
        report.append("in breast cancer risk assessment and personalized screening recommendations.")
        report.append("")
        
        # Add specific insights based on the data
        if 'DensityPercentage' in df.columns:
            high_density_percent = (df['DensityPercentage'] > 50).mean() * 100
            report.append(f"In this cohort, {high_density_percent:.1f}% of patients had breast density above 50%,")
            report.append("which is considered high density. These patients might benefit from supplemental")
            report.append("screening modalities beyond mammography due to the masking effect of dense tissue.")
            report.append("")
        
        report.append("The texture analysis revealed heterogeneity in breast tissue composition that may not")
        report.append("be captured by density measurements alone. Texture features could potentially serve as")
        report.append("imaging biomarkers for tissue characterization and disease risk assessment.")
        report.append("")
        
        # 5. Limitations
        report.append("## 5. Limitations")
        report.append("")
        report.append("Several limitations should be considered when interpreting these results:")
        report.append("")
        report.append("1. The sample size is relatively small, limiting the generalizability of the findings.")
        report.append("2. The automatic segmentation algorithm may not perfectly delineate breast tissue in all cases.")
        report.append("3. The HU ranges used for tissue classification are approximate and may not account for")
        report.append("   all individual variations in tissue composition.")
        report.append("4. The analysis does not account for potential confounding factors such as age, menopausal")
        report.append("   status, and breast size.")
        report.append("")
        
        # 6. Conclusion
        report.append("## 6. Conclusion")
        report.append("")
        report.append("This research demonstrates that local histogram analysis of CT intensity values can provide")
        report.append("valuable information about breast tissue composition and heterogeneity in LDCT images.")
        report.append("The methods developed in this study could be applied to large-scale screening programs to")
        report.append("enable automated assessment of breast density, potentially improving risk stratification")
        report.append("and personalized screening recommendations.")
        report.append("")
        report.append("Future work should focus on validating these findings in larger cohorts, correlating")
        report.append("imaging features with clinical outcomes, and developing machine learning models to")
        report.append("predict breast cancer risk based on quantitative imaging biomarkers.")
        report.append("")
        
        # 7. References
        report.append("## 7. References")
        report.append("")
        report.append("1. Winkel RR, et al. Mammographic density and structural features can individually and jointly contribute to breast cancer risk assessment in mammography screening: a case-control study. BMC Cancer. 2016;16:414.")
        report.append("2. Wengert GJ, et al. Density histogram analysis of unenhanced computed tomography for assessing breast density: correlation with quantitative mammographic measurements. Medicine (Baltimore). 2018;97(36):e12100.")
        report.append("3. Tagliafico A, et al. Mammographic density estimation: one-to-one comparison of digital mammography and digital breast tomosynthesis using fully automated software. Eur Radiol. 2012;22(6):1265-70.")
        report.append("4. Weigel S, et al. Digital mammography screening: sensitivity of the programme dependent on breast density. Eur Radiol. 2017;27(7):2744-51.")
        report.append("")
        
        # Create final report
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.results_dir / "research_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        print(f"Research report saved to {report_path}")
        
        return report_path
