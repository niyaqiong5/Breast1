"""
Breast cancer risk stratification model - clinical feature extraction module
Responsible for extracting and processing features from clinical data
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClinicalFeatureExtractor:
    
    def __init__(self):
        pass
    
    def extract_features(self, clinical_data):
        """
        Extract features from clinical data
        """
        features = {}
        
        try:
            # Convert to dictionary for unified processing
            if isinstance(clinical_data, pd.Series):
                data = clinical_data.to_dict()
            else:
                data = clinical_data
            
            # Basic Features
            features['density_numeric'] = data.get('density_numeric', None)
            features['history'] = data.get('history', None)
            features['age'] = data.get('Age', None)
            features['bmi'] = data.get('BMI', None)
            
            # Derived Features
            
            # Age group characteristics
            if features['age'] is not None:
                features['age_group'] = self._age_to_group(features['age'])
            
            # BMI classification characteristics
            if features['bmi'] is not None:
                features['bmi_category'] = self._bmi_to_category(features['bmi'])
            
            # Interaction characteristics between breast density and age
            if features['density_numeric'] is not None and features['age'] is not None:
                features['density_age_interaction'] = features['density_numeric'] * features['age']
            
            # Interaction characteristics between breast density and BMI
            if features['density_numeric'] is not None and features['bmi'] is not None:
                features['density_bmi_interaction'] = features['density_numeric'] * features['bmi']
            
            # Interaction characteristics between breast density and family history
            if features['density_numeric'] is not None and features['history'] is not None:
                features['density_history_interaction'] = features['density_numeric'] * features['history']
            
            # Standardized for age and BMI (based on typical ranges)
            if features['age'] is not None:
                features['age_normalized'] = (features['age'] - 50) / 15  # Assume mean 50, standard deviation 15
            
            if features['bmi'] is not None:
                features['bmi_normalized'] = (features['bmi'] - 25) / 5  # Assume mean 25, standard deviation 5
            
            logger.info("Successfully extracted clinical features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            return None
    
    def _age_to_group(self, age):
        """
        Convert age to group
        """
        if age < 40:
            return 0
        elif age < 50:
            return 1
        elif age < 60:
            return 2
        elif age < 70:
            return 3
        else:
            return 4
    
    def _bmi_to_category(self, bmi):
        """
        Convert BMI to classification
        """
        if bmi < 18.5:
            return 0  # Low body weight
        elif bmi < 24:
            return 1  # Normal
        elif bmi < 28:
            return 2  # overweight
        else:
            return 3  # obesity

    def process_batch(self, clinical_df):
        
        result_dfs = []
        
        # Keep the original PID and Label columns
        result_df = clinical_df[['PID']].copy()
        if 'label' in clinical_df.columns:
            result_df['label'] = clinical_df['label']
        if 'risk' in clinical_df.columns:
            result_df['risk'] = clinical_df['risk']
        if 'risk_numeric' in clinical_df.columns:
            result_df['risk_numeric'] = clinical_df['risk_numeric']
        
        # Extract clinical characteristics of each patient
        features_list = []
        for _, row in clinical_df.iterrows():
            features = self.extract_features(row)
            if features:
                features['PID'] = row['PID']
                features_list.append(features)
        
        # Create a feature DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
            
            # Merge the feature DataFrame and the result DataFrame
            result_df = pd.merge(result_df, features_df, on='PID', how='left')
            
            logger.info(f"成功为{len(features_list)}个患者提取临床特征")
            return result_df
        else:
            logger.warning("No clinical characteristics of any patient were successfully extracted")
            return result_df  # Returns a DataFrame containing only the PID and the label
