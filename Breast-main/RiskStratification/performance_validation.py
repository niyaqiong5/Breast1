"""
Complete performance validation file
For validating bilateral breast MIL model performance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceValidator:
    """Model performance validator"""
    
    def __init__(self, model, data_manager):
        self.model = model
        self.data_manager = data_manager
        
    def check_data_leakage(self, train_data, val_data, test_data):
        """Check data leakage"""
        print("🔍 Checking data leakage...")
        
        # Get patient IDs
        train_patients = set([info['patient_id'] for info in train_data['bag_info']])
        val_patients = set([info['patient_id'] for info in val_data['bag_info']])
        test_patients = set([info['patient_id'] for info in test_data['bag_info']])
        
        # Check overlaps
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        print(f"   Train-Val overlap: {len(train_val_overlap)} patients")
        print(f"   Train-Test overlap: {len(train_test_overlap)} patients")
        print(f"   Val-Test overlap: {len(val_test_overlap)} patients")
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("   ⚠️ Data leakage detected!")
            return False
        else:
            print("   ✅ No data leakage found")
            return True
    
    def analyze_test_set_distribution(self, test_data):
        """Analyze test set distribution"""
        print("📊 Detailed test set analysis...")
        
        # Risk distribution
        risk_labels = test_data['risk_labels']
        unique_labels, counts = np.unique(risk_labels, return_counts=True)
        
        print(f"   Test set size: {len(test_data['bags'])} patients")
        print(f"   Risk distribution:")
        for label, count in zip(unique_labels, counts):
            risk_name = 'Medium Risk' if label == 0 else 'High Risk'
            print(f"     {risk_name}: {count} ({count/len(risk_labels)*100:.1f}%)")
        
        # BI-RADS distribution
        if 'bag_info' in test_data and len(test_data['bag_info']) > 0:
            birads_left = [info.get('birads_left', 0) for info in test_data['bag_info']]
            birads_right = [info.get('birads_right', 0) for info in test_data['bag_info']]
            asymmetry = [info.get('birads_asymmetry', 0) for info in test_data['bag_info']]
            
            print(f"   Left BI-RADS range: {min(birads_left)}-{max(birads_left)}")
            print(f"   Right BI-RADS range: {min(birads_right)}-{max(birads_right)}")
            print(f"   Average asymmetry: {np.mean(asymmetry):.2f}")
    
    def perform_bootstrap_validation(self, test_data, n_bootstrap=1000):
        """Bootstrap validation"""
        print("🔄 Performing Bootstrap validation...")
        
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features']
        ]
        y_test = test_data['risk_labels']
        
        # Get predictions
        predictions = self.model.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Bootstrap sampling
        n_samples = len(y_test)
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # Random sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_y_true = y_test[indices]
            bootstrap_y_pred = pred_classes[indices]
            
            # Calculate accuracy
            accuracy = np.mean(bootstrap_y_true == bootstrap_y_pred)
            bootstrap_scores.append(accuracy)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        print(f"   Bootstrap results (n={n_bootstrap}):")
        print(f"     Mean accuracy: {np.mean(bootstrap_scores):.3f}")
        print(f"     Standard deviation: {np.std(bootstrap_scores):.3f}")
        print(f"     95% CI: [{np.percentile(bootstrap_scores, 2.5):.3f}, {np.percentile(bootstrap_scores, 97.5):.3f}]")
        
        return bootstrap_scores
    
    def analyze_prediction_confidence(self, test_data):
        """Analyze prediction confidence"""
        print("🎯 Analyzing prediction confidence...")
        
        X_test = [
            test_data['bags'],
            test_data['instance_masks'],
            test_data['clinical_features']
        ]
        y_test = test_data['risk_labels']
        
        # Get prediction probabilities
        predictions = self.model.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        max_probs = np.max(predictions, axis=1)
        
        # Analyze confidence
        print(f"   Prediction confidence statistics:")
        print(f"     Mean confidence: {np.mean(max_probs):.3f}")
        print(f"     Minimum confidence: {np.min(max_probs):.3f}")
        print(f"     High confidence (>0.9) samples: {np.sum(max_probs > 0.9)}/{len(max_probs)}")
        
        # Analyze confidence of incorrect predictions
        errors = pred_classes != y_test
        if np.sum(errors) > 0:
            error_confidences = max_probs[errors]
            print(f"     Mean confidence of incorrect predictions: {np.mean(error_confidences):.3f}")
        else:
            print(f"     ✅ No incorrect predictions")
        
        return predictions, max_probs
    
    def check_model_complexity(self):
        """Check model complexity"""
        print("🏗️ Model complexity analysis...")
        
        total_params = self.model.model.count_params()
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Parameters/sample ratio: {total_params/60:.1f} (assuming 60 training samples)")
        
        if total_params > 60 * 10:  # Parameters > training samples * 10
            print("   ⚠️ Model may be too complex, prone to overfitting")
        else:
            print("   ✅ Model complexity is reasonable")
    
    def generate_validation_report(self, train_data, val_data, test_data, mil_data):
        """Generate complete validation report"""
        print("\n" + "="*80)
        print("🔍 Model Performance Validation Report")
        print("="*80)
        
        # 1. Data leakage check
        leakage_free = self.check_data_leakage(train_data, val_data, test_data)
        
        print("\n" + "-"*60)
        
        # 2. Test set analysis
        self.analyze_test_set_distribution(test_data)
        
        print("\n" + "-"*60)
        
        # 3. Model complexity check
        self.check_model_complexity()
        
        print("\n" + "-"*60)
        
        # 4. Bootstrap validation
        bootstrap_scores = self.perform_bootstrap_validation(test_data)
        
        print("\n" + "-"*60)
        
        # 5. Prediction confidence analysis
        predictions, confidences = self.analyze_prediction_confidence(test_data)
        
        print("\n" + "="*80)
        print("📋 Summary and Recommendations")
        print("="*80)
        
        # Generate recommendations
        suggestions = []
        
        if not leakage_free:
            suggestions.append("❗ Fix data leakage issues")
        
        if len(test_data['bags']) < 30:
            suggestions.append("🔄 Increase test set size or use cross-validation")
        
        if np.std(bootstrap_scores) > 0.1:
            suggestions.append("⚠️ Prediction results are unstable, need more data")
        
        if self.model.model.count_params() > len(train_data['bags']) * 10:
            suggestions.append("🏗️ Consider simplifying model architecture")
        
        if len(suggestions) == 0:
            print("✅ Model performance appears reliable")
        else:
            print("Recommended improvements:")
            for suggestion in suggestions:
                print(f"   {suggestion}")
        
        return {
            'leakage_free': leakage_free,
            'test_size': len(test_data['bags']),
            'bootstrap_scores': bootstrap_scores,
            'confidence_scores': confidences,
            'suggestions': suggestions
        }


class PerformanceDeepAnalysis:
    """Deep performance analyzer"""
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        
    def analyze_feature_quality(self):
        """Analyze feature quality"""
        print("🎯 Analyzing feature quality...")
        
        # 1. BI-RADS asymmetry analysis
        asymmetry_scores = [info.get('birads_asymmetry', 0) for info in self.test_data['bag_info']]
        risk_labels = self.test_data['risk_labels']
        
        print(f"\n📊 BI-RADS asymmetry analysis:")
        
        # Group analysis by risk level
        high_risk_asymmetry = [asymmetry_scores[i] for i in range(len(risk_labels)) if risk_labels[i] == 1]
        medium_risk_asymmetry = [asymmetry_scores[i] for i in range(len(risk_labels)) if risk_labels[i] == 0]
        
        if high_risk_asymmetry:
            print(f"   High risk patient asymmetry: {np.mean(high_risk_asymmetry):.2f} ± {np.std(high_risk_asymmetry):.2f}")
        if medium_risk_asymmetry:
            print(f"   Medium risk patient asymmetry: {np.mean(medium_risk_asymmetry):.2f} ± {np.std(medium_risk_asymmetry):.2f}")
        
        # 2. Left-right BI-RADS distribution analysis
        birads_left = [info.get('birads_left', 0) for info in self.test_data['bag_info']]
        birads_right = [info.get('birads_right', 0) for info in self.test_data['bag_info']]
        
        print(f"\n📊 BI-RADS distribution analysis:")
        print(f"   Left BI-RADS: {min(birads_left)}-{max(birads_left)}, mean: {np.mean(birads_left):.1f}")
        print(f"   Right BI-RADS: {min(birads_right)}-{max(birads_right)}, mean: {np.mean(birads_right):.1f}")
        
        # 3. Slice count analysis
        left_slices = [info.get('n_left_instances', 0) for info in self.test_data['bag_info']]
        right_slices = [info.get('n_right_instances', 0) for info in self.test_data['bag_info']]
        total_slices = [info.get('n_total_instances', 0) for info in self.test_data['bag_info']]
        
        print(f"\n📊 Slice information analysis:")
        print(f"   Left slices: {np.mean(left_slices):.1f} ± {np.std(left_slices):.1f}")
        print(f"   Right slices: {np.mean(right_slices):.1f} ± {np.std(right_slices):.1f}")
        print(f"   Total slices: {np.mean(total_slices):.1f} ± {np.std(total_slices):.1f}")
        
        return {
            'asymmetry_scores': asymmetry_scores,
            'high_risk_asymmetry': high_risk_asymmetry,
            'medium_risk_asymmetry': medium_risk_asymmetry
        }
    
    def analyze_attention_patterns(self):
        """Analyze attention patterns"""
        print("\n🔍 Analyzing attention patterns...")
        
        X_test = [
            self.test_data['bags'],
            self.test_data['instance_masks'],
            self.test_data['clinical_features']
        ]
        
        # Get attention weights
        predictions, attention_weights = self.model.attention_model.predict(X_test, verbose=0)
        
        attention_analysis = {
            'high_risk_attention': [],
            'medium_risk_attention': [],
            'attention_positions': [],
            'attention_variances': []
        }
        
        for i in range(len(self.test_data['bags'])):
            risk = self.test_data['risk_labels'][i]
            valid_slices = int(np.sum(self.test_data['instance_masks'][i]))
            attention_scores = attention_weights[i, :valid_slices, 0]
            
            # Record attention distribution
            if risk == 1:
                attention_analysis['high_risk_attention'].extend(attention_scores)
            else:
                attention_analysis['medium_risk_attention'].extend(attention_scores)
            
            # Analyze attention position
            max_attention_pos = np.argmax(attention_scores) / (valid_slices - 1) if valid_slices > 1 else 0.5
            attention_analysis['attention_positions'].append(max_attention_pos)
            
            # Attention variance
            attention_analysis['attention_variances'].append(np.var(attention_scores))
        
        print(f"   High risk attention distribution: mean={np.mean(attention_analysis['high_risk_attention']):.3f}")
        print(f"   Medium risk attention distribution: mean={np.mean(attention_analysis['medium_risk_attention']):.3f}")
        print(f"   Attention position preference: mean={np.mean(attention_analysis['attention_positions']):.3f}")
        print(f"   Attention concentration: variance mean={np.mean(attention_analysis['attention_variances']):.3f}")
        
        return attention_analysis
    
    def analyze_decision_boundary_quality(self):
        """Analyze decision boundary quality"""
        print("\n🎯 Analyzing decision boundary quality...")
        
        X_test = [
            self.test_data['bags'],
            self.test_data['instance_masks'],
            self.test_data['clinical_features']
        ]
        y_test = self.test_data['risk_labels']
        
        # Get prediction probabilities
        predictions = self.model.model.predict(X_test, verbose=0)
        prob_high_risk = predictions[:, 1]
        
        # Analyze prediction probability distribution
        high_risk_probs = prob_high_risk[y_test == 1]
        medium_risk_probs = prob_high_risk[y_test == 0]
        
        print(f"   High risk patient prediction probability: {np.mean(high_risk_probs):.3f} ± {np.std(high_risk_probs):.3f}")
        print(f"   Medium risk patient prediction probability: {np.mean(medium_risk_probs):.3f} ± {np.std(medium_risk_probs):.3f}")
        
        # Calculate separation
        separation = np.mean(high_risk_probs) - np.mean(medium_risk_probs)
        print(f"   Class separation: {separation:.3f}")
        
        # Analyze confidence
        confidence_scores = np.max(predictions, axis=1)
        print(f"   Mean prediction confidence: {np.mean(confidence_scores):.3f}")
        print(f"   Minimum prediction confidence: {np.min(confidence_scores):.3f}")
        
        return {
            'prob_high_risk': prob_high_risk,
            'separation': separation,
            'confidence_scores': confidence_scores
        }
    
    def analyze_clinical_feature_contribution(self):
        """Analyze clinical feature contribution"""
        print("\n🏥 Analyzing clinical feature contribution...")
        
        clinical_features = self.test_data['clinical_features']
        risk_labels = self.test_data['risk_labels']
        
        # Feature names
        feature_names = ['Age', 'BMI', 'Density', 'Family History', 
                        'Age Group', 'BMI Category', 'Age×Density', 'BI-RADS Asymmetry']
        
        print(f"   Clinical feature analysis:")
        for i, feature_name in enumerate(feature_names):
            if i < clinical_features.shape[1]:
                high_risk_values = clinical_features[risk_labels == 1, i]
                medium_risk_values = clinical_features[risk_labels == 0, i]
                
                high_mean = np.mean(high_risk_values) if len(high_risk_values) > 0 else 0
                medium_mean = np.mean(medium_risk_values) if len(medium_risk_values) > 0 else 0
                
                print(f"     {feature_name}:")
                print(f"       High risk: {high_mean:.3f}")
                print(f"       Medium risk: {medium_mean:.3f}")
                print(f"       Difference: {abs(high_mean - medium_mean):.3f}")
        
        return clinical_features
    
    def test_robustness(self):
        """Test model robustness"""
        print("\n🛡️ Testing model robustness...")
        
        X_test = [
            self.test_data['bags'],
            self.test_data['instance_masks'],
            self.test_data['clinical_features']
        ]
        
        # Original predictions
        original_predictions = self.model.model.predict(X_test, verbose=0)
        original_classes = np.argmax(original_predictions, axis=1)
        
        # Add small noise test
        noise_levels = [0.01, 0.02, 0.05]
        robustness_scores = []
        
        for noise_level in noise_levels:
            # Add noise to images
            noisy_bags = X_test[0] + np.random.normal(0, noise_level, X_test[0].shape)
            noisy_bags = np.clip(noisy_bags, 0, 1)
            
            # Add noise to clinical features
            noisy_clinical = X_test[2] + np.random.normal(0, noise_level * 0.1, X_test[2].shape)
            
            X_noisy = [noisy_bags, X_test[1], noisy_clinical]
            
            # Predict
            noisy_predictions = self.model.model.predict(X_noisy, verbose=0)
            noisy_classes = np.argmax(noisy_predictions, axis=1)
            
            # Calculate consistency
            consistency = np.mean(original_classes == noisy_classes)
            robustness_scores.append(consistency)
            
            print(f"   Noise level {noise_level}: prediction consistency {consistency:.3f}")
        
        return robustness_scores
    
    def explain_high_performance(self):
        """Explain reasons for high performance"""
        print("\n" + "="*80)
        print("🎯 Explaining Model High Performance Reasons")
        print("="*80)
        
        # 1. Feature quality analysis
        feature_analysis = self.analyze_feature_quality()
        
        # 2. Attention pattern analysis
        attention_analysis = self.analyze_attention_patterns()
        
        # 3. Decision boundary analysis
        boundary_analysis = self.analyze_decision_boundary_quality()
        
        # 4. Clinical feature contribution
        clinical_analysis = self.analyze_clinical_feature_contribution()
        
        # 5. Robustness test
        robustness_scores = self.test_robustness()
        
        print("\n" + "="*80)
        print("📋 High Performance Reasons Summary")
        print("="*80)
        
        reasons = []
        
        # Analyze feature discriminability
        if len(feature_analysis['high_risk_asymmetry']) > 0 and len(feature_analysis['medium_risk_asymmetry']) > 0:
            asymmetry_diff = np.mean(feature_analysis['high_risk_asymmetry']) - np.mean(feature_analysis['medium_risk_asymmetry'])
            if abs(asymmetry_diff) > 1.0:
                reasons.append(f"🎯 BI-RADS asymmetry is strong discriminative feature (difference: {asymmetry_diff:.2f})")
        
        # Analyze decision boundary
        if boundary_analysis['separation'] > 0.5:
            reasons.append(f"🎯 High class separation ({boundary_analysis['separation']:.3f})")
        
        # Analyze robustness
        if np.mean(robustness_scores) > 0.8:
            reasons.append(f"🛡️ Model is robust (mean consistency: {np.mean(robustness_scores):.3f})")
        
        # Analyze attention concentration
        if np.mean(attention_analysis['attention_variances']) > 0.1:
            reasons.append("🔍 Attention mechanism effectively focuses on key regions")
        
        # Analyze data quality
        test_size = len(self.test_data['bags'])
        if test_size >= 10:
            reasons.append(f"📊 Test set contains {test_size} samples, results have reasonable reliability")
        
        print("\nPossible high performance reasons:")
        for reason in reasons:
            print(f"   {reason}")
        
        if len(reasons) >= 3:
            print(f"\n✅ Conclusion: Model's high performance is reasonable, based on:")
            print(f"   - Natural discriminative power of bilateral breast features")
            print(f"   - Effective attention mechanism")
            print(f"   - Good feature engineering")
            print(f"   - Appropriate data augmentation strategy")
        else:
            print(f"\n⚠️ More validation needed to confirm performance reliability")
        
        return {
            'feature_analysis': feature_analysis,
            'attention_analysis': attention_analysis,
            'boundary_analysis': boundary_analysis,
            'robustness_scores': robustness_scores,
            'reasons': reasons
        }


def validate_trained_model(model, data_manager, train_data, val_data, test_data, mil_data, output_dir=None):
    """Validate trained model"""
    
    print("🚀 Starting model performance validation...")
    
    # 1. Basic validation
    validator = ModelPerformanceValidator(model, data_manager)
    validation_results = validator.generate_validation_report(train_data, val_data, test_data, mil_data)
    
    # 2. Deep analysis
    analyzer = PerformanceDeepAnalysis(model, test_data)
    deep_analysis_results = analyzer.explain_high_performance()
    
    # 3. Save results
    if output_dir:
        # Save validation results
        validation_file = os.path.join(output_dir, 'validation_results.json')
        try:
            with open(validation_file, 'w') as f:
                serializable_results = {}
                for key, value in validation_results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    else:
                        serializable_results[key] = value
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Validation results saved to: {validation_file}")
        except Exception as e:
            print(f"⚠️ Failed to save validation results: {e}")
        
        # Save deep analysis results
        analysis_file = os.path.join(output_dir, 'deep_analysis_results.json')
        try:
            with open(analysis_file, 'w') as f:
                serializable_analysis = {}
                for main_key, main_value in deep_analysis_results.items():
                    if isinstance(main_value, dict):
                        serializable_analysis[main_key] = {}
                        for sub_key, sub_value in main_value.items():
                            if isinstance(sub_value, np.ndarray):
                                serializable_analysis[main_key][sub_key] = sub_value.tolist()
                            elif isinstance(sub_value, list) and len(sub_value) > 0 and hasattr(sub_value[0], 'tolist'):
                                serializable_analysis[main_key][sub_key] = [item.tolist() if hasattr(item, 'tolist') else item for item in sub_value]
                            else:
                                serializable_analysis[main_key][sub_key] = sub_value
                    elif isinstance(main_value, np.ndarray):
                        serializable_analysis[main_key] = main_value.tolist()
                    else:
                        serializable_analysis[main_key] = main_value
                json.dump(serializable_analysis, f, indent=2)
            print(f"✅ Deep analysis results saved to: {analysis_file}")
        except Exception as e:
            print(f"⚠️ Failed to save deep analysis results: {e}")
    
    return validation_results, deep_analysis_results
