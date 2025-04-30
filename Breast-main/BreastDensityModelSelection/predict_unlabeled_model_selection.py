# Breast density prediction script

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import cv2
import shutil
from tqdm import tqdm

MODEL_PATH = "D:/Desktop/full_model_selection/best_selected_model.pth"
EXCEL_PATH = "D:/Desktop/breast cancer.xlsx"
OUTPUT_DIR = "D:/Desktop/full_model_selection/prediction_results"
DATA_ROOT = "D:/Dataset_only_breast"
SEGMENTATION_ROOT = "D:/Dataset_only_breast_segmentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_and_fix_model_path(model_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Copy the model file to the target location
    shutil.copy2(model_path, target_path)
    print(f"The model files have been copied to: {target_path}")
    return True

def load_patient_data(excel_path):
    
    df = pd.read_excel(excel_path)
    
    # Keep only necessary columns
    df = df[['PID', 'density']]
    
    # Dividing labeled and unlabeled patients
    labeled_patients = df.dropna(subset=['density']).copy()
    unlabeled_patients = df[df['density'].isna()].copy()
    
    print(f"找到 {len(labeled_patients)} 个有密度标签的患者")
    print(f"找到 {len(unlabeled_patients)} 个未标记的患者")
    
    if len(labeled_patients) > 0:
        density_counts = labeled_patients['density'].value_counts()
        print("Density label distribution:")
        print(density_counts)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labeled_patients['density_encoded'] = le.fit_transform(labeled_patients['density'])

        encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"Tag encoder saved to: {encoder_path}")
    
    return labeled_patients, unlabeled_patients

def load_model(model_path, device):
    
    checkpoint = torch.load(model_path, map_location=device)
                    
    model_name == 'BreastDensity3DNet':
    from improved_bd_pipeline_model_selection import BreastDensity3DNet
    model = BreastDensity3DNet().to(device)

        
    # Loading model weights
    if 'model_state_dict' in checkpoint:
         model.load_state_dict(checkpoint['model_state_dict'])
     elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    # Set to evaluation mode
     model.eval()

    if 'accuracy' in checkpoint:
        print(f"Model Accuracy: {checkpoint['accuracy']:.4f}")
        
    return model

def predict_patient(patient_id, model, device, segmentation_root):
    """Predictions for a single patient"""
    
    # Find the segmentation result directory
    seg_path = segmentation_root if isinstance(segmentation_root, Path) else Path(segmentation_root)
    patient_seg_dirs = list(seg_path.glob(f"*{patient_id}*"))
    
    # Use the first matching directory
    patient_seg_dir = patient_seg_dirs[0]
    
    # Find Series Catalog
    series_dirs = [d for d in patient_seg_dir.iterdir() if d.is_dir()]
    
    # Use the first series directory
    series_dir = series_dirs[0]

    breast_mask_path = series_dir / "breast_mask.npy"
    glandular_mask_path = series_dir / "glandular_tissue_mask.npy"
    
    try:
        breast_mask = np.load(breast_mask_path)
        glandular_mask = np.load(glandular_mask_path)
        
        # Find the section containing breast tissue
        breast_slice_indices = np.where(np.sum(breast_mask, axis=(1, 2)) > 0)[0]
            
        # Select the central area of ​​the breast volume
        mid_idx = len(breast_slice_indices) // 2
        start_idx = max(0, mid_idx - 5)
        end_idx = min(breast_mask.shape[0], start_idx + 10)

        selected_indices = breast_slice_indices[start_idx:end_idx]
        
        # Creating a multi-channel 2D input
        processed_breast_mask = np.zeros((10, 256, 256), dtype=np.float32)
        processed_glandular_mask = np.zeros((10, 256, 256), dtype=np.float32)
        
        # Fill available slices
        for i, slice_idx in enumerate(selected_indices):
            if i >= 10:  # Limited to 10 channels
                break
                
            breast_slice = breast_mask[slice_idx]
            glandular_slice = glandular_mask[slice_idx]
            
            # resize
            breast_slice_resized = cv2.resize(breast_slice.astype(np.float32), (256, 256))
            glandular_slice_resized = cv2.resize(glandular_slice.astype(np.float32), (256, 256))
            
            processed_breast_mask[i] = breast_slice_resized
            processed_glandular_mask[i] = glandular_slice_resized
        
        # Create combined input: 20 channels
        input_tensor = np.concatenate([processed_breast_mask, processed_glandular_mask], axis=0)
        
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)
        
        # Using the model to predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                le = pickle.load(f)
        else:
            # Creating a simple label map
            le = type('LabelEncoder', (), {'inverse_transform': lambda self, x: ['A', 'B', 'C', 'D'][x[0]]})()
        
        density_class = le.inverse_transform([predicted.item()])[0]
        
        # Calculate class probabilities
        probs = probabilities.cpu().numpy()[0]
        prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
        
        # Extract volume ratio and other information
        stats = {
            'breast_volume': float(np.sum(breast_mask)),
            'glandular_volume': float(np.sum(glandular_mask)),
            'density_ratio': float(np.sum(glandular_mask) / np.sum(breast_mask)) if np.sum(breast_mask) > 0 else 0
        }
        
        print(f"预测结果: {density_class} (置信度: {max(prob_dict.values()):.4f})")
        
        return {
            'patient_id': patient_id,
            'predicted_class': density_class,
            'class_probabilities': prob_dict,
            'stats': stats
        }
        
    except Exception as e:
        print(f"预测患者 {patient_id} 时出错: {e}")
        return None

def predict_all_unlabeled(unlabeled_patients, model, device, segmentation_root):
    """Predict all unlabeled patients"""    
    results = []
    
    for idx, row in tqdm(unlabeled_patients.iterrows(), total=len(unlabeled_patients), desc="预测进度"):
        patient_id = row['PID']
        
        # Predict density
        prediction = predict_patient(patient_id, model, device, segmentation_root)
        
        if prediction:
            results.append(prediction)
    
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
        
        # save results
        output_file = os.path.join(OUTPUT_DIR, "unlabeled_predictions.csv")
        results_df.to_csv(output_file, index=False)
        
        # Calculate the predicted proportion of rare classes
        rare_count = results_df['predicted_density'].isin(['A','B', 'D']).sum()
        rare_percentage = (rare_count / len(results_df)) * 100
        print(f"成功为 {len(results_df)} 名患者预测乳腺密度")
        
        create_visualizations(results_df, OUTPUT_DIR)
        
        return results_df
    else:
        print("No patient was successfully predicted")
        return None

def create_visualizations(results_df, output_dir):
    """Creating a prediction visualization"""

    charts_dir = os.path.join(output_dir, "prediction_charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. Pie Chart
    plt.figure(figsize=(10, 8))
    density_counts = results_df['predicted_density'].value_counts()
    colors = {'A': '#66c2a5', 'B': '#fc8d62', 'C': '#8da0cb', 'D': '#e78ac3'}
    pie_colors = [colors.get(label, '#cccccc') for label in density_counts.index]
    
    wedges, texts, autotexts = plt.pie(
        density_counts, 
        labels=density_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        explode=[0.05 if label in ['A', 'B', 'D'] else 0 for label in density_counts.index]
    )
    
    plt.setp(autotexts, size=10, weight='bold')
    plt.title('Predicting breast density distribution', fontsize=14)
    plt.savefig(os.path.join(charts_dir, "density_distribution_pie.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Density Ratio Box Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='predicted_density', y='density_ratio', data=results_df, palette=colors)
    sns.stripplot(x='predicted_density', y='density_ratio', data=results_df, 
                 size=4, color='black', alpha=0.5, jitter=True)
    plt.title('Density ratio distribution of different density categories', fontsize=14)
    plt.xlabel('Predicted density class', fontsize=12)
    plt.ylabel('Density ratio (glandular volume/breast volume)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(charts_dir, "density_ratio_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print(f"Start Breast Density Prediction...")
    
    labeled_patients, unlabeled_patients = load_patient_data(EXCEL_PATH)
    
    model = load_model(MODEL_PATH, device)
    
    # Predict all unlabeled patients
    results_df = predict_all_unlabeled(unlabeled_patients, model, device, SEGMENTATION_ROOT)
    
    print("预测完成!")

if __name__ == "__main__":
    main()
